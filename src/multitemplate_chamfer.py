import numpy as np
import cv2
import os
import json
from typing import List, Dict, Tuple
from src.chamfer_matching import extract_edge_coordinates, dense_chamfer_search

def generate_chamfer_template_ensemble(template_binary: np.ndarray) -> List[Dict]:
    """Generates the orthogonal 7-template ensemble."""
    templates = []
    h, w = template_binary.shape
    
    # 1. Baseline
    templates.append({
        "id": "t0_baseline",
        "type": "baseline",
        "image": template_binary.copy()
    })
    
    # 2. Scale
    for scale in [0.95, 1.05]:
        new_w, new_h = int(w * scale), int(h * scale)
        scaled = cv2.resize(template_binary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        padded = np.zeros_like(template_binary)
        y_off = (h - new_h) // 2
        x_off = (w - new_w) // 2
        
        src_y_start = max(0, -y_off)
        src_x_start = max(0, -x_off)
        src_y_end = min(new_h, new_h - max(0, (new_h - h) - (-y_off if y_off < 0 else 0))) # Keep it simple
        src_x_end = min(new_w, new_w - max(0, (new_w - w) - (-x_off if x_off < 0 else 0)))
        
        # A simpler robust centering logic
        dy = (new_h - h) // 2
        dx = (new_w - w) // 2
        
        if scale > 1.0:
            padded = scaled[dy:dy+h, dx:dx+w]
        else:
            padded[-dy:-dy+new_h, -dx:-dx+new_w] = scaled
            
        templates.append({
            "id": f"t{len(templates)}_{'scale0.95' if scale < 1 else 'scale1.05'}",
            "type": "scale",
            "image": padded
        })
        
    # 3. Thickness
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(template_binary, kernel, iterations=1)
    dilated = cv2.dilate(template_binary, kernel, iterations=1)
    
    templates.append({
        "id": f"t{len(templates)}_eroded",
        "type": "thickness",
        "image": eroded
    })
    templates.append({
        "id": f"t{len(templates)}_dilated",
        "type": "thickness",
        "image": dilated
    })
    
    # 4. Rotation
    center = (w // 2, h // 2)
    for angle in [-3.0, 3.0]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(template_binary, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        templates.append({
            "id": f"t{len(templates)}_rot{'+3' if angle > 0 else '-3'}",
            "type": "rotation",
            "image": rotated
        })
        
    return templates

def run_ensemble_chamfer_search(diagram_shape: tuple, template_ensemble: List[Dict], diagram_dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray]]:
    h, w = diagram_shape[:2]
    
    ensemble_score_map = np.full((h, w), np.inf, dtype=np.float32)
    ensemble_coverage_map = np.zeros((h, w), dtype=np.float32)
    ensemble_template_index_map = np.zeros((h, w), dtype=np.int32)
    
    per_template_scores = []
    
    for i, t in enumerate(template_ensemble):
        t_edges = extract_edge_coordinates(t["image"])
        score_map, coverage_map = dense_chamfer_search(diagram_dt, t_edges, t["image"].shape)
        
        per_template_scores.append(score_map)
        
        # Aggregation using MIN
        mask = score_map < ensemble_score_map
        ensemble_score_map[mask] = score_map[mask]
        ensemble_coverage_map[mask] = coverage_map[mask]
        ensemble_template_index_map[mask] = i
        
    return ensemble_score_map, ensemble_coverage_map, ensemble_template_index_map, per_template_scores

def extract_ensemble_detections(ensemble_score_map: np.ndarray, ensemble_coverage_map: np.ndarray, ensemble_template_index_map: np.ndarray, template_ensemble: List[Dict], coverage_threshold: float, distance_threshold: float) -> List[Dict]:
    from src.nms_refinement import extract_local_minima
    from src.nms_refinement import apply_spatial_nms
    
    raw_minima = extract_local_minima(ensemble_score_map, ensemble_coverage_map)
    
    nms_detections = apply_spatial_nms(raw_minima)
    
    for d in nms_detections:
        x, y = d["x"], d["y"]
        winning_idx = ensemble_template_index_map[y, x]
        t = template_ensemble[winning_idx]
        d["winning_template_id"] = t["id"]
        d["augmentation_type"] = t["type"]
        d["ensemble_score"] = float(d["mean_distance"])
        
    return nms_detections

def track_minima_emergence(ensemble_detections: List[Dict], baseline_detections: List[Dict], threshold: float = 40.0) -> List[Dict]:
    """Tracks which detections are genuinely new minima."""
    for ed in ensemble_detections:
        ex, ey = ed["x"], ed["y"]
        min_dist = float('inf')
        for bd in baseline_detections:
            bx, by = bd["x"], bd["y"]
            dist = np.sqrt((ex - bx)**2 + (ey - by)**2)
            if dist < min_dist:
                min_dist = dist
        
        ed["distance_to_nearest_baseline"] = float(min_dist)
        ed["new_minimum"] = bool(min_dist > threshold)
        
    return ensemble_detections
