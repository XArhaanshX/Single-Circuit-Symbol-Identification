"""
Patch Extraction Utility
Extracts candidate crops from classical CV pipeline outputs for Siamese training.
Does NOT modify any classical pipeline files or outputs.
"""

import os
import json
import cv2
import numpy as np
import skimage.morphology


def extract_classical_candidates(project_root: str) -> dict:
    """
    Reads classical pipeline output JSONs and extracts candidate crops
    from the original diagram image.
    """
    diagram_path = os.path.join(project_root, "Data", "circuit_diagram.png")
    template_path = os.path.join(project_root, "Data", "symbol.png")
    topo_json = os.path.join(project_root, "outputs", "diagram", "topology_verification", "topology_statistics.json")
    
    diagram = cv2.imread(diagram_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    
    if diagram is None or template is None:
        raise FileNotFoundError("Could not load diagram or template images.")
    
    th, tw = template.shape[:2]
    
    candidates = []
    
    if os.path.exists(topo_json):
        with open(topo_json, "r") as f:
            topo_data = json.load(f)
        
        for c in topo_data.get("candidates", []):
            cx, cy = c["x"], c["y"]
            rank = c["rank"]
            
            crop = _extract_crop(diagram, cx, cy, th, tw)
            
            candidates.append({
                "crop": crop,
                "x": cx,
                "y": cy,
                "rank": rank,
                "source": "topology",
                "chamfer_similarity": c.get("chamfer_similarity", 0),
                "pca_similarity": c.get("pca_similarity", 0),
                "topology_similarity": c.get("topology_similarity", 0),
                "fused_score": c.get("fused_score_topology", 0),
                "failure_attributions": c.get("failure_attributions", [])
            })
    
    return {
        "template": template,
        "template_binary": (template > 127).astype(np.uint8) * 255,
        "diagram": diagram,
        "candidates": candidates,
        "template_shape": (th, tw)
    }


def _extract_crop(image: np.ndarray, cx: int, cy: int, th: int, tw: int) -> np.ndarray:
    """Extract a crop centered at (cx, cy) with template dimensions."""
    half_h, half_w = th // 2, tw // 2
    h, w = image.shape[:2]
    
    y1 = max(0, cy - half_h)
    y2 = min(h, cy + half_h + (th % 2))
    x1 = max(0, cx - half_w)
    x2 = min(w, cx + half_w + (tw % 2))
    
    crop = np.zeros((th, tw), dtype=np.uint8)
    c_y1 = half_h - (cy - y1)
    c_y2 = half_h + (y2 - cy)
    c_x1 = half_w - (cx - x1)
    c_x2 = half_w + (x2 - cx)
    
    crop[c_y1:c_y2, c_x1:c_x2] = image[y1:y2, x1:x2]
    return crop


def build_positive_set(template_binary: np.ndarray, candidates: list, top_k: int = 3) -> list:
    """
    Constructs positive samples from:
    A) The original template
    B) High-confidence candidate crops (top-K by rank)
    """
    positives = []
    
    # A) Template itself
    positives.append({
        "crop": template_binary.copy(),
        "label": "positive",
        "source": "template",
        "group_id": "template_base"
    })
    
    # B) Top-K candidates as verified positives
    sorted_cands = sorted(candidates, key=lambda c: c["rank"])
    for c in sorted_cands[:top_k]:
        crop_bin = (c["crop"] > 127).astype(np.uint8) * 255
        positives.append({
            "crop": crop_bin,
            "label": "positive",
            "source": f"candidate_rank_{c['rank']}",
            "group_id": f"candidate_{c['x']}_{c['y']}"
        })
    
    return positives


def build_negative_set(candidates: list, bottom_start: int = 10) -> list:
    """
    Constructs hard negatives from low-confidence candidates.
    These are HARD NEGATIVES — the most valuable supervision signal.
    """
    negatives = []
    
    sorted_cands = sorted(candidates, key=lambda c: c["rank"])
    for c in sorted_cands:
        if c["rank"] >= bottom_start:
            crop_bin = (c["crop"] > 127).astype(np.uint8) * 255
            negatives.append({
                "crop": crop_bin,
                "label": "negative",
                "source": f"hard_negative_rank_{c['rank']}",
                "group_id": f"candidate_{c['x']}_{c['y']}",
                "failure_attributions": c.get("failure_attributions", [])
            })
    
    return negatives


def prepare_multichannel_input(crop_gray: np.ndarray, target_size: tuple = (64, 64)) -> np.ndarray:
    """
    Builds the 3-channel symbolic representation:
    Channel 0: Binary image
    Channel 1: Skeleton image
    Channel 2: Edge map (Canny)
    
    This preserves symbolic structure, topology hints, and geometric priors.
    """
    # Resize
    resized = cv2.resize(crop_gray, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Channel 0: Binary
    _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
    
    # Channel 1: Skeleton
    skeleton_bool = skimage.morphology.skeletonize(binary > 0)
    skeleton = (skeleton_bool * 255).astype(np.uint8)
    
    # Channel 2: Edge map
    edges = cv2.Canny(resized, 50, 150)
    
    # Stack into 3-channel image
    multichannel = np.stack([binary, skeleton, edges], axis=0)  # (3, H, W)
    
    return multichannel
