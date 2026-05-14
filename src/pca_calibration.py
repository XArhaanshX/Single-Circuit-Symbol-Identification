import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Dict

def minmax_normalize(dists: np.ndarray) -> np.ndarray:
    min_d, max_d = dists.min(), dists.max()
    if max_d == min_d: return np.ones_like(dists)
    return 1.0 - ((dists - min_d) / (max_d - min_d))

def exp_normalize(dists: np.ndarray, alpha: float) -> np.ndarray:
    return np.exp(-alpha * dists)

def logistic_normalize(dists: np.ndarray, beta: float) -> np.ndarray:
    if len(dists) < 2: return np.ones_like(dists)
    mean_d = np.mean(dists)
    std_d = np.std(dists)
    if std_d == 0: return np.ones_like(dists)
    z = (dists - mean_d) / std_d
    return 1.0 / (1.0 + np.exp(beta * z))

def save_calibration_reranked_detections(output_dir: str, original_img: np.ndarray, detections: list, template_shape: tuple) -> None:
    vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(original_img.shape) == 2 else original_img.copy()
    th, tw = template_shape[:2]
    
    for c in detections:
        x, y = c["x"], c["y"]
        rank = c["new_rank"]
        cv2.rectangle(vis_img, (x, y), (x+tw, y+th), (255, 0, 255), 2)
        
        text1 = f"#{rank} F:{c['fused_score']:.2f}"
        text2 = f"C:{c['chamfer_similarity']:.2f} P:{c['pca_similarity']:.2f}"
        cv2.putText(vis_img, text1, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.putText(vis_img, text2, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
    cv2.imwrite(os.path.join(output_dir, "reranked_detections.png"), vis_img)

def save_calibration_crops(output_dir: str, detections: list) -> None:
    crops_dir = os.path.join(output_dir, "detection_crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    for c in detections:
        rank = c["new_rank"]
        orig = c["normalized_patch"].astype(np.uint8)
        recon = np.clip(c["reconstructed_patch"], 0, 255).astype(np.uint8)
        
        side_by_side = np.hstack((orig, recon))
        side_by_side[:, orig.shape[1]-1:orig.shape[1]+1] = 128
        
        sbs_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)
        
        cv2.putText(sbs_bgr, f"ID:{c['detection_id']} R:{rank}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        filename = f"rank_{rank:03d}_{c['detection_id']}.png"
        cv2.imwrite(os.path.join(crops_dir, filename), sbs_bgr)

def run_calibration_experiments(
    detections: List[Dict],
    candidate_patches: List[np.ndarray],
    pca_model,
    original_img: np.ndarray,
    template_shape: tuple,
    output_dir: str
):
    print("\nRunning Isolated PCA Score Calibration Experiments...")
    
    baseline_dists = np.array([d["mean_distance"] for d in detections])
    baseline_pca_sim = np.array([d["pca_similarity"] for d in detections])
    
    for d in detections:
        if "detection_id" not in d:
            d["detection_id"] = f"det_{d['pca_rank']:03d}"
        d["baseline_rank"] = d["pca_rank"]

    alphas = [0.5, 1.0, 2.0]
    betas = [0.5, 1.0, 2.0]
    weights = [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7)]
    
    calib_out_dir = os.path.join(output_dir, "pca_calibration_experiments")
    os.makedirs(calib_out_dir, exist_ok=True)
    
    global_summary = []
    
    def execute_experiment(norm_type, norm_param, cw, pw):
        if norm_type == "minmax":
            chamfer_sims = minmax_normalize(baseline_dists)
            param_str = "baseline"
        elif norm_type == "exponential":
            chamfer_sims = exp_normalize(baseline_dists, norm_param)
            param_str = f"alpha{norm_param}"
        elif norm_type == "logistic":
            chamfer_sims = logistic_normalize(baseline_dists, norm_param)
            param_str = f"beta{norm_param}"

        exp_name = f"exp_{norm_type}_{param_str}_w{cw}_{pw}"
        exp_dir = os.path.join(calib_out_dir, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        reranked = []
        for i, det in enumerate(detections):
            fused = cw * chamfer_sims[i] + pw * baseline_pca_sim[i]
            nd = det.copy()
            nd["chamfer_similarity"] = float(chamfer_sims[i])
            nd["fused_score"] = float(fused)
            reranked.append(nd)
            
        reranked.sort(key=lambda x: x["fused_score"], reverse=True)
        
        for r, d in enumerate(reranked, start=1):
            d["new_rank"] = r
            d["rank_delta"] = r - d["baseline_rank"]
            
        save_calibration_reranked_detections(exp_dir, original_img, reranked, template_shape)
        save_calibration_crops(exp_dir, reranked)
        
        ranking_table = []
        gb_delta = None
        mr_deltas = []
        for d in reranked:
            ranking_table.append({
                "detection_id": d["detection_id"],
                "rank": d["new_rank"],
                "x": d["x"],
                "y": d["y"],
                "chamfer_similarity": d["chamfer_similarity"],
                "pca_similarity": d["pca_similarity"],
                "fused_score": d["fused_score"],
                "baseline_rank": d["baseline_rank"],
                "rank_delta": d["rank_delta"]
            })
            if 600 < d["x"] < 700:
                gb_delta = d["rank_delta"]
            else:
                mr_deltas.append(d["rank_delta"])
                
        global_summary.append({
            "experiment_name": exp_name,
            "normalization_strategy": norm_type,
            "normalization_parameter": norm_param,
            "fusion_weights": {"chamfer": cw, "pca": pw},
            "gb_rank_delta": gb_delta,
            "avg_mr_rank_delta": float(np.mean(mr_deltas)) if mr_deltas else 0.0,
            "detections": ranking_table
        })

    for cw, pw in weights:
        execute_experiment("minmax", None, cw, pw)
        for a in alphas:
            execute_experiment("exponential", a, cw, pw)
        for b in betas:
            execute_experiment("logistic", b, cw, pw)
            
    with open(os.path.join(calib_out_dir, "global_summary.json"), "w") as f:
        json.dump(global_summary, f, indent=2)
        
    return calib_out_dir, global_summary
