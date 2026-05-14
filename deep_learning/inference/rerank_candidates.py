"""
Inference: Rerank classical pipeline candidates using trained Siamese verifier.
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from deep_learning.utils.patch_extraction import extract_classical_candidates, prepare_multichannel_input
from deep_learning.models.siamese_network import SiameseNetwork
from deep_learning.datasets.transforms import SymbolicTransform
from deep_learning.training.metrics import compute_recall_at_k, compute_precision_at_k, compute_mrr, compute_map_lite


def main():
    print("=" * 60)
    print("SIAMESE RERANKING OF CLASSICAL PIPELINE CANDIDATES")
    print("=" * 60)
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    output_dir = os.path.join(dl_root, "outputs", "ranked_candidates")
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = SiameseNetwork(embedding_dim=64, similarity="cosine").to(device)
    ckpt_path = os.path.join(dl_root, "experiments", "checkpoints", "best_model.pth")
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded model from {ckpt_path}")
    else:
        print(f"[WARNING] No checkpoint found at {ckpt_path}. Using random weights.")
    
    model.eval()
    
    # Extract candidates
    data = extract_classical_candidates(project_root)
    transform = SymbolicTransform()
    
    # Compute template embedding
    template_tensor = transform(data["template_binary"]).unsqueeze(0).to(device)
    with torch.no_grad():
        template_emb = model.forward_single(template_tensor)
    
    # Compute candidate embeddings and similarities
    results = []
    for c in data["candidates"]:
        crop_tensor = transform(c["crop"]).unsqueeze(0).to(device)
        with torch.no_grad():
            cand_emb = model.forward_single(crop_tensor)
            sim = model.compute_similarity(template_emb, cand_emb).item()
        
        # Fused score: 0.45 * chamfer + 0.25 * pca + 0.30 * siamese
        fused = (0.45 * c.get("chamfer_similarity", 0) +
                 0.25 * c.get("pca_similarity", 0) +
                 0.30 * max(0, sim))
        
        results.append({
            "x": c["x"],
            "y": c["y"],
            "original_rank": c["rank"],
            "chamfer_similarity": c.get("chamfer_similarity", 0),
            "pca_similarity": c.get("pca_similarity", 0),
            "siamese_similarity": float(sim),
            "fused_score": float(fused),
            "failure_attributions": c.get("failure_attributions", [])
        })
    
    # Rerank by fused score
    results = sorted(results, key=lambda r: r["fused_score"], reverse=True)
    for i, r in enumerate(results):
        r["siamese_rank"] = i + 1
    
    # Compute retrieval metrics (top-3 candidates are assumed positives)
    rankings = list(range(len(results)))
    ground_truth = set()
    for i, r in enumerate(results):
        if r["original_rank"] <= 3:
            ground_truth.add(i)
    
    recall = compute_recall_at_k(rankings, ground_truth)
    precision = compute_precision_at_k(rankings, ground_truth)
    mrr = compute_mrr(rankings, ground_truth)
    map_score = compute_map_lite(rankings, ground_truth)
    
    # Save
    output = {
        "candidates": results,
        "retrieval_metrics": {
            **recall, **precision,
            "mrr": mrr,
            "map_lite": map_score
        }
    }
    
    out_path = os.path.join(output_dir, "siamese_reranked.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nReranked {len(results)} candidates.")
    print(f"Retrieval metrics: {json.dumps(output['retrieval_metrics'], indent=2)}")
    print(f"\nTop 5 Siamese-Reranked:")
    for r in results[:5]:
        print(f"  Rank {r['siamese_rank']} (x:{r['x']},y:{r['y']}) | "
              f"Fused: {r['fused_score']:.4f} | Siamese: {r['siamese_similarity']:.4f} | "
              f"Orig: #{r['original_rank']}")
    
    print(f"\nResults saved to: {out_path}")
    
    # ========================================================
    # AUTO-GENERATE FINAL VISUALIZATIONS
    # ========================================================
    from deep_learning.utils.visualization import (
        save_final_detection_overlay,
        save_ranked_overlay,
        save_detection_crop_grid,
        save_template_retrieval
    )
    
    viz_dir = os.path.join(dl_root, "outputs", "final_visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    diagram_path = os.path.join(project_root, "Data", "circuit_diagram.png")
    template_path = os.path.join(project_root, "Data", "symbol.png")
    template_shape = data["template_shape"]
    
    print("\nGenerating final visualizations...")
    
    # 1. Full detection overlay
    save_final_detection_overlay(diagram_path, results, template_shape,
                                  os.path.join(viz_dir, "final_siamese_detections_overlay.png"))
    print("  Generated final_siamese_detections_overlay.png")
    
    # 2. Top-K overlays
    for k in [5, 10, 17]:
        save_ranked_overlay(diagram_path, results, template_shape, k,
                            os.path.join(viz_dir, f"top{k}_overlay.png"))
        print(f"  Generated top{k}_overlay.png")
    
    # 3. Detection crop grid
    save_detection_crop_grid(diagram_path, results, template_shape,
                              os.path.join(viz_dir, "detection_crop_grid.png"))
    print("  Generated detection_crop_grid.png")
    
    # 4. Template retrieval visualization
    save_template_retrieval(template_path, diagram_path, results, template_shape,
                             os.path.join(viz_dir, "template_retrieval_ranking.png"))
    print("  Generated template_retrieval_ranking.png")
    
    # 5. Final analysis JSON
    analysis = {
        "total_candidates": len(results),
        "template_shape": list(template_shape),
        "retrieval_metrics": output["retrieval_metrics"],
        "top5_detections": [
            {"rank": r["siamese_rank"], "x": r["x"], "y": r["y"],
             "siamese_similarity": r["siamese_similarity"],
             "fused_score": r["fused_score"],
             "original_rank": r["original_rank"]}
            for r in results[:5]
        ],
        "confidence_statistics": {
            "mean_fused": float(np.mean([r["fused_score"] for r in results])),
            "max_fused": float(max(r["fused_score"] for r in results)),
            "min_fused": float(min(r["fused_score"] for r in results)),
            "mean_siamese": float(np.mean([r["siamese_similarity"] for r in results])),
            "max_siamese": float(max(r["siamese_similarity"] for r in results)),
            "min_siamese": float(min(r["siamese_similarity"] for r in results)),
        },
        "visualization_paths": [
            "final_siamese_detections_overlay.png",
            "top5_overlay.png", "top10_overlay.png", "top17_overlay.png",
            "detection_crop_grid.png",
            "template_retrieval_ranking.png"
        ]
    }
    
    analysis_path = os.path.join(dl_root, "outputs", "final_analysis.json")
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFinal analysis saved to: {analysis_path}")
    print(f"All visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()
