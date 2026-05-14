"""
Retrieval and metric-learning evaluation metrics.
The Siamese verifier is fundamentally evaluated as a semantic retrieval/reranking system,
NOT as a standalone classifier.
"""

import numpy as np
from typing import List, Dict


def compute_pair_distances(pos_dists: np.ndarray, neg_dists: np.ndarray) -> dict:
    """Positive vs negative pair distance statistics."""
    return {
        "pos_mean": float(np.mean(pos_dists)) if len(pos_dists) > 0 else 0,
        "pos_std": float(np.std(pos_dists)) if len(pos_dists) > 0 else 0,
        "neg_mean": float(np.mean(neg_dists)) if len(neg_dists) > 0 else 0,
        "neg_std": float(np.std(neg_dists)) if len(neg_dists) > 0 else 0,
        "separation": float(np.mean(neg_dists) - np.mean(pos_dists)) if len(pos_dists) > 0 and len(neg_dists) > 0 else 0
    }


def compute_triplet_margin(d_pos: np.ndarray, d_neg: np.ndarray, margin: float = 0.5) -> dict:
    """Average margin satisfaction."""
    violations = np.sum(d_pos - d_neg + margin > 0)
    return {
        "avg_margin": float(np.mean(d_neg - d_pos)),
        "violation_rate": float(violations / len(d_pos)) if len(d_pos) > 0 else 0
    }


def compute_embedding_separation(pos_embeddings: np.ndarray, neg_embeddings: np.ndarray) -> dict:
    """Inter-class vs intra-class distance ratio."""
    if len(pos_embeddings) < 2 or len(neg_embeddings) < 2:
        return {"ratio": 0.0}
    
    from scipy.spatial.distance import pdist
    intra_pos = np.mean(pdist(pos_embeddings))
    intra_neg = np.mean(pdist(neg_embeddings))
    
    from scipy.spatial.distance import cdist
    inter = np.mean(cdist(pos_embeddings, neg_embeddings))
    
    intra_avg = (intra_pos + intra_neg) / 2
    return {
        "intra_class": float(intra_avg),
        "inter_class": float(inter),
        "ratio": float(inter / (intra_avg + 1e-8))
    }


def compute_recall_at_k(rankings: List[int], ground_truth: set, k_values: List[int] = [1, 3, 5, 10]) -> dict:
    """
    Recall@K: fraction of true MR symbols within top-K ranked candidates.
    Rankings is a list of candidate indices sorted by score (best first).
    Ground_truth is the set of indices that are true positives.
    """
    results = {}
    for k in k_values:
        top_k = set(rankings[:k])
        hits = len(top_k & ground_truth)
        recall = hits / len(ground_truth) if ground_truth else 0
        results[f"recall@{k}"] = float(recall)
    return results


def compute_precision_at_k(rankings: List[int], ground_truth: set, k_values: List[int] = [1, 3, 5, 10]) -> dict:
    """Precision@K: fraction of top-K that are true positives."""
    results = {}
    for k in k_values:
        top_k = set(rankings[:k])
        hits = len(top_k & ground_truth)
        precision = hits / k if k > 0 else 0
        results[f"precision@{k}"] = float(precision)
    return results


def compute_mrr(rankings: List[int], ground_truth: set) -> float:
    """Mean Reciprocal Rank: 1/rank of first correct result."""
    for i, idx in enumerate(rankings):
        if idx in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def compute_map_lite(rankings: List[int], ground_truth: set) -> float:
    """Lightweight mean average precision for candidate ranking."""
    if not ground_truth:
        return 0.0
    
    hits = 0
    sum_precision = 0.0
    
    for i, idx in enumerate(rankings):
        if idx in ground_truth:
            hits += 1
            sum_precision += hits / (i + 1)
    
    return sum_precision / len(ground_truth)
