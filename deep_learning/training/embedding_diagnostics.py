"""
Embedding Stability Diagnostics.
Mandatory to verify that semantic structure is genuinely emerging in embedding space.
Detects embedding collapse, overcompression, and unstable manifold formation.
"""

import os
import numpy as np
import torch


class EmbeddingDiagnostics:
    """Tracks embedding health metrics across training."""
    
    def __init__(self, collapse_threshold: float = 0.01):
        self.collapse_threshold = collapse_threshold
        self.history = []
    
    def compute_embedding_variance(self, embeddings: np.ndarray) -> float:
        """Per-dimension variance. Collapse -> variance -> 0."""
        return float(np.mean(np.var(embeddings, axis=0)))
    
    def compute_pairwise_distance_variance(self, embeddings: np.ndarray) -> float:
        """Variance of all pairwise distances. Collapse -> 0."""
        from scipy.spatial.distance import pdist
        if len(embeddings) < 2:
            return 0.0
        dists = pdist(embeddings)
        return float(np.var(dists))
    
    def compute_centroid_separation(self, pos_emb: np.ndarray, neg_emb: np.ndarray) -> float:
        """L2 distance between positive and negative cluster centroids."""
        if len(pos_emb) == 0 or len(neg_emb) == 0:
            return 0.0
        pos_centroid = np.mean(pos_emb, axis=0)
        neg_centroid = np.mean(neg_emb, axis=0)
        return float(np.linalg.norm(pos_centroid - neg_centroid))
    
    def compute_intra_class_spread(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Average within-class pairwise distance."""
        from scipy.spatial.distance import pdist
        spreads = []
        for lbl in np.unique(labels):
            cls_emb = embeddings[labels == lbl]
            if len(cls_emb) >= 2:
                spreads.append(float(np.mean(pdist(cls_emb))))
        return float(np.mean(spreads)) if spreads else 0.0
    
    def compute_inter_class_spread(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Average between-class pairwise distance."""
        from scipy.spatial.distance import cdist
        unique = np.unique(labels)
        if len(unique) < 2:
            return 0.0
        dists = []
        for i, l1 in enumerate(unique):
            for l2 in unique[i+1:]:
                e1 = embeddings[labels == l1]
                e2 = embeddings[labels == l2]
                if len(e1) > 0 and len(e2) > 0:
                    dists.append(float(np.mean(cdist(e1, e2))))
        return float(np.mean(dists)) if dists else 0.0
    
    def detect_collapse(self, embeddings: np.ndarray) -> bool:
        """Returns True if embedding variance falls below threshold."""
        var = self.compute_embedding_variance(embeddings)
        if var < self.collapse_threshold:
            print(f"[WARNING] Embedding collapse detected! Variance: {var:.6f}")
            return True
        return False
    
    def run_all(self, embeddings: np.ndarray, labels: np.ndarray) -> dict:
        """Run all diagnostics and return results."""
        pos_emb = embeddings[labels == 1]
        neg_emb = embeddings[labels == 0]
        
        metrics = {
            "embedding_variance": self.compute_embedding_variance(embeddings),
            "pairwise_dist_variance": self.compute_pairwise_distance_variance(embeddings),
            "centroid_separation": self.compute_centroid_separation(pos_emb, neg_emb),
            "intra_class_spread": self.compute_intra_class_spread(embeddings, labels),
            "inter_class_spread": self.compute_inter_class_spread(embeddings, labels),
            "collapsed": self.detect_collapse(embeddings)
        }
        
        self.history.append(metrics)
        return metrics
    
    def snapshot_embeddings(self, epoch: int, embeddings: np.ndarray, labels: np.ndarray, output_dir: str):
        """Save embedding snapshot for temporal evolution analysis."""
        snap_dir = os.path.join(output_dir, "embedding_snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        np.savez(os.path.join(snap_dir, f"epoch_{epoch:03d}.npz"),
                 embeddings=embeddings, labels=labels)
