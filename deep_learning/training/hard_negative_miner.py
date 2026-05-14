"""
Online Hard Negative Miner.
Dynamically identifies the most informative triplets within each batch.
"""

import torch
import torch.nn.functional as F


class OnlineHardNegativeMiner:
    """
    Supports hardest, semi-hard, and all-triplet mining strategies.
    Static hard negatives alone may not fully shape the embedding manifold
    because the remaining false positives are structurally subtle.
    """
    
    def __init__(self, strategy: str = "semihard", margin: float = 0.5):
        self.strategy = strategy
        self.margin = margin
    
    def mine(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Mine triplets from a batch of embeddings.
        Returns list of (anchor_idx, positive_idx, negative_idx) tuples.
        """
        if self.strategy == "hardest":
            return self.mine_hardest(embeddings, labels)
        elif self.strategy == "semihard":
            return self.mine_semihard(embeddings, labels)
        else:
            return self.mine_all(embeddings, labels)
    
    def mine_hardest(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Selects the negative with smallest distance to anchor."""
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        triplets = []
        
        for i in range(len(labels)):
            pos_mask = (labels == labels[i]) & (torch.arange(len(labels), device=labels.device) != i)
            neg_mask = labels != labels[i]
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            # Hardest positive: farthest positive
            pos_dists = dist_matrix[i][pos_mask]
            hardest_pos_idx = torch.where(pos_mask)[0][pos_dists.argmax()]
            
            # Hardest negative: nearest negative
            neg_dists = dist_matrix[i][neg_mask]
            hardest_neg_idx = torch.where(neg_mask)[0][neg_dists.argmin()]
            
            triplets.append((i, hardest_pos_idx.item(), hardest_neg_idx.item()))
        
        return triplets
    
    def mine_semihard(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Selects negatives farther than positive but within margin boundary."""
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)
        triplets = []
        
        for i in range(len(labels)):
            pos_mask = (labels == labels[i]) & (torch.arange(len(labels), device=labels.device) != i)
            neg_mask = labels != labels[i]
            
            if not pos_mask.any() or not neg_mask.any():
                continue
            
            pos_dists = dist_matrix[i][pos_mask]
            d_ap = pos_dists.max()
            
            neg_dists = dist_matrix[i][neg_mask]
            # Semi-hard: d_ap < d_an < d_ap + margin
            semihard_mask = (neg_dists > d_ap) & (neg_dists < d_ap + self.margin)
            
            if semihard_mask.any():
                semihard_dists = neg_dists[semihard_mask]
                semihard_neg = torch.where(neg_mask)[0][semihard_mask][semihard_dists.argmin()]
            else:
                # Fallback to hardest
                semihard_neg = torch.where(neg_mask)[0][neg_dists.argmin()]
            
            hardest_pos = torch.where(pos_mask)[0][pos_dists.argmax()]
            triplets.append((i, hardest_pos.item(), semihard_neg.item()))
        
        return triplets
    
    def mine_all(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Returns all valid triplets."""
        triplets = []
        for i in range(len(labels)):
            for j in range(len(labels)):
                if labels[j] != labels[i] or j == i:
                    continue
                for k in range(len(labels)):
                    if labels[k] == labels[i]:
                        continue
                    triplets.append((i, j, k))
        return triplets
