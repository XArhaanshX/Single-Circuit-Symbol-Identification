"""
Loss functions for metric learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """L = y * d² + (1-y) * max(0, margin - d)²"""
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        dist = F.pairwise_distance(z_a, z_b)
        loss = label * dist.pow(2) + (1 - label) * F.relu(self.margin - dist).pow(2)
        return loss.mean()


class TripletMarginLoss(nn.Module):
    """L = max(0, d(a,p) - d(a,n) + margin)"""
    
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        d_pos = F.pairwise_distance(anchor, positive)
        d_neg = F.pairwise_distance(anchor, negative)
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()
