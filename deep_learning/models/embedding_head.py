"""
Embedding Head — MLP projection with dropout and L2 normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingHead(nn.Module):
    """MLP projection: input_dim -> 128 -> embedding_dim, with L2 normalization."""
    
    def __init__(self, input_dim: int = 256, embedding_dim: int = 64, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        return x
