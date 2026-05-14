"""
Siamese Network — shared-weight dual encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_learning.models.backbone import TinyCNN
from deep_learning.models.embedding_head import EmbeddingHead


class SiameseNetwork(nn.Module):
    """
    Shared-weight dual encoder.
    Takes (img_a, img_b), produces (z_a, z_b).
    Supports cosine and euclidean similarity.
    """
    
    def __init__(self, backbone=None, embedding_head=None, 
                 embedding_dim: int = 64, similarity: str = "cosine"):
        super().__init__()
        self.backbone = backbone or TinyCNN(in_channels=3)
        backbone_dim = 256  # TinyCNN output
        self.embedding_head = embedding_head or EmbeddingHead(
            input_dim=backbone_dim, embedding_dim=embedding_dim
        )
        self.similarity = similarity
    
    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding for a single image."""
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        return embedding
    
    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor):
        """Forward pass for a pair."""
        z_a = self.forward_single(img_a)
        z_b = self.forward_single(img_b)
        return z_a, z_b
    
    def compute_similarity(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two embedding vectors."""
        if self.similarity == "cosine":
            return F.cosine_similarity(z_a, z_b, dim=1)
        else:
            return -torch.sqrt(torch.sum((z_a - z_b) ** 2, dim=1) + 1e-8)
    
    def compute_distance(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        """Compute distance between two embedding vectors."""
        if self.similarity == "cosine":
            return 1.0 - F.cosine_similarity(z_a, z_b, dim=1)
        else:
            return torch.sqrt(torch.sum((z_a - z_b) ** 2, dim=1) + 1e-8)
