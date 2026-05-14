"""
Backbone architectures for Siamese embedding.
TinyCNN is the default — deliberately small for the tiny dataset.
"""

import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    """4-layer CNN backbone. Input: 3x64x64, Output: 256-dim feature vector."""
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3x64x64 -> 32x32x32
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2: 32x32x32 -> 64x16x16
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3: 64x16x16 -> 128x8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4: 128x8x8 -> 256x4x4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)  # (B, 256)


class ResNet18Backbone(nn.Module):
    """Optional frozen ResNet18 with modified first conv for 3-channel symbolic input."""
    
    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        # Keep the 3-channel input (our symbolic channels match)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.output_dim = 512
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.view(x.size(0), -1)  # (B, 512)
