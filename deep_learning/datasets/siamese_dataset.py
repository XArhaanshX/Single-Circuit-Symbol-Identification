"""
Siamese and Triplet PyTorch Datasets.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from deep_learning.datasets.transforms import SymbolicTransform


class SiameseDataset(Dataset):
    """Returns (img1, img2, label) pairs. label=1 positive, label=0 negative."""
    
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform or SymbolicTransform()
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        crop1, crop2, label = self.pairs[idx]
        t1 = self.transform(crop1)
        t2 = self.transform(crop2)
        return t1, t2, torch.tensor(label, dtype=torch.float32)


class TripletDataset(Dataset):
    """Returns (anchor, positive, negative) triplets."""
    
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform or SymbolicTransform()
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor, pos, neg = self.triplets[idx]
        t_a = self.transform(anchor)
        t_p = self.transform(pos)
        t_n = self.transform(neg)
        return t_a, t_p, t_n


class EmbeddingDataset(Dataset):
    """Returns (image, label, group_id) for embedding extraction."""
    
    def __init__(self, samples, transform=None):
        self.samples = samples  # List of dicts with "crop", "label", "group_id"
        self.transform = transform or SymbolicTransform()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        t = self.transform(s["crop"])
        label = 1 if s["label"] == "positive" else 0
        return t, label, s.get("group_id", "unknown")
