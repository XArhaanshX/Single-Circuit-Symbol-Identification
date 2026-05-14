"""
Group-Aware Split Manager.
Evaluation integrity is critical due to the tiny dataset size and aggressive augmentation reuse.
All augmentations originating from the same base crop must remain within the SAME split.
"""

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


class SplitManager:
    """
    Assigns each base crop a unique group_id.
    All augmented variants sharing a group_id are kept in the same split.
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.groups = defaultdict(list)  # group_id -> list of sample indices
    
    def assign_groups(self, samples: List[Dict]) -> List[Dict]:
        """Tags every sample with its source group."""
        self.groups.clear()
        for i, s in enumerate(samples):
            gid = s.get("group_id", f"auto_{i}")
            s["group_id"] = gid
            self.groups[gid].append(i)
        return samples
    
    def split_by_group(self, samples: List[Dict],
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Performs group-aware splitting.
        No group straddles train/val/test boundaries.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        self.assign_groups(samples)
        
        group_ids = list(self.groups.keys())
        random.seed(self.seed)
        random.shuffle(group_ids)
        
        n_groups = len(group_ids)
        n_train = max(1, int(n_groups * train_ratio))
        n_val = max(1, int(n_groups * val_ratio))
        
        train_groups = set(group_ids[:n_train])
        val_groups = set(group_ids[n_train:n_train + n_val])
        test_groups = set(group_ids[n_train + n_val:])
        
        train, val, test = [], [], []
        for s in samples:
            gid = s["group_id"]
            if gid in train_groups:
                s["split"] = "train"
                train.append(s)
            elif gid in val_groups:
                s["split"] = "val"
                val.append(s)
            else:
                s["split"] = "test"
                test.append(s)
        
        return train, val, test
    
    def verify_no_leakage(self, train: List[Dict], val: List[Dict], test: List[Dict]) -> bool:
        """Post-split assertion: zero group overlap across splits."""
        train_gids = set(s["group_id"] for s in train)
        val_gids = set(s["group_id"] for s in val)
        test_gids = set(s["group_id"] for s in test)
        
        assert train_gids.isdisjoint(val_gids), "LEAKAGE: train/val overlap!"
        assert train_gids.isdisjoint(test_gids), "LEAKAGE: train/test overlap!"
        assert val_gids.isdisjoint(test_gids), "LEAKAGE: val/test overlap!"
        
        print(f"Split verification PASSED: {len(train_gids)} train, {len(val_gids)} val, {len(test_gids)} test groups. Zero overlap.")
        return True
    
    def save_split_manifest(self, train, val, test, output_dir: str):
        """Persists split assignment for reproducibility."""
        os.makedirs(output_dir, exist_ok=True)
        manifest = {
            "seed": self.seed,
            "train_groups": list(set(s["group_id"] for s in train)),
            "val_groups": list(set(s["group_id"] for s in val)),
            "test_groups": list(set(s["group_id"] for s in test)),
            "train_count": len(train),
            "val_count": len(val),
            "test_count": len(test)
        }
        with open(os.path.join(output_dir, "split_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
