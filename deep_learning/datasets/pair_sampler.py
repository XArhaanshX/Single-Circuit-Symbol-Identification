"""
Pair and Triplet Sampler for Siamese/Triplet training.
"""

import random
import numpy as np
from typing import List, Dict, Tuple


def generate_positive_pairs(positives: List[Dict], n_pairs: int = 50) -> List[Tuple]:
    """Generate (template, true_MR) positive pairs."""
    pairs = []
    if len(positives) < 2:
        return pairs
    for _ in range(n_pairs):
        i, j = random.sample(range(len(positives)), 2)
        pairs.append((positives[i]["crop"], positives[j]["crop"], 1))
    return pairs


def generate_negative_pairs(positives: List[Dict], negatives: List[Dict], n_pairs: int = 50) -> List[Tuple]:
    """Generate (template, false_positive) negative pairs."""
    pairs = []
    if not positives or not negatives:
        return pairs
    for _ in range(n_pairs):
        pos = random.choice(positives)
        neg = random.choice(negatives)
        pairs.append((pos["crop"], neg["crop"], 0))
    return pairs


def generate_triplets(positives: List[Dict], negatives: List[Dict], n_triplets: int = 50) -> List[Tuple]:
    """
    Generate (anchor=template, positive=true_MR, negative=hard_FP) triplets.
    Hard negative mining: prioritize negatives with highest source rank (closest to true MR).
    """
    triplets = []
    if len(positives) < 2 or not negatives:
        return triplets
    
    # Sort negatives by difficulty (lower rank = harder)
    sorted_negs = sorted(negatives, key=lambda n: n.get("rank", 999))
    hard_negs = sorted_negs[:max(1, len(sorted_negs) // 2)]
    
    for _ in range(n_triplets):
        anchor = random.choice(positives)
        pos = random.choice(positives)
        while pos is anchor and len(positives) > 1:
            pos = random.choice(positives)
        neg = random.choice(hard_negs)
        
        triplets.append((anchor["crop"], pos["crop"], neg["crop"]))
    
    return triplets
