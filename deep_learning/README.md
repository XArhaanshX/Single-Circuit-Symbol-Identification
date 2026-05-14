# Deep Learning Branch — Siamese Semantic Verification

## Overview

This is a **completely isolated** research branch that implements a Siamese metric-learning verifier on top of the existing classical CV proposal engine.

**Architecture:**
```
Classical CV Proposals → Siamese Neural Verification → Final Semantic Ranking
```

## Dependencies

```
torch
torchvision
scikit-learn
scikit-image
scipy
matplotlib
numpy
opencv-python
```

## Quick Start

```bash
# 1. Train the Siamese verifier
python deep_learning/training/train.py

# 2. Rerank classical candidates
python deep_learning/inference/rerank_candidates.py

# 3. Visualize embeddings
python deep_learning/inference/visualize_embeddings.py
```

## Design Philosophy

- **NOT** full object detection, YOLO, segmentation, or end-to-end.
- Learns semantic similarity between template and candidate regions.
- Uses 3-channel symbolic input: binary + skeleton + edge map.
- Tiny CNN backbone (deliberately small for tiny dataset).
- Triplet loss with online hard negative mining.
- Group-aware split management to prevent augmentation leakage.
- Embedding stability diagnostics to detect collapse.
- Retrieval metrics: Recall@K, Precision@K, MRR, mAP-lite.
