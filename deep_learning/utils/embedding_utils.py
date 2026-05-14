"""Embedding utilities: batch extraction, distance matrix, silhouette."""
import numpy as np
import torch

def extract_all_embeddings(model, dataset, device, batch_size=32):
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embs, lbls = [], []
    with torch.no_grad():
        for imgs, labels, _ in loader:
            e = model.forward_single(imgs.to(device)).cpu().numpy()
            embs.append(e); lbls.append(labels.numpy())
    return np.concatenate(embs), np.concatenate(lbls)

def compute_distance_matrix(embeddings):
    from scipy.spatial.distance import cdist
    return cdist(embeddings, embeddings)

def compute_silhouette_score(embeddings, labels):
    from sklearn.metrics import silhouette_score
    if len(np.unique(labels)) < 2 or len(embeddings) < 3:
        return 0.0
    return float(silhouette_score(embeddings, labels))
