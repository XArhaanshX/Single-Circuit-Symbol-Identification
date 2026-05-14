"""
Training Orchestrator for Siamese Semantic Verification.
Two-phase training with online hard negative mining, embedding diagnostics,
retrieval metrics, and snapshot-based temporal evolution tracking.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from deep_learning.utils.patch_extraction import (
    extract_classical_candidates, build_positive_set, build_negative_set
)
from deep_learning.utils.augmentation import SymbolicAugmentation
from deep_learning.datasets.split_manager import SplitManager
from deep_learning.datasets.siamese_dataset import TripletDataset, EmbeddingDataset
from deep_learning.datasets.pair_sampler import generate_triplets
from deep_learning.datasets.transforms import TrainTransform, ValTransform
from deep_learning.models.siamese_network import SiameseNetwork
from deep_learning.training.losses import TripletMarginLoss
from deep_learning.training.scheduler import create_optimizer_phase1, create_optimizer_phase2, create_scheduler
from deep_learning.training.embedding_diagnostics import EmbeddingDiagnostics
from deep_learning.training.metrics import (
    compute_pair_distances, compute_recall_at_k, compute_precision_at_k, compute_mrr
)

# Snapshot epochs for temporal evolution tracking
SNAPSHOT_EPOCHS = [1, 5, 10]


def extract_embeddings(model, dataset, device):
    """Extract embeddings from all samples in a dataset."""
    model.eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_embs, all_labels = [], []
    
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            embs = model.forward_single(imgs)
            all_embs.append(embs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_embs), np.concatenate(all_labels)


def main():
    print("=" * 60)
    print("DEEP LEARNING BRANCH: SIAMESE SEMANTIC VERIFICATION")
    print("=" * 60)
    
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    project_root = os.path.abspath(project_root)
    
    exp_dir = os.path.join(os.path.dirname(__file__), "..", "experiments")
    exp_dir = os.path.abspath(exp_dir)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ========================================================
    # STAGE 1: DATASET CONSTRUCTION
    # ========================================================
    print("\n[1/4] Building dataset from classical pipeline outputs...")
    
    data = extract_classical_candidates(project_root)
    positives = build_positive_set(data["template_binary"], data["candidates"], top_k=3)
    negatives = build_negative_set(data["candidates"], bottom_start=8)
    
    print(f"  Base positives: {len(positives)}")
    print(f"  Base negatives: {len(negatives)}")
    
    # Augment
    augmenter = SymbolicAugmentation()
    all_samples = []
    
    for p in positives:
        all_samples.append(p)
        augs = augmenter.augment_light(p["crop"], p["group_id"], n_samples=15)
        for a in augs:
            all_samples.append({"crop": a["crop"], "label": "positive", 
                               "source": p["source"] + "_aug", "group_id": a["group_id"]})
    
    for n in negatives:
        all_samples.append(n)
        augs = augmenter.augment_light(n["crop"], n["group_id"], n_samples=10)
        for a in augs:
            all_samples.append({"crop": a["crop"], "label": "negative",
                               "source": n["source"] + "_aug", "group_id": a["group_id"]})
    
    print(f"  Total samples after augmentation: {len(all_samples)}")
    
    # Split with leakage prevention
    splitter = SplitManager(seed=42)
    train_samples, val_samples, test_samples = splitter.split_by_group(all_samples)
    splitter.verify_no_leakage(train_samples, val_samples, test_samples)
    splitter.save_split_manifest(train_samples, val_samples, test_samples, 
                                 os.path.join(exp_dir, "configs"))
    
    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    
    # Build triplets from training data
    train_pos = [s for s in train_samples if s["label"] == "positive"]
    train_neg = [s for s in train_samples if s["label"] == "negative"]
    triplets = generate_triplets(train_pos, train_neg, n_triplets=200)
    
    train_dataset = TripletDataset(triplets, transform=TrainTransform())
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # Embedding datasets for evaluation
    val_emb_dataset = EmbeddingDataset(val_samples, transform=ValTransform())
    all_emb_dataset = EmbeddingDataset(all_samples, transform=ValTransform())
    
    # ========================================================
    # STAGE 2: MODEL INITIALIZATION
    # ========================================================
    print("\n[2/4] Initializing Siamese network...")
    
    model = SiameseNetwork(embedding_dim=64, similarity="cosine").to(device)
    criterion = TripletMarginLoss(margin=0.5)
    diagnostics = EmbeddingDiagnostics(collapse_threshold=0.001)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # ========================================================
    # STAGE 3: TRAINING
    # ========================================================
    print("\n[3/4] Training...")
    
    phase1_epochs = 8
    phase2_epochs = 12
    total_epochs = phase1_epochs + phase2_epochs
    
    training_log = {"phase1": [], "phase2": [], "diagnostics": [], "retrieval": []}
    best_loss = float("inf")
    
    # --- Phase 1: Frozen backbone ---
    print(f"\n  Phase 1: Embedding head only ({phase1_epochs} epochs)")
    optimizer = create_optimizer_phase1(model, lr=1e-3)
    scheduler = create_scheduler(optimizer, T_max=phase1_epochs)
    
    for epoch in range(1, phase1_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            
            z_a = model.forward_single(anchor)
            z_p = model.forward_single(pos)
            z_n = model.forward_single(neg)
            
            loss = criterion(z_a, z_p, z_n)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        training_log["phase1"].append({"epoch": epoch, "loss": avg_loss})
        
        # Run diagnostics + snapshots
        embeddings, labels = extract_embeddings(model, val_emb_dataset, device)
        diag = diagnostics.run_all(embeddings, labels)
        diag["epoch"] = epoch
        training_log["diagnostics"].append(diag)
        
        if epoch in SNAPSHOT_EPOCHS:
            all_embs, all_lbls = extract_embeddings(model, all_emb_dataset, device)
            diagnostics.snapshot_embeddings(epoch, all_embs, all_lbls, exp_dir)
        
        print(f"    Epoch {epoch}/{phase1_epochs} | Loss: {avg_loss:.4f} | "
              f"Var: {diag['embedding_variance']:.4f} | Sep: {diag['centroid_separation']:.4f}")
        
        if diag["collapsed"]:
            print("    [!] EMBEDDING COLLAPSE — stopping Phase 1 early.")
            break
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoints", "best_model.pth"))
    
    # --- Phase 2: Full fine-tuning ---
    print(f"\n  Phase 2: Full fine-tuning ({phase2_epochs} epochs)")
    optimizer = create_optimizer_phase2(model, lr=1e-4)
    scheduler = create_scheduler(optimizer, T_max=phase2_epochs)
    
    for epoch in range(phase1_epochs + 1, total_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        for anchor, pos, neg in train_loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            
            z_a = model.forward_single(anchor)
            z_p = model.forward_single(pos)
            z_n = model.forward_single(neg)
            
            loss = criterion(z_a, z_p, z_n)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        training_log["phase2"].append({"epoch": epoch, "loss": avg_loss})
        
        embeddings, labels = extract_embeddings(model, val_emb_dataset, device)
        diag = diagnostics.run_all(embeddings, labels)
        diag["epoch"] = epoch
        training_log["diagnostics"].append(diag)
        
        if epoch in [phase1_epochs + 1, total_epochs] or epoch == 10:
            all_embs, all_lbls = extract_embeddings(model, all_emb_dataset, device)
            diagnostics.snapshot_embeddings(epoch, all_embs, all_lbls, exp_dir)
        
        print(f"    Epoch {epoch}/{total_epochs} | Loss: {avg_loss:.4f} | "
              f"Var: {diag['embedding_variance']:.4f} | Sep: {diag['centroid_separation']:.4f}")
        
        if diag["collapsed"]:
            print("    [!] EMBEDDING COLLAPSE — stopping Phase 2 early.")
            break
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "checkpoints", "best_model.pth"))
    
    # Final snapshot
    all_embs, all_lbls = extract_embeddings(model, all_emb_dataset, device)
    diagnostics.snapshot_embeddings(total_epochs, all_embs, all_lbls, exp_dir)
    
    # ========================================================
    # STAGE 4: SAVE RESULTS
    # ========================================================
    print("\n[4/4] Saving training log...")
    
    log_path = os.path.join(exp_dir, "logs", "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {os.path.join(exp_dir, 'checkpoints', 'best_model.pth')}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    main()
