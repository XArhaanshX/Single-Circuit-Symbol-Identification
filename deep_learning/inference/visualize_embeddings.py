"""
Embedding Visualization: t-SNE, UMAP, temporal evolution, hard-negative analysis.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

def load_snapshots(snap_dir):
    snaps = {}
    if not os.path.exists(snap_dir):
        return snaps
    for f in sorted(os.listdir(snap_dir)):
        if f.endswith(".npz"):
            d = np.load(os.path.join(snap_dir, f))
            epoch = int(f.replace("epoch_", "").replace(".npz", ""))
            snaps[epoch] = {"embeddings": d["embeddings"], "labels": d["labels"]}
    return snaps

def plot_tsne(embeddings, labels, title, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    if len(embeddings) < 5:
        return
    tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1), random_state=42)
    proj = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, color, name in [(1, "green", "Positive"), (0, "red", "Negative")]:
        mask = labels == lbl
        ax.scatter(proj[mask, 0], proj[mask, 1], c=color, label=name, alpha=0.7, s=40)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    dl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    snap_dir = os.path.join(dl_root, "experiments", "embedding_snapshots")
    out_dir = os.path.join(dl_root, "outputs", "embeddings")
    evo_dir = os.path.join(out_dir, "embedding_evolution")
    os.makedirs(evo_dir, exist_ok=True)
    
    snaps = load_snapshots(snap_dir)
    if not snaps:
        print("No embedding snapshots found.")
        return
    
    print(f"Found {len(snaps)} snapshots: epochs {list(snaps.keys())}")
    
    # Generate t-SNE for each epoch
    for epoch, data in snaps.items():
        plot_tsne(data["embeddings"], data["labels"],
                  f"t-SNE Epoch {epoch}", os.path.join(evo_dir, f"tsne_epoch_{epoch:03d}.png"))
        print(f"  Generated tsne_epoch_{epoch:03d}.png")
    
    # Centroid separation evolution
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    epochs_list, seps = [], []
    for epoch in sorted(snaps.keys()):
        d = snaps[epoch]
        pos = d["embeddings"][d["labels"] == 1]
        neg = d["embeddings"][d["labels"] == 0]
        if len(pos) > 0 and len(neg) > 0:
            sep = float(np.linalg.norm(np.mean(pos, axis=0) - np.mean(neg, axis=0)))
            epochs_list.append(epoch)
            seps.append(sep)
    
    if epochs_list:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_list, seps, "bo-", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Centroid Separation (L2)")
        ax.set_title("Centroid Separation Evolution")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(evo_dir, "centroid_separation_evolution.png"), dpi=150)
        plt.close()
        print("  Generated centroid_separation_evolution.png")
    
    # Distance histograms from final snapshot
    final_epoch = max(snaps.keys())
    final = snaps[final_epoch]
    pos_emb = final["embeddings"][final["labels"] == 1]
    neg_emb = final["embeddings"][final["labels"] == 0]
    
    if len(pos_emb) > 0 and len(neg_emb) > 0:
        from scipy.spatial.distance import cdist
        template_emb = pos_emb[0:1]  # First positive as template proxy
        pos_dists = cdist(template_emb, pos_emb[1:]).flatten() if len(pos_emb) > 1 else np.array([])
        neg_dists = cdist(template_emb, neg_emb).flatten()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        if len(pos_dists) > 0:
            ax.hist(pos_dists, bins=15, alpha=0.6, color="green", label="Positive")
        ax.hist(neg_dists, bins=15, alpha=0.6, color="red", label="Negative")
        ax.set_xlabel("Distance to Template")
        ax.set_ylabel("Count")
        ax.set_title("Positive vs Negative Distance Distribution")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "distance_histograms.png"), dpi=150)
        plt.close()
        print("  Generated distance_histograms.png")
    
    print("\nEmbedding visualization complete.")

if __name__ == "__main__":
    main()
