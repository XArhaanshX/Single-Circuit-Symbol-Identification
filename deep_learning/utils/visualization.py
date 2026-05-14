"""
Visualization utilities for the deep learning branch.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_embedding_clusters(embeddings, labels, save_path):
    from sklearn.manifold import TSNE
    if len(embeddings) < 5:
        return
    tsne = TSNE(n_components=2, perplexity=min(5, len(embeddings)-1), random_state=42)
    proj = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    for lbl, c, n in [(1, "green", "Positive"), (0, "red", "Negative")]:
        m = labels == lbl
        ax.scatter(proj[m, 0], proj[m, 1], c=c, label=n, alpha=0.7, s=40)
    ax.set_title("Embedding Clusters (t-SNE)")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def save_similarity_grid(template_emb, candidate_embs, save_path):
    from scipy.spatial.distance import cdist
    dists = cdist(template_emb.reshape(1, -1), candidate_embs).flatten()
    sims = 1.0 / (1.0 + dists)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(range(len(sims)), sims, color="steelblue")
    ax.set_xlabel("Candidate Index"); ax.set_ylabel("Similarity")
    ax.set_title("Template-Candidate Similarity"); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def save_distance_histograms(pos_dists, neg_dists, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    if len(pos_dists) > 0:
        ax.hist(pos_dists, bins=15, alpha=0.6, color="green", label="Positive")
    if len(neg_dists) > 0:
        ax.hist(neg_dists, bins=15, alpha=0.6, color="red", label="Negative")
    ax.set_xlabel("Distance"); ax.set_ylabel("Count")
    ax.set_title("Pos vs Neg Distance Distribution")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()

def save_embedding_stability_analysis(diag_history, save_path):
    if not diag_history:
        return
    epochs = [d["epoch"] for d in diag_history]
    var = [d["embedding_variance"] for d in diag_history]
    sep = [d["centroid_separation"] for d in diag_history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, var, "b-o"); ax1.set_title("Embedding Variance"); ax1.set_xlabel("Epoch")
    ax1.axhline(y=0.01, color="red", linestyle="--", label="Collapse Threshold")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, sep, "g-o"); ax2.set_title("Centroid Separation"); ax2.set_xlabel("Epoch")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def save_retrieval_metrics_dashboard(metrics_history, save_path):
    if not metrics_history:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Retrieval Metrics Dashboard", fontsize=14)
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_title("Placeholder: Recall@K"); axes[0, 0].text(0.5, 0.5, "No data yet", ha="center")
    axes[0, 1].set_title("Placeholder: Precision@K"); axes[0, 1].text(0.5, 0.5, "No data yet", ha="center")
    axes[1, 0].set_title("Placeholder: MRR"); axes[1, 0].text(0.5, 0.5, "No data yet", ha="center")
    axes[1, 1].set_title("Placeholder: Ranking Quality"); axes[1, 1].text(0.5, 0.5, "No data yet", ha="center")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()


# =====================================================================
# FINAL DETECTION OVERLAY VISUALIZATIONS
# =====================================================================

MIN_VISUALIZATION_SCORE = 0.05


def _get_color(rank, total):
    """Green for top, yellow for mid, red for bottom."""
    frac = rank / max(total, 1)
    if frac <= 0.3:
        return (0, 220, 0)
    elif frac <= 0.6:
        return (0, 220, 220)
    else:
        return (0, 0, 220)


def save_final_detection_overlay(diagram_path, candidates, template_shape, save_path):
    """Overlays ALL reranked detections on the circuit diagram with color-coded boxes."""
    import cv2
    diagram = cv2.imread(diagram_path)
    if diagram is None:
        return
    th, tw = template_shape
    n = len(candidates)
    for c in candidates:
        if c["fused_score"] < MIN_VISUALIZATION_SCORE:
            continue
        rank = c["siamese_rank"]
        color = _get_color(rank, n)
        cx, cy = c["x"], c["y"]
        x1, y1 = max(0, cx - tw // 2), max(0, cy - th // 2)
        x2, y2 = min(diagram.shape[1], x1 + tw), min(diagram.shape[0], y1 + th)
        cv2.rectangle(diagram, (x1, y1), (x2, y2), color, 2)
        label = f"#{rank} S:{c['siamese_similarity']:.2f} F:{c['fused_score']:.2f}"
        cv2.putText(diagram, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    cv2.putText(diagram, "GREEN=High  YELLOW=Mid  RED=Low",
                (10, diagram.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, diagram)


def save_ranked_overlay(diagram_path, candidates, template_shape, top_k, save_path):
    """Overlay only top-K detections on the diagram."""
    import cv2
    diagram = cv2.imread(diagram_path)
    if diagram is None:
        return
    th, tw = template_shape
    shown = [c for c in candidates if c["siamese_rank"] <= top_k]
    for c in shown:
        rank = c["siamese_rank"]
        color = _get_color(rank, top_k)
        cx, cy = c["x"], c["y"]
        x1, y1 = max(0, cx - tw // 2), max(0, cy - th // 2)
        x2, y2 = min(diagram.shape[1], x1 + tw), min(diagram.shape[0], y1 + th)
        cv2.rectangle(diagram, (x1, y1), (x2, y2), color, 2)
        label = f"#{rank} S:{c['siamese_similarity']:.2f}"
        cv2.putText(diagram, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.putText(diagram, f"Top-{top_k} Siamese Detections", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(save_path, diagram)


def save_detection_crop_grid(diagram_path, candidates, template_shape, save_path):
    """Grid of ranked candidate crops with similarity scores."""
    import cv2
    diagram = cv2.imread(diagram_path, cv2.IMREAD_GRAYSCALE)
    if diagram is None:
        return
    th, tw = template_shape
    n = len(candidates)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    cell_w, cell_h = tw + 20, th + 50
    grid = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 30
    for i, c in enumerate(candidates):
        row, col = divmod(i, cols)
        cx, cy = c["x"], c["y"]
        x1, y1 = max(0, cx - tw // 2), max(0, cy - th // 2)
        x2, y2 = min(diagram.shape[1], x1 + tw), min(diagram.shape[0], y1 + th)
        crop = diagram[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_bgr = cv2.cvtColor(cv2.resize(crop, (tw, th)), cv2.COLOR_GRAY2BGR)
        gx, gy = col * cell_w + 10, row * cell_h + 5
        grid[gy:gy + th, gx:gx + tw] = crop_bgr
        color = _get_color(c["siamese_rank"], n)
        cv2.rectangle(grid, (gx - 1, gy - 1), (gx + tw, gy + th), color, 1)
        cv2.putText(grid, f"#{c['siamese_rank']} F:{c['fused_score']:.2f}",
                    (gx, gy + th + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(grid, f"S:{c['siamese_similarity']:.2f}",
                    (gx, gy + th + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (180, 180, 180), 1)
    cv2.imwrite(save_path, grid)


def save_template_retrieval(template_path, diagram_path, candidates, template_shape, save_path):
    """Template on left, retrieved candidates sorted by similarity on right."""
    import cv2
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    diagram = cv2.imread(diagram_path, cv2.IMREAD_GRAYSCALE)
    if template is None or diagram is None:
        return
    th, tw = template_shape
    cell = max(th, tw) + 10
    n = min(len(candidates), 10)
    canvas = np.ones((cell + 40, cell * (n + 2), 3), dtype=np.uint8) * 30
    t_bgr = cv2.cvtColor(cv2.resize(template, (tw, th)), cv2.COLOR_GRAY2BGR)
    canvas[5:5 + th, 5:5 + tw] = t_bgr
    cv2.putText(canvas, "Template", (5, th + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.arrowedLine(canvas, (tw + 15, th // 2 + 5), (tw + 35, th // 2 + 5),
                    (255, 255, 255), 2)
    sorted_c = sorted(candidates, key=lambda c: c["siamese_similarity"], reverse=True)[:n]
    for i, c in enumerate(sorted_c):
        cx, cy = c["x"], c["y"]
        x1, y1 = max(0, cx - tw // 2), max(0, cy - th // 2)
        x2, y2 = min(diagram.shape[1], x1 + tw), min(diagram.shape[0], y1 + th)
        crop = diagram[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_bgr = cv2.cvtColor(cv2.resize(crop, (tw, th)), cv2.COLOR_GRAY2BGR)
        gx = (i + 2) * cell
        canvas[5:5 + th, gx:gx + tw] = crop_bgr
        color = _get_color(i + 1, n)
        cv2.rectangle(canvas, (gx - 1, 4), (gx + tw, 5 + th), color, 1)
        cv2.putText(canvas, f"S:{c['siamese_similarity']:.2f}", (gx, th + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        cv2.putText(canvas, f"#{c['siamese_rank']}", (gx, th + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    cv2.imwrite(save_path, canvas)
