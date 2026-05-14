"""
Embedding Search — precompute template embedding, rank by distance.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from deep_learning.utils.patch_extraction import extract_classical_candidates
from deep_learning.models.siamese_network import SiameseNetwork
from deep_learning.datasets.transforms import SymbolicTransform

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork(embedding_dim=64).to(device)
    ckpt = os.path.join(dl_root, "experiments", "checkpoints", "best_model.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    transform = SymbolicTransform()
    data = extract_classical_candidates(project_root)
    t_tensor = transform(data["template_binary"]).unsqueeze(0).to(device)
    with torch.no_grad():
        t_emb = model.forward_single(t_tensor).cpu().numpy()[0]
    embeddings = []
    for c in data["candidates"]:
        c_tensor = transform(c["crop"]).unsqueeze(0).to(device)
        with torch.no_grad():
            c_emb = model.forward_single(c_tensor).cpu().numpy()[0]
        embeddings.append(c_emb)
    embeddings = np.array(embeddings)
    dists = np.linalg.norm(embeddings - t_emb, axis=1)
    ranked = np.argsort(dists)
    out_dir = os.path.join(dl_root, "outputs", "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "all_embeddings.npz"),
             template=t_emb, candidates=embeddings, distances=dists, rankings=ranked)
    print("Embedding search complete.")
    for i, idx in enumerate(ranked[:5]):
        c = data["candidates"][idx]
        print(f"  Rank {i+1}: (x:{c['x']},y:{c['y']}) dist={dists[idx]:.4f}")

if __name__ == "__main__":
    main()
