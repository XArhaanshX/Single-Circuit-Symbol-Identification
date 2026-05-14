import numpy as np
import cv2
from sklearn.decomposition import PCA

PCA_TEMPLATE_SIZE = (64, 64)
WHITEN_PCA = True
PCA_TAU = 2000.0  # Configurable tau for similarity

def center_foreground_centroid(binary_img: np.ndarray, target_size: tuple = PCA_TEMPLATE_SIZE) -> np.ndarray:
    """
    Computes foreground pixel coordinates, estimates centroid,
    and translates foreground to canvas center, preserving binary structure.
    PCA should learn structural variability, NOT alignment variability.
    """
    canvas = np.zeros(target_size, dtype=np.uint8)
    
    y_coords, x_coords = np.where(binary_img > 0)
    if len(x_coords) == 0:
        return canvas
        
    cx = int(np.mean(x_coords))
    cy = int(np.mean(y_coords))
    
    target_cx = target_size[1] // 2
    target_cy = target_size[0] // 2
    
    dx = target_cx - cx
    dy = target_cy - cy
    
    for x, y in zip(x_coords, y_coords):
        nx = x + dx
        ny = y + dy
        if 0 <= nx < target_size[1] and 0 <= ny < target_size[0]:
            canvas[ny, nx] = 255
            
    return canvas

def generate_augmented_templates(template_binary: np.ndarray) -> list:
    """
    Construct a robust MR appearance manifold from a single reference symbol
    using controlled augmentations.
    """
    augmented = []
    
    rotations = [-8, -4, 0, 4, 8]
    scales = [0.90, 0.95, 1.0, 1.05, 1.10]
    translations = [(0,0), (-2,0), (2,0), (0,-2), (0,2)]
    
    h, w = template_binary.shape
    
    # We want to scale and rotate around the foreground centroid, so let's find it.
    y_coords, x_coords = np.where(template_binary > 0)
    if len(x_coords) > 0:
        center = (int(np.mean(x_coords)), int(np.mean(y_coords)))
    else:
        center = (w // 2, h // 2)
    
    for rot in rotations:
        for scale in scales:
            M = cv2.getRotationMatrix2D(center, rot, scale)
            # Use INTER_NEAREST to preserve binary structure
            warped = cv2.warpAffine(template_binary, M, (w, h), flags=cv2.INTER_NEAREST)
            
            for thick in ["erode", "none", "dilate"]:
                thick_img = warped.copy()
                if thick == "erode":
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    thick_img = cv2.erode(thick_img, kernel, iterations=1)
                elif thick == "dilate":
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    thick_img = cv2.dilate(thick_img, kernel, iterations=1)
                
                for tx, ty in translations:
                    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
                    trans_img = cv2.warpAffine(thick_img, M_trans, (w, h), flags=cv2.INTER_NEAREST)
                    
                    # Normalize alignment before adding to manifold
                    normalized = center_foreground_centroid(trans_img, PCA_TEMPLATE_SIZE)
                    augmented.append(normalized)
                    
    return augmented

def build_pca_subspace(augmented_templates: list):
    """
    Constructs PCA model from augmented templates.
    """
    flattened = [img.flatten() for img in augmented_templates]
    X = np.array(flattened, dtype=np.float32)
    
    n_components = min(len(augmented_templates) - 1, 10)
    pca = PCA(n_components=n_components, whiten=WHITEN_PCA)
    pca.fit(X)
    
    return pca

def extract_candidate_patches(diagram_no_wire: np.ndarray, detections: list, template_shape: tuple) -> list:
    """
    Extract candidate patches from diagram_no_wire, using Stage 3.5 bounds + padding.
    """
    patches = []
    th, tw = template_shape[:2]
    img_h, img_w = diagram_no_wire.shape[:2]
    pad = 10
    
    for det in detections:
        x, y = det["x"], det["y"]
        
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + tw + pad)
        y2 = min(img_h, y + th + pad)
        
        crop = diagram_no_wire[y1:y2, x1:x2]
        
        # Normalize alignment
        normalized = center_foreground_centroid(crop, PCA_TEMPLATE_SIZE)
        patches.append(normalized)
        
    return patches

def compute_reconstruction_error(candidate_patch: np.ndarray, pca_model: PCA) -> float:
    """
    Computes L2 reconstruction error.
    """
    flat = candidate_patch.flatten().astype(np.float32).reshape(1, -1)
    proj = pca_model.transform(flat)
    recon = pca_model.inverse_transform(proj)
    
    error = np.linalg.norm(flat - recon)
    return float(error)

def rerank_detections(detections: list, candidate_patches: list, pca_model: PCA) -> list:
    """
    Fuses Chamfer distance and PCA similarity, and re-ranks detections.
    """
    reranked = []
    
    chamfer_scores = [d["mean_distance"] for d in detections]
    min_c = min(chamfer_scores)
    max_c = max(chamfer_scores) if max(chamfer_scores) > min_c else min_c + 1e-5
    
    for i, (det, patch) in enumerate(zip(detections, candidate_patches)):
        error = compute_reconstruction_error(patch, pca_model)
        
        # PCA similarity: exp(-error / tau)
        pca_sim = np.exp(-error / PCA_TAU)
        
        # Chamfer similarity: LOWER distance = HIGHER similarity
        # Min-Max inversion: 1.0 is best (min dist), 0.0 is worst (max dist)
        chamfer_sim = 1.0 - ((det["mean_distance"] - min_c) / (max_c - min_c))
        
        fused_score = 0.7 * chamfer_sim + 0.3 * pca_sim
        
        new_det = det.copy()
        new_det["pca_error"] = error
        new_det["pca_similarity"] = float(pca_sim)
        new_det["chamfer_similarity"] = float(chamfer_sim)
        new_det["fused_score"] = float(fused_score)
        new_det["normalized_patch"] = patch
        new_det["reconstructed_patch"] = pca_model.inverse_transform(pca_model.transform(patch.flatten().astype(np.float32).reshape(1, -1))).reshape(PCA_TEMPLATE_SIZE)
        reranked.append(new_det)
        
    # Re-rank by fused score DESCENDING
    reranked = sorted(reranked, key=lambda x: x["fused_score"], reverse=True)
    
    for rank, d in enumerate(reranked, start=1):
        d["pca_rank"] = rank
        
    return reranked
