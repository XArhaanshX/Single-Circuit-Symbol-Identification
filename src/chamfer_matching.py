import cv2
import numpy as np

def compute_distance_transform(edge_image: np.ndarray) -> np.ndarray:
    """
    Computes the distance transform of the inverted edge image.
    Chamfer matching works by placing template edges over the diagram
    and sampling distances to the nearest diagram edges.
    LOWER distance means better geometric similarity.
    """
    inverted = cv2.bitwise_not(edge_image)
    dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 3)
    return dist_transform

def extract_edge_coordinates(edge_image: np.ndarray) -> list:
    """
    Extracts all template edge coordinates as a list of (x, y) tuples.
    """
    y_coords, x_coords = np.where(edge_image > 0)
    return list(zip(x_coords, y_coords))

def dense_chamfer_search(distance_transform: np.ndarray, template_edge_points: list, template_shape: tuple) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs exhaustive sliding-window translation search.
    Computes mean distance and coverage ratio for each valid window.
    
    Args:
        distance_transform: Diagram distance transform.
        template_edge_points: List of (x, y) coordinates of template edges.
        template_shape: (height, width) of the template.
        
    Returns:
        score_map, coverage_map
    """
    th, tw = template_shape[:2]
    dh, dw = distance_transform.shape[:2]
    
    score_map = np.full((dh, dw), np.inf, dtype=np.float32)
    coverage_map = np.zeros((dh, dw), dtype=np.float32)
    
    num_pts = len(template_edge_points)
    if num_pts == 0:
        return score_map, coverage_map
        
    # Extract arrays for faster indexing
    px = np.array([p[0] for p in template_edge_points])
    py = np.array([p[1] for p in template_edge_points])
    
    # Iterate over all valid top-left (dx, dy) translation windows
    for dy in range(dh - th + 1):
        for dx in range(dw - tw + 1):
            sample_y = py + dy
            sample_x = px + dx
            
            sampled_distances = distance_transform[sample_y, sample_x]
            
            mean_distance = np.mean(sampled_distances)
            coverage_ratio = np.mean(sampled_distances < 2.0)
            
            score_map[dy, dx] = mean_distance
            coverage_map[dy, dx] = coverage_ratio
            
    return score_map, coverage_map

def extract_top_matches(score_map: np.ndarray, coverage_map: np.ndarray, k: int = 20) -> list:
    """
    Extracts the top K matches using the lowest mean distance.
    (Without aggressive pruning / NMS).
    """
    flat_indices = np.argsort(score_map.ravel())
    top_indices = flat_indices[:k]
    
    matches = []
    for idx in top_indices:
        y, x = np.unravel_index(idx, score_map.shape)
        if np.isinf(score_map[y, x]):
            continue
        matches.append({
            "x": int(x),
            "y": int(y),
            "mean_distance": float(score_map[y, x]),
            "coverage_ratio": float(coverage_map[y, x])
        })
    return matches
