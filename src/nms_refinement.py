import numpy as np
import scipy.ndimage

NMS_WINDOW_SIZE = 15
MIN_COVERAGE_RATIO = 0.75
NMS_DISTANCE_THRESHOLD = 40

def extract_local_minima(score_map: np.ndarray, coverage_map: np.ndarray) -> list:
    """
    Identifies spatially distinct low-score basins using a minimum filter,
    and applies a coverage ratio threshold.
    """
    # Find local minima in the score map using a sliding window
    neighborhood_min = scipy.ndimage.minimum_filter(score_map, size=NMS_WINDOW_SIZE)
    
    # A point is a local minimum if it equals the minimum in its neighborhood
    local_min_mask = (score_map == neighborhood_min) & (~np.isinf(score_map))
    
    # Filter by minimum structural coverage
    coverage_mask = coverage_map >= MIN_COVERAGE_RATIO
    
    # Combine masks
    valid_minima_mask = local_min_mask & coverage_mask
    
    y_coords, x_coords = np.where(valid_minima_mask)
    
    minima = []
    for y, x in zip(y_coords, x_coords):
        minima.append({
            "x": int(x),
            "y": int(y),
            "mean_distance": float(score_map[y, x]),
            "coverage_ratio": float(coverage_map[y, x])
        })
        
    return minima

def apply_spatial_nms(candidate_matches: list) -> list:
    """
    Suppresses neighboring duplicate detections within NMS_DISTANCE_THRESHOLD.
    Iteratively keeps the strongest remaining match.
    """
    # 1. Sort matches by: mean_distance ASC, coverage_ratio DESC
    sorted_matches = sorted(candidate_matches, key=lambda m: (m["mean_distance"], -m["coverage_ratio"]))
    
    consolidated = []
    
    # 2. Iteratively suppress
    for match in sorted_matches:
        mx, my = match["x"], match["y"]
        
        is_duplicate = False
        for c in consolidated:
            cx, cy = c["x"], c["y"]
            # Euclidean distance
            dist = np.sqrt((mx - cx)**2 + (my - cy)**2)
            if dist < NMS_DISTANCE_THRESHOLD:
                is_duplicate = True
                break
                
        if not is_duplicate:
            # Assign rank based on insertion order (since it's sorted by mean_distance)
            match_copy = match.copy()
            match_copy["rank"] = len(consolidated) + 1
            consolidated.append(match_copy)
            
    return consolidated
