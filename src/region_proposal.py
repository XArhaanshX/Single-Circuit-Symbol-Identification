import cv2
import numpy as np

def compute_template_reference(template_binary: np.ndarray) -> int:
    """
    Computes the template reference scale.
    Per PRD constraints, this MUST be the foreground pixel count.
    
    Args:
        template_binary: Preprocessed binary template image.
        
    Returns:
        Number of foreground pixels in the template.
    """
    return int(np.count_nonzero(template_binary))

def extract_connected_components(binary_image: np.ndarray) -> tuple[list, np.ndarray, int]:
    """
    Extracts connected components from a binary image using 8-connectivity.
    Explicitly skips the background component (label 0).
    
    Args:
        binary_image: Binary image (e.g., no_wire diagram).
        
    Returns:
        tuple: (list of candidate dicts, labels image, number of labels)
    """
    # Use 8-connectivity as recommended for circuit diagrams
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    components = []
    # Skip label 0 (background)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # foreground pixels
        area = stats[i, cv2.CC_STAT_AREA] 
        cx, cy = centroids[i]
        
        components.append({
            "label": i,
            "bbox": (x, y, w, h),
            "area": area,
            "centroid": (cx, cy)
        })
        
    return components, labels_img, num_labels

def filter_candidates(components: list, template_foreground_pixels: int) -> dict:
    """
    Systematically filters candidates based on geometric properties.
    Progressively filters and returns intermediate stages.
    
    Args:
        components: List of all connected component dicts.
        template_foreground_pixels: Reference scale (foreground pixel count).
        
    Returns:
        Dict containing lists of candidates at each filtering stage.
    """
    
    # 1. Area Filter
    # 0.5 * template_foreground_pixels < component_foreground_pixels < 4.0 * template_foreground_pixels
    # Rejects tiny text fragments and giant merged regions based strictly on foreground mass.
    min_area = 0.5 * template_foreground_pixels
    max_area = 4.0 * template_foreground_pixels
    
    after_area = []
    for c in components:
        if min_area < c["area"] < max_area:
            after_area.append(c)
            
    # 2. Aspect Ratio Filter
    # 0.3 < width / height < 2.5
    after_aspect = []
    for c in after_area:
        x, y, w, h = c["bbox"]
        if h > 0:
            aspect_ratio = w / h
            if 0.3 < aspect_ratio < 2.5:
                c["aspect_ratio"] = aspect_ratio
                after_aspect.append(c)
                
    # 3. Density Filter
    # 0.15 < foreground_pixels / bbox_area < 0.85
    after_density = []
    for c in after_aspect:
        x, y, w, h = c["bbox"]
        bbox_area = w * h
        if bbox_area > 0:
            density = c["area"] / bbox_area
            if 0.15 < density < 0.85:
                c["density"] = density
                after_density.append(c)
                
    return {
        "all_components": components,
        "after_area": after_area,
        "after_aspect": after_aspect,
        "after_density": after_density,  # These are the final candidates
        "thresholds": {
            "area_min_multiplier": 0.5,
            "area_max_multiplier": 4.0,
            "aspect_ratio_min": 0.3,
            "aspect_ratio_max": 2.5,
            "density_min": 0.15,
            "density_max": 0.85
        }
    }
