import os
import cv2
import json
import numpy as np
from skimage.morphology import skeletonize

from src.region_proposal import extract_connected_components, filter_candidates

def is_inside_mr_region(centroid: tuple, mr_reference_regions: list) -> bool:
    """
    Checks if a centroid falls within any of the defined MR reference regions.
    Args:
        centroid: (cx, cy)
        mr_reference_regions: List of bounding boxes (x, y, w, h)
    """
    cx, cy = centroid
    for (rx, ry, rw, rh) in mr_reference_regions:
        if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
            return True
    return False

def run_continuity_experiments(no_wire_img: np.ndarray, original_img: np.ndarray, template_area: int, mr_reference_regions: list = None) -> tuple[dict, dict]:
    """
    Runs systematic local morphological closing experiments to recover MR coil continuity.
    
    Args:
        no_wire_img: Binary image with horizontal buses removed.
        original_img: Original diagram image for overlay generation.
        template_area: Reference foreground pixel count.
        mr_reference_regions: List of bounding boxes (x, y, w, h) of known MR symbols.
        
    Returns:
        tuple: (experiment_summary_list, overlay_images_dict)
    """
    if mr_reference_regions is None:
        mr_reference_regions = []
        
    experiments = [
        {"name": "rect_3x3", "kernel": cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))},
        {"name": "rect_5x3", "kernel": cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))},
        {"name": "rect_3x5", "kernel": cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))},
        {"name": "ellipse_3x3", "kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))},
        {"name": "ellipse_5x5", "kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))}
    ]
    
    base_out_dir = os.path.join("outputs", "diagram", "continuity_experiments")
    
    summary = []
    overlays_for_grid = {}
    
    for exp in experiments:
        name = exp["name"]
        kernel = exp["kernel"]
        
        exp_out_dir = os.path.join(base_out_dir, name)
        os.makedirs(exp_out_dir, exist_ok=True)
        
        # 1. Apply local morphological closing
        closed_binary = cv2.morphologyEx(no_wire_img, cv2.MORPH_CLOSE, kernel)
        
        # 2. Generate skeletonized output
        binary_bool = closed_binary > 0
        skeleton = skeletonize(binary_bool).astype(np.uint8) * 255
        
        # 3. Extract connected components
        components, labels_img, num_labels = extract_connected_components(closed_binary)
        
        # 4. Pass components through UNCHANGED Stage 2 filtering
        filter_results = filter_candidates(components, template_area)
        final_candidates = filter_results["after_density"]
        
        # 5. Generate and save images
        cv2.imwrite(os.path.join(exp_out_dir, "01_closed_binary.png"), closed_binary)
        
        # Colored components
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]
        colored_components = colors[labels_img]
        cv2.imwrite(os.path.join(exp_out_dir, "02_connected_components.png"), colored_components)
        
        cv2.imwrite(os.path.join(exp_out_dir, "04_skeleton.png"), skeleton)
        
        # Overlay
        if len(original_img.shape) == 2:
            vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = original_img.copy()
            
        mr_candidate_count = 0
        for c in final_candidates:
            x, y, w, h = c["bbox"]
            centroid = c["centroid"]
            
            # Check MR Region overlap
            is_mr = is_inside_mr_region(centroid, mr_reference_regions)
            if is_mr or len(mr_reference_regions) == 0: 
                mr_candidate_count += 1
                color = (0, 255, 0) # Green for MR
            else:
                color = (0, 165, 255) # Orange for other
                
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
            
        cv2.imwrite(os.path.join(exp_out_dir, "03_candidate_overlay.png"), vis_img)
        overlays_for_grid[name] = vis_img
        
        # Compute stats for failure detection
        largest_area = max([c["area"] for c in components]) if components else 0
        
        # Failure logic: if largest component grows dramatically 
        # (e.g. > 10x template area implies bus reconnection)
        is_failure = largest_area > (10 * template_area)
        notes = "Failure: Catastrophic bus reconnection." if is_failure else "Stable."
        
        exp_summary = {
            "kernel_type": name.split('_')[0],
            "kernel_size": [kernel.shape[1], kernel.shape[0]],
            "total_components": len(components),
            "final_candidates": len(final_candidates),
            "largest_component_area": int(largest_area),
            "mr_region_candidate_count": mr_candidate_count,
            "notes": notes
        }
        
        summary.append(exp_summary)
        
    # Save component evolution summary
    summary_path = os.path.join("outputs", "diagram", "component_evolution_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    return summary, overlays_for_grid
