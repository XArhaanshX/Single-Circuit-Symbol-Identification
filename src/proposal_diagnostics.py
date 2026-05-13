import csv
import json

def compute_component_metadata(components: list, template_foreground_pixels: int) -> list:
    """
    Evaluates every component against the standard filters and determines
    its exact rejection stage to understand WHY MR symbols are not surviving.
    
    Args:
        components: List of all connected component dicts extracted in Stage 2.
        template_foreground_pixels: The reference scale from the template.
        
    Returns:
        List of enriched component dicts containing their full filter history.
    """
    min_area = 0.5 * template_foreground_pixels
    max_area = 4.0 * template_foreground_pixels
    
    enriched_components = []
    
    for c in components:
        x, y, w, h = c["bbox"]
        bbox_area = w * h
        foreground_area = c["area"]
        
        # Compute geometric properties safely
        aspect_ratio = w / h if h > 0 else 0
        density = foreground_area / bbox_area if bbox_area > 0 else 0
        
        # Test filters independently to log exactly what fails
        passed_area = min_area < foreground_area < max_area
        passed_aspect = 0.3 < aspect_ratio < 2.5
        passed_density = 0.15 < density < 0.85
        
        # Determine rejection stage (simulating the sequential pipeline)
        rejected_stage = None
        if not passed_area:
            rejected_stage = "area"
        elif not passed_aspect:
            rejected_stage = "aspect"
        elif not passed_density:
            rejected_stage = "density"
            
        enriched_c = {
            "label": c["label"],
            "bbox": c["bbox"],
            "foreground_area": foreground_area,
            "bbox_area": bbox_area,
            "aspect_ratio": aspect_ratio,
            "density": density,
            "centroid": c["centroid"],
            "passed_area_filter": passed_area,
            "passed_aspect_filter": passed_aspect,
            "passed_density_filter": passed_density,
            "rejected_stage": rejected_stage
        }
        enriched_components.append(enriched_c)
        
    return enriched_components

def save_component_ranking_table(metadata: list, output_path: str):
    """
    Saves a CSV ranking table sorted by foreground area descending.
    """
    sorted_meta = sorted(metadata, key=lambda x: x["foreground_area"], reverse=True)
    
    with open(output_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["label", "foreground_area", "aspect_ratio", "density", "rejected_stage"])
        for c in sorted_meta:
            writer.writerow([
                c["label"], 
                c["foreground_area"], 
                f"{c['aspect_ratio']:.3f}", 
                f"{c['density']:.3f}", 
                c["rejected_stage"] if c["rejected_stage"] else "survived"
            ])

def run_relaxed_filter_experiment(components: list, template_foreground_pixels: int) -> list:
    """
    Temporarily tests RELAXED thresholds to see if MR symbols become recoverable.
    Original: Area max 4.0 -> Relaxed: 10.0
    Original: Aspect max 2.5 -> Relaxed: 6.0
    """
    min_area = 0.5 * template_foreground_pixels
    max_area = 10.0 * template_foreground_pixels
    
    relaxed_candidates = []
    
    for c in components:
        x, y, w, h = c["bbox"]
        foreground_area = c["area"]
        bbox_area = w * h
        aspect_ratio = w / h if h > 0 else 0
        density = foreground_area / bbox_area if bbox_area > 0 else 0
        
        if min_area < foreground_area < max_area:
            if 0.3 < aspect_ratio < 6.0:
                if 0.15 < density < 0.85:
                    c_copy = c.copy()
                    c_copy["aspect_ratio"] = aspect_ratio
                    c_copy["density"] = density
                    relaxed_candidates.append(c_copy)
                    
    return relaxed_candidates
