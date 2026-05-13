import os
import json
from src.utils import safe_load_image, save_preprocessing_stats
from src.preprocessing import preprocess_template, preprocess_diagram
from src.visualization import (
    save_debug_images, 
    save_pipeline_overview, 
    save_intensity_histogram,
    save_colored_components,
    save_candidate_overlays,
    save_final_overlay,
    save_candidate_crops,
    save_diagnostic_overlays,
    save_large_components_overlay,
    save_all_component_crops,
    save_continuity_comparison_grid
)
from src.region_proposal import (
    compute_template_reference,
    extract_connected_components,
    filter_candidates
)
from src.proposal_diagnostics import (
    compute_component_metadata,
    save_component_ranking_table,
    run_relaxed_filter_experiment
)
from src.continuity_recovery import run_continuity_experiments

def main():
    print("=" * 50)
    print("STAGE 1: PREPROCESSING PIPELINE")
    print("=" * 50)

    # Output directories
    TEMPLATE_OUT_DIR = os.path.join("outputs", "template")
    DIAGRAM_OUT_DIR = os.path.join("outputs", "diagram")

    # Image Paths
    TEMPLATE_PATH = os.path.join("Data", "symbol.png")
    DIAGRAM_PATH = os.path.join("Data", "circuit_diagram.png")

    try:
        # --------------------------------------------------
        # 1. PROCESS TEMPLATE
        # --------------------------------------------------
        print("\n[1/2] Processing Template...")
        template_image = safe_load_image(TEMPLATE_PATH)
        template_results = preprocess_template(template_image)
        
        save_debug_images(TEMPLATE_OUT_DIR, template_results["images"])
        save_intensity_histogram(TEMPLATE_OUT_DIR, "histogram_gray.png", template_results["images"]["02_gray.png"])
        save_pipeline_overview(TEMPLATE_OUT_DIR, "template_pipeline_overview.png", template_results["images"])
        save_preprocessing_stats(template_results["stats"], os.path.join(TEMPLATE_OUT_DIR, "preprocessing_stats.json"))
        
        # --------------------------------------------------
        # 2. PROCESS DIAGRAM
        # --------------------------------------------------
        print("\n[2/2] Processing Diagram...")
        diagram_image = safe_load_image(DIAGRAM_PATH)
        diagram_results = preprocess_diagram(diagram_image)
        
        save_debug_images(DIAGRAM_OUT_DIR, diagram_results["images"])
        save_intensity_histogram(DIAGRAM_OUT_DIR, "histogram_gray.png", diagram_results["images"]["02_gray.png"])
        save_pipeline_overview(DIAGRAM_OUT_DIR, "diagram_pipeline_overview.png", diagram_results["images"])
        save_preprocessing_stats(diagram_results["stats"], os.path.join(DIAGRAM_OUT_DIR, "preprocessing_stats.json"))
        
        print("\n" + "=" * 50)
        print("STAGE 2: REGION PROPOSAL")
        print("=" * 50)
        
        template_binary = template_results["images"]["04_binary.png"]
        diagram_no_wire = diagram_results["images"]["06_no_wire.png"]
        diagram_original = diagram_results["images"]["01_original.png"]
        
        # Compute template reference (foreground pixels)
        template_ref_pixels = compute_template_reference(template_binary)
        print(f"Template Reference Scale (Foreground Pixels): {template_ref_pixels}")
        
        # Extract components from diagram_no_wire
        components, labels_img, num_labels = extract_connected_components(diagram_no_wire)
        print(f"Initial Connected Components (excluding background): {len(components)}")
        
        # Save colored components visualization
        save_colored_components(DIAGRAM_OUT_DIR, num_labels, labels_img)
        
        # Progressively filter candidates
        filter_results = filter_candidates(components, template_ref_pixels)
        
        all_comps = filter_results["all_components"]
        after_area = filter_results["after_area"]
        after_aspect = filter_results["after_aspect"]
        final_candidates = filter_results["after_density"]
        
        # Save progressive overlays
        save_candidate_overlays(DIAGRAM_OUT_DIR, "01_all_components.png", diagram_original, all_comps, (200, 200, 200))
        save_candidate_overlays(DIAGRAM_OUT_DIR, "02_after_area_filter.png", diagram_original, after_area, (255, 165, 0))
        save_candidate_overlays(DIAGRAM_OUT_DIR, "03_after_aspect_filter.png", diagram_original, after_aspect, (255, 255, 0))
        save_candidate_overlays(DIAGRAM_OUT_DIR, "04_after_density_filter.png", diagram_original, final_candidates, (0, 255, 0))
        
        # Save final overlay with text and crop candidates
        save_final_overlay(DIAGRAM_OUT_DIR, "final_candidate_overlay.png", diagram_original, final_candidates)
        save_candidate_crops(DIAGRAM_OUT_DIR, diagram_original, final_candidates)
        
        # Compute Stage 2 Statistics
        avg_area = sum(c["area"] for c in final_candidates) / len(final_candidates) if final_candidates else 0
        avg_density = sum(c["density"] for c in final_candidates) / len(final_candidates) if final_candidates else 0
        
        print(f"  - Total components: {len(all_comps)}")
        print(f"  - After area filtering: {len(after_area)}")
        print(f"  - After aspect filtering: {len(after_aspect)}")
        print(f"  - After density filtering (FINAL): {len(final_candidates)}")
        print(f"  - Average candidate area: {avg_area:.1f} pixels")
        print(f"  - Average candidate density: {avg_density:.3f}")
        
        # Save JSON metadata
        rp_stats = {
            "total_components": len(all_comps),
            "after_area_filter": len(after_area),
            "after_aspect_filter": len(after_aspect),
            "after_density_filter": len(final_candidates),
            "final_candidate_count": len(final_candidates),
            "average_candidate_area": avg_area,
            "average_candidate_density": avg_density,
            "template_area_reference": template_ref_pixels,
            "filter_thresholds": filter_results["thresholds"],
            "final_candidates_metadata": [
                {
                    "label": int(c["label"]),
                    "bbox": [int(x) for x in c["bbox"]],
                    "area": int(c["area"]),
                    "aspect_ratio": float(c["aspect_ratio"]),
                    "density": float(c["density"]),
                    "centroid": [float(x) for x in c["centroid"]],
                    "crop_path": c.get("crop_path", "")
                } for c in final_candidates
            ]
        }
        
        with open(os.path.join(DIAGRAM_OUT_DIR, "region_proposal_stats.json"), "w") as f:
            json.dump(rp_stats, f, indent=2)
            
        print("\n" + "=" * 50)
        print("STAGE 2 COMPLETED SUCCESSFULLY")
        print("Outputs saved to: outputs/diagram/")
        print("=" * 50)
        
        # --------------------------------------------------
        # STAGE 2.5: PROPOSAL DIAGNOSTICS
        # --------------------------------------------------
        print("\n" + "=" * 50)
        print("STAGE 2.5: PROPOSAL DIAGNOSTICS")
        print("=" * 50)
        
        # 1. Compute full metadata and rejection history for all components
        components_metadata = compute_component_metadata(all_comps, template_ref_pixels)
        
        # 2. Save metadata JSON
        metadata_path = os.path.join(DIAGRAM_OUT_DIR, "all_components_metadata.json")
        with open(metadata_path, "w") as f:
            # Clean up metadata for JSON serialization
            json_safe_metadata = [
                {
                    "label": int(c["label"]),
                    "bbox": [int(x) for x in c["bbox"]],
                    "foreground_area": int(c["foreground_area"]),
                    "bbox_area": int(c["bbox_area"]),
                    "aspect_ratio": float(c["aspect_ratio"]),
                    "density": float(c["density"]),
                    "centroid": [float(x) for x in c["centroid"]],
                    "passed_area_filter": bool(c["passed_area_filter"]),
                    "passed_aspect_filter": bool(c["passed_aspect_filter"]),
                    "passed_density_filter": bool(c["passed_density_filter"]),
                    "rejected_stage": c["rejected_stage"]
                } for c in components_metadata
            ]
            json.dump(json_safe_metadata, f, indent=2)
            
        # 3. Save CSV ranking table
        csv_path = os.path.join(DIAGRAM_OUT_DIR, "component_statistics.csv")
        save_component_ranking_table(components_metadata, csv_path)
        
        # 4. Save ALL component crops
        save_all_component_crops(DIAGRAM_OUT_DIR, diagram_original, all_comps)
        
        # 5. Diagnostic Overlays
        save_diagnostic_overlays(DIAGRAM_OUT_DIR, diagram_original, components_metadata)
        save_large_components_overlay(DIAGRAM_OUT_DIR, diagram_original, components_metadata, template_ref_pixels)
        
        # 6. Relaxed Filter Experiment
        relaxed_candidates = run_relaxed_filter_experiment(all_comps, template_ref_pixels)
        save_candidate_overlays(DIAGRAM_OUT_DIR, "relaxed_filter_candidates.png", diagram_original, relaxed_candidates, (255, 0, 255))
        
        # 7. Print Diagnostics
        rejected_area = sum(1 for c in components_metadata if c["rejected_stage"] == "area")
        rejected_aspect = sum(1 for c in components_metadata if c["rejected_stage"] == "aspect")
        rejected_density = sum(1 for c in components_metadata if c["rejected_stage"] == "density")
        
        areas = [c["foreground_area"] for c in components_metadata]
        densities = [c["density"] for c in components_metadata]
        
        largest_area = max(areas) if areas else 0
        median_area = sorted(areas)[len(areas)//2] if areas else 0
        mean_density = sum(densities) / len(densities) if densities else 0
        
        print("\nComponent Count Analysis:")
        print(f"  - Total components: {len(all_comps)}")
        print(f"  - Rejected by area: {rejected_area}")
        print(f"  - Rejected by aspect: {rejected_aspect}")
        print(f"  - Rejected by density: {rejected_density}")
        print(f"  - Surviving (original filters): {len(final_candidates)}")
        print(f"  - Surviving (RELAXED filters): {len(relaxed_candidates)}")
        
        print("\nStructural Analysis:")
        print(f"  - Largest component area: {largest_area}")
        print(f"  - Median component area: {median_area}")
        print(f"  - Mean component density: {mean_density:.3f}")
        
        print("\nStage 2.5 diagnostics saved to outputs/diagram/")
        
        # --------------------------------------------------
        # STAGE 2.75: SYMBOL CONTINUITY RECOVERY
        # --------------------------------------------------
        print("\n" + "=" * 50)
        print("STAGE 2.75: SYMBOL CONTINUITY RECOVERY")
        print("=" * 50)
        
        # Approximate MR reference regions can be defined here based on diagram knowledge.
        # Format: (x, y, w, h)
        MR_REFERENCE_REGIONS = [] 
        
        exp_summary, overlays_grid = run_continuity_experiments(
            diagram_no_wire, 
            diagram_original, 
            template_ref_pixels,
            mr_reference_regions=MR_REFERENCE_REGIONS
        )
        
        save_continuity_comparison_grid(DIAGRAM_OUT_DIR, diagram_no_wire, overlays_grid)
        
        print("\nContinuity Experiments Summary:")
        for exp in exp_summary:
            print(f"  Kernel: {exp['kernel_type']} {exp['kernel_size']} -> Candidates: {exp['final_candidates']} | Largest CC Area: {exp['largest_component_area']} | Status: {exp['notes']}")
            
        print("\nStage 2.75 completed. Check outputs/diagram/continuity_experiments/ and continuity_comparison_grid.png")

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
