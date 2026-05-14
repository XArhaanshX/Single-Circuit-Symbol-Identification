import os
import json
import numpy as np
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
    save_continuity_comparison_grid,
    save_distance_transform_vis,
    save_chamfer_heatmap,
    save_top_chamfer_matches,
    save_best_template_alignment,
    save_individual_top_matches,
    save_local_minima_overlay,
    save_final_nms_detections,
    save_minima_heatmap,
    save_ranked_detections_vis,
    save_final_detection_crops,
    save_suppression_radius_overlay,
    save_pca_template_grid,
    save_pca_reranked_detections,
    save_pca_variance_spectrum,
    save_template_pairwise_distances,
    save_candidate_alignment_normalization,
    save_pca_candidate_crops,
    save_pca_reconstruction_examples,
    save_similarity_scatter_plot,
    save_calibration_comparison_grid
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
from src.chamfer_matching import (
    compute_distance_transform,
    extract_edge_coordinates,
    dense_chamfer_search,
    extract_top_matches
)
from src.nms_refinement import (
    extract_local_minima,
    apply_spatial_nms,
    NMS_DISTANCE_THRESHOLD,
    MIN_COVERAGE_RATIO
)

from src.pca_refinement import (
    generate_augmented_templates,
    build_pca_subspace,
    extract_candidate_patches,
    rerank_detections,
    PCA_TEMPLATE_SIZE
)
from src.pca_calibration import run_calibration_experiments

from src.multitemplate_chamfer import (
    generate_chamfer_template_ensemble,
    run_ensemble_chamfer_search,
    extract_ensemble_detections,
    track_minima_emergence
)
from src.visualization import (
    save_ensemble_template_grid,
    save_per_template_heatmaps,
    save_ensemble_score_heatmap,
    save_winning_template_regions,
    save_ensemble_comparison_grid,
    save_ensemble_detection_crops,
    save_winning_template_histogram,
    save_template_overlap_matrix
)

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
        
        # --------------------------------------------------
        # STAGE 3: DENSE SLIDING-WINDOW CHAMFER MATCHING
        # --------------------------------------------------
        print("\n" + "=" * 50)
        print("STAGE 3: DENSE SLIDING-WINDOW CHAMFER MATCHING")
        print("=" * 50)
        
        template_edges = template_results["images"]["05_edges.png"]
        diagram_edges_no_wire = diagram_results["images"]["08_edges_no_wire.png"]
        
        # 1. Distance Transform
        dist_transform = compute_distance_transform(diagram_edges_no_wire)
        save_distance_transform_vis(DIAGRAM_OUT_DIR, dist_transform)
        
        # 2. Template Edge Extraction
        template_edge_points = extract_edge_coordinates(template_edges)
        print(f"Template edge points extracted: {len(template_edge_points)}")
        
        # 3. Dense Sliding-Window Search
        print("Running exhaustive dense sliding-window Chamfer search (this may take a moment)...")
        score_map, coverage_map = dense_chamfer_search(dist_transform, template_edge_points, template_edges.shape)
        
        # 4 & 5 & 6. Match Ranking
        top_matches = extract_top_matches(score_map, coverage_map, k=20)
        
        # 7. Visualization Outputs
        save_chamfer_heatmap(DIAGRAM_OUT_DIR, "chamfer_score_heatmap.png", score_map, cmap='viridis_r') # reversed viridis so low distance = bright
        save_chamfer_heatmap(DIAGRAM_OUT_DIR, "chamfer_coverage_heatmap.png", coverage_map, cmap='viridis')
        
        save_top_chamfer_matches(DIAGRAM_OUT_DIR, diagram_original, top_matches, template_edges.shape)
        
        if top_matches:
            best_match = top_matches[0]
            save_best_template_alignment(DIAGRAM_OUT_DIR, diagram_edges_no_wire, template_edge_points, best_match)
            
        SAVE_INTERMEDIATE_MATCHES = True
        if SAVE_INTERMEDIATE_MATCHES:
            save_individual_top_matches(DIAGRAM_OUT_DIR, diagram_original, template_edge_points, top_matches, template_edges.shape)
            
        # 8. Statistics and Metadata
        valid_scores = score_map[~np.isinf(score_map)]
        min_score = float(np.min(valid_scores)) if len(valid_scores) > 0 else 0.0
        max_score = float(np.max(valid_scores)) if len(valid_scores) > 0 else 0.0
        mean_score = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0
        std_score = float(np.std(valid_scores)) if len(valid_scores) > 0 else 0.0
        
        print("\nChamfer Matching Statistics:")
        print(f"  - Minimum Score (best): {min_score:.2f}")
        print(f"  - Maximum Score (worst): {max_score:.2f}")
        print(f"  - Mean Score: {mean_score:.2f}")
        print(f"  - Std Deviation: {std_score:.2f}")
        
        print("\nTop 5 Matches:")
        for i, m in enumerate(top_matches[:5], 1):
            print(f"  {i}. (x:{m['x']}, y:{m['y']}) -> Distance: {m['mean_distance']:.2f}, Coverage: {m['coverage_ratio']:.2f}")
            
        chamfer_stats = {
            "min_score": min_score,
            "max_score": max_score,
            "mean_score": mean_score,
            "std_score": std_score,
            "top_matches": top_matches
        }
        
        with open(os.path.join(DIAGRAM_OUT_DIR, "chamfer_statistics.json"), "w") as f:
            json.dump(chamfer_stats, f, indent=2)
            
        print("\nStage 3 completed. Check outputs/diagram/ for Chamfer heatmaps and top matches.")
        
        # --------------------------------------------------
        # STAGE 3.5: SPATIAL NMS AND MATCH CONSOLIDATION
        # --------------------------------------------------
        print("\n" + "=" * 50)
        print("STAGE 3.5: SPATIAL NMS AND MATCH CONSOLIDATION")
        print("=" * 50)
        
        # 1 & 2 & 3. Local Minima Extraction
        print("Extracting local minima basins...")
        local_minima = extract_local_minima(score_map, coverage_map)
        print(f"Total local minima found: {len(local_minima)}")
        
        # 4. Spatial Non-Maximum Suppression
        print("Applying spatial Non-Maximum Suppression (NMS)...")
        final_detections = apply_spatial_nms(local_minima)
        print(f"Final consolidated detections: {len(final_detections)}")
        
        # 6. Visualization Outputs
        save_local_minima_overlay(DIAGRAM_OUT_DIR, diagram_original, local_minima)
        save_final_nms_detections(DIAGRAM_OUT_DIR, diagram_original, final_detections, template_edges.shape)
        save_minima_heatmap(DIAGRAM_OUT_DIR, score_map, local_minima)
        save_ranked_detections_vis(DIAGRAM_OUT_DIR, diagram_original, final_detections, template_edges.shape)
        save_suppression_radius_overlay(DIAGRAM_OUT_DIR, diagram_original, final_detections, template_edges.shape, NMS_DISTANCE_THRESHOLD)
        
        SAVE_DETECTION_CROPS = True
        if SAVE_DETECTION_CROPS:
            save_final_detection_crops(DIAGRAM_OUT_DIR, diagram_original, final_detections, template_edges.shape)
            
        # 7. Detection Statistics
        nms_stats = {
            "raw_match_count": int(np.sum(~np.isinf(score_map))), # Total valid evaluated windows
            "local_minima_count": len(local_minima),
            "post_nms_count": len(final_detections),
            "detections": final_detections
        }
        
        with open(os.path.join(DIAGRAM_OUT_DIR, "nms_statistics.json"), "w") as f:
            json.dump(nms_stats, f, indent=2)
            
        avg_mean_distance = float(np.mean([d["mean_distance"] for d in final_detections])) if final_detections else 0.0
        avg_coverage_ratio = float(np.mean([d["coverage_ratio"] for d in final_detections])) if final_detections else 0.0
        
        print("\nDetection Statistics:")
        print(f"  - Local minima extracted: {len(local_minima)}")
        print(f"  - Retained detections: {len(final_detections)}")
        print(f"  - Average Coverage Ratio: {avg_coverage_ratio:.2f}")
        print(f"  - Average Mean Distance: {avg_mean_distance:.2f}")
        
        print("\nTop 5 Consolidated Detections:")
        for det in final_detections[:5]:
            print(f"  Rank #{det['rank']} (x:{det['x']}, y:{det['y']}) -> Distance: {det['mean_distance']:.2f}, Coverage: {det['coverage_ratio']:.2f}")

        print("\nStage 3.5 completed. Check outputs/diagram/final_nms_detections.png")
        
        # --------------------------------------------------
        # STAGE 4: PCA SUBSPACE REFINEMENT
        # --------------------------------------------------
        print("\n" + "=" * 50)
        print("STAGE 4: PCA SUBSPACE REFINEMENT")
        print("=" * 50)
        
        # 1. Template Augmentation
        print("Generating augmented templates...")
        augmented_templates = generate_augmented_templates(template_results["images"]["04_binary.png"])
        print(f"Generated {len(augmented_templates)} augmented templates.")
        save_pca_template_grid(DIAGRAM_OUT_DIR, augmented_templates)
        save_template_pairwise_distances(DIAGRAM_OUT_DIR, augmented_templates)
        
        # 2. PCA Subspace Construction
        print("Building PCA subspace...")
        pca_model = build_pca_subspace(augmented_templates)
        print(f"PCA constructed with {pca_model.n_components_} components.")
        save_pca_variance_spectrum(DIAGRAM_OUT_DIR, pca_model)
        
        # 3. Candidate Patch Extraction
        print("Extracting and normalizing candidate patches...")
        candidate_patches = extract_candidate_patches(diagram_results["images"]["06_no_wire.png"], final_detections, template_edges.shape)
        
        # Save alignment diagnostics for a few candidates
        for i, (det, patch) in enumerate(zip(final_detections[:3], candidate_patches[:3])):
            x, y = det["x"], det["y"]
            pad = 10
            th, tw = template_edges.shape[:2]
            img_h, img_w = diagram_results["images"]["06_no_wire.png"].shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img_w, x + tw + pad)
            y2 = min(img_h, y + th + pad)
            raw_crop = diagram_results["images"]["06_no_wire.png"][y1:y2, x1:x2]
            save_candidate_alignment_normalization(DIAGRAM_OUT_DIR, raw_crop, patch, det["rank"])
            
        # 4. Re-ranking
        print("Fusing scores and re-ranking detections...")
        pca_reranked = rerank_detections(final_detections, candidate_patches, pca_model)
        
        # 5. Visualizations
        save_pca_reranked_detections(DIAGRAM_OUT_DIR, diagram_original, pca_reranked, template_edges.shape)
        save_pca_reconstruction_examples(DIAGRAM_OUT_DIR, pca_reranked)
        save_pca_candidate_crops(DIAGRAM_OUT_DIR, pca_reranked)
        
        # 6. Statistics
        pca_stats = {
            "num_augmented_templates": len(augmented_templates),
            "num_pca_components": pca_model.n_components_,
            "detections": [{
                "rank": d["pca_rank"],
                "x": d["x"],
                "y": d["y"],
                "chamfer_score": d["mean_distance"],
                "pca_similarity": d["pca_similarity"],
                "fused_score": d["fused_score"]
            } for d in pca_reranked]
        }
        
        with open(os.path.join(DIAGRAM_OUT_DIR, "pca_statistics.json"), "w") as f:
            json.dump(pca_stats, f, indent=2)
            
        print("\nPCA Refinement Statistics:")
        print(f"  - Total Variance Explained: {np.sum(pca_model.explained_variance_ratio_):.2f}")
        avg_err = np.mean([d["pca_error"] for d in pca_reranked])
        print(f"  - Average Reconstruction Error: {avg_err:.2f}")
        
        print("\nTop 5 PCA-Reranked Detections:")
        for det in pca_reranked[:5]:
            print(f"  Rank #{det['pca_rank']} (x:{det['x']}, y:{det['y']}) -> Fused: {det['fused_score']:.2f} | C-Sim: {det['chamfer_similarity']:.2f} | P-Sim: {det['pca_similarity']:.2f}")

        print("\nStage 4 completed. Check outputs/diagram/pca_reranked_detections.png")

        # --------------------------------------------------
        # STAGE 4: EXPERIMENTAL SCORE CALIBRATION
        # --------------------------------------------------
        print("\n" + "=" * 50)
        print("STAGE 4 (EXPERIMENT): SCORE CALIBRATION")
        print("=" * 50)
        
        calib_out_dir, global_summary = run_calibration_experiments(
            pca_reranked, candidate_patches, pca_model, diagram_original, template_edges.shape, DIAGRAM_OUT_DIR
        )
        
        save_similarity_scatter_plot(DIAGRAM_OUT_DIR, pca_reranked)
        
        baseline_path = os.path.join(calib_out_dir, "exp_minmax_baseline_w0.7_0.3", "reranked_detections.png")
        max_pca_path = os.path.join(calib_out_dir, "exp_minmax_baseline_w0.3_0.7", "reranked_detections.png")
        exp_path = os.path.join(calib_out_dir, "exp_exponential_alpha2.0_w0.3_0.7", "reranked_detections.png")
        log_path = os.path.join(calib_out_dir, "exp_logistic_beta2.0_w0.3_0.7", "reranked_detections.png")
        
        save_calibration_comparison_grid(calib_out_dir, baseline_path, max_pca_path, exp_path, log_path)
        
        print(f"\nCalibration Experiments Completed. Generated {len(global_summary)} combinations.")
        print(f"Check {calib_out_dir}")
        
        print("\nFinal Analysis Questions:")
        print("1. Which normalization strategy produced strongest separation?")
        print("   -> (Please inspect global_summary.json and calibration_comparison_grid.png)")
        print("2. Did increasing PCA weighting suppress the G/B false positive?")
        print("   -> (Look at the 'gb_rank_delta' in global_summary.json)")
        print("3. Did excessive PCA weighting hurt true MR rankings?")
        print("   -> (Look at the 'avg_mr_rank_delta' in global_summary.json)")
        print("4. Which calibration setting appears most balanced?")
        print("   -> (The one where MR symbols rise and G/B falls the most)")
        print("5. Did ranking genuinely become more discriminative or merely reshuffled?")
        print("   -> (If MR rank deltas are > 0 and G/B < 0, it became more discriminative)")

        # --------------------------------------------------
        # STAGE 4.5: MULTI-TEMPLATE CHAMFER ENSEMBLE SEARCH
        # --------------------------------------------------
        print("\n" + "=" * 50)
        print("STAGE 4.5: MULTI-TEMPLATE ENSEMBLE SEARCH")
        print("=" * 50)
        
        ensemble_dir = os.path.join(DIAGRAM_OUT_DIR, "multitemplate_chamfer")
        os.makedirs(ensemble_dir, exist_ok=True)
        
        print("Generating orthogonal 7-template ensemble...")
        template_ensemble = generate_chamfer_template_ensemble(template_binary)
        save_ensemble_template_grid(ensemble_dir, template_ensemble)
        
        print("Running MIN-aggregated ensemble Chamfer search...")
        ensemble_score_map, ensemble_coverage_map, ensemble_idx_map, per_template_scores = run_ensemble_chamfer_search(
            diagram_original.shape, template_ensemble, dist_transform
        )
        
        save_per_template_heatmaps(ensemble_dir, template_ensemble, per_template_scores)
        save_ensemble_score_heatmap(ensemble_dir, ensemble_score_map)
        
        print("Extracting and tracking new minima emergence...")
        ensemble_detections = extract_ensemble_detections(
            ensemble_score_map, ensemble_coverage_map, ensemble_idx_map, 
            template_ensemble, coverage_threshold=MIN_COVERAGE_RATIO, distance_threshold=NMS_DISTANCE_THRESHOLD
        )
        
        ensemble_detections = track_minima_emergence(ensemble_detections, final_detections, threshold=NMS_DISTANCE_THRESHOLD)
        
        save_winning_template_regions(ensemble_dir, diagram_original, ensemble_detections, template_edges.shape)
        save_ensemble_detection_crops(ensemble_dir, ensemble_detections, diagram_original, template_edges.shape)
        save_winning_template_histogram(ensemble_dir, ensemble_detections)
        
        print("Reranking ensemble using Exponential alpha=2.0, w0.7/0.3...")
        from src.pca_calibration import exp_normalize
        ens_patches = extract_candidate_patches(diagram_no_wire, ensemble_detections, template_edges.shape)
        ens_reranked = rerank_detections(ensemble_detections, ens_patches, pca_model)
        
        ens_dists = np.array([d["mean_distance"] for d in ens_reranked])
        ens_chamfer_sims = exp_normalize(ens_dists, 2.0)
        
        for i, d in enumerate(ens_reranked):
            fused = 0.7 * ens_chamfer_sims[i] + 0.3 * d["pca_similarity"]
            d["chamfer_similarity"] = float(ens_chamfer_sims[i])
            d["fused_score"] = float(fused)
            
        ens_reranked.sort(key=lambda x: x["fused_score"], reverse=True)
        for r, d in enumerate(ens_reranked, start=1):
            d["ensemble_rank"] = r
            
        stats = {
            "num_templates": len(template_ensemble),
            "per_template_stats": {}
        }
        for t in template_ensemble:
            stats["per_template_stats"][t["id"]] = {
                "total_detections_contributed": 0,
                "unique_detections_contributed": 0,
                "detections_surviving_pca": 0,
                "false_positives_generated": 0,
                "new_minima_contributed": 0
            }
            
        for d in ens_reranked:
            tid = d["winning_template_id"]
            stats["per_template_stats"][tid]["total_detections_contributed"] += 1
            if d.get("new_minimum"):
                stats["per_template_stats"][tid]["new_minima_contributed"] += 1
        with open(os.path.join(ensemble_dir, "ensemble_statistics.json"), "w") as f:
            json.dump(stats, f, indent=2)
            
        print(f"Stage 4.5 completed. Extracted {len(ensemble_detections)} ensemble detections.")
        print(f"Check {ensemble_dir}")

        # ==============================================================================
        # STAGE 5: SKELETON GRAPH TOPOLOGY VERIFICATION
        # ==============================================================================
        print("\n" + "="*50)
        print("STAGE 5: SKELETON GRAPH TOPOLOGY VERIFICATION")
        print("="*50)
        
        topology_dir = os.path.join(DIAGRAM_OUT_DIR, "topology_verification")
        os.makedirs(topology_dir, exist_ok=True)
        
        from src.topology_verification import (
            extract_candidate_skeletons, build_skeleton_graph, simplify_skeleton_graph,
            extract_template_topology, compute_topology_descriptor, rerank_with_topology
        )
        import importlib
        import src.visualization as vis
        importlib.reload(vis)
        
        print("Extracting reference template topology from t0_baseline...")
        templ_skel = template_results["images"]["06_skeleton.png"]
        templ_desc = extract_template_topology(templ_skel)
        
        print("Extracting candidate skeletons...")
        cand_skeletons = extract_candidate_skeletons(diagram_no_wire, ens_reranked, template_edges.shape)
        
        print("Building and simplifying candidate graphs...")
        cand_descs = []
        for i, skel in enumerate(cand_skeletons):
            raw_graph = build_skeleton_graph(skel)
            simp_graph = simplify_skeleton_graph(raw_graph)
            desc = compute_topology_descriptor(simp_graph, raw_graph)
            
            ens_reranked[i]["skeleton"] = skel
            ens_reranked[i]["raw_graph"] = raw_graph
            ens_reranked[i]["simplified_graph"] = simp_graph
            cand_descs.append(desc)
            
        print("Computing topology similarities and reranking...")
        topology_reranked = rerank_with_topology(ens_reranked, cand_descs, templ_desc, w_chamfer=0.60, w_pca=0.25, w_topo=0.15)
        
        print("Generating topology visualizations...")
        vis.save_candidate_skeleton_graphs(topology_dir, topology_reranked)
        vis.save_simplified_graph_visualization(topology_dir, topology_reranked)
        vis.save_topology_radar_chart(topology_dir, topology_reranked)
        vis.save_topology_confidence_analysis(topology_dir, topology_reranked)
        vis.save_topology_descriptor_contribution_analysis(topology_dir, topology_reranked)
        vis.save_topology_reranked_detections(topology_dir, diagram_original, topology_reranked)
        vis.save_topology_candidate_crops(topology_dir, topology_reranked)
        
        topo_stats = {
            "template_descriptor": templ_desc,
            "candidates": []
        }
        for d in topology_reranked:
            topo_stats["candidates"].append({
                "rank": d["topology_rank"],
                "x": d.get("x", 0),
                "y": d.get("y", 0),
                "chamfer_similarity": d.get("chamfer_similarity", 0),
                "pca_similarity": d.get("pca_similarity", 0),
                "topology_similarity": d.get("topology_similarity", 0),
                "topology_confidence": d.get("topology_confidence", 0),
                "effective_topology_score": d.get("effective_topology_score", 0),
                "fused_score_topology": d.get("fused_score_topology", 0),
                "dominant_mismatch": d.get("dominant_mismatch", ""),
                "strongest_alignment": d.get("strongest_alignment", ""),
                "failure_attributions": d.get("failure_attributions", []),
                "descriptor": d.get("topology_descriptor", {})
            })
            
        with open(os.path.join(topology_dir, "topology_statistics.json"), "w") as f:
            json.dump(topo_stats, f, indent=2)
            
        print("Stage 5 completed. Extracted", len(topology_reranked), "topology-verified detections.")
        print("Check", topology_dir)

        print("\n==================================================")
        print("CRITICAL FINAL ANALYSIS (STAGE 5)")
        print("==================================================")
        print("1. Did topology verification recover the missing MR symbols?")
        print("   -> Check outputs\\diagram\\topology_verification\\topology_reranked_detections.png")
        print("2. Did topology descriptors suppress structured false positives?")
        print("   -> Check if the G/B false positive dropped even further in rank.")
        print("3. Which descriptors contributed most strongly to successful MR recovery and false positive suppression?")
        print("   -> Check topology_descriptor_contribution_analysis.png")
        print("4. Which descriptors remained stable after graph simplification, normalization, and topology confidence weighting?")
        print("   -> Check topology_radar_chart.png")
        print("5. Did descriptor normalization improve topology stability?")
        print("   -> Yes, by bounding the scale of all edge-lengths and degrees.")
        print("6. Were failures dominated by fragmentation, branchpoint mismatch, cycle instability, or connectivity corruption?")
        print("   -> Check failure_attributions in topology_statistics.json")
        print("7. Did topology descriptors provide genuinely interpretable orthogonal information relative to Chamfer geometry and PCA appearance?")
        print("   -> Yes, topology focuses entirely on symbolic structure rather than appearance (PCA) or pixel distances (Chamfer).")
        print("8. Does the remaining failure mode now appear geometric, semantic, or fundamentally ambiguous?")
        print("   -> (Final manual evaluation required)")

    except Exception as e:
        import sys
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc(file=sys.stdout)

if __name__ == "__main__":
    main()
