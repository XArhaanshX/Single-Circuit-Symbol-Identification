import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def save_debug_images(output_dir: str, images_dict: dict) -> None:
    """
    Saves intermediate images sequentially to the output directory.
    
    Args:
        output_dir: Directory where the images will be saved.
        images_dict: Dictionary mapping filenames (e.g., '01_original.png') to image arrays.
    """
    os.makedirs(output_dir, exist_ok=True)
    for filename, img in images_dict.items():
        if img is None:
            continue
            
        path = os.path.join(output_dir, filename)
        
        # Ensure image is in a format cv2 can write
        if len(img.shape) == 2:
            # Grayscale or binary
            cv2.imwrite(path, img)
        elif len(img.shape) == 3:
            # Color (ensure BGR for cv2)
            cv2.imwrite(path, img)
        else:
            print(f"Warning: Unexpected image shape {img.shape} for {filename}, skipping.")

def save_pipeline_overview(output_dir: str, name: str, images_dict: dict) -> None:
    """
    Generates and saves a combined matplotlib subplot grid displaying all 
    intermediate stages side-by-side.
    
    Args:
        output_dir: Directory where the overview image will be saved.
        name: Name of the output file (e.g., 'template_pipeline_overview.png').
        images_dict: Dictionary mapping titles to image arrays.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_images = len(images_dict)
    if num_images == 0:
        return
        
    cols = 3
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = np.array(axes).flatten()
    
    for ax, (title, img) in zip(axes, images_dict.items()):
        if img is None:
            ax.axis('off')
            continue
            
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            # Convert BGR to RGB for matplotlib
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
        ax.set_title(title)
        ax.axis('off')
        
    # Hide any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name), dpi=150)
    plt.close(fig)

def save_intensity_histogram(output_dir: str, name: str, image: np.ndarray) -> None:
    """
    Generates and saves a grayscale intensity histogram.
    
    Args:
        output_dir: Directory where the histogram will be saved.
        name: Name of the output file (e.g., 'histogram_gray.png').
        image: Grayscale image array to compute the histogram from.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(image.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.7)
    ax.set_title("Grayscale Intensity Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', alpha=0.75)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name), dpi=150)
    plt.close(fig)

def save_colored_components(output_dir: str, num_labels: int, labels_img: np.ndarray) -> None:
    """
    Creates components_colored.png mapping each component ID to a distinct color on a black background.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Map labels to distinct colors
    # We use a random colormap, label 0 is black
    np.random.seed(42) # For reproducibility
    colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0] # Background is black
    
    colored_components = colors[labels_img]
    
    path = os.path.join(output_dir, "components_colored.png")
    cv2.imwrite(path, colored_components)

def save_candidate_overlays(output_dir: str, name: str, original_img: np.ndarray, candidates: list, color: tuple) -> None:
    """
    Overlays a list of candidate bounding boxes onto the original diagram image.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(original_img.shape) == 2:
        vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = original_img.copy()
        
    for c in candidates:
        x, y, w, h = c["bbox"]
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
        
    path = os.path.join(output_dir, name)
    cv2.imwrite(path, vis_img)

def save_final_overlay(output_dir: str, name: str, original_img: np.ndarray, candidates: list) -> None:
    """
    Overlays final bounding boxes along with their component ID text.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(original_img.shape) == 2:
        vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = original_img.copy()
        
    for c in candidates:
        x, y, w, h = c["bbox"]
        label = c["label"]
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis_img, f"ID:{label}", (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    path = os.path.join(output_dir, name)
    cv2.imwrite(path, vis_img)

def save_candidate_crops(output_dir: str, original_img: np.ndarray, candidates: list) -> None:
    """
    Crops final candidate regions (with 5px padding clipped to bounds) from the ORIGINAL diagram.
    Updates the candidate dictionary with the crop_path.
    """
    crops_dir = os.path.join(output_dir, "candidate_crops")
    os.makedirs(crops_dir, exist_ok=True)
    img_h, img_w = original_img.shape[:2]
    
    for i, c in enumerate(candidates):
        x, y, w, h = c["bbox"]
        
        # 5px padding safely clipped to image bounds
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        
        crop = original_img[y1:y2, x1:x2]
        
        filename = f"candidate_{i+1:03d}.png"
        path = os.path.join(crops_dir, filename)
        cv2.imwrite(path, crop)
        
        # update candidate with crop path
        c["crop_path"] = path

def save_continuity_comparison_grid(output_dir: str, original_no_wire: np.ndarray, experiment_results: dict) -> None:
    """
    Generates a side-by-side comparison grid of all continuity recovery experiments.
    Shows the original no_wire image alongside the candidate overlays from each experiment.
    
    Args:
        output_dir: Directory to save the grid.
        original_no_wire: The original no_wire binary image.
        experiment_results: Dictionary mapping kernel_name -> overlaid candidate image.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    num_experiments = len(experiment_results)
    num_plots = num_experiments + 1 # +1 for original no_wire
    
    cols = 2
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4))
    axes = np.array(axes).flatten()
    
    # Plot original no_wire
    axes[0].imshow(original_no_wire, cmap='gray')
    axes[0].set_title("Original no_wire (Stage 1)")
    axes[0].axis('off')
    
    # Plot each experiment
    for i, (kernel_name, overlay_img) in enumerate(experiment_results.items(), start=1):
        if len(overlay_img.shape) == 3:
            axes[i].imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        else:
            axes[i].imshow(overlay_img, cmap='gray')
            
        axes[i].set_title(f"Experiment: {kernel_name}")
        axes[i].axis('off')
        
    # Hide any unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "continuity_comparison_grid.png"), dpi=150)
    plt.close(fig)

def save_diagnostic_overlays(output_dir: str, original_img: np.ndarray, components_metadata: list) -> None:
    """
    Generates color-coded overlays for filter failures.
    Red: rejected by area
    Orange: rejected by aspect
    Yellow: rejected by density
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(original_img.shape) == 2:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = original_img.copy()
        
    img_area = img_rgb.copy()
    img_aspect = img_rgb.copy()
    img_density = img_rgb.copy()
    
    # BGR format
    color_red = (0, 0, 255)
    color_orange = (0, 165, 255) 
    color_yellow = (0, 255, 255)
    
    for c in components_metadata:
        x, y, w, h = c["bbox"]
        stage = c["rejected_stage"]
        
        if stage == "area":
            cv2.rectangle(img_area, (x, y), (x+w, y+h), color_red, 2)
            cv2.putText(img_area, str(c["label"]), (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_red, 1)
        elif stage == "aspect":
            cv2.rectangle(img_aspect, (x, y), (x+w, y+h), color_orange, 2)
            cv2.putText(img_aspect, str(c["label"]), (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_orange, 1)
        elif stage == "density":
            cv2.rectangle(img_density, (x, y), (x+w, y+h), color_yellow, 2)
            cv2.putText(img_density, str(c["label"]), (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_yellow, 1)
            
    cv2.imwrite(os.path.join(output_dir, "rejected_by_area.png"), img_area)
    cv2.imwrite(os.path.join(output_dir, "rejected_by_aspect.png"), img_aspect)
    cv2.imwrite(os.path.join(output_dir, "rejected_by_density.png"), img_density)

def save_large_components_overlay(output_dir: str, original_img: np.ndarray, components_metadata: list, template_area: int) -> None:
    """
    Overlays ALL large connected components where foreground_area > template_area_reference.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(original_img.shape) == 2:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = original_img.copy()
        
    color = (255, 0, 255) # Magenta for large structures
    
    for c in components_metadata:
        if c["foreground_area"] > template_area:
            x, y, w, h = c["bbox"]
            cv2.rectangle(img_rgb, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img_rgb, f"ID:{c['label']} Area:{c['foreground_area']}", (x, max(y-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
    cv2.imwrite(os.path.join(output_dir, "large_components_overlay.png"), img_rgb)

def save_all_component_crops(output_dir: str, original_img: np.ndarray, components: list) -> None:
    """
    Saves cropped PNGs of EVERY component before filtering into all_component_crops/.
    """
    crops_dir = os.path.join(output_dir, "all_component_crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    img_h, img_w = original_img.shape[:2]
    
    for c in components:
        x, y, w, h = c["bbox"]
        label = c["label"]
        
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        
        crop = original_img[y1:y2, x1:x2]
        
        filename = f"component_{label:03d}.png"
        path = os.path.join(crops_dir, filename)
        cv2.imwrite(path, crop)

def save_distance_transform_vis(output_dir: str, dist_transform: np.ndarray) -> None:
    """Saves normalized distance transform."""
    os.makedirs(output_dir, exist_ok=True)
    norm_dt = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(os.path.join(output_dir, "distance_transform.png"), norm_dt)

def save_chamfer_heatmap(output_dir: str, name: str, score_map: np.ndarray, cmap: str = 'viridis') -> None:
    """Saves a matplotlib heatmap of the score_map."""
    os.makedirs(output_dir, exist_ok=True)
    valid_mask = ~np.isinf(score_map)
    if not np.any(valid_mask):
        return
        
    min_val = np.min(score_map[valid_mask])
    max_val = np.max(score_map[valid_mask])
    
    if max_val > min_val:
        normalized = (score_map - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(score_map)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    masked_data = np.ma.masked_where(~valid_mask, normalized)
    cax = ax.imshow(masked_data, cmap=cmap)
    fig.colorbar(cax)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, name), dpi=150)
    plt.close(fig)

def save_top_chamfer_matches(output_dir: str, original_img: np.ndarray, matches: list, template_shape: tuple) -> None:
    """Overlays the top Chamfer matches on the diagram."""
    os.makedirs(output_dir, exist_ok=True)
    th, tw = template_shape[:2]
    vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(original_img.shape) == 2 else original_img.copy()
    
    for c in matches:
        x, y = c["x"], c["y"]
        cv2.rectangle(vis_img, (x, y), (x+tw, y+th), (0, 0, 255), 2)
        
    cv2.imwrite(os.path.join(output_dir, "top_chamfer_matches.png"), vis_img)

def save_best_template_alignment(output_dir: str, diagram_edges: np.ndarray, template_edge_points: list, best_match: dict) -> None:
    """Overlays the best translated template edges onto the diagram edges."""
    os.makedirs(output_dir, exist_ok=True)
    vis_img = cv2.cvtColor(diagram_edges, cv2.COLOR_GRAY2BGR)
    
    dx, dy = best_match["x"], best_match["y"]
    for (px, py) in template_edge_points:
        nx, ny = px + dx, py + dy
        if 0 <= ny < vis_img.shape[0] and 0 <= nx < vis_img.shape[1]:
            vis_img[ny, nx] = [0, 0, 255] # Red template edges
            
    cv2.imwrite(os.path.join(output_dir, "best_template_alignment.png"), vis_img)

def save_individual_top_matches(output_dir: str, original_img: np.ndarray, template_edge_points: list, matches: list, template_shape: tuple) -> None:
    """Saves individual crops of the top matches with template edge overlays."""
    matches_dir = os.path.join(output_dir, "top_matches")
    os.makedirs(matches_dir, exist_ok=True)
    
    th, tw = template_shape[:2]
    img_h, img_w = original_img.shape[:2]
    pad = 10
    
    for i, m in enumerate(matches):
        x, y = m["x"], m["y"]
        
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + tw + pad), min(img_h, y + th + pad)
        
        crop_base = original_img[y1:y2, x1:x2]
        crop_vis = cv2.cvtColor(crop_base, cv2.COLOR_GRAY2BGR) if len(crop_base.shape) == 2 else crop_base.copy()
        
        for (px, py) in template_edge_points:
            nx = (px + x) - x1
            ny = (py + y) - y1
            if 0 <= ny < crop_vis.shape[0] and 0 <= nx < crop_vis.shape[1]:
                crop_vis[ny, nx] = [0, 0, 255]
                
        cv2.putText(crop_vis, f"D:{m['mean_distance']:.1f} C:{m['coverage_ratio']:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        filename = f"match_{i+1:03d}.png"
        cv2.imwrite(os.path.join(matches_dir, filename), crop_vis)

def save_local_minima_overlay(output_dir: str, original_img: np.ndarray, minima: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(original_img.shape) == 2 else original_img.copy()
    
    for m in minima:
        x, y = m["x"], m["y"]
        cv2.circle(vis_img, (x, y), 3, (0, 165, 255), -1) # Orange dots for all minima
        
    cv2.imwrite(os.path.join(output_dir, "local_minima_overlay.png"), vis_img)

def save_final_nms_detections(output_dir: str, original_img: np.ndarray, detections: list, template_shape: tuple) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(original_img.shape) == 2 else original_img.copy()
    th, tw = template_shape[:2]
    
    for c in detections:
        x, y = c["x"], c["y"]
        rank = c["rank"]
        cv2.rectangle(vis_img, (x, y), (x+tw, y+th), (0, 255, 0), 2)
        cv2.putText(vis_img, f"#{rank} D:{c['mean_distance']:.1f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imwrite(os.path.join(output_dir, "final_nms_detections.png"), vis_img)

def save_minima_heatmap(output_dir: str, score_map: np.ndarray, minima: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    # Create a blank black image
    heatmap = np.zeros(score_map.shape, dtype=np.float32)
    
    for m in minima:
        x, y = m["x"], m["y"]
        # Inverse distance so better minima are brighter
        heatmap[y, x] = 1.0 / (m["mean_distance"] + 1e-5)
        
    # Dilate slightly so single pixels are visible
    kernel = np.ones((5,5), np.uint8)
    heatmap = cv2.dilate(heatmap, kernel)
    
    # Normalize for visualization
    if np.max(heatmap) > 0:
        heatmap = (heatmap / np.max(heatmap)) * 255
    heatmap = heatmap.astype(np.uint8)
    
    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    cv2.imwrite(os.path.join(output_dir, "minima_heatmap.png"), colored_heatmap)

def save_ranked_detections_vis(output_dir: str, original_img: np.ndarray, detections: list, template_shape: tuple) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(original_img.shape) == 2 else original_img.copy()
    th, tw = template_shape[:2]
    
    for c in detections:
        x, y = c["x"], c["y"]
        rank = c["rank"]
        # Draw box and detailed text
        cv2.rectangle(vis_img, (x, y), (x+tw, y+th), (255, 0, 0), 2)
        text = f"#{rank} D:{c['mean_distance']:.2f} C:{c['coverage_ratio']:.2f}"
        cv2.putText(vis_img, text, (x, y + th + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
    cv2.imwrite(os.path.join(output_dir, "ranked_detections.png"), vis_img)

def save_final_detection_crops(output_dir: str, original_img: np.ndarray, detections: list, template_shape: tuple) -> None:
    crops_dir = os.path.join(output_dir, "final_detection_crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    th, tw = template_shape[:2]
    img_h, img_w = original_img.shape[:2]
    pad = 20
    
    for c in detections:
        x, y = c["x"], c["y"]
        rank = c["rank"]
        
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(img_w, x + tw + pad), min(img_h, y + th + pad)
        
        crop = original_img[y1:y2, x1:x2]
        crop_vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR) if len(crop.shape) == 2 else crop.copy()
        
        # draw box relative to crop
        rx, ry = x - x1, y - y1
        cv2.rectangle(crop_vis, (rx, ry), (rx+tw, ry+th), (0, 255, 0), 2)
        
        # Draw text
        cv2.putText(crop_vis, f"#{rank} D:{c['mean_distance']:.1f} C:{c['coverage_ratio']:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        filename = f"detection_{rank:03d}.png"
        cv2.imwrite(os.path.join(crops_dir, filename), crop_vis)

def save_suppression_radius_overlay(output_dir: str, original_img: np.ndarray, detections: list, template_shape: tuple, radius: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(original_img.shape) == 2 else original_img.copy()
    th, tw = template_shape[:2]
    
    for c in detections:
        x, y = c["x"], c["y"]
        # Center of the detection box
        cx, cy = x + tw // 2, y + th // 2
        
        # Draw center point
        cv2.circle(vis_img, (cx, cy), 3, (0, 0, 255), -1)
        # Draw suppression radius
        cv2.circle(vis_img, (cx, cy), radius, (0, 255, 255), 2)
        
    cv2.imwrite(os.path.join(output_dir, "suppression_radius_overlay.png"), vis_img)

def save_pca_template_grid(output_dir: str, templates: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n = min(len(templates), 25)
    cols = 5
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2 * rows))
    axes = axes.flatten()
    
    for i in range(n):
        axes[i].imshow(templates[i], cmap='gray')
        axes[i].axis('off')
        
    for i in range(n, len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_template_grid.png"), dpi=150)
    plt.close(fig)

def save_pca_reranked_detections(output_dir: str, original_img: np.ndarray, detections: list, template_shape: tuple) -> None:
    os.makedirs(output_dir, exist_ok=True)
    vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(original_img.shape) == 2 else original_img.copy()
    th, tw = template_shape[:2]
    
    for c in detections:
        x, y = c["x"], c["y"]
        rank = c["pca_rank"]
        cv2.rectangle(vis_img, (x, y), (x+tw, y+th), (255, 0, 255), 2)
        
        text1 = f"#{rank} F:{c['fused_score']:.2f}"
        text2 = f"C:{c['chamfer_similarity']:.2f} P:{c['pca_similarity']:.2f}"
        cv2.putText(vis_img, text1, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        cv2.putText(vis_img, text2, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
        
    cv2.imwrite(os.path.join(output_dir, "pca_reranked_detections.png"), vis_img)

def save_pca_variance_spectrum(output_dir: str, pca_model) -> None:
    os.makedirs(output_dir, exist_ok=True)
    evr = pca_model.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    ax1.bar(range(1, len(evr) + 1), evr, alpha=0.7, color='b', label='Individual')
    ax1.set_xlabel('Principal Components')
    ax1.set_ylabel('Explained Variance Ratio', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumulative) + 1), cumulative, 'r-o', label='Cumulative')
    ax2.set_ylabel('Cumulative Explained Variance', color='r')
    ax2.tick_params('y', colors='r')
    
    plt.title('PCA Explained Variance Spectrum\n(1-2 components dominating indicates insufficient augmentation)')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_variance_spectrum.png"), dpi=150)
    plt.close(fig)

def save_template_pairwise_distances(output_dir: str, templates: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n = min(len(templates), 100)
    flat = np.array([t.flatten() for t in templates[:n]], dtype=np.float32)
    
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(flat[i] - flat[j])
            
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(dist_matrix, cmap='viridis')
    fig.colorbar(cax)
    plt.title('Template Pairwise L2 Distances\n(Validates low-entropy coherent manifold)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "template_pairwise_distances.png"), dpi=150)
    plt.close(fig)

def save_candidate_alignment_normalization(output_dir: str, raw_patch: np.ndarray, normalized_patch: np.ndarray, rank: int) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw_patch, cmap='gray')
    axes[0].set_title('Raw Candidate Patch')
    axes[0].axis('off')
    
    axes[1].imshow(normalized_patch, cmap='gray')
    axes[1].set_title('Centroid-Normalized Patch')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"candidate_alignment_normalization_{rank:03d}.png"), dpi=150)
    plt.close(fig)

def save_pca_candidate_crops(output_dir: str, detections: list) -> None:
    crops_dir = os.path.join(output_dir, "pca_candidate_crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    for c in detections:
        rank = c["pca_rank"]
        orig = c["normalized_patch"]
        recon = c["reconstructed_patch"]
        
        recon_vis = np.clip(recon, 0, 255).astype(np.uint8)
        orig_vis = orig.astype(np.uint8)
        
        side_by_side = np.hstack((orig_vis, recon_vis))
        side_by_side[:, orig_vis.shape[1]-1:orig_vis.shape[1]+1] = 128
        
        sbs_bgr = cv2.cvtColor(side_by_side, cv2.COLOR_GRAY2BGR)
        
        cv2.putText(sbs_bgr, f"R:{rank} F:{c['fused_score']:.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(sbs_bgr, f"Orig | Recon", (5, sbs_bgr.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        filename = f"pca_detection_{rank:03d}.png"
        cv2.imwrite(os.path.join(crops_dir, filename), sbs_bgr)
        
def save_pca_reconstruction_examples(output_dir: str, detections: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    n = min(len(detections), 6)
    
    fig, axes = plt.subplots(n, 2, figsize=(6, 2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for i in range(n):
        idx = i if i < 3 else len(detections) - (n - i)
        det = detections[idx]
        orig = det["normalized_patch"]
        recon = det["reconstructed_patch"]
        
        axes[i, 0].imshow(orig, cmap='gray')
        axes[i, 0].set_title(f"Rank {det['pca_rank']} Orig")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(recon, cmap='gray')
        axes[i, 1].set_title(f"Recon (Err: {det['pca_error']:.1f})")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pca_reconstruction_examples.png"), dpi=150)
    plt.close(fig)

def save_similarity_scatter_plot(output_dir: str, detections: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for d in detections:
        x = d["chamfer_similarity"]
        y = d["pca_similarity"]
        ax.scatter(x, y, color='blue')
        ax.annotate(f"{d['detection_id']}\nR:{d['pca_rank']}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=9)
        
    ax.set_xlabel('Chamfer Similarity')
    ax.set_ylabel('PCA Similarity')
    ax.set_title('Chamfer vs PCA Similarity\n(Does PCA provide orthogonal discriminative signal?)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chamfer_vs_pca_scatter.png"), dpi=150)
    plt.close(fig)

def save_calibration_comparison_grid(output_dir: str, baseline_path: str, pca_path: str, exp_path: str, log_path: str) -> None:
    def add_title(path, text):
        img = cv2.imread(path)
        if img is None: 
            return np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img
        
    baseline = add_title(baseline_path, "Baseline")
    max_pca = add_title(pca_path, "Strongest PCA")
    exp_img = add_title(exp_path, "Strongest Exp")
    log_img = add_title(log_path, "Strongest Logistic")
    
    h1, w1 = baseline.shape[:2]
    h2, w2 = max_pca.shape[:2]
    h3, w3 = exp_img.shape[:2]
    h4, w4 = log_img.shape[:2]
    
    if h1 == h2 == h3 == h4 and w1 == w2 == w3 == w4:
        top = np.hstack((baseline, max_pca))
        bottom = np.hstack((exp_img, log_img))
        grid = np.vstack((top, bottom))
        cv2.imwrite(os.path.join(output_dir, "calibration_comparison_grid.png"), grid)

def save_ensemble_template_grid(output_dir: str, templates: list) -> None:
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, t in enumerate(templates):
        axes[i].imshow(t["image"], cmap='gray')
        axes[i].set_title(f"{t['id']}\n({t['type']})")
        axes[i].axis('off')
    for j in range(len(templates), 8):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "template_ensemble_grid.png"), dpi=150)
    plt.close(fig)

def save_per_template_heatmaps(output_dir: str, templates: list, score_maps: list) -> None:
    hm_dir = os.path.join(output_dir, "per_template_heatmaps")
    os.makedirs(hm_dir, exist_ok=True)
    for i, (t, sm) in enumerate(zip(templates, score_maps)):
        plt.figure(figsize=(10, 8))
        sm_clip = np.clip(sm, 0, np.percentile(sm[np.isfinite(sm)], 95))
        plt.imshow(sm_clip, cmap='viridis_r')
        plt.colorbar(label='Chamfer Distance')
        plt.title(f"Score Map: {t['id']}")
        plt.axis('off')
        plt.savefig(os.path.join(hm_dir, f"score_map_{t['id']}.png"), dpi=150)
        plt.close()

def save_ensemble_score_heatmap(output_dir: str, ensemble_score_map: np.ndarray) -> None:
    plt.figure(figsize=(10, 8))
    sm_clip = np.clip(ensemble_score_map, 0, np.percentile(ensemble_score_map[np.isfinite(ensemble_score_map)], 95))
    plt.imshow(sm_clip, cmap='viridis_r')
    plt.colorbar(label='Min Chamfer Distance')
    plt.title('Ensemble MIN Score Map')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "ensemble_score_heatmap.png"), dpi=150)
    plt.close()

def save_winning_template_regions(output_dir: str, original_img: np.ndarray, detections: list, template_shape: tuple) -> None:
    vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(original_img.shape) == 2 else original_img.copy()
    th, tw = template_shape[:2]
    for c in detections:
        x, y = c["x"], c["y"]
        cv2.rectangle(vis_img, (x, y), (x+tw, y+th), (0, 255, 255), 2)
        text1 = f"{c['winning_template_id']}"
        text2 = f"New: {'Yes' if c.get('new_minimum') else 'No'}"
        cv2.putText(vis_img, text1, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(vis_img, text2, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    cv2.imwrite(os.path.join(output_dir, "winning_template_regions.png"), vis_img)

def save_ensemble_comparison_grid(output_dir: str, baseline_path: str, raw_ensemble_path: str, pca_ensemble_path: str) -> None:
    def add_title(path, text):
        img = cv2.imread(path)
        if img is None: return np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img
    
    b = add_title(baseline_path, "Baseline (Single Template)")
    r = add_title(raw_ensemble_path, "Raw Ensemble")
    p = add_title(pca_ensemble_path, "PCA Reranked Ensemble")
    
    if b.shape == r.shape == p.shape:
        grid = np.hstack((b, r, p))
        cv2.imwrite(os.path.join(output_dir, "ensemble_vs_baseline_comparison.png"), grid)

def save_ensemble_detection_crops(output_dir: str, detections: list, original_img: np.ndarray, template_shape: tuple) -> None:
    crops_dir = os.path.join(output_dir, "detection_crops")
    os.makedirs(crops_dir, exist_ok=True)
    th, tw = template_shape[:2]
    for c in detections:
        x, y = c["x"], c["y"]
        if y+th <= original_img.shape[0] and x+tw <= original_img.shape[1]:
            crop = original_img[y:y+th, x:x+tw]
            cv2.imwrite(os.path.join(crops_dir, f"crop_x{x}_y{y}_{c['winning_template_id']}.png"), crop)

def save_winning_template_histogram(output_dir: str, detections: list) -> None:
    import collections
    counts = collections.Counter(d["winning_template_id"] for d in detections)
    labels = list(counts.keys())
    values = list(counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Winning Detection Frequency')
    plt.title('Winning Template Frequency Histogram')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "winning_template_histogram.png"), dpi=150)
    plt.close()

def save_template_overlap_matrix(output_dir: str, templates: list, detections: list) -> None:
    # A simplified overlap matrix based on spatial proximity of detections
    # In a true ensemble, min aggregation only picks one per pixel.
    # To see overlap, we can check how close detections from different templates would be.
    # For now, we will just create a placeholder matrix since true independent NMS per template is not saved.
    plt.figure(figsize=(8, 6))
    plt.text(0.5, 0.5, "Overlap Matrix Placeholder\n(Requires independent per-template NMS)", ha='center', va='center')
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, "template_overlap_matrix.png"), dpi=150)
    plt.close()

# ==============================================================================
# STAGE 5: TOPOLOGY VERIFICATION VISUALIZATIONS
# ==============================================================================

def save_candidate_skeleton_graphs(output_dir, candidates):
    out_dir = os.path.join(output_dir, "candidate_skeleton_graphs")
    os.makedirs(out_dir, exist_ok=True)
    for i, c in enumerate(candidates):
        if "skeleton" not in c:
            continue
        skel = c["skeleton"]
        vis = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
        if "raw_graph" in c:
            for y, x in c["raw_graph"]["endpoints"]:
                cv2.circle(vis, (x, y), 2, (255, 0, 0), -1)
            for y, x in c["raw_graph"]["branchpoints"]:
                cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(out_dir, f"graph_cand_{i+1:03d}.png"), vis)

def save_simplified_graph_visualization(output_dir, candidates):
    out_dir = os.path.join(output_dir, "simplified_graphs")
    os.makedirs(out_dir, exist_ok=True)
    for i, c in enumerate(candidates):
        if "skeleton" not in c or "simplified_graph" not in c:
            continue
        skel = c["skeleton"]
        simp = c["simplified_graph"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(skel, cmap='gray')
        ax1.set_title("Raw Skeleton")
        ax1.axis('off')
        vis_s = np.zeros((*skel.shape, 3), dtype=np.uint8)
        for edge in simp["edges"]:
            for pt in edge["path"]:
                vis_s[pt[0], pt[1]] = (0, 255, 0)
        for y, x in simp["endpoints"]:
            cv2.circle(vis_s, (x, y), 2, (255, 0, 0), -1)
        for y, x in simp["branchpoints"]:
            cv2.circle(vis_s, (x, y), 2, (0, 0, 255), -1)
        ax2.imshow(vis_s)
        ax2.set_title("Simplified Graph")
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"simplified_{i+1:03d}.png"), dpi=100)
        plt.close(fig)

def save_topology_radar_chart(output_dir, candidates):
    os.makedirs(output_dir, exist_ok=True)
    if len(candidates) < 2:
        return
    keys = ["endpoint_count", "branchpoint_count", "connected_component_count", "estimated_cycle_count"]
    top_vals = [candidates[0].get("topology_descriptor", {}).get(k, 0) for k in keys]
    bot_vals = [candidates[-1].get("topology_descriptor", {}).get(k, 0) for k in keys]
    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width/2, top_vals, width, label='Rank 1')
    ax.bar(x + width/2, bot_vals, width, label='Lowest Rank')
    ax.set_ylabel('Count')
    ax.set_title('Topology Descriptor Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=15)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topology_radar_chart.png"), dpi=150)
    plt.close()

def save_topology_confidence_analysis(output_dir, candidates):
    os.makedirs(output_dir, exist_ok=True)
    ids = [f"#{c.get('topology_rank', i)}" for i, c in enumerate(candidates)]
    sims = [c.get("topology_similarity", 0) for c in candidates]
    confs = [c.get("topology_confidence", 0) for c in candidates]
    effs = [c.get("effective_topology_score", 0) for c in candidates]
    x = np.arange(len(ids))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, sims, width, label='Raw Similarity')
    ax.bar(x, confs, width, label='Confidence')
    ax.bar(x + width, effs, width, label='Effective Score')
    ax.set_title('Topology Confidence Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topology_confidence_analysis.png"), dpi=150)
    plt.close()

def save_topology_descriptor_contribution_analysis(output_dir, candidates):
    os.makedirs(output_dir, exist_ok=True)
    if not candidates:
        return
    worst = candidates[-1]
    contribs = worst.get("descriptor_contributions", {})
    if not contribs:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(contribs.keys(), contribs.values(), color='red', alpha=0.6)
    ax.set_title("Descriptor Mismatch (Lowest Rank)")
    ax.set_ylabel("Normalized Difference")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topology_descriptor_contribution_analysis.png"), dpi=150)
    plt.close()

def save_topology_reranked_detections(output_dir, original_img, candidates):
    os.makedirs(output_dir, exist_ok=True)
    if len(original_img.shape) == 2:
        vis_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    else:
        vis_img = original_img.copy()
    for c in candidates:
        x, y = c.get("x", 0), c.get("y", 0)
        rank = c.get("topology_rank", 0)
        score = c.get("fused_score_topology", 0.0)
        cv2.circle(vis_img, (x, y), 20, (255, 0, 255), 2)
        cv2.putText(vis_img, f"R{rank}:{score:.2f}", (x+22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.imwrite(os.path.join(output_dir, "topology_reranked_detections.png"), vis_img)

def save_topology_candidate_crops(output_dir, candidates):
    out_dir = os.path.join(output_dir, "topology_candidate_crops")
    os.makedirs(out_dir, exist_ok=True)
    for i, c in enumerate(candidates):
        if "skeleton" not in c:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.imshow(c["skeleton"], cmap='gray')
        ax1.set_title(f"Skeleton (Rank {c.get('topology_rank', '?')})")
        ax1.axis('off')
        lines = [
            f"Fused: {c.get('fused_score_topology', 0):.3f}",
            f"Topo Sim: {c.get('topology_similarity', 0):.3f}",
            f"Confidence: {c.get('topology_confidence', 0):.3f}",
            f"Mismatch: {c.get('dominant_mismatch', 'N/A')}",
            f"Failures: {', '.join(c.get('failure_attributions', [])) or 'None'}"
        ]
        ax2.text(0.05, 0.95, '\n'.join(lines), transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"topo_cand_{i+1:03d}.png"), dpi=100)
        plt.close(fig)
