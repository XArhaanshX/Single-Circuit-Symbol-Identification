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
