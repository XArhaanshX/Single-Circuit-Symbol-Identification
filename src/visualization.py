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
