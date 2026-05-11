import os
import cv2
import json
import numpy as np

def safe_load_image(image_path: str, color_mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Safely loads an image using OpenCV.
    Raises FileNotFoundError if the image does not exist or cannot be read.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
    
    image = cv2.imread(image_path, color_mode)
    if image is None:
        raise ValueError(f"Failed to decode image at: {image_path}. Ensure it is a valid image file.")
    
    return image

def save_preprocessing_stats(stats: dict, output_path: str) -> None:
    """
    Saves the structured statistics dictionary to a formatted JSON file.
    
    This metadata system is required for reproducibility, parameter tracking,
    future benchmarking, and debugging preprocessing regressions.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
