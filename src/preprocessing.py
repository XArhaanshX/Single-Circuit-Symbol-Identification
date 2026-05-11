import cv2
import numpy as np
import skimage.morphology

# ==================================================
# CONSTANTS CONFIGURATION
# ==================================================
MEDIAN_KERNEL_SIZE = 3

CANNY_LOW = 50
CANNY_HIGH = 150

ADAPTIVE_BLOCKSIZE = 21
ADAPTIVE_C = 10

WIRE_KERNEL_WIDTH = 50
WIRE_KERNEL_HEIGHT = 1

def _compute_statistics(binary_image: np.ndarray, edges: np.ndarray, skeleton: np.ndarray, wire_removed_pixels: int = None) -> dict:
    """Helper to compute standard preprocessing statistics."""
    height, width = binary_image.shape
    total_pixels = height * width
    
    # Binary images from OpenCV are typically 0 and 255.
    foreground_count = int(np.count_nonzero(binary_image))
    foreground_ratio = foreground_count / total_pixels if total_pixels > 0 else 0
    
    edge_count = int(np.count_nonzero(edges))
    skeleton_count = int(np.count_nonzero(skeleton))
    
    stats = {
        "image_dimensions": {
            "height": height,
            "width": width
        },
        "foreground_pixel_count": foreground_count,
        "foreground_ratio": foreground_ratio,
        "edge_pixel_count": edge_count,
        "skeleton_pixel_count": skeleton_count,
        "parameters": {
            "median_kernel_size": MEDIAN_KERNEL_SIZE,
            "canny_low": CANNY_LOW,
            "canny_high": CANNY_HIGH,
            "adaptive_blocksize": ADAPTIVE_BLOCKSIZE,
            "adaptive_c": ADAPTIVE_C,
            "wire_kernel_width": WIRE_KERNEL_WIDTH,
            "wire_kernel_height": WIRE_KERNEL_HEIGHT
        }
    }
    
    if wire_removed_pixels is not None:
        stats["wire_mask_removed_percentage"] = (wire_removed_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
    return stats

def preprocess_template(image: np.ndarray) -> dict:
    """
    Preprocesses the template image.
    
    Operations:
    1. Grayscale
    2. Median Blur
    3. Otsu Binarization
    4. Canny Edges
    5. Boolean Skeletonization
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 2. Median Blur
    denoised = cv2.medianBlur(gray, MEDIAN_KERNEL_SIZE)
    
    # 3. Otsu Binarization
    # cv2.THRESH_BINARY_INV is used because symbols are usually dark on light background,
    # and we want foreground (symbol) to be white (255) and background to be black (0).
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Edge Extraction
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    
    # 5. Skeletonization
    # SKELETONIZATION MUST USE BOOLEAN CONVERSION
    binary_bool = binary > 0
    skeleton_bool = skimage.morphology.skeletonize(binary_bool)
    skeleton = (skeleton_bool * 255).astype(np.uint8)
    
    # Compute quantitative statistics
    stats = _compute_statistics(binary, edges, skeleton)
    
    return {
        "images": {
            "01_original.png": image,
            "02_gray.png": gray,
            "03_denoised.png": denoised,
            "04_binary.png": binary,
            "05_edges.png": edges,
            "06_skeleton.png": skeleton
        },
        "stats": stats
    }

def preprocess_diagram(image: np.ndarray) -> dict:
    """
    Preprocesses the circuit diagram image, explicitly removing horizontal bus wires.
    
    Operations:
    1. Grayscale
    2. Median Blur
    3. Adaptive Thresholding
    4. Horizontal Wire Removal
    5. Canny Edges (Original and No-Wire)
    6. Boolean Skeletonization on No-Wire
    """
    # 1. Grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 2. Median Blur
    denoised = cv2.medianBlur(gray, MEDIAN_KERNEL_SIZE)
    
    # 3. Adaptive Thresholding
    # Gaussian adaptive thresholding handles uneven illumination across the large scan.
    binary = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCKSIZE,
        ADAPTIVE_C
    )
    
    # 4. Horizontal Wire Removal
    # IMPORTANT VALIDATION: 
    #   GOOD RESULT: long horizontal bus wires mostly removed, MR symbol loops preserved, local geometry intact.
    #   BAD RESULT: loops broken, symbols fragmented, aggressive erosion, entire symbols removed.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, 
        (WIRE_KERNEL_WIDTH, WIRE_KERNEL_HEIGHT)
    )
    wire_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    no_wire = cv2.subtract(binary, wire_mask)
    
    wire_removed_pixels = int(np.count_nonzero(wire_mask))
    
    # 5. Dual Edge Extraction
    edges_original = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    # Using no_wire for edge extraction might require it to be converted back to something canny likes,
    # or apply canny to the original grayscale where the wire mask is applied.
    # However, user requested `edges_no_wire = cv2.Canny(no_wire, ...)`
    edges_no_wire = cv2.Canny(no_wire, CANNY_LOW, CANNY_HIGH)
    
    # 6. Skeletonization
    # Apply to binary no-wire image using boolean conversion
    binary_bool = no_wire > 0
    skeleton_bool = skimage.morphology.skeletonize(binary_bool)
    skeleton = (skeleton_bool * 255).astype(np.uint8)
    
    # Compute quantitative statistics
    stats = _compute_statistics(no_wire, edges_no_wire, skeleton, wire_removed_pixels)
    
    return {
        "images": {
            "01_original.png": image,
            "02_gray.png": gray,
            "03_denoised.png": denoised,
            "04_binary.png": binary,
            "05_wire_mask.png": wire_mask,
            "06_no_wire.png": no_wire,
            "07_edges_original.png": edges_original,
            "08_edges_no_wire.png": edges_no_wire,
            "09_skeleton.png": skeleton
        },
        "stats": stats
    }
