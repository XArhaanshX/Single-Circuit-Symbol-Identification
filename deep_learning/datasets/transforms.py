"""
Symbolic Transforms for Siamese Training.
Converts grayscale crops into 3-channel [binary, skeleton, edge_map] tensors.
"""

import cv2
import numpy as np
import torch
import skimage.morphology


TARGET_SIZE = (64, 64)


def center_foreground(binary_img: np.ndarray, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """Centers foreground pixels on a clean canvas — preserves binary structure."""
    canvas = np.zeros(target_size, dtype=np.uint8)
    y_coords, x_coords = np.where(binary_img > 0)
    if len(x_coords) == 0:
        return canvas
    cx, cy = int(np.mean(x_coords)), int(np.mean(y_coords))
    tcx, tcy = target_size[1] // 2, target_size[0] // 2
    dx, dy = tcx - cx, tcy - cy
    for x, y in zip(x_coords, y_coords):
        nx, ny = x + dx, y + dy
        if 0 <= nx < target_size[1] and 0 <= ny < target_size[0]:
            canvas[ny, nx] = 255
    return canvas


class SymbolicTransform:
    """Converts a grayscale crop into a 3-channel symbolic tensor."""
    
    def __init__(self, target_size=TARGET_SIZE):
        self.target_size = target_size
    
    def __call__(self, crop_gray: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(crop_gray, self.target_size, interpolation=cv2.INTER_NEAREST)
        _, binary = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)
        
        centered = center_foreground(binary, self.target_size)
        
        # Channel 0: Binary
        ch_binary = centered.astype(np.float32) / 255.0
        
        # Channel 1: Skeleton
        skel_bool = skimage.morphology.skeletonize(centered > 0)
        ch_skeleton = skel_bool.astype(np.float32)
        
        # Channel 2: Edge map
        ch_edges = cv2.Canny(centered, 50, 150).astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(np.stack([ch_binary, ch_skeleton, ch_edges], axis=0))
        return tensor


class TrainTransform(SymbolicTransform):
    """Training transform — same symbolic channels, augmentation handled upstream."""
    pass


class ValTransform(SymbolicTransform):
    """Validation transform — clean, no augmentation."""
    pass
