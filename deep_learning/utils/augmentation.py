"""
Symbolic Augmentation
Reuses the same physically motivated transforms from src/pca_refinement.py:
scale jitter, rotation, thickness morphology, translation.
"""

import cv2
import numpy as np
import itertools


class SymbolicAugmentation:
    """Generates controlled augmentations for binary symbolic images."""
    
    def __init__(self, 
                 rotations=(-8, -4, 0, 4, 8),
                 scales=(0.90, 0.95, 1.0, 1.05, 1.10),
                 thicknesses=("erode", "none", "dilate"),
                 translations=((0,0), (-2,0), (2,0), (0,-2), (0,2))):
        self.rotations = rotations
        self.scales = scales
        self.thicknesses = thicknesses
        self.translations = translations
    
    def augment(self, binary_img: np.ndarray, group_id: str) -> list:
        """
        Generate all augmented variants for a single base crop.
        All variants share the same group_id for split isolation.
        """
        augmented = []
        h, w = binary_img.shape[:2]
        
        y_coords, x_coords = np.where(binary_img > 0)
        if len(x_coords) > 0:
            center = (int(np.mean(x_coords)), int(np.mean(y_coords)))
        else:
            center = (w // 2, h // 2)
        
        for rot, scale in itertools.product(self.rotations, self.scales):
            M = cv2.getRotationMatrix2D(center, rot, scale)
            warped = cv2.warpAffine(binary_img, M, (w, h), flags=cv2.INTER_NEAREST)
            
            for thick in self.thicknesses:
                thick_img = warped.copy()
                if thick == "erode":
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    thick_img = cv2.erode(thick_img, kernel, iterations=1)
                elif thick == "dilate":
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                    thick_img = cv2.dilate(thick_img, kernel, iterations=1)
                
                for tx, ty in self.translations:
                    M_t = np.float32([[1, 0, tx], [0, 1, ty]])
                    final = cv2.warpAffine(thick_img, M_t, (w, h), flags=cv2.INTER_NEAREST)
                    
                    augmented.append({
                        "crop": final,
                        "group_id": group_id,
                        "aug_params": {"rot": rot, "scale": scale, "thick": thick, "tx": tx, "ty": ty}
                    })
        
        return augmented
    
    def augment_light(self, binary_img: np.ndarray, group_id: str, n_samples: int = 10) -> list:
        """Generate a small random subset of augmentations."""
        all_augs = self.augment(binary_img, group_id)
        if len(all_augs) <= n_samples:
            return all_augs
        indices = np.random.choice(len(all_augs), n_samples, replace=False)
        return [all_augs[i] for i in indices]
