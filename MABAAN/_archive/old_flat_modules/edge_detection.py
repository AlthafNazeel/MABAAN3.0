"""Morphology-adaptive edge detection for MABAAN."""

import cv2
import numpy as np
from typing import List
from skimage.measure import regionprops, label


def compute_shape_descriptors(mask: np.ndarray) -> dict:
    """
    Compute shape descriptors for morphology-adaptive processing.
    
    Args:
        mask: Binary segmentation mask
        
    Returns:
        Dictionary with circularity, solidity, and complexity scores
    """
    if mask.sum() == 0:
        return {'circularity': 1.0, 'solidity': 1.0, 'complexity': 0.0}
    
    labeled = label(mask > 0)
    props = regionprops(labeled)
    
    if len(props) == 0:
        return {'circularity': 1.0, 'solidity': 1.0, 'complexity': 0.0}
    
    circularities = []
    solidities = []
    
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        
        # Circularity: 4*pi*area / perimeter^2 (1.0 = perfect circle)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(min(circularity, 1.0))
        
        # Solidity: area / convex_hull_area (1.0 = convex shape)
        if prop.convex_area > 0:
            solidity = area / prop.convex_area
            solidities.append(solidity)
    
    avg_circularity = np.mean(circularities) if circularities else 1.0
    avg_solidity = np.mean(solidities) if solidities else 1.0
    
    # Complexity: higher for irregular cells
    complexity = 1.0 - (avg_circularity * avg_solidity)
    
    return {
        'circularity': avg_circularity,
        'solidity': avg_solidity,
        'complexity': complexity
    }


def compute_morphological_gradient(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Compute morphological gradient (dilation - erosion)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if image.dtype in [np.float32, np.float64]:
        img = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    
    gradient = cv2.dilate(img, kernel).astype(float) - cv2.erode(img, kernel).astype(float)
    return gradient / (gradient.max() + 1e-8)


def multi_scale_edge_fusion(image: np.ndarray, scales: List[int] = [3, 5, 7]) -> np.ndarray:
    """
    Fuse edges at multiple scales with weighted combination.
    
    Larger scales capture coarse boundaries, smaller scales capture fine details.
    """
    edges = [compute_morphological_gradient(image, s) for s in scales]
    weights = [0.2, 0.35, 0.45]  # Emphasize larger scales
    fused = sum(w * e for w, e in zip(weights, edges))
    return fused / (fused.max() + 1e-8)


class MorphologyAdaptiveEdgeDetector:
    """Edge detector using multi-scale morphological operations."""
    
    def __init__(self, scales: List[int] = [3, 5, 7]):
        self.scales = scales
    
    def detect(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using multi-scale fusion."""
        return multi_scale_edge_fusion(image, self.scales)
