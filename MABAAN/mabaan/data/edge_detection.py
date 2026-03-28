"""
Morphology-adaptive edge detection for MABAAN.

Computes shape descriptors (circularity, solidity, eccentricity, etc.),
instance-level complexity scoring, pixel-level morphology weight maps,
and multi-scale morphological edge fusion.
"""

import numpy as np
import cv2
from skimage.measure import regionprops, label
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# Shape descriptors (legacy, image-level)
# ---------------------------------------------------------------------------

def compute_shape_descriptors(mask):
    """Compute shape descriptors for cell morphology analysis.

    Returns dict with circularity, solidity, and complexity (0=circular, 1=irregular).
    """
    if mask.sum() == 0:
        return {'circularity': 1.0, 'solidity': 1.0, 'complexity': 0.0}
    labeled = label(mask > 0)
    props = regionprops(labeled)
    if len(props) == 0:
        return {'circularity': 1.0, 'solidity': 1.0, 'complexity': 0.0}
    circularities, solidities = [], []
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(min(circularity, 1.0))
        if prop.convex_area > 0:
            solidities.append(area / prop.convex_area)
    avg_circ = np.mean(circularities) if circularities else 1.0
    avg_sol = np.mean(solidities) if solidities else 1.0
    complexity = 1.0 - (avg_circ * avg_sol)
    return {'circularity': avg_circ, 'solidity': avg_sol, 'complexity': complexity}


# ---------------------------------------------------------------------------
# Instance-level descriptors (upgraded)
# ---------------------------------------------------------------------------

def compute_instance_descriptors(instance_map):
    """Compute rich morphology descriptors per cell instance.

    Args:
        instance_map: Labeled integer array where each cell has a unique ID.

    Returns:
        List of dicts, one per instance, with keys:
        label, area, perimeter, circularity, solidity, eccentricity,
        extent, aspect_ratio, roughness.
    """
    descriptors = []
    for region in regionprops(instance_map):
        area = float(region.area)
        perimeter = float(region.perimeter) if region.perimeter > 0 else 1.0
        circularity = min((4.0 * np.pi * area) / (perimeter ** 2 + 1e-8), 1.0)
        solidity = float(region.solidity) if region.solidity is not None else 1.0
        eccentricity = float(region.eccentricity) if hasattr(region, 'eccentricity') else 0.0
        extent = float(region.extent) if hasattr(region, 'extent') else 1.0
        major = float(region.major_axis_length) if region.major_axis_length > 0 else 1.0
        minor = float(region.minor_axis_length) if region.minor_axis_length > 0 else 1.0
        aspect_ratio = major / (minor + 1e-8)
        roughness = (perimeter ** 2) / (4.0 * np.pi * area + 1e-8)

        descriptors.append({
            'label': region.label,
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'solidity': solidity,
            'eccentricity': eccentricity,
            'extent': extent,
            'aspect_ratio': aspect_ratio,
            'roughness': roughness,
        })
    return descriptors


def _normalize_descriptor(v, lo, hi):
    """Normalize a descriptor value to [0, 1] range."""
    return float(np.clip((v - lo) / (hi - lo + 1e-8), 0.0, 1.0))


def compute_instance_complexity(desc):
    """Compute a single complexity score for one cell instance.

    Higher = more morphologically complex (harder to segment).
    Combines inverse circularity, inverse solidity, eccentricity,
    aspect ratio, and roughness.
    """
    c1 = 1.0 - np.clip(desc['circularity'], 0, 1)
    c2 = 1.0 - np.clip(desc['solidity'], 0, 1)
    c3 = np.clip(desc['eccentricity'], 0, 1)
    c4 = _normalize_descriptor(desc['aspect_ratio'], 1.0, 5.0)
    c5 = _normalize_descriptor(desc['roughness'], 1.0, 8.0)

    complexity = 0.25 * c1 + 0.20 * c2 + 0.20 * c3 + 0.15 * c4 + 0.20 * c5
    return float(np.clip(complexity, 0.0, 1.0))


def build_morphology_weight_map(instance_map, boundary_map=None,
                                alpha=2.0, boundary_boost=2.0):
    """Build a pixel-level morphology weight map from instance labels.

    Each instance's pixels get weight ``1 + alpha * complexity``.
    If ``boundary_map`` is provided, boundary pixels of complex cells
    receive additional ``boundary_boost`` emphasis.

    Args:
        instance_map: Labeled integer array (0 = background).
        boundary_map: Optional binary boundary mask.
        alpha: Scaling factor for complexity → weight conversion.
        boundary_boost: Extra multiplier for boundary regions of complex cells.

    Returns:
        Float32 weight map clipped to [1.0, 6.0].
    """
    descs = compute_instance_descriptors(instance_map)
    weight_map = np.ones(instance_map.shape, dtype=np.float32)

    for desc in descs:
        comp = compute_instance_complexity(desc)
        region_mask = (instance_map == desc['label'])
        weight_map[region_mask] = 1.0 + alpha * comp

    if boundary_map is not None:
        bnd = (boundary_map > 0).astype(np.float32)
        weight_map = weight_map + boundary_boost * bnd * (weight_map - 1.0)

    return np.clip(weight_map, 1.0, 6.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Boundary map generation
# ---------------------------------------------------------------------------

def make_boundary_map(mask, mode='thick', thickness=3, soft=False, sigma=3.0):
    """Generate boundary target from binary mask.

    Args:
        mask: Binary numpy array.
        mode: 'thick' (morphological gradient) or 'thin' (inner contour).
        thickness: Kernel size for thick boundaries.
        soft: If True, generate soft Gaussian-weighted boundary band.
        sigma: Width of Gaussian decay for soft boundaries.
    """
    mask = (mask > 0).astype(np.uint8)

    if soft:
        edge = find_boundaries(mask, mode='inner').astype(np.uint8)
        dist = distance_transform_edt(1 - edge)
        soft_bnd = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        soft_bnd[mask == 0] *= 0.7
        return soft_bnd.astype(np.float32)

    if mode == 'thin':
        bnd = find_boundaries(mask, mode='inner').astype(np.uint8)
    else:
        kernel = np.ones((thickness, thickness), np.uint8)
        dil = cv2.dilate(mask, kernel, iterations=1)
        ero = cv2.erode(mask, kernel, iterations=1)
        bnd = (dil - ero) > 0

    return bnd.astype(np.float32)


# ---------------------------------------------------------------------------
# Morphological gradient edge detection (for input channel)
# ---------------------------------------------------------------------------

def compute_morphological_gradient(image, kernel_size=3):
    """Compute morphological gradient (dilation - erosion) for edge detection."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if image.dtype in [np.float32, np.float64]:
        img = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    gradient = cv2.dilate(img, kernel).astype(float) - cv2.erode(img, kernel).astype(float)
    return gradient / (gradient.max() + 1e-8)


def multi_scale_edge_fusion(image, scales=(3, 5, 7)):
    """Fuse edges detected at multiple scales with weighted combination."""
    edges = [compute_morphological_gradient(image, s) for s in scales]
    weights = [0.2, 0.35, 0.45]
    fused = sum(w * e for w, e in zip(weights, edges))
    return fused / (fused.max() + 1e-8)


class MorphologyAdaptiveEdgeDetector:
    """Multi-scale morphological edge detector."""

    def __init__(self, scales=(3, 5, 7)):
        self.scales = scales

    def detect(self, image):
        """Detect edges using multi-scale morphological gradient fusion."""
        return multi_scale_edge_fusion(image, self.scales)
