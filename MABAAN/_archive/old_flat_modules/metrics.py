"""Evaluation metrics for MABAAN."""

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Dice coefficient."""
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    return ((2 * inter + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)).item()


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute Intersection over Union (Jaccard Index)."""
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return ((inter + 1e-6) / (union + 1e-6)).item()


def compute_boundary_f1(pred: np.ndarray, target: np.ndarray, tolerance: int = 2) -> float:
    """
    Compute F1 score for boundary detection.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        tolerance: Pixel tolerance for boundary matching
        
    Returns:
        Boundary F1 score
    """
    pred_boundary = find_boundaries(pred > 0.5, mode='thick')
    target_boundary = find_boundaries(target > 0.5, mode='thick')
    
    if pred_boundary.sum() == 0 and target_boundary.sum() == 0:
        return 1.0
    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return 0.0
    
    # Dilate boundaries for tolerance
    from scipy.ndimage import binary_dilation
    structure = np.ones((tolerance * 2 + 1, tolerance * 2 + 1))
    
    target_dilated = binary_dilation(target_boundary, structure)
    pred_dilated = binary_dilation(pred_boundary, structure)
    
    precision = (pred_boundary & target_dilated).sum() / (pred_boundary.sum() + 1e-6)
    recall = (target_boundary & pred_dilated).sum() / (target_boundary.sum() + 1e-6)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_hausdorff_95(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute 95th percentile Hausdorff distance.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        HD95 distance in pixels (inf if either mask is empty)
    """
    pred_boundary = find_boundaries(pred > 0.5, mode='thick')
    target_boundary = find_boundaries(target > 0.5, mode='thick')
    
    if pred_boundary.sum() == 0 or target_boundary.sum() == 0:
        return float('inf')
    
    pred_points = np.argwhere(pred_boundary)
    target_points = np.argwhere(target_boundary)
    
    # Compute directed distances
    d1 = np.array([np.min(np.linalg.norm(target_points - p, axis=1)) for p in pred_points])
    d2 = np.array([np.min(np.linalg.norm(pred_points - p, axis=1)) for p in target_points])
    
    all_distances = np.concatenate([d1, d2])
    return np.percentile(all_distances, 95)


def evaluate_sample(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute all metrics for a single sample.
    
    Args:
        pred: Predicted mask (0-1 range or binary)
        target: Ground truth mask (0-1 range or binary)
        
    Returns:
        Dictionary with dice, iou, boundary_f1, hausdorff_95
    """
    pred_t = torch.from_numpy(pred).float()
    target_t = torch.from_numpy(target).float()
    
    return {
        'dice': compute_dice(pred_t, target_t),
        'iou': compute_iou(pred_t, target_t),
        'boundary_f1': compute_boundary_f1(pred, target),
        'hausdorff_95': compute_hausdorff_95(pred, target)
    }
