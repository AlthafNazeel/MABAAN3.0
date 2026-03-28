"""
Evaluation metrics for MABAAN.

Includes Dice, IoU, Boundary F1, Hausdorff Distance, HD95, ASSD,
and per-image batch metric computation.
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance as sp_dist
from skimage.segmentation import find_boundaries


def compute_dice(pred, target, threshold=0.5):
    """Compute Dice coefficient (works with both tensors and numpy)."""
    if hasattr(pred, 'float'):  # torch tensor
        pred_bin = (pred > threshold).float()
        inter = (pred_bin * target).sum()
        return ((2 * inter + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)).item()
    else:
        pred_bin = (pred > threshold).astype(float)
        inter = (pred_bin * target).sum()
        return (2 * inter + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)


def compute_iou(pred, target, threshold=0.5):
    """Compute Intersection over Union."""
    if hasattr(pred, 'float'):
        pred_bin = (pred > threshold).float()
        inter = (pred_bin * target).sum()
        union = pred_bin.sum() + target.sum() - inter
        return ((inter + 1e-6) / (union + 1e-6)).item()
    else:
        pred_bin = (pred > threshold).astype(float)
        inter = (pred_bin * target).sum()
        union = pred_bin.sum() + target.sum() - inter
        return (inter + 1e-6) / (union + 1e-6)


def compute_boundary_f1(pred, target, threshold=0.5):
    """Compute Boundary F1 score."""
    pred_b = find_boundaries((pred > threshold).astype(np.uint8), mode='thick').astype(np.uint8)
    target_b = find_boundaries((target > 0).astype(np.uint8), mode='thick').astype(np.uint8)
    tp = np.sum(pred_b & target_b)
    fp = np.sum(pred_b & ~target_b)
    fn = np.sum(~pred_b & target_b)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return 2 * prec * rec / (prec + rec + 1e-6)


def compute_hausdorff_95(pred, target, threshold=0.5):
    """Compute 95th percentile Hausdorff distance."""
    pred_b = find_boundaries((pred > threshold).astype(np.uint8), mode='thick')
    target_b = find_boundaries((target > 0).astype(np.uint8), mode='thick')
    pred_pts = np.argwhere(pred_b)
    target_pts = np.argwhere(target_b)
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float('inf') if len(pred_pts) != len(target_pts) else 0.0
    if len(pred_pts) > 1000:
        pred_pts = pred_pts[::len(pred_pts) // 1000]
    if len(target_pts) > 1000:
        target_pts = target_pts[::len(target_pts) // 1000]
    d_matrix = sp_dist.cdist(pred_pts, target_pts)
    d1 = np.min(d_matrix, axis=1)
    d2 = np.min(d_matrix, axis=0)
    return max(np.percentile(d1, 95), np.percentile(d2, 95))


def compute_hausdorff_distance(pred, target, threshold=0.5):
    """Compute full Hausdorff distance."""
    pred_b = find_boundaries((pred > threshold).astype(np.uint8), mode='thick')
    target_b = find_boundaries((target > 0).astype(np.uint8), mode='thick')
    pred_pts = np.argwhere(pred_b)
    target_pts = np.argwhere(target_b)
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float('inf') if len(pred_pts) != len(target_pts) else 0.0
    d1 = directed_hausdorff(pred_pts, target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]
    return max(d1, d2)


def compute_assd(pred, target, threshold=0.5):
    """Compute Average Symmetric Surface Distance.

    Measures the mean distance between predicted and ground-truth contours
    in both directions, then averages them.
    """
    pred_b = find_boundaries((pred > threshold).astype(np.uint8), mode='thick')
    target_b = find_boundaries((target > 0).astype(np.uint8), mode='thick')
    pred_pts = np.argwhere(pred_b)
    target_pts = np.argwhere(target_b)

    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float('inf') if len(pred_pts) != len(target_pts) else 0.0

    # Subsample if too many points
    if len(pred_pts) > 2000:
        pred_pts = pred_pts[::len(pred_pts) // 2000]
    if len(target_pts) > 2000:
        target_pts = target_pts[::len(target_pts) // 2000]

    D = sp_dist.cdist(pred_pts, target_pts)
    d1 = D.min(axis=1).mean()
    d2 = D.min(axis=0).mean()
    return float((d1 + d2) / 2.0)


def compute_all_metrics(pred, target, threshold=0.5):
    """Compute all metrics for a single prediction-target pair."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    pred_bin = (pred_flat > threshold)
    target_bin = (target_flat > 0)
    tp = np.sum(pred_bin & target_bin)
    fp = np.sum(pred_bin & ~target_bin)
    fn = np.sum(~pred_bin & target_bin)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {
        'dice': dice,
        'iou': iou,
        'boundary_f1': compute_boundary_f1(pred, target, threshold),
        'hausdorff': compute_hausdorff_distance(pred, target, threshold),
        'hausdorff_95': compute_hausdorff_95(pred, target, threshold),
        'assd': compute_assd(pred, target, threshold),
    }


# ---------------------------------------------------------------------------
# Per-image batch metrics (upgraded)
# ---------------------------------------------------------------------------

def dice_np(pred, target, threshold=0.5):
    """Compute Dice for a single numpy image."""
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)
    inter = (pred_bin & target_bin).sum()
    denom = pred_bin.sum() + target_bin.sum()
    return (2.0 * inter + 1e-6) / (denom + 1e-6)


def iou_np(pred, target, threshold=0.5):
    """Compute IoU for a single numpy image."""
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > 0.5).astype(np.uint8)
    inter = (pred_bin & target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)


def batch_metrics(pred_batch, target_batch, threshold=0.5):
    """Compute per-image Dice and IoU, then average across the batch.

    Args:
        pred_batch: Tensor or numpy array of shape (B, 1, H, W).
        target_batch: Tensor or numpy array of shape (B, 1, H, W).

    Returns:
        (avg_dice, avg_iou) as floats.
    """
    if hasattr(pred_batch, 'detach'):
        pred_batch = pred_batch.detach().cpu().numpy()
    if hasattr(target_batch, 'detach'):
        target_batch = target_batch.detach().cpu().numpy()

    dices, ious = [], []
    for i in range(pred_batch.shape[0]):
        p = pred_batch[i, 0]
        t = target_batch[i, 0]
        dices.append(dice_np(p, t, threshold))
        ious.append(iou_np(p, t, threshold))
    return float(np.mean(dices)), float(np.mean(ious))
