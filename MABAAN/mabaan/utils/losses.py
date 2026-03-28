"""
Loss functions for MABAAN.

Includes DiceLoss, per-sample DiceLoss, weighted losses, edge consistency loss,
and the upgraded MorphologyAwareLossV2 with pixel-level weight maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Basic loss components
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Standard Dice loss (batch-level)."""

    def forward(self, pred, target, smooth=1e-6):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        inter = (pred_flat * target_flat).sum()
        return 1 - (2 * inter + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)


class DiceLossPerSample(nn.Module):
    """Dice loss computed per-sample for individual weighting."""

    def forward(self, pred, target, smooth=1e-6):
        B = pred.shape[0]
        losses = []
        for i in range(B):
            p = pred[i].reshape(-1)
            t = target[i].reshape(-1)
            inter = (p * t).sum()
            losses.append(1 - (2 * inter + smooth) / (p.sum() + t.sum() + smooth))
        return torch.stack(losses)


# ---------------------------------------------------------------------------
# Weighted loss components (pixel-level weight map support)
# ---------------------------------------------------------------------------

class WeightedBCELoss(nn.Module):
    """BCE loss with optional per-pixel weight map.

    Uses binary_cross_entropy_with_logits for AMP compatibility.
    Input should be raw logits (pre-sigmoid).
    """

    def forward(self, logits, target, weight_map=None):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        if weight_map is not None:
            bce = bce * weight_map
        return bce.mean()


class WeightedDiceLoss(nn.Module):
    """Dice loss with optional per-pixel weight map.

    Weights are applied to both pred and target before computing
    the weighted intersection and union.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target, weight_map=None):
        if weight_map is None:
            weight_map = torch.ones_like(pred)

        pred_w = pred * weight_map
        target_w = target * weight_map

        pred_w = pred_w.view(pred.size(0), -1)
        target_w = target_w.view(target.size(0), -1)

        inter = (pred_w * target_w).sum(dim=1)
        denom = pred_w.sum(dim=1) + target_w.sum(dim=1)
        loss = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
        return loss.mean()


# ---------------------------------------------------------------------------
# Edge consistency loss
# ---------------------------------------------------------------------------

class EdgeConsistencyLoss(nn.Module):
    """Encourage alignment between segmentation edges and boundary prediction.

    Derives soft edges from the segmentation output using Sobel filters
    and computes L1 loss against the boundary prediction.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, seg_pred, bnd_pred):
        gx = F.conv2d(seg_pred, self.sobel_x, padding=1)
        gy = F.conv2d(seg_pred, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        grad_mag = grad_mag / (grad_mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return F.l1_loss(grad_mag, bnd_pred)


# ---------------------------------------------------------------------------
# Legacy loss (preserved for backward compatibility)
# ---------------------------------------------------------------------------

class MorphologyAwareLoss(nn.Module):
    """Combined BCE + Dice + Boundary loss with per-sample morphology weighting.

    Args:
        bce_w: Weight for mask BCE loss.
        dice_w: Weight for mask Dice loss.
        boundary_w: Weight for boundary BCE loss.
        complexity_scale: Scale factor for complexity weighting (0 = no weighting).
    """

    def __init__(self, bce_w=0.4, dice_w=0.3, boundary_w=0.3, complexity_scale=0.5):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.dice_ps = DiceLossPerSample()
        self.bce_w, self.dice_w, self.boundary_w = bce_w, dice_w, boundary_w
        self.complexity_scale = complexity_scale

    def forward(self, preds, targets, complexity=None):
        B = preds['mask'].shape[0]
        # Per-sample BCE for mask
        bce_mask = self.bce(preds['mask'], targets['mask']).view(B, -1).mean(dim=1)
        # Per-sample Dice
        dice = self.dice_ps(preds['mask'], targets['mask'])
        # Per-sample boundary BCE
        bce_bnd = self.bce(preds['boundary'], targets['boundary']).view(B, -1).mean(dim=1)
        # Combined per-sample loss
        total_ps = self.bce_w * bce_mask + self.dice_w * dice + self.boundary_w * bce_bnd
        # Per-sample complexity weighting
        if complexity is not None and self.complexity_scale > 0:
            sample_weights = 1.0 + self.complexity_scale * complexity.to(total_ps.device)
            total_ps = total_ps * sample_weights
        total = total_ps.mean()
        return {
            'total': total,
            'bce': bce_mask.mean(),
            'dice': dice.mean(),
            'boundary': bce_bnd.mean(),
        }


# ---------------------------------------------------------------------------
# Upgraded loss (pixel-level weight maps, BCE+Dice for both branches)
# ---------------------------------------------------------------------------

class MorphologyAwareLossV2(nn.Module):
    """Upgraded loss with:
    - BCE + Dice for BOTH segmentation and boundary branches
    - Pixel-level morphology weight maps (not just sample-level)
    - Stronger boundary emphasis for complex cells
    - Optional edge consistency loss

    Args:
        seg_bce_w: Weight for segmentation BCE.
        seg_dice_w: Weight for segmentation Dice.
        bnd_bce_w: Weight for boundary BCE.
        bnd_dice_w: Weight for boundary Dice.
        cons_w: Weight for edge consistency loss.
        use_consistency: Whether to include consistency loss.
    """

    def __init__(self, seg_bce_w=0.25, seg_dice_w=0.35,
                 bnd_bce_w=0.20, bnd_dice_w=0.15,
                 cons_w=0.05, use_consistency=False):
        super().__init__()
        self.seg_bce = WeightedBCELoss()
        self.seg_dice = WeightedDiceLoss()
        self.bnd_bce = WeightedBCELoss()
        self.bnd_dice = WeightedDiceLoss()
        self.consistency = EdgeConsistencyLoss() if use_consistency else None

        self.seg_bce_w = seg_bce_w
        self.seg_dice_w = seg_dice_w
        self.bnd_bce_w = bnd_bce_w
        self.bnd_dice_w = bnd_dice_w
        self.cons_w = cons_w
        self.use_consistency = use_consistency

    def forward(self, preds, targets):
        # Use logits for BCE (AMP-safe), sigmoid outputs for Dice
        seg_logits = preds['logits']
        bnd_logits = preds.get('boundary_logits', None)
        seg_prob = preds['mask']        # already sigmoid'd
        bnd_prob = preds['boundary']    # already sigmoid'd
        seg_gt = targets['mask']
        bnd_gt = targets['boundary']
        weight_map = targets.get('weight_map', None)

        # Segmentation losses with morphology weighting
        seg_bce = self.seg_bce(seg_logits, seg_gt, weight_map=weight_map)
        seg_dice = self.seg_dice(seg_prob, seg_gt, weight_map=weight_map)

        # Boundary losses with stronger morphology emphasis
        bnd_weight_map = None
        if weight_map is not None:
            bnd_weight_map = 1.0 + 1.5 * (weight_map - 1.0)

        # Use boundary logits for BCE if available, else fall back to probs
        if bnd_logits is not None:
            bnd_bce = self.bnd_bce(bnd_logits, bnd_gt, weight_map=bnd_weight_map)
        else:
            bnd_bce = F.binary_cross_entropy_with_logits(
                torch.logit(bnd_prob.clamp(1e-6, 1 - 1e-6)), bnd_gt, reduction='none'
            ).mean()
        bnd_dice = self.bnd_dice(bnd_prob, bnd_gt, weight_map=bnd_weight_map)

        # Optional consistency loss (uses sigmoid outputs)
        cons = torch.tensor(0.0, device=seg_prob.device)
        if self.use_consistency and self.consistency is not None:
            cons = self.consistency(seg_prob, bnd_prob)

        total = (
            self.seg_bce_w * seg_bce
            + self.seg_dice_w * seg_dice
            + self.bnd_bce_w * bnd_bce
            + self.bnd_dice_w * bnd_dice
            + self.cons_w * cons
        )

        return {
            'total': total,
            'seg_bce': seg_bce,
            'seg_dice': seg_dice,
            'bnd_bce': bnd_bce,
            'bnd_dice': bnd_dice,
            'cons': cons,
        }
