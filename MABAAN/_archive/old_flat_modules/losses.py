"""Loss functions for MABAAN."""

from typing import Dict, Optional

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)


class MorphologyAwareLoss(nn.Module):
    """
    Combined loss with morphology-adaptive weighting.
    
    Components:
    1. BCE Loss: Pixel-wise classification
    2. Dice Loss: Overlap-based loss
    3. Boundary Loss: Emphasizes boundary pixels
    
    Morphology adaptation: Samples with higher complexity (irregular cells)
    receive higher weight to focus learning on difficult cases.
    """
    
    def __init__(self, bce_weight: float = 0.4, dice_weight: float = 0.3, 
                 boundary_weight: float = 0.3, complexity_scale: float = 0.5):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.dice = DiceLoss()
        self.weights = (bce_weight, dice_weight, boundary_weight)
        self.complexity_scale = complexity_scale
    
    def forward(self, preds: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                complexity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            preds: Dictionary with 'mask' and 'boundary' predictions
            targets: Dictionary with 'mask' and 'boundary' ground truth
            complexity: Optional tensor of complexity scores per sample
            
        Returns:
            Dictionary with total loss and individual components
        """
        bce = self.bce(preds['mask'], targets['mask']).mean()
        dice = self.dice(preds['mask'], targets['mask'])
        boundary = self.bce(preds['boundary'], targets['boundary']).mean()
        
        total = self.weights[0] * bce + self.weights[1] * dice + self.weights[2] * boundary
        
        # Apply complexity weighting
        if complexity is not None and self.complexity_scale > 0:
            complexity_weight = 1.0 + self.complexity_scale * complexity.mean()
            total = total * complexity_weight
        
        return {'total': total, 'bce': bce, 'dice': dice, 'boundary': boundary}
