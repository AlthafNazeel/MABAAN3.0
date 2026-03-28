"""
MABAANUNet — Morphology-Adaptive Boundary-Aware Attention Network.

Contains both legacy MABAANUNet and upgraded MABAANUNetV2 with
boundary-feature fusion and attention gates on skip connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .decoder import MABAANDecoder, MABAANDecoderV2
from .attention import BoundaryFusionHead


class MABAANUNet(nn.Module):
    """MABAAN U-Net with dual heads for mask and boundary prediction (legacy v1)."""

    def __init__(self, encoder_name="resnet34", in_channels=4, classes=1,
                 encoder_weights="imagenet", reduction=16):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels
        self.decoder = MABAANDecoder(enc_channels, reduction)
        self.mask_head = nn.Conv2d(32, classes, 1)
        self.boundary_head = nn.Conv2d(32, classes, 1)

    def forward(self, x):
        features = self.encoder(x)
        dec_out = self.decoder(features)
        dec_up = F.interpolate(dec_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        logits = self.mask_head(dec_up)
        mask = torch.sigmoid(logits)
        boundary = torch.sigmoid(self.boundary_head(dec_up))
        return {'mask': mask, 'boundary': boundary, 'logits': logits}


class MABAANUNetV2(nn.Module):
    """Upgraded MABAAN U-Net with:
    - Attention gates on skip connections
    - Boundary-feature fusion back into segmentation
    - Separate boundary branch with dedicated features

    Args:
        encoder_name: SMP encoder backbone name.
        in_channels: Number of input channels (4 = 3 image + 1 edge, or 3).
        classes: Number of output classes.
        encoder_weights: Pretrained weights ('imagenet' or None).
        reduction: Attention reduction ratio.
    """

    def __init__(self, encoder_name="resnet34", in_channels=4, classes=1,
                 encoder_weights="imagenet", reduction=16):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights
        )
        enc_channels = self.encoder.out_channels
        self.decoder = MABAANDecoderV2(enc_channels, reduction)

        # Boundary branch: dedicated feature extraction + head
        self.boundary_branch = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.boundary_head = nn.Conv2d(32, classes, 1)

        # Boundary → segmentation fusion
        self.fusion = BoundaryFusionHead(32, mid_ch=32)
        self.seg_head = nn.Conv2d(32, classes, 1)

    def forward(self, x):
        features = self.encoder(x)
        dec_out = self.decoder(features)
        dec_up = F.interpolate(dec_out, size=x.shape[2:], mode='bilinear', align_corners=False)

        # Boundary branch
        bnd_feat = self.boundary_branch(dec_up)
        bnd_logits = self.boundary_head(bnd_feat)

        # Fuse boundary features into segmentation
        fused_feat = self.fusion(dec_up, bnd_feat)
        seg_logits = self.seg_head(fused_feat)

        return {
            'mask': torch.sigmoid(seg_logits),
            'boundary': torch.sigmoid(bnd_logits),
            'logits': seg_logits,
            'boundary_logits': bnd_logits,
        }
