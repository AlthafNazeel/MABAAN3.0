"""
Boundary-Aware Attention modules for MABAAN.

Contains Channel Attention, Spatial Attention, Attention Gates for
skip connections, and the combined BoundaryAwareAttentionBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """Spatial attention using average and max pooling."""

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.conv(torch.cat([avg_out, max_out], dim=1))


class BoundaryAwareAttentionBlock(nn.Module):
    """Combined channel + spatial + boundary-aware attention.

    Uses a learnable gamma parameter to control boundary attention strength.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(ca)
        boundary_mask = self.boundary_conv(x)
        out = sa + self.gamma * (sa * boundary_mask)
        return out


class AttentionGate(nn.Module):
    """Attention Gate for skip connections (Attention U-Net style).

    Uses decoder (gating) features to filter encoder (skip) features,
    suppressing irrelevant background and emphasizing boundary structures.

    Args:
        F_g: Number of channels in the gating signal (decoder features).
        F_l: Number of channels in the skip connection (encoder features).
        F_int: Number of intermediate channels.
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (lower resolution, upsampled).
            x: Skip connection from encoder.
        Returns:
            Attention-filtered skip features.
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Align spatial dimensions if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class BoundaryFusionHead(nn.Module):
    """Fuse boundary features back into the segmentation path.

    Projects both segmentation and boundary features into a shared space,
    concatenates them, and applies refinement convolutions.

    Args:
        in_ch: Number of input channels for both seg and boundary features.
        mid_ch: Number of channels in the fusion layers.
    """

    def __init__(self, in_ch, mid_ch=32):
        super().__init__()
        self.seg_proj = nn.Conv2d(in_ch, mid_ch, 1)
        self.bnd_proj = nn.Conv2d(in_ch, mid_ch, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, seg_feat, bnd_feat):
        s = self.seg_proj(seg_feat)
        b = self.bnd_proj(bnd_feat)
        return self.fuse(torch.cat([s, b], dim=1))
