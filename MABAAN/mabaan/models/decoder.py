"""
MABAAN Decoder with Boundary-Aware Attention.

Custom U-Net decoder that applies BoundaryAwareAttentionBlock
at each decoder stage for enhanced boundary segmentation.
Includes both legacy (v1) and upgraded (v2) decoder with attention gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import BoundaryAwareAttentionBlock, AttentionGate


class MABAANDecoder(nn.Module):
    """U-Net decoder with attention blocks at each stage (legacy v1)."""

    def __init__(self, encoder_channels, reduction=16):
        super().__init__()
        dec_channels = [256, 128, 64, 32]
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        for i, dc in enumerate(dec_channels):
            in_ch = encoder_channels[-(i + 1)] + (
                encoder_channels[-(i + 2)] if i < len(dec_channels) - 1 else encoder_channels[0]
            )
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, dc, 3, padding=1, bias=False),
                nn.BatchNorm2d(dc),
                nn.ReLU(inplace=True),
                nn.Conv2d(dc, dc, 3, padding=1, bias=False),
                nn.BatchNorm2d(dc),
                nn.ReLU(inplace=True),
            ))
            self.attention_blocks.append(BoundaryAwareAttentionBlock(dc, reduction))

    def forward(self, features):
        x = features[-1]
        for i, (block, att) in enumerate(zip(self.blocks, self.attention_blocks)):
            skip = features[-(i + 2)] if i < len(self.blocks) - 1 else features[0]
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            x = att(x)
        return x


class DecoderBlock(nn.Module):
    """Single decoder block: two 3x3 convolutions with BN and ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class MABAANDecoderV2(nn.Module):
    """Upgraded U-Net decoder with:
    - Attention Gates on skip connections
    - BoundaryAwareAttentionBlock after each decoder stage

    Args:
        encoder_channels: Tuple of encoder output channel counts (from SMP).
        reduction: Reduction ratio for attention modules.
    """

    def __init__(self, encoder_channels, reduction=16):
        super().__init__()
        dec_channels = [256, 128, 64, 32]

        self.blocks = nn.ModuleList()
        self.att_gates = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()

        for i, dc in enumerate(dec_channels):
            # Gating signal channels = previous decoder output (or bottleneck)
            g_ch = encoder_channels[-(i + 1)] if i == 0 else dec_channels[i - 1]
            # Skip channels = encoder features at this level
            skip_ch = encoder_channels[-(i + 2)] if i < len(dec_channels) - 1 else encoder_channels[0]
            # Attention gate intermediate channels
            ag_int = max(skip_ch // 4, 16)

            self.att_gates.append(AttentionGate(g_ch, skip_ch, ag_int))

            in_ch = (encoder_channels[-(i + 1)] if i == 0 else dec_channels[i - 1]) + skip_ch
            self.blocks.append(DecoderBlock(in_ch, dc))
            self.attention_blocks.append(BoundaryAwareAttentionBlock(dc, reduction))

    def forward(self, features):
        x = features[-1]
        for i, (block, ag, att) in enumerate(
            zip(self.blocks, self.att_gates, self.attention_blocks)
        ):
            skip = features[-(i + 2)] if i < len(self.blocks) - 1 else features[0]
            # Apply attention gate: decoder guides which encoder features pass
            skip = ag(x, skip)
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            x = att(x)
        return x
