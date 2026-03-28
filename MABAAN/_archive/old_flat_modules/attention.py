"""Boundary-aware attention modules for MABAAN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel attention (SE-block style) to weight feature channels.
    Helps focus on channels most relevant for boundary detection.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial attention to highlight important regions (like boundaries).
    Uses both channel max and average to compute attention map.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))


class BoundaryAwareAttentionBlock(nn.Module):
    """
    Boundary-Aware Attention Block - Core MABAAN component.
    
    This module enhances features using boundary-guided attention:
    1. Combines decoder features with skip connection features
    2. Applies channel attention to weight important feature channels
    3. Applies spatial attention guided by boundary information
    4. Refines features to emphasize boundary regions
    
    Args:
        in_channels: Number of input channels from decoder
        skip_channels: Number of channels from skip connection
        reduction: Channel reduction factor for attention
    """
    
    def __init__(self, in_channels: int, skip_channels: int, reduction: int = 16):
        super().__init__()
        
        # Combine decoder and skip features
        self.combine = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Channel attention
        self.channel_attn = ChannelAttention(in_channels, reduction)
        
        # Spatial attention
        self.spatial_attn = SpatialAttention(kernel_size=7)
        
        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Decoder features (upsampled)
            skip: Skip connection features from encoder
            
        Returns:
            Boundary-aware refined features
        """
        # Resize skip if needed
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Combine features
        combined = torch.cat([x, skip], dim=1)
        x = self.combine(combined)
        
        # Apply attention
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        
        return self.refine(x)
