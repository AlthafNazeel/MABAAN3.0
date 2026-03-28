"""MABAAN U-Net model architecture."""

from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from .attention import BoundaryAwareAttentionBlock


class MABAANDecoder(nn.Module):
    """
    Custom decoder with BoundaryAwareAttentionBlocks at each stage.
    Replaces standard U-Net decoder to add morphology-adaptive attention.
    """
    
    def __init__(self, encoder_channels: Tuple[int, ...], 
                 decoder_channels: Tuple[int, ...] = (256, 128, 64, 32),
                 reduction: int = 16):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Build decoder stages
        in_ch = encoder_channels[0]  # Start from bottleneck
        for skip_ch, out_ch in zip(encoder_channels[1:], decoder_channels):
            self.upsamples.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            self.blocks.append(
                BoundaryAwareAttentionBlock(out_ch, skip_ch, reduction)
            )
            in_ch = out_ch
        
        self.out_channels = decoder_channels[-1]
    
    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of encoder features [bottleneck, skip4, skip3, skip2, skip1]
            
        Returns:
            Decoded features
        """
        x = features[0]  # Bottleneck
        
        for i, (upsample, block) in enumerate(zip(self.upsamples, self.blocks)):
            x = upsample(x)
            skip = features[i + 1]
            x = block(x, skip)
        
        return x


class MABAANUNet(nn.Module):
    """
    MABAAN: Morphology-Adaptive Boundary-Aware Attention Network
    
    Combines:
    1. ResNet34 encoder (pretrained on ImageNet)
    2. Custom decoder with BoundaryAwareAttentionBlocks
    3. Dual output heads for mask and boundary prediction
    
    Input: 4 channels (3 image + 1 edge map)
    Output: Mask prediction + Boundary prediction
    """
    
    def __init__(self, encoder_name: str = "resnet34", 
                 encoder_weights: str = "imagenet",
                 in_channels: int = 4, 
                 classes: int = 1, 
                 reduction: int = 16):
        super().__init__()
        
        # Use SMP's encoder
        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights
        )
        
        # Get encoder channel sizes: [in_ch, 64, 64, 128, 256, 512] for ResNet34
        encoder_channels = self.encoder.out_channels
        
        # Reversed (excluding first): [512, 256, 128, 64, 64]
        reversed_channels = encoder_channels[:0:-1]
        
        # Decoder with 4 stages
        decoder_channels = (256, 128, 64, 32)
        self.decoder = MABAANDecoder(
            encoder_channels=reversed_channels,
            decoder_channels=decoder_channels,
            reduction=reduction
        )
        
        final_channels = decoder_channels[-1]
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, 3, padding=1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, classes, 1)
        )
        
        # Boundary head
        self.boundary_head = nn.Sequential(
            nn.Conv2d(final_channels, final_channels, 3, padding=1),
            nn.BatchNorm2d(final_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(final_channels, 1, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Extract multi-scale features
        features = self.encoder(x)
        features = features[::-1]  # Reverse: [bottleneck, skip4, skip3, skip2, skip1]
        
        # Decode with attention
        decoded = self.decoder(features)
        
        # Upsample to input resolution
        if decoded.shape[2:] != x.shape[2:]:
            decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # Generate outputs
        logits = self.seg_head(decoded)
        mask = self.sigmoid(logits)
        boundary = self.sigmoid(self.boundary_head(decoded))
        
        return {'mask': mask, 'boundary': boundary, 'logits': logits}
