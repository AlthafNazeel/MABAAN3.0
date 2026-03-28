"""
MABAAN — Morphology-Adaptive Boundary-Aware Attention Network.

A deep learning framework for brightfield microscopy cell segmentation
with boundary-aware attention, morphology-adaptive loss weighting,
and multi-metric evaluation.
"""

__version__ = "0.2.0"

from .config import Config
from .models import MABAANUNet, MABAANUNetV2
from .utils import (
    MorphologyAwareLoss,
    MorphologyAwareLossV2,
    seed_everything,
    train_model,
    train_model_v2,
)
