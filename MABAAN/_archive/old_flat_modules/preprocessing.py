"""Image preprocessing functions for MABAAN."""

import cv2
import numpy as np


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Apply Z-score normalization to image."""
    image = image.astype(np.float32)
    return (image - image.mean()) / (image.std() + 1e-8)


def reduce_noise(image: np.ndarray, d: int = 9, 
                 sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Apply bilateral filtering for noise reduction while preserving edges."""
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, 
                     tile_size: int = 8) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def preprocess_image(image: np.ndarray, apply_clahe: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline: grayscale -> denoise -> enhance -> normalize.
    
    Args:
        image: Input image (grayscale or RGB)
        apply_clahe: Whether to apply contrast enhancement
        
    Returns:
        Preprocessed image (float32, normalized)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    
    denoised = reduce_noise(image)
    enhanced = enhance_contrast(denoised) if apply_clahe else denoised
    return normalize_image(enhanced)
