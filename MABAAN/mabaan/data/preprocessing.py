"""
Image preprocessing functions for MABAAN.

Provides normalization, denoising (bilateral filter), contrast enhancement (CLAHE),
illumination correction, and configurable preprocessing pipelines.
"""

import numpy as np
import cv2


def normalize_image(image):
    """Normalize image to zero mean and unit variance (z-score)."""
    image = image.astype(np.float32)
    return (image - image.mean()) / (image.std() + 1e-8)


def normalize_minmax(image):
    """Normalize image to [0, 1] range."""
    image = image.astype(np.float32)
    mn, mx = image.min(), image.max()
    return (image - mn) / (mx - mn + 1e-8)


def reduce_noise(image, d=9, sigma_color=75, sigma_space=75):
    """Apply bilateral filter for edge-preserving noise reduction."""
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def enhance_contrast(image, clip_limit=2.0, tile_size=8):
    """Apply CLAHE for adaptive contrast enhancement."""
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)


def illumination_correction(image, kernel_size=51):
    """Correct uneven illumination by dividing by smoothed background estimate."""
    image = image.astype(np.float32)
    bg = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    corrected = image / (bg + 1e-6)
    corrected = corrected - corrected.min()
    corrected = corrected / (corrected.max() + 1e-8)
    return corrected


def preprocess_image(image, apply_clahe=True):
    """Legacy preprocessing pipeline: grayscale -> denoise -> CLAHE -> normalize."""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    denoised = reduce_noise(image)
    enhanced = enhance_contrast(denoised) if apply_clahe else denoised
    return normalize_image(enhanced)


def preprocess_image_v2(image, use_clahe=True, use_denoise=False,
                        use_illum_corr=False, norm='zscore'):
    """Configurable preprocessing pipeline for ablation experiments.

    Args:
        image: Input grayscale image.
        use_clahe: Apply CLAHE contrast enhancement.
        use_denoise: Apply bilateral denoising.
        use_illum_corr: Apply illumination correction.
        norm: Normalization mode ('zscore' or 'minmax').
    """
    x = image.astype(np.float32)

    if use_denoise:
        x = cv2.bilateralFilter(
            np.uint8(np.clip(x, 0, 255)), 5, 50, 50
        ).astype(np.float32)

    if use_clahe:
        x = enhance_contrast(x).astype(np.float32)

    if use_illum_corr:
        x = illumination_correction(x)

    if norm == 'minmax':
        x = normalize_minmax(x)
    elif norm == 'zscore':
        x = normalize_image(x)

    return x.astype(np.float32)
