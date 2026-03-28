"""
PyTorch Dataset for LIVECell with edge maps, augmentation, and morphology weight maps.

Integrates preprocessing, edge detection, boundary extraction, augmentation,
and instance-level morphology complexity scoring into each sample.
"""

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries

from .preprocessing import preprocess_image, preprocess_image_v2
from .edge_detection import (
    compute_shape_descriptors,
    MorphologyAdaptiveEdgeDetector,
    make_boundary_map,
    build_morphology_weight_map,
)


# ---------------------------------------------------------------------------
# Augmentation helpers (requires albumentations)
# ---------------------------------------------------------------------------

def build_train_augment():
    """Build training augmentation pipeline using albumentations."""
    try:
        import albumentations as A
    except ImportError:
        print("Warning: albumentations not installed. No augmentation will be applied.")
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.10, rotate_limit=20,
            border_mode=cv2.BORDER_REFLECT_101, p=0.5
        ),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(p=0.2),
    ], additional_targets={
        'boundary': 'mask',
        'weight_map': 'mask',
    })


def build_val_augment():
    """Build validation augmentation pipeline (identity, but compatible format)."""
    try:
        import albumentations as A
    except ImportError:
        return None

    return A.Compose([], additional_targets={
        'boundary': 'mask',
        'weight_map': 'mask',
    })


# ---------------------------------------------------------------------------
# Original dataset (preserved for backward compatibility)
# ---------------------------------------------------------------------------

class LiveCellDataset(Dataset):
    """PyTorch Dataset that returns 4-channel input (3 image + 1 edge),
    mask, boundary, and complexity score."""

    def __init__(self, loader, img_ids, img_size=256, edge_detector=None, max_samples=None):
        self.loader = loader
        self.img_ids = img_ids[:max_samples] if max_samples else img_ids
        self.img_size = img_size
        self.edge_detector = edge_detector or MorphologyAdaptiveEdgeDetector()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = self.loader.load_image(img_id)
        if image is None:
            return {
                'input': torch.zeros(4, self.img_size, self.img_size),
                'mask': torch.zeros(1, self.img_size, self.img_size),
                'boundary': torch.zeros(1, self.img_size, self.img_size),
                'complexity': torch.tensor(0.0),
            }
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = self.loader.generate_mask(img_id)
        preprocessed = preprocess_image(image)
        edge_map = self.edge_detector.detect(preprocessed)
        binary_mask = (mask > 0).astype(np.float32)
        boundary = find_boundaries(binary_mask > 0, mode='thick').astype(np.float32)
        shape_info = compute_shape_descriptors(mask)
        complexity = shape_info['complexity']
        # Resize all to target size
        img_r = cv2.resize(preprocessed, (self.img_size, self.img_size))
        edge_r = cv2.resize(edge_map, (self.img_size, self.img_size))
        mask_r = cv2.resize(binary_mask, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_NEAREST)
        bnd_r = cv2.resize(boundary, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST)
        # Stack: 3-ch image + 1-ch edge
        combined = np.concatenate(
            [np.stack([img_r] * 3, axis=0), edge_r[np.newaxis, ...]], axis=0
        ).astype(np.float32)
        return {
            'input': torch.from_numpy(combined),
            'mask': torch.from_numpy(mask_r[np.newaxis, ...].astype(np.float32)),
            'boundary': torch.from_numpy(bnd_r[np.newaxis, ...].astype(np.float32)),
            'complexity': torch.tensor(complexity, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Upgraded dataset with augmentation + pixel-level morphology weight maps
# ---------------------------------------------------------------------------

class LiveCellDatasetV2(Dataset):
    """Upgraded LIVECell dataset with:
    - Instance-level morphology weight maps
    - Data augmentation
    - Safe INTER_NEAREST for masks/boundaries
    - Configurable boundary generation
    - Optional edge channel toggle

    Args:
        loader: LIVECellLoader instance.
        img_ids: List of image IDs.
        img_size: Target spatial resolution.
        max_samples: Limit dataset size (for debugging).
        augment: Albumentations Compose object, or None.
        use_edge_channel: Whether to include the 4th edge input channel.
        boundary_mode: 'thick' or 'thin'.
        boundary_thickness: Kernel size for boundary generation.
        morph_alpha: Morphology weight scaling factor.
        morph_boundary_boost: Extra boundary emphasis for complex cells.
    """

    def __init__(self, loader, img_ids, img_size=256, max_samples=None,
                 augment=None, use_edge_channel=True,
                 boundary_mode='thick', boundary_thickness=3,
                 morph_alpha=2.0, morph_boundary_boost=2.0):
        self.loader = loader
        self.img_ids = img_ids[:max_samples] if max_samples else img_ids
        self.img_size = img_size
        self.augment = augment
        self.use_edge_channel = use_edge_channel
        self.boundary_mode = boundary_mode
        self.boundary_thickness = boundary_thickness
        self.morph_alpha = morph_alpha
        self.morph_boundary_boost = morph_boundary_boost
        self.edge_detector = MorphologyAdaptiveEdgeDetector()

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        image = self.loader.load_image(img_id)

        n_ch = 4 if self.use_edge_channel else 3
        if image is None:
            return {
                'input': torch.zeros(n_ch, self.img_size, self.img_size),
                'mask': torch.zeros(1, self.img_size, self.img_size),
                'boundary': torch.zeros(1, self.img_size, self.img_size),
                'weight_map': torch.ones(1, self.img_size, self.img_size),
                'complexity': torch.tensor(0.0),
            }

        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Preprocess
        preprocessed = preprocess_image_v2(
            image, use_clahe=True, use_denoise=False,
            use_illum_corr=False, norm='zscore'
        )

        # Instance map and binary mask
        binary_mask, instance_map = self.loader.generate_instance_map(img_id)

        # Boundary from binary mask
        boundary = make_boundary_map(
            binary_mask, mode=self.boundary_mode,
            thickness=self.boundary_thickness, soft=False
        )

        # Pixel-level morphology weight map from instances
        weight_map = build_morphology_weight_map(
            instance_map,
            boundary_map=boundary,
            alpha=self.morph_alpha,
            boundary_boost=self.morph_boundary_boost,
        )

        # Edge map for input channel
        edge_map = self.edge_detector.detect(preprocessed)

        # Resize with safe interpolation
        img_r = cv2.resize(preprocessed, (self.img_size, self.img_size),
                           interpolation=cv2.INTER_LINEAR)
        mask_r = cv2.resize(binary_mask.astype(np.uint8),
                            (self.img_size, self.img_size),
                            interpolation=cv2.INTER_NEAREST).astype(np.float32)
        bnd_r = cv2.resize(boundary.astype(np.uint8),
                           (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST).astype(np.float32)
        wgt_r = cv2.resize(weight_map,
                           (self.img_size, self.img_size),
                           interpolation=cv2.INTER_NEAREST).astype(np.float32)
        edge_r = cv2.resize(edge_map, (self.img_size, self.img_size),
                            interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # Augment (image + mask + boundary + weight map together)
        if self.augment is not None:
            aug = self.augment(
                image=img_r,
                mask=mask_r,
                boundary=bnd_r,
                weight_map=wgt_r,
            )
            img_r = aug['image']
            mask_r = aug['mask']
            bnd_r = aug['boundary']
            wgt_r = aug['weight_map']

        # Scalar complexity (max weight - 1, clipped)
        complexity_scalar = float(np.clip(wgt_r.max() - 1.0, 0.0, 1.0))

        # Build input tensor
        image_3 = np.stack([img_r] * 3, axis=0).astype(np.float32)
        if self.use_edge_channel:
            inp = np.concatenate([image_3, edge_r[None, ...]], axis=0).astype(np.float32)
        else:
            inp = image_3

        return {
            'input': torch.from_numpy(inp),
            'mask': torch.from_numpy(mask_r[None, ...].astype(np.float32)),
            'boundary': torch.from_numpy(bnd_r[None, ...].astype(np.float32)),
            'weight_map': torch.from_numpy(wgt_r[None, ...].astype(np.float32)),
            'complexity': torch.tensor(complexity_scalar, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Fast dataset: loads precomputed .npz files (no CPU work per sample)
# ---------------------------------------------------------------------------

class LiveCellDatasetFast(Dataset):
    """Ultra-fast LIVECell dataset that loads precomputed .npz files.

    All expensive CPU operations (preprocessing, edge detection, instance maps,
    morphology weight maps, boundary maps) are done ONCE by the precompute
    script and saved to disk. This dataset just reads the files.

    Args:
        precomputed_dir: Path to split directory (e.g. '/kaggle/working/precomputed/train').
        augment: Albumentations Compose object, or None.
        use_edge_channel: Whether to include the 4th edge input channel.
    """

    def __init__(self, precomputed_dir, augment=None, use_edge_channel=True):
        import json as _json
        self.precomputed_dir = Path(precomputed_dir)
        self.augment = augment
        self.use_edge_channel = use_edge_channel

        # Load the list of valid IDs
        ids_path = self.precomputed_dir / "ids.json"
        with open(ids_path, 'r') as f:
            self.ids = _json.load(f)

        print(f"  LiveCellDatasetFast: {len(self.ids)} samples from {precomputed_dir}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        data = np.load(self.precomputed_dir / f"{img_id}.npz")

        img_r = data['image']
        edge_r = data['edge']
        mask_r = data['mask']
        bnd_r = data['boundary']
        wgt_r = data['weight_map']
        complexity = float(data['complexity'])

        # Augment (image + mask + boundary + weight map together)
        if self.augment is not None:
            aug = self.augment(
                image=img_r,
                mask=mask_r,
                boundary=bnd_r,
                weight_map=wgt_r,
            )
            img_r = aug['image']
            mask_r = aug['mask']
            bnd_r = aug['boundary']
            wgt_r = aug['weight_map']

        # Build input tensor
        image_3 = np.stack([img_r] * 3, axis=0).astype(np.float32)
        if self.use_edge_channel:
            inp = np.concatenate([image_3, edge_r[None, ...]], axis=0).astype(np.float32)
        else:
            inp = image_3

        return {
            'input': torch.from_numpy(inp),
            'mask': torch.from_numpy(mask_r[None, ...].astype(np.float32)),
            'boundary': torch.from_numpy(bnd_r[None, ...].astype(np.float32)),
            'weight_map': torch.from_numpy(wgt_r[None, ...].astype(np.float32)),
            'complexity': torch.tensor(complexity, dtype=torch.float32),
        }

