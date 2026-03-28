"""
Precompute all expensive CPU operations for MABAAN training.

Run once before training to precompute:
- Preprocessed images
- Edge maps
- Binary masks and instance maps
- Boundary maps
- Morphology weight maps
- Complexity scores

Saves everything as .npy files for fast loading during training.
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm.auto import tqdm

# Add MABAAN package to path (adjust for Kaggle)
sys.path.insert(0, '/kaggle/working/MABAAN')

from mabaan.config import Config
from mabaan.data import (
    LIVECellLoader,
    preprocess_image_v2,
    MorphologyAdaptiveEdgeDetector,
    make_boundary_map,
    build_morphology_weight_map,
)


def precompute_split(loader, split_name, output_dir, config, edge_detector):
    """Precompute all data for one split (train/val/test)."""
    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    img_ids = loader.get_image_ids()
    print(f"\nPrecomputing {split_name}: {len(img_ids)} images → {split_dir}")

    valid_ids = []

    for img_id in tqdm(img_ids, desc=split_name):
        image = loader.load_image(img_id)
        if image is None:
            continue

        # Convert to grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Preprocess
        preprocessed = preprocess_image_v2(
            image, use_clahe=True, use_denoise=False,
            use_illum_corr=False, norm='zscore'
        )

        # Instance map and binary mask
        binary_mask, instance_map = loader.generate_instance_map(img_id)

        # Boundary map
        boundary = make_boundary_map(
            binary_mask,
            mode=config.BOUNDARY_MODE,
            thickness=config.BOUNDARY_THICKNESS,
            soft=False,
        )

        # Morphology weight map
        weight_map = build_morphology_weight_map(
            instance_map,
            boundary_map=boundary,
            alpha=config.MORPH_ALPHA,
            boundary_boost=config.MORPH_BOUNDARY_BOOST,
        )

        # Edge map
        edge_map = edge_detector.detect(preprocessed)

        # Complexity scalar
        complexity = float(np.clip(weight_map.max() - 1.0, 0.0, 1.0))

        # Resize everything to target size
        sz = config.IMG_SIZE
        img_r = cv2.resize(preprocessed, (sz, sz), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        edge_r = cv2.resize(edge_map, (sz, sz), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mask_r = cv2.resize(binary_mask.astype(np.uint8), (sz, sz), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        bnd_r = cv2.resize(boundary.astype(np.uint8), (sz, sz), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        wgt_r = cv2.resize(weight_map, (sz, sz), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        # Save as compressed .npz (one file per sample)
        sample_path = split_dir / f"{img_id}.npz"
        np.savez_compressed(
            sample_path,
            image=img_r,
            edge=edge_r,
            mask=mask_r,
            boundary=bnd_r,
            weight_map=wgt_r,
            complexity=np.float32(complexity),
        )
        valid_ids.append(img_id)

    # Save the list of valid IDs
    with open(split_dir / "ids.json", 'w') as f:
        json.dump(valid_ids, f)

    print(f"  Saved {len(valid_ids)} samples to {split_dir}")
    return valid_ids


def main():
    config = Config()
    output_dir = '/kaggle/working/precomputed'
    edge_detector = MorphologyAdaptiveEdgeDetector()

    print("=" * 60)
    print("MABAAN Dataset Precomputation")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Image size: {config.IMG_SIZE}")

    for split in ['train', 'val', 'test']:
        loader = LIVECellLoader(config.DATA_PATH, split=split)
        precompute_split(loader, split, output_dir, config, edge_detector)

    print("\n" + "=" * 60)
    print("Precomputation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
