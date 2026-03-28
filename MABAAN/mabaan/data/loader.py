"""
LIVECell dataset loader for COCO-format annotations.

Handles the non-standard dict-based annotation format used by the LIVECell dataset
and resolves image paths across cell-type subdirectories.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from PIL import Image


class LIVECellLoader:
    """Loads LIVECell dataset with COCO-format annotations."""

    def __init__(self, data_path, split='train'):
        self.data_path = Path(data_path)
        self.split = split
        self.annotations_dir = self.data_path / "annotations" / "LIVECell"
        self.images_dir = self.data_path / "images"
        if split == 'test':
            self.img_subdir = self.images_dir / "livecell_test_images"
        else:
            self.img_subdir = self.images_dir / "livecell_train_val_images"
        json_path = self.annotations_dir / f"livecell_coco_{split}.json"
        print(f"Loading {split} annotations...")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.images = {img['id']: img for img in self.data['images']}
        self.img_to_anns = defaultdict(list)
        for ann_id, ann_data in self.data['annotations'].items():
            if isinstance(ann_data, dict):
                ann_data['id'] = int(ann_id)
                self.img_to_anns[ann_data['image_id']].append(ann_data)
        print(f"  Loaded {len(self.images)} images")

    def get_image_ids(self):
        """Return list of all image IDs in this split."""
        return list(self.images.keys())

    def get_image_path(self, img_id):
        """Resolve full path for an image, searching cell-type subdirectories."""
        info = self.images[img_id]
        filename = info['file_name']
        cell_type = filename.split('_')[0]
        for p in [
            self.img_subdir / cell_type / filename,
            self.images_dir / "livecell_train_val_images" / cell_type / filename,
            self.images_dir / "livecell_test_images" / cell_type / filename,
        ]:
            if p.exists():
                return p
        return None

    def load_image(self, img_id):
        """Load image as numpy array."""
        path = self.get_image_path(img_id)
        if path and path.exists():
            return np.array(Image.open(path))
        return None

    def generate_mask(self, img_id):
        """Generate binary segmentation mask from polygon annotations."""
        info = self.images[img_id]
        h, w = info['height'], info['width']
        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in self.img_to_anns.get(img_id, []):
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                for polygon in ann['segmentation']:
                    if len(polygon) >= 6:
                        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(mask, [pts], 1)
        return mask

    def generate_instance_map(self, img_id):
        """Generate instance-level segmentation map and binary mask.

        Returns:
            binary_mask: uint8 array (0 background, 1 foreground).
            instance_map: int32 array where each cell gets a unique label.
        """
        info = self.images[img_id]
        h, w = info['height'], info['width']
        instance_map = np.zeros((h, w), dtype=np.int32)

        inst_id = 1
        for ann in self.img_to_anns.get(img_id, []):
            if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                temp = np.zeros((h, w), dtype=np.uint8)
                for polygon in ann['segmentation']:
                    if len(polygon) >= 6:
                        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                        cv2.fillPoly(temp, [pts], 1)
                instance_map[temp > 0] = inst_id
                inst_id += 1

        binary_mask = (instance_map > 0).astype(np.uint8)
        return binary_mask, instance_map
