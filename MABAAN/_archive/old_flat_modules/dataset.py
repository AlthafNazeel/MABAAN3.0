"""LIVECELL dataset loader for MABAAN."""

import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict
from PIL import Image

import torch
from torch.utils.data import Dataset
from skimage.segmentation import find_boundaries

from .preprocessing import preprocess_image
from .edge_detection import MorphologyAdaptiveEdgeDetector, compute_shape_descriptors


class LIVECellLoader:
    """Custom loader for LIVECELL dataset (COCO format with non-standard annotations)."""
    
    def __init__(self, data_path: str, split: str = 'train'):
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
        
        # LIVECELL uses dict for annotations, not list
        for ann_id, ann_data in self.data['annotations'].items():
            if isinstance(ann_data, dict):
                ann_data['id'] = int(ann_id)
                self.img_to_anns[ann_data['image_id']].append(ann_data)
        
        print(f"  Loaded {len(self.images)} images")
    
    def get_image_ids(self) -> List[int]:
        """Get all image IDs."""
        return list(self.images.keys())
    
    def get_image_path(self, img_id: int) -> Optional[Path]:
        """Get path to image file."""
        info = self.images[img_id]
        filename = info['file_name']
        cell_type = filename.split('_')[0]
        
        possible_paths = [
            self.img_subdir / cell_type / filename,
            self.images_dir / "livecell_train_val_images" / cell_type / filename,
            self.images_dir / "livecell_test_images" / cell_type / filename,
        ]
        
        for p in possible_paths:
            if p.exists():
                return p
        return None
    
    def load_image(self, img_id: int) -> Optional[np.ndarray]:
        """Load image by ID."""
        path = self.get_image_path(img_id)
        if path and path.exists():
            return np.array(Image.open(path))
        return None
    
    def generate_mask(self, img_id: int) -> np.ndarray:
        """Generate segmentation mask from annotations."""
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


class LiveCellDataset(Dataset):
    """PyTorch Dataset for LIVECELL with edge maps and complexity scores."""
    
    def __init__(self, loader: LIVECellLoader, img_ids: List[int], 
                 img_size: int = 256, edge_detector: Optional[MorphologyAdaptiveEdgeDetector] = None,
                 max_samples: Optional[int] = None):
        self.loader = loader
        self.img_ids = img_ids[:max_samples] if max_samples else img_ids
        self.img_size = img_size
        self.edge_detector = edge_detector or MorphologyAdaptiveEdgeDetector()
    
    def __len__(self) -> int:
        return len(self.img_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_id = self.img_ids[idx]
        
        image = self.loader.load_image(img_id)
        if image is None:
            return {
                'input': torch.zeros(4, self.img_size, self.img_size),
                'mask': torch.zeros(1, self.img_size, self.img_size),
                'boundary': torch.zeros(1, self.img_size, self.img_size),
                'complexity': torch.tensor(0.0)
            }
        
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        mask = self.loader.generate_mask(img_id)
        preprocessed = preprocess_image(image)
        edge_map = self.edge_detector.detect(preprocessed)
        binary_mask = (mask > 0).astype(np.float32)
        boundary = find_boundaries(binary_mask > 0, mode='thick').astype(np.float32)
        
        # Compute shape complexity
        shape_info = compute_shape_descriptors(mask)
        complexity = shape_info['complexity']
        
        # Resize
        img_resized = cv2.resize(preprocessed, (self.img_size, self.img_size))
        edge_resized = cv2.resize(edge_map, (self.img_size, self.img_size))
        mask_resized = cv2.resize(binary_mask, (self.img_size, self.img_size))
        boundary_resized = cv2.resize(boundary, (self.img_size, self.img_size))
        
        # Combine: 3 image channels + 1 edge channel
        combined = np.concatenate([
            np.stack([img_resized] * 3, axis=0),
            edge_resized[np.newaxis, ...]
        ], axis=0).astype(np.float32)
        
        return {
            'input': torch.from_numpy(combined),
            'mask': torch.from_numpy(mask_resized[np.newaxis, ...].astype(np.float32)),
            'boundary': torch.from_numpy(boundary_resized[np.newaxis, ...].astype(np.float32)),
            'complexity': torch.tensor(complexity, dtype=torch.float32)
        }
