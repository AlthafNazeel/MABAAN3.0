"""Shared cell content for all v4 notebooks."""

INSTALL_CELL = "!pip install -q segmentation-models-pytorch pycocotools"

IMPORTS_CELL = """import os, random, warnings, time, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops, label
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import segmentation_models_pytorch as smp
warnings.filterwarnings('ignore')
print("Libraries imported successfully!")"""

SETUP_CELL = """def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")"""

CONFIG_CELL = """class Config:
    DATA_PATH = "/kaggle/input/livecell"
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 4  # 3 image + 1 edge
    BATCH_SIZE = 4
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10
    IMG_SIZE = 256
    MAX_SAMPLES = None
    NUM_WORKERS = 0
    ATTENTION_REDUCTION = 16

config = Config()
print(f"Data path: {config.DATA_PATH}")"""

PREPROCESSING_CELL = """def normalize_image(image):
    image = image.astype(np.float32)
    return (image - image.mean()) / (image.std() + 1e-8)

def reduce_noise(image, d=9, sigma_color=75, sigma_space=75):
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def enhance_contrast(image, clip_limit=2.0, tile_size=8):
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    return clahe.apply(image)

def preprocess_image(image, apply_clahe=True):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    denoised = reduce_noise(image)
    enhanced = enhance_contrast(denoised) if apply_clahe else denoised
    return normalize_image(enhanced)

print("Preprocessing functions defined")"""

EDGE_DETECTION_CELL = """def compute_shape_descriptors(mask):
    if mask.sum() == 0:
        return {'circularity': 1.0, 'solidity': 1.0, 'complexity': 0.0}
    labeled = label(mask > 0)
    props = regionprops(labeled)
    if len(props) == 0:
        return {'circularity': 1.0, 'solidity': 1.0, 'complexity': 0.0}
    circularities, solidities = [], []
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            circularities.append(min(circularity, 1.0))
        if prop.convex_area > 0:
            solidities.append(area / prop.convex_area)
    avg_circ = np.mean(circularities) if circularities else 1.0
    avg_sol = np.mean(solidities) if solidities else 1.0
    complexity = 1.0 - (avg_circ * avg_sol)
    return {'circularity': avg_circ, 'solidity': avg_sol, 'complexity': complexity}

def compute_morphological_gradient(image, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if image.dtype in [np.float32, np.float64]:
        img = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    gradient = cv2.dilate(img, kernel).astype(float) - cv2.erode(img, kernel).astype(float)
    return gradient / (gradient.max() + 1e-8)

def multi_scale_edge_fusion(image, scales=[3, 5, 7]):
    edges = [compute_morphological_gradient(image, s) for s in scales]
    weights = [0.2, 0.35, 0.45]
    fused = sum(w * e for w, e in zip(weights, edges))
    return fused / (fused.max() + 1e-8)

class MorphologyAdaptiveEdgeDetector:
    def __init__(self, scales=[3, 5, 7]):
        self.scales = scales
    def detect(self, image):
        return multi_scale_edge_fusion(image, self.scales)

print("Edge detection with shape analysis defined")"""

LOADER_CELL = """class LIVECellLoader:
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
        return list(self.images.keys())

    def get_image_path(self, img_id):
        info = self.images[img_id]
        filename = info['file_name']
        cell_type = filename.split('_')[0]
        for p in [self.img_subdir / cell_type / filename,
                  self.images_dir / "livecell_train_val_images" / cell_type / filename,
                  self.images_dir / "livecell_test_images" / cell_type / filename]:
            if p.exists():
                return p
        return None

    def load_image(self, img_id):
        path = self.get_image_path(img_id)
        if path and path.exists():
            return np.array(Image.open(path))
        return None

    def generate_mask(self, img_id):
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

print("LIVECellLoader defined")"""

DATASET_CELL = """class LiveCellDataset(Dataset):
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
            return {'input': torch.zeros(4, self.img_size, self.img_size),
                    'mask': torch.zeros(1, self.img_size, self.img_size),
                    'boundary': torch.zeros(1, self.img_size, self.img_size),
                    'complexity': torch.tensor(0.0)}
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = self.loader.generate_mask(img_id)
        preprocessed = preprocess_image(image)
        edge_map = self.edge_detector.detect(preprocessed)
        binary_mask = (mask > 0).astype(np.float32)
        boundary = find_boundaries(binary_mask > 0, mode='thick').astype(np.float32)
        shape_info = compute_shape_descriptors(mask)
        complexity = shape_info['complexity']
        img_r = cv2.resize(preprocessed, (self.img_size, self.img_size))
        edge_r = cv2.resize(edge_map, (self.img_size, self.img_size))
        mask_r = cv2.resize(binary_mask, (self.img_size, self.img_size))
        bnd_r = cv2.resize(boundary, (self.img_size, self.img_size))
        combined = np.concatenate([np.stack([img_r]*3, axis=0), edge_r[np.newaxis, ...]], axis=0).astype(np.float32)
        return {'input': torch.from_numpy(combined),
                'mask': torch.from_numpy(mask_r[np.newaxis, ...].astype(np.float32)),
                'boundary': torch.from_numpy(bnd_r[np.newaxis, ...].astype(np.float32)),
                'complexity': torch.tensor(complexity, dtype=torch.float32)}

print("LiveCellDataset defined")"""

METRICS_CELL = """def compute_dice(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    return ((2*inter + 1e-6) / (pred_bin.sum() + target.sum() + 1e-6)).item()

def compute_iou(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    inter = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - inter
    return ((inter + 1e-6) / (union + 1e-6)).item()

def compute_boundary_f1(pred, target, threshold=0.5):
    pred_b = find_boundaries((pred > threshold).astype(np.uint8), mode='thick').astype(np.uint8)
    target_b = find_boundaries((target > 0).astype(np.uint8), mode='thick').astype(np.uint8)
    tp = np.sum(pred_b & target_b)
    fp = np.sum(pred_b & ~target_b)
    fn = np.sum(~pred_b & target_b)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    return 2 * prec * rec / (prec + rec + 1e-6)

def compute_hausdorff_95(pred, target, threshold=0.5):
    from scipy.spatial import distance
    pred_b = find_boundaries((pred > threshold).astype(np.uint8), mode='thick')
    target_b = find_boundaries((target > 0).astype(np.uint8), mode='thick')
    pred_pts = np.argwhere(pred_b)
    target_pts = np.argwhere(target_b)
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float('inf') if len(pred_pts) != len(target_pts) else 0.0
    if len(pred_pts) > 1000:
        pred_pts = pred_pts[::len(pred_pts)//1000]
    if len(target_pts) > 1000:
        target_pts = target_pts[::len(target_pts)//1000]
    d_matrix = distance.cdist(pred_pts, target_pts)
    d1 = np.min(d_matrix, axis=1)
    d2 = np.min(d_matrix, axis=0)
    return max(np.percentile(d1, 95), np.percentile(d2, 95))

def compute_hausdorff_distance(pred, target, threshold=0.5):
    pred_b = find_boundaries((pred > threshold).astype(np.uint8), mode='thick')
    target_b = find_boundaries((target > 0).astype(np.uint8), mode='thick')
    pred_pts = np.argwhere(pred_b)
    target_pts = np.argwhere(target_b)
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return float('inf') if len(pred_pts) != len(target_pts) else 0.0
    d1 = directed_hausdorff(pred_pts, target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]
    return max(d1, d2)

def compute_all_metrics(pred, target, threshold=0.5):
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    pred_bin = (pred_flat > threshold)
    target_bin = (target_flat > 0)
    tp = np.sum(pred_bin & target_bin)
    fp = np.sum(pred_bin & ~target_bin)
    fn = np.sum(~pred_bin & target_bin)
    dice = (2*tp) / (2*tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {'dice': dice, 'iou': iou,
            'boundary_f1': compute_boundary_f1(pred, target, threshold),
            'hausdorff': compute_hausdorff_distance(pred, target, threshold),
            'hausdorff_95': compute_hausdorff_95(pred, target, threshold)}

print("Metrics defined (Dice, IoU, Boundary F1, HD, HD95)")"""

EVAL_FUNCTIONS_CELL = """@torch.no_grad()
def run_inference(model, dl, device):
    model.eval()
    preds, tgts, complexities = [], [], []
    for batch in tqdm(dl, desc="Inference"):
        out = model(batch['input'].to(device))
        preds.extend(out['mask'].cpu().numpy())
        tgts.extend(batch['mask'].numpy())
        complexities.extend(batch['complexity'].numpy())
    return preds, tgts, complexities

def evaluate_model(preds, tgts, model_name="Model"):
    all_metrics = []
    for p, t in tqdm(zip(preds, tgts), total=len(preds), desc=f"Evaluating {model_name}"):
        all_metrics.append(compute_all_metrics(p[0], t[0]))
    df = pd.DataFrame(all_metrics)
    hd_valid = df['hausdorff'].replace([np.inf, -np.inf], np.nan).dropna()
    hd95_valid = df['hausdorff_95'].replace([np.inf, -np.inf], np.nan).dropna()
    print(f"\\n{'='*60}")
    print(f"{model_name} EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Dice:         {df['dice'].mean():.4f} ± {df['dice'].std():.4f}")
    print(f"  IoU:          {df['iou'].mean():.4f} ± {df['iou'].std():.4f}")
    print(f"  Boundary F1:  {df['boundary_f1'].mean():.4f} ± {df['boundary_f1'].std():.4f}")
    if len(hd_valid) > 0:
        print(f"  HD:           {hd_valid.mean():.2f} ± {hd_valid.std():.2f}")
    if len(hd95_valid) > 0:
        print(f"  HD95:         {hd95_valid.mean():.2f} ± {hd95_valid.std():.2f}")
    print(f"{'='*60}")
    return df

def morphology_stratified_evaluation(preds, tgts, complexities, model_name="Model"):
    bins = {'Circular (c<0.2)': [], 'Moderate (0.2≤c≤0.5)': [], 'Irregular (c>0.5)': []}
    bin_keys = list(bins.keys())
    for p, t, c in tqdm(zip(preds, tgts, complexities), total=len(preds), desc="Stratified eval"):
        metrics = compute_all_metrics(p[0], t[0])
        if c < 0.2:
            bins[bin_keys[0]].append(metrics)
        elif c <= 0.5:
            bins[bin_keys[1]].append(metrics)
        else:
            bins[bin_keys[2]].append(metrics)
    print(f"\\n{'='*70}")
    print(f"{model_name} — Morphology-Stratified Evaluation")
    print(f"{'='*70}")
    print(f"{'Bin':<25} {'N':>5} {'Dice':>12} {'IoU':>12} {'BF1':>12} {'HD95':>12}")
    print("-"*70)
    results = {}
    for bin_name, metrics_list in bins.items():
        if len(metrics_list) == 0:
            print(f"{bin_name:<25} {'0':>5} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")
            continue
        df = pd.DataFrame(metrics_list)
        hd95 = df['hausdorff_95'].replace([np.inf, -np.inf], np.nan).dropna()
        hd95_str = f"{hd95.mean():.2f}±{hd95.std():.2f}" if len(hd95) > 0 else "N/A"
        print(f"{bin_name:<25} {len(df):>5} "
              f"{df['dice'].mean():.4f}±{df['dice'].std():.4f} "
              f"{df['iou'].mean():.4f}±{df['iou'].std():.4f} "
              f"{df['boundary_f1'].mean():.4f}±{df['boundary_f1'].std():.4f} "
              f"{hd95_str:>12}")
        results[bin_name] = df
    print(f"{'='*70}")
    return results

print("Evaluation functions defined (including morphology-stratified)")"""

DATA_LOADING_CELL = """print("Loading datasets...")
train_loader = LIVECellLoader(config.DATA_PATH, split='train')
val_loader = LIVECellLoader(config.DATA_PATH, split='val')
test_loader = LIVECellLoader(config.DATA_PATH, split='test')

train_dataset = LiveCellDataset(train_loader, train_loader.get_image_ids(),
                                config.IMG_SIZE, max_samples=config.MAX_SAMPLES)
val_dataset = LiveCellDataset(val_loader, val_loader.get_image_ids(),
                              config.IMG_SIZE, max_samples=config.MAX_SAMPLES)
test_dataset = LiveCellDataset(test_loader, test_loader.get_image_ids(),
                               config.IMG_SIZE, max_samples=config.MAX_SAMPLES)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                        num_workers=config.NUM_WORKERS, pin_memory=True),
    'val': DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                      num_workers=config.NUM_WORKERS, pin_memory=True),
    'test': DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                       num_workers=config.NUM_WORKERS, pin_memory=True)
}
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")"""

SAMPLE_VIZ_CELL = """batch = next(iter(dataloaders['train']))
fig, ax = plt.subplots(1, 5, figsize=(20, 4))
ax[0].imshow(batch['input'][0, 0].numpy(), cmap='gray'); ax[0].set_title('Image')
ax[1].imshow(batch['input'][0, 3].numpy(), cmap='hot'); ax[1].set_title('Edge Map')
ax[2].imshow(batch['mask'][0, 0].numpy(), cmap='gray'); ax[2].set_title('Mask')
ax[3].imshow(batch['boundary'][0, 0].numpy(), cmap='hot'); ax[3].set_title('Boundary')
ax[4].bar(['Complexity'], [batch['complexity'][0].item()]); ax[4].set_title('Shape Complexity'); ax[4].set_ylim(0, 1)
for a in ax[:4]: a.axis('off')
plt.tight_layout(); plt.show()"""
