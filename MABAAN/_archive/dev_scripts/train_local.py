"""
MABAAN Local Training Script
Run this from the MABAAN directory: python train_local.py
"""

import os
import sys
import random
import warnings
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add mabaan to path
sys.path.insert(0, str(Path(__file__).parent))

from mabaan.config import Config
from mabaan.dataset import LIVECellLoader, LiveCellDataset
from mabaan.model import MABAANUNet
from mabaan.losses import MorphologyAwareLoss
from mabaan.training import train_model

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    print("=" * 60)
    print("MABAAN Local Training")
    print("=" * 60)
    
    # Configuration
    config = Config()
    
    # ============================================
    # UPDATE THESE PATHS FOR YOUR LOCAL SETUP
    # ============================================
    config.DATA_PATH = r"D:\Althaf\IIT\Final Year\FYP\Implementation Workspace\LIVECell_dataset_2021"
    
    # Local settings - adjust based on your GPU memory
    config.BATCH_SIZE = 4          # Reduce if GPU memory issues
    config.NUM_WORKERS = 2         # Set to 0 if issues on Windows
    config.NUM_EPOCHS = 30
    config.MAX_SAMPLES = None      # Set to 500 for quick testing
    
    # Device setup
    if torch.cuda.is_available():
        config.DEVICE = torch.device('cuda')
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        config.DEVICE = torch.device('cpu')
        print("Running on CPU (training will be slow)")
        config.BATCH_SIZE = 2  # Reduce for CPU
    
    print(f"\nConfiguration:")
    print(f"  Data path: {config.DATA_PATH}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Batch size: {config.BATCH_SIZE}")
    print(f"  Epochs: {config.NUM_EPOCHS}")
    print(f"  Image size: {config.IMG_SIZE}")
    
    # Set seed
    set_seed(config.SEED)
    
    # Check data path exists
    if not Path(config.DATA_PATH).exists():
        print(f"\nERROR: Data path not found: {config.DATA_PATH}")
        print("Please update config.DATA_PATH in this script.")
        return
    
    # Load datasets
    print("\n" + "-" * 60)
    print("Loading datasets...")
    train_loader = LIVECellLoader(config.DATA_PATH, split='train')
    val_loader = LIVECellLoader(config.DATA_PATH, split='val')
    
    train_ids = train_loader.get_image_ids()
    val_ids = val_loader.get_image_ids()
    
    print(f"Train images: {len(train_ids)}")
    print(f"Val images: {len(val_ids)}")
    
    # Create datasets
    train_dataset = LiveCellDataset(
        train_loader, train_ids,
        img_size=config.IMG_SIZE,
        max_samples=config.MAX_SAMPLES
    )
    val_dataset = LiveCellDataset(
        val_loader, val_ids,
        img_size=config.IMG_SIZE,
        max_samples=config.MAX_SAMPLES
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if config.DEVICE.type == 'cuda' else False
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True if config.DEVICE.type == 'cuda' else False
        )
    }
    
    print(f"Train batches: {len(dataloaders['train'])}")
    print(f"Val batches: {len(dataloaders['val'])}")
    
    # Create model
    print("\n" + "-" * 60)
    print("Creating model...")
    model = MABAANUNet(
        encoder_name=config.ENCODER,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        classes=config.NUM_CLASSES,
        reduction=config.ATTENTION_REDUCTION
    ).to(config.DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Create loss, optimizer, scheduler
    criterion = MorphologyAwareLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
        boundary_weight=config.BOUNDARY_WEIGHT,
        complexity_scale=config.COMPLEXITY_SCALE
    )
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Train
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)
    
    start_time = time.time()
    
    tracker = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        num_epochs=config.NUM_EPOCHS,
        patience=config.EARLY_STOPPING_PATIENCE,
        save_path='best_model.pth'
    )
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Best Val Dice: {max(tracker.data['val_dice']):.4f}")
    print(f"Best Val IoU: {max(tracker.data['val_iou']):.4f}")
    print(f"\nSaved: best_model.pth, training_curves.png")


if __name__ == "__main__":
    main()
