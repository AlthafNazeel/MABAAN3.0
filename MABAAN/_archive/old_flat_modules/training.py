"""Training utilities for MABAAN."""

from typing import Dict, Optional
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm

from .metrics import compute_dice, compute_iou


class MetricTracker:
    """Track and plot training metrics."""
    
    def __init__(self):
        self.data = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': []
        }
    
    def update(self, train_loss: float, val_loss: float, 
               train_dice: float, val_dice: float,
               train_iou: float, val_iou: float):
        """Record metrics for one epoch."""
        self.data['train_loss'].append(train_loss)
        self.data['val_loss'].append(val_loss)
        self.data['train_dice'].append(train_dice)
        self.data['val_dice'].append(val_dice)
        self.data['train_iou'].append(train_iou)
        self.data['val_iou'].append(val_iou)
    
    def plot(self, save_path: str = 'training_curves.png'):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        epochs = range(1, len(self.data['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.data['train_loss'], 'b-', label='Train')
        axes[0].plot(epochs, self.data['val_loss'], 'r-', label='Val')
        axes[0].set_title('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        
        # Dice
        axes[1].plot(epochs, self.data['train_dice'], 'b-', label='Train')
        axes[1].plot(epochs, self.data['val_dice'], 'r-', label='Val')
        axes[1].set_title('Dice Score')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Dice')
        
        # IoU
        axes[2].plot(epochs, self.data['train_iou'], 'b-', label='Train')
        axes[2].plot(epochs, self.data['val_iou'], 'r-', label='Val')
        axes[2].set_title('Mean IoU')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('IoU')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                criterion: nn.Module, optimizer: Optimizer,
                device: torch.device) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss, total_dice, total_iou, count = 0, 0, 0, 0
    
    pbar = tqdm(dataloader, desc="Train", leave=False)
    for batch in pbar:
        inputs = batch['input'].to(device)
        targets = {k: batch[k].to(device) for k in ['mask', 'boundary']}
        complexity = batch['complexity'].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss_dict = criterion(outputs, targets, complexity)
        loss = loss_dict['total']
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        batch_dice = compute_dice(outputs['mask'], targets['mask'])
        batch_iou = compute_iou(outputs['mask'], targets['mask'])
        
        total_loss += batch_loss
        total_dice += batch_dice
        total_iou += batch_iou
        count += 1
        
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'dice': f'{batch_dice:.4f}',
            'iou': f'{batch_iou:.4f}'
        })
    
    return total_loss / count, total_dice / count, total_iou / count


@torch.no_grad()
def val_epoch(model: nn.Module, dataloader: DataLoader,
              criterion: nn.Module, device: torch.device) -> tuple:
    """Validate for one epoch."""
    model.eval()
    total_loss, total_dice, total_iou, count = 0, 0, 0, 0
    
    pbar = tqdm(dataloader, desc="Val  ", leave=False)
    for batch in pbar:
        inputs = batch['input'].to(device)
        targets = {k: batch[k].to(device) for k in ['mask', 'boundary']}
        complexity = batch['complexity'].to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets, complexity)['total'].item()
        dice = compute_dice(outputs['mask'], targets['mask'])
        iou = compute_iou(outputs['mask'], targets['mask'])
        
        total_loss += loss
        total_dice += dice
        total_iou += iou
        count += 1
        
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'dice': f'{dice:.4f}',
            'iou': f'{iou:.4f}'
        })
    
    return total_loss / count, total_dice / count, total_iou / count


def train_model(model: nn.Module, dataloaders: Dict[str, DataLoader],
                criterion: nn.Module, optimizer: Optimizer,
                scheduler: _LRScheduler, device: torch.device,
                num_epochs: int = 30, patience: int = 10,
                save_path: str = 'best_model.pth') -> MetricTracker:
    """
    Full training loop with early stopping.
    
    Args:
        model: The neural network to train
        dataloaders: Dict with 'train' and 'val' DataLoaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        save_path: Path to save best model
        
    Returns:
        MetricTracker with training history
    """
    tracker = MetricTracker()
    best_dice = 0
    wait = 0
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        # Train and validate
        train_loss, train_dice, train_iou = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device
        )
        val_loss, val_dice, val_iou = val_epoch(
            model, dataloaders['val'], criterion, device
        )
        
        scheduler.step()
        tracker.update(train_loss, val_loss, train_dice, val_dice, train_iou, val_iou)
        
        # Print metrics
        print(f"Train - Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f">>> Saved best model (Val Dice: {best_dice:.4f}, Val IoU: {val_iou:.4f})")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    tracker.plot()
    return tracker
