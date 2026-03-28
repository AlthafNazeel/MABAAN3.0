"""
Training utilities for MABAAN.

Includes MetricTracker for logging, train/val epoch functions (legacy and v2),
and the main training loop with early stopping, AMP, gradient clipping,
multi-criteria checkpointing, and complexity logging.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from .metrics import compute_dice, compute_iou, batch_metrics, compute_boundary_f1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed=42):
    """Set all random seeds for full reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """DataLoader worker init function for reproducible data loading."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Metric tracker
# ---------------------------------------------------------------------------

class MetricTracker:
    """Tracks and plots training metrics across epochs."""

    def __init__(self):
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice': [], 'val_dice': [],
            'train_iou': [], 'val_iou': [],
            'val_bf1': [],
            'avg_complexity': [],
        }

    def update(self, tl, vl, td, vd, ti, vi, avg_c=0.0, vbf1=0.0):
        self.history['train_loss'].append(tl)
        self.history['val_loss'].append(vl)
        self.history['train_dice'].append(td)
        self.history['val_dice'].append(vd)
        self.history['train_iou'].append(ti)
        self.history['val_iou'].append(vi)
        self.history['val_bf1'].append(vbf1)
        self.history['avg_complexity'].append(avg_c)

    def plot(self):
        """Plot training curves: Loss, Dice, IoU, BF1, and Complexity."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        e = range(1, len(self.history['train_loss']) + 1)
        for ax, key, title in [
            (axes[0, 0], 'loss', 'Loss'),
            (axes[0, 1], 'dice', 'Dice'),
            (axes[0, 2], 'iou', 'IoU'),
        ]:
            ax.plot(e, self.history[f'train_{key}'], 'b-o', markersize=3, label='Train')
            ax.plot(e, self.history[f'val_{key}'], 'r-o', markersize=3, label='Val')
            ax.legend()
            ax.set_title(title)
            ax.set_xlabel('Epoch')

        axes[1, 0].plot(e, self.history['val_bf1'], 'm-o', markersize=3)
        axes[1, 0].set_title('Val Boundary F1')
        axes[1, 0].set_xlabel('Epoch')

        axes[1, 1].plot(e, self.history['avg_complexity'], 'g-o', markersize=3)
        axes[1, 1].set_title('Avg Complexity')
        axes[1, 1].set_xlabel('Epoch')

        axes[1, 2].axis('off')
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Legacy training functions (preserved for backward compatibility)
# ---------------------------------------------------------------------------

def train_epoch(model, dl, criterion, optimizer, device):
    """Run one training epoch, returning loss, dice, iou, and avg complexity."""
    model.train()
    total_loss, total_dice, total_iou, total_complexity, n = 0, 0, 0, 0, 0
    for batch in tqdm(dl, desc="Train", leave=False):
        inp = batch['input'].to(device)
        masks = {'mask': batch['mask'].to(device), 'boundary': batch['boundary'].to(device)}
        complexity = batch['complexity'].to(device)
        out = model(inp)
        losses = criterion(out, masks, complexity)
        optimizer.zero_grad()
        losses['total'].backward()
        optimizer.step()
        total_loss += losses['total'].item() * inp.size(0)
        total_dice += compute_dice(out['mask'].detach(), masks['mask']) * inp.size(0)
        total_iou += compute_iou(out['mask'].detach(), masks['mask']) * inp.size(0)
        total_complexity += complexity.mean().item() * inp.size(0)
        n += inp.size(0)
    return total_loss / n, total_dice / n, total_iou / n, total_complexity / n


def val_epoch(model, dl, criterion, device):
    """Run one validation epoch, returning loss, dice, iou, and avg complexity."""
    model.eval()
    total_loss, total_dice, total_iou, total_complexity, n = 0, 0, 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Val", leave=False):
            inp = batch['input'].to(device)
            masks = {'mask': batch['mask'].to(device), 'boundary': batch['boundary'].to(device)}
            complexity = batch['complexity'].to(device)
            out = model(inp)
            losses = criterion(out, masks, complexity)
            total_loss += losses['total'].item() * inp.size(0)
            total_dice += compute_dice(out['mask'], masks['mask']) * inp.size(0)
            total_iou += compute_iou(out['mask'], masks['mask']) * inp.size(0)
            total_complexity += complexity.mean().item() * inp.size(0)
            n += inp.size(0)
    return total_loss / n, total_dice / n, total_iou / n, total_complexity / n


def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=30, patience=10, save_path='best_model.pth',
                resume_from=None):
    """Full training loop with early stopping, DataParallel-aware saving, and checkpoint resume.

    Args:
        resume_from: Path to a checkpoint to resume from. If provided, loads the
                     state dict into the model and runs one val epoch to establish
                     the baseline best_val_loss before training continues.
    """
    tracker = MetricTracker()
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"  Resuming from checkpoint: {resume_from}")
        state_dict = torch.load(resume_from, map_location=device)
        # Strip 'module.' prefix if saved with DataParallel
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)
        # Run one val epoch to get baseline val_loss
        vl, vd, vi, vc = val_epoch(model, dataloaders['val'], criterion, device)
        best_val_loss = vl
        print(f"  Checkpoint loaded. Baseline val_loss={vl:.4f}, Dice={vd:.4f}, IoU={vi:.4f}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        tl, td, ti, tc = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        vl, vd, vi, vc = val_epoch(model, dataloaders['val'], criterion, device)
        scheduler.step()
        tracker.update(tl, vl, td, vd, ti, vi, tc)
        print(f"  Train - Loss:{tl:.4f} Dice:{td:.4f} IoU:{ti:.4f} Complexity:{tc:.3f}")
        print(f"  Val   - Loss:{vl:.4f} Dice:{vd:.4f} IoU:{vi:.4f}")
        if vl < best_val_loss:
            best_val_loss = vl
            epochs_no_improve = 0
            # Save without DataParallel wrapper
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"  *** Best model saved (val_loss={vl:.4f}) ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch + 1}")
                break
    return tracker


# ---------------------------------------------------------------------------
# Upgraded training functions (AMP, grad clipping, multi-checkpoint)
# ---------------------------------------------------------------------------

def train_epoch_v2(model, dl, criterion, optimizer, device,
                   scaler=None, grad_clip=1.0):
    """Upgraded training epoch with AMP and gradient clipping.

    Args:
        scaler: GradScaler instance (None to disable AMP).
        grad_clip: Max gradient norm (0 to disable).
    """
    try:
        from torch.amp import autocast
        amp_device = 'cuda'
    except ImportError:
        from torch.cuda.amp import autocast
        amp_device = None

    model.train()
    total_loss, total_dice, total_iou, n = 0.0, 0.0, 0.0, 0

    for batch in tqdm(dl, desc="Train", leave=False):
        inp = batch['input'].to(device)
        targets = {
            'mask': batch['mask'].to(device),
            'boundary': batch['boundary'].to(device),
        }
        if 'weight_map' in batch:
            targets['weight_map'] = batch['weight_map'].to(device)

        optimizer.zero_grad(set_to_none=True)

        ctx = autocast(amp_device, enabled=(scaler is not None)) if amp_device else autocast(enabled=(scaler is not None))
        with ctx:
            out = model(inp)
            losses = criterion(out, targets)

        if scaler is not None:
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses['total'].backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        dice_b, iou_b = batch_metrics(out['mask'], targets['mask'])
        bs = inp.size(0)
        total_loss += losses['total'].item() * bs
        total_dice += dice_b * bs
        total_iou += iou_b * bs
        n += bs

    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def val_epoch_v2(model, dl, criterion, device, threshold=0.5):
    """Upgraded validation epoch tracking Boundary F1 alongside Dice/IoU."""
    model.eval()
    total_loss, total_dice, total_iou, total_bf1, n = 0.0, 0.0, 0.0, 0.0, 0

    for batch in tqdm(dl, desc="Val", leave=False):
        inp = batch['input'].to(device)
        targets = {
            'mask': batch['mask'].to(device),
            'boundary': batch['boundary'].to(device),
        }
        if 'weight_map' in batch:
            targets['weight_map'] = batch['weight_map'].to(device)

        out = model(inp)
        losses = criterion(out, targets)

        dice_b, iou_b = batch_metrics(out['mask'], targets['mask'], threshold=threshold)

        # Compute boundary F1 per image
        pred_np = out['mask'].cpu().numpy()
        tgt_np = targets['mask'].cpu().numpy()
        bf1_scores = [
            compute_boundary_f1(pred_np[i, 0], tgt_np[i, 0], threshold=threshold)
            for i in range(pred_np.shape[0])
        ]
        bf1_b = float(np.mean(bf1_scores))

        bs = inp.size(0)
        total_loss += losses['total'].item() * bs
        total_dice += dice_b * bs
        total_iou += iou_b * bs
        total_bf1 += bf1_b * bs
        n += bs

    return total_loss / n, total_dice / n, total_iou / n, total_bf1 / n


def train_model_v2(model, dataloaders, criterion, optimizer, scheduler, device,
                   num_epochs=30, patience=10, use_amp=True, grad_clip=1.0,
                   resume_from=None):
    """Upgraded training loop with:
    - AMP (mixed precision)
    - Gradient clipping
    - Multi-criteria checkpointing (best by loss, dice, boundary F1)
    - Boundary F1 tracking during validation
    - Comprehensive epoch logging

    Args:
        use_amp: Enable mixed precision training.
        grad_clip: Maximum gradient norm (0 to disable).
        resume_from: Path to checkpoint to resume from.
    """
    try:
        from torch.amp import GradScaler
    except ImportError:
        from torch.cuda.amp import GradScaler
    from .evaluation import find_best_threshold

    scaler = GradScaler('cuda', enabled=use_amp) if use_amp else None
    tracker = MetricTracker()

    print(f"  AMP: {'ON' if use_amp else 'OFF'} | Grad clip: {grad_clip}")
    print(f"  Device: {device} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)} x{torch.cuda.device_count()}")

    best = {
        'val_loss': float('inf'),
        'val_dice': -1.0,
        'val_bf1': -1.0,
    }
    no_improve = 0

    # Resume from checkpoint if provided
    if resume_from and os.path.exists(resume_from):
        print(f"  Resuming from checkpoint: {resume_from}")
        state_dict = torch.load(resume_from, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(new_state_dict)
        print("  Checkpoint loaded.")

    # Find initial best threshold
    best_thr, _ = find_best_threshold(model, dataloaders['val'], device)
    print(f"Initial best threshold: {best_thr:.2f}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        tr_loss, tr_dice, tr_iou = train_epoch_v2(
            model, dataloaders['train'], criterion, optimizer, device,
            scaler=scaler, grad_clip=grad_clip,
        )

        val_loss, val_dice, val_iou, val_bf1 = val_epoch_v2(
            model, dataloaders['val'], criterion, device, threshold=best_thr,
        )

        if scheduler is not None:
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        tracker.update(
            tr_loss, val_loss, tr_dice, val_dice, tr_iou, val_iou,
            avg_c=0.0, vbf1=val_bf1,
        )

        print(f"  Train - Loss:{tr_loss:.4f} Dice:{tr_dice:.4f} IoU:{tr_iou:.4f}")
        print(f"  Val   - Loss:{val_loss:.4f} Dice:{val_dice:.4f} IoU:{val_iou:.4f} BF1:{val_bf1:.4f}")

        improved = False

        # Save best by each criterion
        _model_sd = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()

        if val_loss < best['val_loss']:
            best['val_loss'] = val_loss
            torch.save(_model_sd, 'best_by_loss.pth')
            print(f"  *** Best by loss saved (val_loss={val_loss:.4f}) ***")
            improved = True

        if val_dice > best['val_dice']:
            best['val_dice'] = val_dice
            torch.save(_model_sd, 'best_by_dice.pth')
            print(f"  *** Best by Dice saved (val_dice={val_dice:.4f}) ***")
            improved = True

        if val_bf1 > best['val_bf1']:
            best['val_bf1'] = val_bf1
            torch.save(_model_sd, 'best_by_bf1.pth')
            print(f"  *** Best by BF1 saved (val_bf1={val_bf1:.4f}) ***")
            improved = True

        if improved:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    return tracker
