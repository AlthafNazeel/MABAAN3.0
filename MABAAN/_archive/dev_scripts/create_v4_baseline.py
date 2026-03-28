"""Generate MABAAN_v4_Baseline.ipynb — Plain SMP U-Net (no attention, no edge, standard loss)."""
import sys; sys.path.insert(0, '.')
from nb_helper import md_cell, code_cell, make_notebook, save_notebook
from shared_cells import *

BASELINE_MODEL_CELL = """class BaselineUNet(nn.Module):
    \"\"\"Plain SMP U-Net with ResNet34. No attention, no edge channel, standard loss.\"\"\"
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet"):
        super().__init__()
        self.model = smp.Unet(encoder_name=encoder_name, in_channels=3, classes=1,
                              encoder_weights=encoder_weights)
    def forward(self, x):
        # Use only RGB channels (ignore edge channel if present)
        rgb = x[:, :3]
        logits = self.model(rgb)
        mask = torch.sigmoid(logits)
        return {'mask': mask, 'boundary': mask, 'logits': logits}

print("BaselineUNet defined (no attention, no edge channel)")"""

BASELINE_LOSS_CELL = """class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1e-6):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        inter = (pred_flat * target_flat).sum()
        return 1 - (2*inter + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

class BaselineLoss(nn.Module):
    \"\"\"Standard BCE + Dice loss without boundary head or complexity weighting.\"\"\"
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_w, self.dice_w = bce_w, dice_w

    def forward(self, preds, targets, complexity=None):
        bce = self.bce(preds['mask'], targets['mask'])
        dice = self.dice(preds['mask'], targets['mask'])
        total = self.bce_w * bce + self.dice_w * dice
        return {'total': total, 'bce': bce, 'dice': dice, 'boundary': torch.tensor(0.0)}

print("BaselineLoss defined (BCE + Dice, no complexity weighting)")"""

BASELINE_TRAINING_CELL = """class MetricTracker:
    def __init__(self):
        self.history = {'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': [],
                        'train_iou': [], 'val_iou': []}
    def update(self, tl, vl, td, vd, ti, vi):
        self.history['train_loss'].append(tl); self.history['val_loss'].append(vl)
        self.history['train_dice'].append(td); self.history['val_dice'].append(vd)
        self.history['train_iou'].append(ti); self.history['val_iou'].append(vi)
    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        e = range(1, len(self.history['train_loss'])+1)
        for ax, key, t in [(axes[0], 'loss', 'Loss'), (axes[1], 'dice', 'Dice'), (axes[2], 'iou', 'IoU')]:
            ax.plot(e, self.history[f'train_{key}'], 'b-o', markersize=3, label='Train')
            ax.plot(e, self.history[f'val_{key}'], 'r-o', markersize=3, label='Val')
            ax.legend(); ax.set_title(t); ax.set_xlabel('Epoch')
        plt.tight_layout(); plt.show()

def train_epoch(model, dl, criterion, optimizer, device):
    model.train()
    total_loss, total_dice, total_iou, n = 0, 0, 0, 0
    for batch in tqdm(dl, desc="Train", leave=False):
        inp = batch['input'].to(device)
        masks = {'mask': batch['mask'].to(device), 'boundary': batch['boundary'].to(device)}
        out = model(inp)
        losses = criterion(out, masks)
        optimizer.zero_grad(); losses['total'].backward(); optimizer.step()
        total_loss += losses['total'].item() * inp.size(0)
        total_dice += compute_dice(out['mask'].detach(), masks['mask']) * inp.size(0)
        total_iou += compute_iou(out['mask'].detach(), masks['mask']) * inp.size(0)
        n += inp.size(0)
    return total_loss/n, total_dice/n, total_iou/n

def val_epoch(model, dl, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou, n = 0, 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dl, desc="Val", leave=False):
            inp = batch['input'].to(device)
            masks = {'mask': batch['mask'].to(device), 'boundary': batch['boundary'].to(device)}
            out = model(inp)
            losses = criterion(out, masks)
            total_loss += losses['total'].item() * inp.size(0)
            total_dice += compute_dice(out['mask'], masks['mask']) * inp.size(0)
            total_iou += compute_iou(out['mask'], masks['mask']) * inp.size(0)
            n += inp.size(0)
    return total_loss/n, total_dice/n, total_iou/n

def train_model(model, dataloaders, criterion, optimizer, scheduler, device,
                num_epochs=30, patience=10, save_path='best_model.pth'):
    tracker = MetricTracker()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        print(f"\\nEpoch {epoch+1}/{num_epochs}")
        tl, td, ti = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        vl, vd, vi = val_epoch(model, dataloaders['val'], criterion, device)
        scheduler.step()
        tracker.update(tl, vl, td, vd, ti, vi)
        print(f"  Train - Loss:{tl:.4f} Dice:{td:.4f} IoU:{ti:.4f}")
        print(f"  Val   - Loss:{vl:.4f} Dice:{vd:.4f} IoU:{vi:.4f}")
        if vl < best_val_loss:
            best_val_loss = vl
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  *** Best model saved (val_loss={vl:.4f}) ***")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    return tracker

print("Training functions defined")"""

BASELINE_EXEC_CELL = """model = BaselineUNet(encoder_name=config.ENCODER, encoder_weights=config.ENCODER_WEIGHTS).to(DEVICE)
criterion = BaselineLoss(bce_w=0.5, dice_w=0.5)
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
print(f"Baseline model params: {sum(p.numel() for p in model.parameters()):,}")
tracker = train_model(model, dataloaders, criterion, optimizer, scheduler, DEVICE,
                      num_epochs=config.NUM_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE,
                      save_path='baseline_best.pth')
tracker.plot()"""

BASELINE_EVAL_CELL = """model.load_state_dict(torch.load('baseline_best.pth', map_location=DEVICE))
preds, tgts, complexities = run_inference(model, dataloaders['test'], DEVICE)
baseline_df = evaluate_model(preds, tgts, model_name="Baseline U-Net")
baseline_strat = morphology_stratified_evaluation(preds, tgts, complexities, model_name="Baseline U-Net")"""

BASELINE_SAVE_CELL = """import pickle
results = {'metrics_df': baseline_df, 'complexities': complexities,
           'stratified': {k: v.to_dict() for k, v in baseline_strat.items()}}
with open('baseline_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Baseline results saved to baseline_results.pkl")"""

cells = [
    md_cell("# MABAAN v4 — Baseline: Plain SMP U-Net\n\nThis notebook trains a standard U-Net (ResNet34 encoder) without:\n- Boundary-aware attention\n- Edge channel input\n- Morphology-weighted loss\n\nUsed to answer **RQ1** (boundary improvement) and **RQ2** (attention effect)."),
    code_cell(INSTALL_CELL),
    code_cell(IMPORTS_CELL),
    code_cell(SETUP_CELL),
    code_cell(CONFIG_CELL),
    md_cell("## Preprocessing & Edge Detection"),
    code_cell(PREPROCESSING_CELL),
    code_cell(EDGE_DETECTION_CELL),
    md_cell("## Dataset"),
    code_cell(LOADER_CELL),
    code_cell(DATASET_CELL),
    md_cell("## Baseline Model (Standard SMP U-Net)"),
    code_cell(BASELINE_MODEL_CELL),
    md_cell("## Loss Function (Standard BCE + Dice)"),
    code_cell(BASELINE_LOSS_CELL),
    md_cell("## Metrics & Evaluation"),
    code_cell(METRICS_CELL),
    code_cell(EVAL_FUNCTIONS_CELL),
    md_cell("## Data Loading"),
    code_cell(DATA_LOADING_CELL),
    code_cell(SAMPLE_VIZ_CELL),
    md_cell("## Training"),
    code_cell(BASELINE_TRAINING_CELL),
    code_cell(BASELINE_EXEC_CELL),
    md_cell("## Official Evaluation on Test Set"),
    code_cell(BASELINE_EVAL_CELL),
    md_cell("## Save Results"),
    code_cell(BASELINE_SAVE_CELL),
]

nb = make_notebook(cells)
save_notebook(nb, r"d:\Althaf\IIT\Final Year\FYP\Implementation Workspace\MABAAN\notebooks\MABAAN_v4_Baseline.ipynb")
