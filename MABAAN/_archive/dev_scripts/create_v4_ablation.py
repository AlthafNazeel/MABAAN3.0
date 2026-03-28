"""Generate MABAAN_v4_Ablation.ipynb — MABAAN architecture but complexity_scale=0."""
import sys; sys.path.insert(0, '.')
from nb_helper import md_cell, code_cell, make_notebook, save_notebook
from shared_cells import *

# Reuse attention and model from the main notebook
ATTENTION_CELL = """class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.ReLU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1).unsqueeze(-1)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False), nn.Sigmoid())
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.conv(torch.cat([avg_out, max_out], dim=1))

class BoundaryAwareAttentionBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels), nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        ca = self.channel_att(x)
        sa = self.spatial_att(ca)
        boundary_mask = self.boundary_conv(x)
        return sa + self.gamma * (sa * boundary_mask)

print("Boundary-aware attention module defined")"""

MODEL_CELL = """class MABAANDecoder(nn.Module):
    def __init__(self, encoder_channels, reduction=16):
        super().__init__()
        dec_channels = [256, 128, 64, 32]
        self.blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        for i, dc in enumerate(dec_channels):
            in_ch = encoder_channels[-(i+1)] + (encoder_channels[-(i+2)] if i < len(dec_channels)-1 else encoder_channels[0])
            self.blocks.append(nn.Sequential(
                nn.Conv2d(in_ch, dc, 3, padding=1, bias=False), nn.BatchNorm2d(dc), nn.ReLU(inplace=True),
                nn.Conv2d(dc, dc, 3, padding=1, bias=False), nn.BatchNorm2d(dc), nn.ReLU(inplace=True)))
            self.attention_blocks.append(BoundaryAwareAttentionBlock(dc, reduction))
    def forward(self, features):
        x = features[-1]
        for i, (block, att) in enumerate(zip(self.blocks, self.attention_blocks)):
            skip = features[-(i+2)] if i < len(self.blocks)-1 else features[0]
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            x = att(x)
        return x

class MABAANUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", in_channels=4, classes=1, encoder_weights="imagenet", reduction=16):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=in_channels, depth=5, weights=encoder_weights)
        enc_channels = self.encoder.out_channels
        self.decoder = MABAANDecoder(enc_channels, reduction)
        self.mask_head = nn.Conv2d(32, classes, 1)
        self.boundary_head = nn.Conv2d(32, classes, 1)
    def forward(self, x):
        features = self.encoder(x)
        dec_out = self.decoder(features)
        dec_up = F.interpolate(dec_out, size=x.shape[2:], mode='bilinear', align_corners=False)
        mask = torch.sigmoid(self.mask_head(dec_up))
        boundary = torch.sigmoid(self.boundary_head(dec_up))
        return {'mask': mask, 'boundary': boundary, 'logits': self.mask_head(dec_up)}

print("MABAANUNet model defined (same architecture, ablation is in the loss)")"""

ABLATION_LOSS_CELL = """class DiceLoss(nn.Module):
    def forward(self, pred, target, smooth=1e-6):
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        inter = (pred_flat * target_flat).sum()
        return 1 - (2*inter + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

class MorphologyAwareLoss(nn.Module):
    \"\"\"ABLATION: Same loss structure but complexity_scale=0 (no morphology weighting).\"\"\"
    def __init__(self, bce_w=0.4, dice_w=0.3, boundary_w=0.3, complexity_scale=0.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_w, self.dice_w, self.boundary_w = bce_w, dice_w, boundary_w
        self.complexity_scale = complexity_scale  # Fixed at 0.0

    def forward(self, preds, targets, complexity=None):
        bce = self.bce(preds['mask'], targets['mask'])
        dice = self.dice(preds['mask'], targets['mask'])
        boundary = self.bce(preds['boundary'], targets['boundary'])
        total = self.bce_w * bce + self.dice_w * dice + self.boundary_w * boundary
        # complexity_scale=0 so no morphology weighting applied
        return {'total': total, 'bce': bce, 'dice': dice, 'boundary': boundary}

print("Ablation loss defined (complexity_scale=0, no morphology weighting)")"""

ABLATION_TRAINING_CELL = """class MetricTracker:
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
        complexity = batch['complexity'].to(device)
        out = model(inp)
        losses = criterion(out, masks, complexity)
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
            complexity = batch['complexity'].to(device)
            out = model(inp)
            losses = criterion(out, masks, complexity)
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

ABLATION_EXEC_CELL = """model = MABAANUNet(encoder_name=config.ENCODER, in_channels=config.IN_CHANNELS,
                   encoder_weights=config.ENCODER_WEIGHTS, reduction=config.ATTENTION_REDUCTION).to(DEVICE)
criterion = MorphologyAwareLoss(bce_w=0.4, dice_w=0.3, boundary_w=0.3, complexity_scale=0.0)
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
print(f"Ablation model params: {sum(p.numel() for p in model.parameters()):,}")
print("NOTE: This is MABAAN architecture with complexity_scale=0 (no morphology weighting)")
tracker = train_model(model, dataloaders, criterion, optimizer, scheduler, DEVICE,
                      num_epochs=config.NUM_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE,
                      save_path='ablation_best.pth')
tracker.plot()"""

ABLATION_EVAL_CELL = """model.load_state_dict(torch.load('ablation_best.pth', map_location=DEVICE))
preds, tgts, complexities = run_inference(model, dataloaders['test'], DEVICE)
ablation_df = evaluate_model(preds, tgts, model_name="MABAAN Ablation (no morph weight)")
ablation_strat = morphology_stratified_evaluation(preds, tgts, complexities, model_name="MABAAN Ablation")"""

ABLATION_SAVE_CELL = """import pickle
results = {'metrics_df': ablation_df, 'complexities': complexities,
           'stratified': {k: v.to_dict() for k, v in ablation_strat.items()}}
with open('ablation_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Ablation results saved to ablation_results.pkl")"""

cells = [
    md_cell("# MABAAN v4 — Ablation: No Morphology Weighting\\n\\nThis notebook trains the full MABAAN architecture (with attention + edge channel)\\nbut with `complexity_scale=0` — disabling morphology-weighted loss.\\n\\nUsed to answer **RQ3** (effect of morphology-weighted loss).\\nCompare with the Main notebook (complexity_scale=0.5) to isolate the loss contribution."),
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
    md_cell("## Model Architecture (Same as Full MABAAN)"),
    code_cell(ATTENTION_CELL),
    code_cell(MODEL_CELL),
    md_cell("## Loss Function (complexity_scale = 0)"),
    code_cell(ABLATION_LOSS_CELL),
    md_cell("## Metrics & Evaluation"),
    code_cell(METRICS_CELL),
    code_cell(EVAL_FUNCTIONS_CELL),
    md_cell("## Data Loading"),
    code_cell(DATA_LOADING_CELL),
    code_cell(SAMPLE_VIZ_CELL),
    md_cell("## Training (Ablation — No Morphology Weighting)"),
    code_cell(ABLATION_TRAINING_CELL),
    code_cell(ABLATION_EXEC_CELL),
    md_cell("## Official Evaluation on Test Set"),
    code_cell(ABLATION_EVAL_CELL),
    md_cell("## Save Results"),
    code_cell(ABLATION_SAVE_CELL),
]

nb = make_notebook(cells)
save_notebook(nb, r"d:\Althaf\IIT\Final Year\FYP\Implementation Workspace\MABAAN\notebooks\MABAAN_v4_Ablation.ipynb")
