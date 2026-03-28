"""Generate the slim MABAAN_v4_Main.ipynb that imports from the mabaan package."""
import json
import sys
sys.path.insert(0, '.')
from nb_helper import md_cell, code_cell, make_notebook, save_notebook

cells = [
    # ===== HEADER =====
    md_cell("""# MABAAN v4 — Morphology-Adaptive Boundary-Aware Attention Network

Full model training and evaluation for cell segmentation on LIVECell.

**Features:**
- ResNet34 encoder with 4-channel input (image + edge map)
- Boundary-aware attention at every decoder stage
- Per-sample morphology complexity weighting in loss
- Morphology-stratified evaluation (Circular / Moderate / Irregular)"""),

    # ===== INSTALL =====
    code_cell("!pip install -q segmentation-models-pytorch pycocotools"),

    # ===== IMPORTS =====
    code_cell("""import os, sys, random, warnings, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Add project root to path (adjust for Kaggle if needed)
sys.path.insert(0, '/kaggle/working/MABAAN')

from mabaan.config import Config
from mabaan.data import LIVECellLoader, LiveCellDataset
from mabaan.models import MABAANUNet
from mabaan.utils import (
    MorphologyAwareLoss, compute_dice, compute_iou,
    run_inference, evaluate_model, morphology_stratified_evaluation,
    train_model,
)
from mabaan.utils.visualization import plot_sample_batch, plot_predictions, plot_complexity_analysis

warnings.filterwarnings('ignore')
print("All imports successful!")"""),

    # ===== SETUP =====
    code_cell("""def set_seed(seed=42):
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
    print("GPU count:", torch.cuda.device_count())
    print(f"GPU: {torch.cuda.get_device_name(0)}")"""),

    # ===== CONFIG =====
    code_cell("""config = Config()
# Override for this run:
# config.NUM_EPOCHS = 0       # Set to 0 to skip training
# config.MAX_SAMPLES = 100    # Limit samples for testing
print(f"Data path: {config.DATA_PATH}")
print(f"Encoder: {config.ENCODER}, Epochs: {config.NUM_EPOCHS}, Batch: {config.BATCH_SIZE}")"""),

    # ===== DATA LOADING =====
    md_cell("## Data Loading"),

    code_cell("""print("Loading datasets...")
train_loader = LIVECellLoader(config.DATA_PATH, split='train')
val_loader   = LIVECellLoader(config.DATA_PATH, split='val')
test_loader  = LIVECellLoader(config.DATA_PATH, split='test')

train_dataset = LiveCellDataset(train_loader, train_loader.get_image_ids(),
                                config.IMG_SIZE, max_samples=config.MAX_SAMPLES)
val_dataset   = LiveCellDataset(val_loader, val_loader.get_image_ids(),
                                config.IMG_SIZE, max_samples=config.MAX_SAMPLES)
test_dataset  = LiveCellDataset(test_loader, test_loader.get_image_ids(),
                                config.IMG_SIZE, max_samples=config.MAX_SAMPLES)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                        num_workers=config.NUM_WORKERS, pin_memory=True),
    'val':   DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True),
    'test':  DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                        num_workers=config.NUM_WORKERS, pin_memory=True),
}
print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")"""),

    # ===== SAMPLE VISUALIZATION =====
    code_cell("""batch = next(iter(dataloaders['train']))
plot_sample_batch(batch)"""),

    # ===== TRAINING =====
    md_cell("## Training"),

    code_cell("""model = MABAANUNet(
    encoder_name=config.ENCODER,
    in_channels=config.IN_CHANNELS,
    encoder_weights=config.ENCODER_WEIGHTS,
    reduction=config.ATTENTION_REDUCTION,
).to(DEVICE)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

criterion = MorphologyAwareLoss(bce_w=0.4, dice_w=0.3, boundary_w=0.3, complexity_scale=0.5)
optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

tracker = train_model(model, dataloaders, criterion, optimizer, scheduler, DEVICE,
                      num_epochs=config.NUM_EPOCHS, patience=config.EARLY_STOPPING_PATIENCE,
                      save_path='mabaan_best.pth')"""),

    code_cell("tracker.plot()"),

    # ===== CHECKPOINT RESUME =====
    md_cell("## Resume from Checkpoint (skip if just trained)"),

    code_cell("""# If training already completed and checkpoint exists, run this cell to load it.
CHECKPOINT_PATH = 'mabaan_best.pth'

if os.path.exists(CHECKPOINT_PATH):
    print(f"Loading checkpoint from {CHECKPOINT_PATH}")
    model = MABAANUNet(
        encoder_name=config.ENCODER, in_channels=config.IN_CHANNELS,
        encoder_weights=config.ENCODER_WEIGHTS, reduction=config.ATTENTION_REDUCTION,
    ).to(DEVICE)
    # Handle DataParallel 'module.' key prefixes
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Checkpoint loaded! Model params: {sum(p.numel() for p in model.parameters()):,}")
else:
    print(f"No checkpoint found at {CHECKPOINT_PATH}. Run training first.")"""),

    # ===== EVALUATION =====
    md_cell("## Official Evaluation on Test Set"),

    code_cell("""# Load best model (handles DataParallel keys)
state_dict = torch.load('mabaan_best.pth', map_location=DEVICE)
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

if isinstance(model, nn.DataParallel):
    model = model.module
model.load_state_dict(new_state_dict)
model.eval()
print("Best model loaded for evaluation")

preds, tgts, complexities = run_inference(model, dataloaders['test'], DEVICE)
mabaan_df = evaluate_model(preds, tgts, model_name="MABAAN (Full)")
mabaan_strat = morphology_stratified_evaluation(preds, tgts, complexities, model_name="MABAAN (Full)")"""),

    # ===== VISUALIZATIONS =====
    md_cell("## Prediction Visualization"),

    code_cell("plot_predictions(preds, tgts, complexities, title='MABAAN Predictions on Test Set')"),

    md_cell("## Complexity vs Metrics Analysis"),

    code_cell("plot_complexity_analysis(preds, tgts, complexities, mabaan_df)"),

    # ===== SAVE RESULTS =====
    md_cell("## Save Results"),

    code_cell("""results = {
    'metrics_df': mabaan_df,
    'complexities': complexities,
    'stratified': {k: v.to_dict() for k, v in mabaan_strat.items()},
}
with open('mabaan_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Results saved to mabaan_results.pkl")"""),

    # ===== RESEARCH QUESTIONS =====
    md_cell("""## Research Question 1: Boundary Metric Improvement

**RQ1: Does MABAAN improve boundary segmentation metrics compared to a standard U-Net baseline?**

The results above show MABAAN's performance on Dice, IoU, Boundary F1, HD, and HD95.
To answer RQ1 definitively, compare these metrics with the Baseline U-Net results
from `MABAAN_v4_Baseline.ipynb`. Key metrics to compare:

- **Boundary F1**: Higher = better boundary delineation
- **HD95**: Lower = boundaries are closer to ground truth
- **Dice/IoU**: Improved or maintained overall segmentation"""),

    md_cell("""## Research Question 2: Effect of Boundary-Aware Attention

**RQ2: How does the boundary-aware attention mechanism affect segmentation quality?**

Compare MABAAN (this notebook) vs Baseline (no attention, no edge channel):
- The attention mechanism modulates features using boundary-sensitive spatial and channel gates
- The edge channel provides explicit boundary information to the encoder
- Improvements in Boundary F1 and HD95 relative to baseline demonstrate attention's contribution

See the Baseline notebook for direct comparison values."""),

    md_cell("""## Research Question 3: Effect of Morphology-Weighted Loss

**RQ3: Does the morphology-weighted loss improve segmentation of complex cell shapes?**

Compare MABAAN (complexity_scale=0.5) vs Ablation (complexity_scale=0):
- Per-sample complexity weighting gives higher loss weight to irregular cells
- The morphology-stratified evaluation above shows per-bin performance
- If MABAAN outperforms the ablation in the "Irregular" bin, the loss is beneficial

See the Ablation notebook for comparison. Key evidence: improvement in HD95 and BF1 for irregular cells."""),

    md_cell("""## Research Question 4: Robustness Across Morphologies

**RQ4: How robust is MABAAN across different cell morphologies?**

Evidence from the morphology-stratified evaluation:
- Performance across Circular, Moderate, and Irregular bins
- Smaller variance = more robust
- The scatter and box plots above visualize this relationship
- A robust model maintains high Dice/IoU even for complex morphologies"""),
]

nb = make_notebook(cells)
save_notebook(nb, r"d:\Althaf\IIT\Final Year\FYP\Implementation Workspace\MABAAN\notebooks\MABAAN_v4_Main.ipynb")
