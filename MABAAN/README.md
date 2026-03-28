# MABAAN — Morphology-Adaptive Boundary-Aware Attention Network

A deep learning framework for cell segmentation that adapts to cell morphology using boundary-aware attention mechanisms and morphology-weighted loss.

## Key Features

- **Boundary-Aware Attention** — Channel, spatial, and boundary-specific attention at every decoder stage
- **Morphology-Adaptive Edge Detection** — Multi-scale morphological gradient fusion as a 4th input channel
- **Per-Sample Complexity Weighting** — Loss scaled by shape complexity (circularity × solidity) to emphasize irregular cells
- **Morphology-Stratified Evaluation** — Performance broken down by Circular / Moderate / Irregular cell shapes
- **Comprehensive Metrics** — Dice, IoU, Boundary F1, Hausdorff Distance, HD95

## Project Structure

```
MABAAN/
├── mabaan/                          # Main package
│   ├── __init__.py
│   ├── config.py                    # Training & model configuration
│   ├── data/                        # Data loading & preprocessing
│   │   ├── preprocessing.py         # Normalize, denoise, CLAHE
│   │   ├── edge_detection.py        # Shape descriptors, multi-scale edge fusion
│   │   ├── loader.py                # LIVECell COCO annotation loader
│   │   └── dataset.py               # PyTorch Dataset (4-ch input)
│   ├── models/                      # Model architecture
│   │   ├── attention.py             # Channel/Spatial/Boundary attention
│   │   ├── decoder.py               # MABAAN decoder with attention blocks
│   │   └── mabaan_unet.py           # Full MABAANUNet model
│   └── utils/                       # Training & evaluation
│       ├── losses.py                # DiceLoss, MorphologyAwareLoss
│       ├── metrics.py               # Dice, IoU, BF1, HD, HD95
│       ├── evaluation.py            # Inference, evaluation, stratified eval
│       ├── training.py              # Train/val loops, MetricTracker
│       └── visualization.py         # Plotting functions
├── notebooks/
│   └── MABAAN_v4_Main.ipynb         # Main training & evaluation notebook
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Using the notebook (Kaggle)

1. Upload the `mabaan/` package to your Kaggle working directory
2. Open `notebooks/MABAAN_v4_Main.ipynb`
3. Set `Config.DATA_PATH` to your LIVECell dataset path
4. Run all cells

### Using the package directly

```python
from mabaan.config import Config
from mabaan.data import LIVECellLoader, LiveCellDataset
from mabaan.models import MABAANUNet
from mabaan.utils import MorphologyAwareLoss, train_model

config = Config()
model = MABAANUNet(
    encoder_name=config.ENCODER,
    in_channels=config.IN_CHANNELS,
    encoder_weights=config.ENCODER_WEIGHTS,
)
criterion = MorphologyAwareLoss(complexity_scale=0.5)
```

## Dataset

This project uses the [LIVECell](https://sartorius-research.github.io/LIVECell/) dataset with COCO-format annotations.

## Research Questions

| RQ  | Question | Evidence |
|-----|----------|----------|
| RQ1 | Does MABAAN improve boundary metrics vs baseline U-Net? | Boundary F1, HD95 comparison |
| RQ2 | How does boundary-aware attention affect quality? | MABAAN vs baseline (no attention) |
| RQ3 | Does morphology-weighted loss help complex cells? | MABAAN vs ablation (complexity_scale=0) |
| RQ4 | How robust is MABAAN across morphologies? | Stratified evaluation across bins |

## License

This project is for academic research purposes.
