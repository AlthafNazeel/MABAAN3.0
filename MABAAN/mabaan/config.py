"""
Configuration for MABAAN training and evaluation.
"""


class Config:
    """Default configuration for MABAAN experiments."""

    # Data
    DATA_PATH = "/kaggle/input/datasets/althafnazeell/livecell"
    IMG_SIZE = 256
    MAX_SAMPLES = None
    NUM_WORKERS = 4

    # Model
    ENCODER = "resnet34"
    ENCODER_WEIGHTS = "imagenet"
    IN_CHANNELS = 4  # 3 image channels + 1 edge channel
    ATTENTION_REDUCTION = 16

    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    EARLY_STOPPING_PATIENCE = 10

    # Experimental switches
    USE_AMP = True                      # Mixed precision training
    GRAD_CLIP = 1.0                     # Gradient clipping max norm (0 = disabled)
    USE_AUGMENTATION = True             # Data augmentation
    USE_EDGE_CHANNEL = True             # 4th edge-map input channel
    BOUNDARY_THICKNESS = 3              # Boundary label thickness in pixels
    BOUNDARY_MODE = 'thick'             # 'thick' or 'thin'
    USE_CONSISTENCY_LOSS = False        # Edge consistency loss between branches
    MORPHOLOGY_WEIGHTING_MODE = 'pixel' # 'pixel', 'sample', or 'none'
    SEED = 42

    # Loss weights
    SEG_BCE_W = 0.25
    SEG_DICE_W = 0.35
    BND_BCE_W = 0.20
    BND_DICE_W = 0.15
    CONS_W = 0.05                       # Consistency loss weight (if enabled)
    MORPH_ALPHA = 2.0                   # Morphology weight map scaling
    MORPH_BOUNDARY_BOOST = 2.0          # Extra boundary emphasis for complex cells
