"""
config.py  —  Global configuration for the GNSS Spoofing Detection System v3
=============================================================================
PRINCIPLE CHANGES FROM v2.1:
  [C1] NEW: VAL_RATIO = 0.20
       Training files now split into train (80%) + val (20%) by TIME, not
       by file. The validation set is used for early stopping and threshold
       selection.  This prevents the model from overfitting to training-set
       loss and ensures the saved checkpoint generalises.
  [C2] REVISED: TRAIN_FILES now includes ds1 and ds2 (high-power attacks).
       Previously only ds4 (hardest traction) was in training.
       Adding ds1 (switching) and ds2 (+10 dB) gives the model exposure to
       two qualitatively different attack fingerprints, reducing the extreme
       mismatch between train and test attack diversity.
  [C3] REVISED: TEST_FILES is now explicit.
       ds3/ds5/ds6/ds7/ds8 remain strictly zero-shot holdout.
       cleanStatic80 is the clean test reference.
  [C4] NEW: SPOOF_THRESHOLD = 0.35
       Default inference threshold for softmax probability of spoofing.
       argmax (=threshold 0.5) is replaced everywhere by softmax+threshold
       to allow tuning independently of the training loss.
  [C5] REVISED: HIDDEN_DIM 128→64, LSTM_HIDDEN 64→32, DROPOUT 0.3→0.5,
       WEIGHT_DECAY 1e-4→5e-4.
       With only ~3000 training windows, HIDDEN_DIM=128 causes loss→0.000
       (training memorisation). Smaller model + stronger regularisation
       keeps training loss at a healthy ~0.05–0.15.
  [C6] REVISED: PATIENCE 20→15, and early stopping now monitors val DR
       instead of train loss (implemented in train_eval.py).
"""

import os

# ── 1. Directory Layout ────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR  = r'D:\李小双博士资料\2.观测数据\SDR Dataset\TEXBAT file'
DATA_PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DATASET_DIR   = os.path.join(BASE_DIR, 'data', 'dataset')
MODEL_DIR     = os.path.join(BASE_DIR, 'models')
LOG_BASE_DIR  = os.path.join(BASE_DIR, 'logs')
FIG_DIR       = os.path.join(BASE_DIR, 'figures')

for _d in [DATA_PROC_DIR, DATASET_DIR, MODEL_DIR, LOG_BASE_DIR, FIG_DIR]:
    os.makedirs(_d, exist_ok=True)


# ── 2. SDR Signal Parameters ───────────────────────────────────────────────────
FS_RAW          = 25e6
DECIMATION      = 5
FS_WORK         = FS_RAW / DECIMATION
INT_TIME_MS     = 4
DOPPLER_MAX_HZ  = 6000
DOPPLER_STEP_HZ = 50     # 从250降到50，Doppler精度从±125Hz提升到±25Hz
CN0_THRESHOLD   = 37.0   # 高于噪声本底 ~34.5，低于弱真实信号 ~38
SNAP_STRIDE_SEC = 0.5


# ── 3. Dataset Definition ──────────────────────────────────────────────────────
#  File                 | Label | Attack type
#  ---------------------|-------|---------------------------------------------
#  cleanStatic80.bin    |   0   | No attack — short static baseline (test only)
#  cleanStatic.bin      |   0   | No attack — full static baseline
#  cleanDynamic.bin     |   0   | No attack — dynamic (vehicle) baseline
#  ds1.bin              |   1   | Static, switch, HIGH power
#  ds2.bin              |   1   | Static, time-push, +10 dB power
#  ds3.bin              |   1   | Static, time-push, power-MATCHED
#  ds4.bin              |   1   | Static, PVT-push, power-MATCHED (hardest static)
#  ds5.bin              |   1   | Dynamic, switch, HIGH power
#  ds6.bin              |   1   | Dynamic, PVT-push, power-MATCHED (hardest dynamic)
#  ds7.bin              |   1   | Static, SCER carrier-phase-aligned (synthesised)
#  ds8.bin              |   1   | Static, SCER replay (synthesised from ds7)
DATA_FILES = {
    'cleanStatic80.bin': 0,
    'cleanStatic.bin':   0,
    'cleanDynamic.bin':  0,
    'ds1.bin': 1,
    'ds2.bin': 1,
    'ds3.bin': 1,
    'ds4.bin': 1,
    'ds5.bin': 1,
    'ds6.bin': 1,
    'ds7.bin': 1,
    'ds8.bin': 1,
}

# [C2] TRAIN_FILES: three clean + three spoofed covering different attack types.
#   - cleanStatic + cleanDynamic: authentic signal diversity (static + moving)
#   - ds1: switching attack — abrupt CN0 jump, easiest to detect, good anchor
#   - ds2: time-push +10 dB — power anomaly, teaches CN0 sensitivity
#   - ds4: PVT-push matched power — hardest traction, teaches subtle drift
# Rationale: training on three qualitatively different spoofed scenarios forces
# the model to learn general features (geometric inconsistency, temporal drift)
# rather than memorising a single attack fingerprint.
TRAIN_FILES = {
    'cleanStatic.bin',
    'cleanDynamic.bin',
    'ds1.bin',
    'ds2.bin',
    'ds4.bin',
}

# [C3] TEST_FILES: strictly zero-shot — none of these appear in TRAIN_FILES.
#   ds3: matched-power time-push (like ds2 but harder)
#   ds5: dynamic switching (like ds1 but moving platform)
#   ds6: dynamic matched-power PVT (hardest overall)
#   ds7/ds8: synthesised SCER attacks (most covert)
#   cleanStatic80: clean reference for FAR measurement
TEST_FILES = {
    'cleanStatic80.bin',
    'ds3.bin',
    'ds5.bin',
    'ds6.bin',
    'ds7.bin',
    'ds8.bin',
}

# [C1] VAL_RATIO: fraction of each training file held out as validation.
# Split is by TIME (last VAL_RATIO of each file's timeline), not random,
# to avoid temporal leakage between train and val.
VAL_RATIO = 0.20

# Attack onset timestamps (seconds into recording).
# Windows ending before this time in a spoofed file get label=0.
ATTACK_ONSET_SEC = {
    'ds1.bin': 98.0,
    'ds2.bin': 90.0,
    'ds3.bin': 90.0,
    'ds4.bin': 90.0,
    'ds5.bin': 98.0,
    'ds6.bin': 90.0,
    'ds7.bin': 100.0,
    'ds8.bin': 100.0,
}


# ── 4. Feature Dimensions ──────────────────────────────────────────────────────
NODE_FEATURE_DIM = 6
EDGE_FEATURE_DIM = 4
TEMPORAL_WINDOW  = 10


# ── 5. Normalisation Constants ─────────────────────────────────────────────────
CN0_MEAN            = 45.0
CN0_STD             = 8.0
DOP_SCALE           = 3000.0
CN0_RATE_SCALE      = 3.0
DOP_RATE_SCALE      = 200.0
PEAK_RATIO_SCALE    = 20.0
DIFF_CN0_STD        = 8.0
DIFF_DOP_SCALE      = 500.0
DIFF_CP_STD         = 0.3
DIFF_DOP_RATE_SCALE = 100.0


# ── 6. Model Hyper-parameters ──────────────────────────────────────────────────
EPOCHS       = 120
BATCH_SIZE   = 32
LR           = 3e-4        # slightly lower LR for more stable convergence
# [C5] Reduced capacity to prevent loss→0 memorisation on small dataset
HIDDEN_DIM   = 64          # was 128
GNN_HEADS    = 4
DROPOUT      = 0.5         # was 0.3
LSTM_HIDDEN  = 32          # was 64
LSTM_LAYERS  = 2
WEIGHT_DECAY = 5e-4        # was 1e-4
# [C6] Early stop monitors val DR; patience reduced because val DR is noisier
PATIENCE     = 15          # was 20

# [C4] Inference threshold: probability of spoofing above which we predict 1.
# Lower value → more sensitive (higher DR, higher FAR).
# Tune on val set; 0.35 is a good starting point for this imbalance ratio.
SPOOF_THRESHOLD = 0.35     # new parameter


# ── 7. Ablation & Comparison Flags ────────────────────────────────────────────
ABLATION_NO_TEMPORAL = False   # True → GNN-only (no LSTM)
RUN_BASELINE_MLP     = False    # True → also train/evaluate per-channel MLP