"""
config.py  —  Global configuration for the GNSS Spoofing Detection System v2
=============================================================================
All hyper-parameters, file paths, and signal-processing constants are
centralised here.  Import this module from every other source file.

Design principle: every "magic number" must appear exactly once, here.
"""

import os

# ── 1. Directory Layout ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_RAW_DIR  = r'D:\李小双博士资料\2.观测数据\SDR Dataset\TEXBAT file'
DATA_PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DATASET_DIR   = os.path.join(BASE_DIR, 'data', 'dataset')
MODEL_DIR     = os.path.join(BASE_DIR, 'models')
LOG_BASE_DIR  = os.path.join(BASE_DIR, 'logs')
FIG_DIR       = os.path.join(BASE_DIR, 'figures')

for _d in [DATA_PROC_DIR, DATASET_DIR, MODEL_DIR, LOG_BASE_DIR, FIG_DIR]:
    os.makedirs(_d, exist_ok=True)


# ── 2. SDR Signal Parameters ───────────────────────────────────────────────────
FS_RAW         = 25e6
DECIMATION     = 5
FS_WORK        = FS_RAW / DECIMATION   # 5.0 MHz
INT_TIME_MS    = 4
DOPPLER_MAX_HZ   = 6000
DOPPLER_STEP_HZ  = 250
CN0_THRESHOLD    = 30.0
SNAP_STRIDE_SEC  = 0.5


# ── 3. Dataset Definition ──────────────────────────────────────────────────────
# Maps filename → binary label (0 = authentic, 1 = spoofed).
DATA_FILES = {
    'cleanStatic80.bin':  0,
    'cleanStatic.bin':    0,
    'cleanDynamic.bin':   0,
    'ds1.bin':            1,
    'ds2.bin':            1,
    'ds3.bin':            1,
    'ds4.bin':            1,
    'ds5.bin':            1,
    'ds6.bin':            1,
    'ds7.bin':            1,
    'ds8.bin':            1,
}

# ── 3b. Time-resolved attack onset (seconds from file start) ──────────────────
# TEXBAT recordings: spoofing does NOT start at t=0. The first ~90-100s are
# clean authentic signal to let receivers warm up. Windows before the onset
# should be labelled clean (0) even in attack files.
# Values derived from TEXBAT documentation (Humphreys 2012, Table I text).
# For clean files, onset = infinity (never attacked).
ATTACK_ONSET_SEC = {
    'cleanStatic80.bin':  float('inf'),
    'cleanStatic.bin':    float('inf'),
    'cleanDynamic.bin':   float('inf'),
    'ds1.bin':            98.0,    # switch occurs ~98 s
    'ds2.bin':            90.0,    # time push starts ~90 s
    'ds3.bin':            90.0,    # matched-power time push ~90 s
    'ds4.bin':            90.0,    # matched-power position push ~90 s
    'ds5.bin':            98.0,    # dynamic switch ~98 s
    'ds6.bin':            90.0,    # dynamic position push ~90 s
    'ds7.bin':            100.0,   # SCER onset ~100 s
    'ds8.bin':            100.0,   # SCER replay onset ~100 s
}

TRAIN_FILES = {'cleanStatic.bin', 'cleanDynamic.bin', 'ds4.bin'}


# ── 4. Feature Dimensions ──────────────────────────────────────────────────────
NODE_FEATURE_DIM = 6
EDGE_FEATURE_DIM = 4
TEMPORAL_WINDOW  = 10


# ── 5. Normalisation Constants ─────────────────────────────────────────────────
CN0_MEAN           = 45.0
CN0_STD            = 8.0
DOP_SCALE          = 3000.0
CN0_RATE_SCALE     = 3.0
DOP_RATE_SCALE     = 200.0
PEAK_RATIO_SCALE   = 20.0
DIFF_CN0_STD       = 8.0
DIFF_DOP_SCALE     = 500.0
DIFF_CP_STD        = 0.3
DIFF_DOP_RATE_SCALE = 100.0


# ── 6. Model Hyper-parameters ──────────────────────────────────────────────────
EPOCHS       = 100
BATCH_SIZE   = 32
LR           = 5e-4
WEIGHT_DECAY = 1e-4
HIDDEN_DIM   = 128
GNN_HEADS    = 4
DROPOUT      = 0.3
LSTM_HIDDEN  = 64
LSTM_LAYERS  = 2
PATIENCE     = 20


# ── 7. Ablation & Baseline Flags ───────────────────────────────────────────────
# Set these to run different experiments:
#   ABLATION_NO_TEMPORAL = True   → GNN-only (no LSTM), tests spatial contribution
#   RUN_BASELINE_MLP     = True   → also train/evaluate a per-channel MLP baseline
ABLATION_NO_TEMPORAL = False
RUN_BASELINE_MLP     = True