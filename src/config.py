"""
config.py  —  Global configuration for the GNSS Spoofing Detection System v2
=============================================================================
All hyper-parameters, file paths, and signal-processing constants are
centralised here.  Import this module from every other source file.

Design principle: every "magic number" must appear exactly once, here.
"""

import os

# ── 1. Directory Layout ────────────────────────────────────────────────────────
# Base directory is the project root (one level above this file's src/ folder).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw TEXBAT .bin files — **edit this path to match your local storage**.
DATA_RAW_DIR = r'D:\李小双博士资料\2.观测数据\SDR Dataset\TEXBAT file'

# Intermediate CSV files produced by the feature extractor.
DATA_PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# PyTorch Geometric .pt dataset files.
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'dataset')

# Trained model checkpoints.
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Runtime logs (one timestamped sub-folder per training run).
LOG_BASE_DIR = os.path.join(BASE_DIR, 'logs')

# Paper-quality figures.
FIG_DIR = os.path.join(BASE_DIR, 'figures')

# Auto-create all directories on import.
for _d in [DATA_PROC_DIR, DATASET_DIR, MODEL_DIR, LOG_BASE_DIR, FIG_DIR]:
    os.makedirs(_d, exist_ok=True)


# ── 2. SDR Signal Parameters ───────────────────────────────────────────────────
# Original hardware sampling rate of the TEXBAT recordings.
FS_RAW = 25e6           # Hz  (25 MSPS, interleaved I/Q int16 pairs)

# Decimation factor applied before correlation.
# Work sampling rate = 25 MHz / 5 = 5 MHz.
# At 5 MHz, one C/A chip spans ≈ 4.89 samples — enough resolution for
# sub-chip code-phase estimation via parabolic interpolation.
DECIMATION = 5
FS_WORK = FS_RAW / DECIMATION   # 5.0 MHz

# Coherent integration time (in milliseconds).
# 4 ms gives CN0 sensitivity down to ~28 dB-Hz while keeping computational
# cost manageable.  Longer integration improves sensitivity but aliases
# Doppler if the receiver is highly dynamic.
INT_TIME_MS = 4

# Doppler search grid: ±6000 Hz in 250 Hz steps.
DOPPLER_MAX_HZ   = 6000
DOPPLER_STEP_HZ  = 250

# Minimum CN0 (dB-Hz) for a satellite to be considered visible/tracked.
CN0_THRESHOLD = 30.0   # raised from 35 dBHz to capture weaker spoofed sats

# Snapshot stride: one feature snapshot every 0.5 seconds.
SNAP_STRIDE_SEC = 0.5


# ── 3. Dataset Definition ──────────────────────────────────────────────────────
# Maps filename → binary label (0 = authentic, 1 = spoofed).
# This is the ground-truth label table for the complete TEXBAT dataset.
#
#  File                 | Label | Scenario description
#  ---------------------|-------|--------------------------------------------------
#  cleanStatic80.bin    |   0   | 80-second static authentic baseline
#  cleanStatic.bin      |   0   | Full-length static authentic baseline
#  cleanDynamic.bin     |   0   | Dynamic (vehicle) authentic baseline
#  ds1.bin              |   1   | Static,  switching,   high power
#  ds2.bin              |   1   | Static,  time-push,   +10 dB
#  ds3.bin              |   1   | Static,  time-push,   power-matched
#  ds4.bin              |   1   | Static,  PVT-push,    power-matched (hardest static)
#  ds5.bin              |   1   | Dynamic, switching,   high power
#  ds6.bin              |   1   | Dynamic, PVT-push,    power-matched (hardest dynamic)
#  ds7.bin              |   1   | Static,  carrier-phase-aligned SCER (synthesised)
#  ds8.bin              |   1   | Static,  SCER replay  (synthesised from ds7)
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

# Training split: only these files are used to fit the model.
# Everything else is held out as zero-shot test scenarios.
#
# Rationale for this split:
#   • cleanStatic + cleanDynamic  → authentic signal topology (static + dynamic)
#   • ds4                         → the hardest *known* covert static attack
# This forces the model to generalise to ds1/ds2/ds3/ds5/ds6/ds7/ds8 without
# ever seeing them during training — a strict evaluation protocol.
TRAIN_FILES = {'cleanStatic.bin', 'cleanDynamic.bin', 'ds4.bin'}


# ── 4. Feature Dimensions ──────────────────────────────────────────────────────
# Node feature vector (per satellite per snapshot):
#   [0] CN0_norm          — carrier-to-noise density, z-score normalised
#   [1] Doppler_norm      — absolute Doppler, tanh-compressed
#   [2] CodePhase_norm    — fractional chip offset from correlation peak (0–1)
#   [3] CN0_rate_norm     — first difference of CN0 over consecutive snapshots
#   [4] Doppler_rate_norm — first difference of Doppler (acceleration proxy)
#   [5] PeakRatio_norm    — peak-to-sidelobe ratio of correlation function
NODE_FEATURE_DIM = 6

# Edge feature vector (inter-satellite single difference i → j):
#   [0] dCN0         — normalised CN0 difference
#   [1] dDoppler     — normalised Doppler difference
#   [2] dCodePhase   — normalised code-phase difference
#   [3] dDoppler_rate — normalised Doppler-rate difference
EDGE_FEATURE_DIM = 4

# Temporal window: number of consecutive snapshots stacked into one sample.
# A window of 10 snapshots × 0.5 s = 5 seconds captures slow traction attacks.
TEMPORAL_WINDOW = 10


# ── 5. Normalisation Constants ─────────────────────────────────────────────────
# These are derived from the cleanStatic baseline statistics.
# Changing them requires re-building the dataset.

# Node: CN0  (z-score: subtract mean, divide by std)
CN0_MEAN = 45.0     # dB-Hz  (typical strong-signal level)
CN0_STD  = 8.0      # dB-Hz  (wider std to accommodate near-threshold signals)

# Node: absolute Doppler (tanh compression, scale = saturation knee)
DOP_SCALE = 3000.0  # Hz  (±3 kHz covers LEO + Earth rotation for L1)

# Node: code phase (already in [0,1] chip fraction from peak interpolation)
#        → no additional normalisation needed; stored as-is.

# Node: CN0 rate (change per 0.5-s step, z-score)
CN0_RATE_SCALE = 3.0    # dB-Hz / snapshot  (3σ clip)

# Node: Doppler rate (Hz per 0.5-s step, tanh compression)
DOP_RATE_SCALE = 200.0  # Hz/snapshot  (typical ionospheric + dynamic range)

# Node: peak-to-sidelobe ratio (tanh compression)
PEAK_RATIO_SCALE = 20.0  # linear ratio units

# Edge: single-differenced CN0 (divide by std of differences)
DIFF_CN0_STD = 8.0      # dB-Hz

# Edge: single-differenced Doppler (tanh, tighter scale than absolute)
DIFF_DOP_SCALE = 500.0  # Hz  (relative Doppler between sat pairs is small)

# Edge: single-differenced code phase (divide by expected std)
DIFF_CP_STD = 0.3       # chip fractions

# Edge: single-differenced Doppler rate
DIFF_DOP_RATE_SCALE = 100.0  # Hz/snapshot


# ── 6. Model Hyper-parameters ──────────────────────────────────────────────────
EPOCHS      = 100
BATCH_SIZE  = 32        # smaller batch for temporal graphs (higher variance → better generalisation)
LR          = 5e-4      # Adam learning rate
WEIGHT_DECAY = 1e-4     # L2 regularisation

HIDDEN_DIM  = 128       # GNN hidden channel width
GNN_HEADS   = 4         # TransformerConv attention heads
DROPOUT     = 0.3

# LSTM temporal module
LSTM_HIDDEN = 64
LSTM_LAYERS = 2

# Early stopping: halt if validation loss does not improve for this many epochs.
PATIENCE = 20
