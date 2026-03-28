# GNSS Spoofing Detection — ST-GNN v2

## Project Structure

```
gnss_gnn_v2/
├── src/
│   ├── __init__.py
│   ├── config.py          # All hyper-parameters and paths (edit DATA_RAW_DIR here)
│   ├── extractor.py       # SDR → 6-feature CSV (C/N0, Doppler, CodePhase, rates, PeakRatio)
│   ├── graph_builder.py   # CSV → temporal PyG graph windows (T=10 snapshots)
│   └── model.py           # Spatial GNN encoder + bidirectional LSTM classifier
│
├── batch_extract.py       # Step 1: extract features from all .bin files
├── train_eval.py          # Step 2 & 3: build dataset, train, evaluate
│
├── data/
│   ├── processed/         # Generated CSVs (one per .bin file)
│   └── dataset/           # Cached PyG dataset (.pt file)
│
├── models/                # Saved model checkpoints (.pth)
└── logs/                  # Per-run logs and plots
```

## Quick Start

```bash
# 1. Edit DATA_RAW_DIR in src/config.py to point to your TEXBAT .bin files

# 2. Extract features (runs once, ~30–60 min per file on CPU)
python batch_extract.py

# 3. Train and evaluate
python train_eval.py
```

## What Changed from v1

| Aspect | v1 | v2 |
|---|---|---|
| Node features | 2 (CN0, Doppler) | **6** (+ CodePhase, CN0_rate, Dop_rate, PeakRatio) |
| Edge features | 2 (ΔCN0, ΔDop) | **4** (+ ΔCodePhase, ΔDop_rate) |
| Temporal model | None (single-epoch GNN) | **Bidirectional LSTM** over T=10 snapshots |
| CodePhase | Not extracted | **Parabolic interpolation** for sub-chip resolution |
| Peak ratio | Not extracted | **Peak/sidelobe ratio** — sensitive to multipath/spoofing |
| Training set | cleanStatic + ds4 | cleanStatic + **cleanDynamic** + ds4 |

## Key Design Decisions

### 6-Node Features
- **CN0**: baseline power level; ds2 shows +10 dB anomaly
- **Doppler**: carrier frequency; all attacks must fake this
- **CodePhase**: fractional chip offset; traction attacks shift this slowly
- **CN0_rate**: temporal CN0 gradient; switching attacks (ds1/ds5) show sharp jumps
- **Dop_rate**: Doppler acceleration; fake dynamics (ds5/ds6) violate physical motion
- **PeakRatio**: correlation peak sharpness; signal quality monitor (SQM)

### Temporal Window (T=10, 5 seconds)
A single 0.5-second snapshot misses slow traction attacks (ds3/ds4/ds7).
The 5-second window allows the LSTM to track code-phase drift trajectories.

### Bidirectional LSTM
Traction attacks are directional trends. Bidirectional LSTM detects both
the onset (forward direction) and the confirmation (backward direction).

## Evaluation Protocol

Training set: `cleanStatic.bin`, `cleanDynamic.bin`, `ds4.bin`

Test set (zero-shot): `ds1`, `ds2`, `ds3`, `ds5`, `ds6`, `ds7`, `ds8`,
`cleanStatic80`

Target: **DR > 90%** and **FAR < 5%** on all test scenarios.
