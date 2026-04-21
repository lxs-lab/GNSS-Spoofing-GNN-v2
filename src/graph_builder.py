"""
graph_builder.py  —  CSV → PyTorch Geometric temporal graph dataset (v3)
========================================================================
PRINCIPLE CHANGES FROM v2.1:
  [G1] THREE-WAY SPLIT: train / val / test
       Previously: train_mask=True → training, False → test.
       Now: each sample carries a 'split' attribute with value
       'train', 'val', or 'test'.
       Val comes from the LAST VAL_RATIO fraction of each TRAIN_FILE's
       timeline. Test comes from TEST_FILES. Split is by time, not random,
       to prevent temporal leakage.
  [G2] VAL SET used for early stopping (not the full test set).
       This stops the model from indirectly overfitting to test scenarios.
  [G3] Label logic unchanged from v2.1: window label=1 only if the window
       ends after ATTACK_ONSET_SEC for that file.
  [G4] Attribute naming unchanged from v2.1: nx_{t}, ei_{t}, ea_{t}.
"""

import os, random
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

try:
    from src import config
except ImportError:
    from .. import config


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _norm_cn0(v):        return (v - config.CN0_MEAN) / config.CN0_STD
def _norm_doppler(v):    return np.tanh(v / config.DOP_SCALE)
def _norm_code_phase(v): return v - 0.5
def _norm_cn0_rate(v):   return np.clip(v / config.CN0_RATE_SCALE, -1.0, 1.0)
def _norm_dop_rate(v):   return np.tanh(v / config.DOP_RATE_SCALE)
def _norm_peak_ratio(v): return np.tanh(v / config.PEAK_RATIO_SCALE)

def _norm_diff_cn0(v):        return v / config.DIFF_CN0_STD
def _norm_diff_doppler(v):    return np.tanh(v / config.DIFF_DOP_SCALE)
def _norm_diff_code_phase(v): return v / config.DIFF_CP_STD
def _norm_diff_dop_rate(v):   return np.tanh(v / config.DIFF_DOP_RATE_SCALE)


# ── Single-epoch graph construction ──────────────────────────────────────────

def _build_epoch_graph(group: pd.DataFrame):
    if len(group) < 4:
        return None

    cn0        = group['CN0_dBHz'].values.astype(np.float32)
    doppler    = group['Doppler_Hz'].values.astype(np.float32)
    code_phase = group['CodePhase_chips'].values.astype(np.float32)
    cn0_rate   = group['CN0_rate'].values.astype(np.float32)
    dop_rate   = group['Dop_rate'].values.astype(np.float32)
    peak_ratio = group['PeakRatio'].values.astype(np.float32)

    x = np.stack([
        _norm_cn0(cn0), _norm_doppler(doppler), _norm_code_phase(code_phase),
        _norm_cn0_rate(cn0_rate), _norm_dop_rate(dop_rate),
        _norm_peak_ratio(peak_ratio),
    ], axis=1).astype(np.float32)

    N = len(cn0)
    src_list, dst_list, edge_attrs = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            src_list.append(i)
            dst_list.append(j)
            edge_attrs.append([
                _norm_diff_cn0(cn0[i] - cn0[j]),
                _norm_diff_doppler(doppler[i] - doppler[j]),
                _norm_diff_code_phase(code_phase[i] - code_phase[j]),
                _norm_diff_dop_rate(dop_rate[i] - dop_rate[j]),
            ])

    return Data(
        x          = torch.tensor(x, dtype=torch.float),
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long),
        edge_attr  = torch.tensor(edge_attrs, dtype=torch.float),
    )


# ── Dataset class ─────────────────────────────────────────────────────────────

class GNSSGraphDataset(InMemoryDataset):

    def __init__(self, root=None, transform=None, pre_transform=None):
        root = root or config.DATASET_DIR
        super().__init__(root, transform, pre_transform)
        cached = self.processed_paths[0]
        if os.path.exists(cached):
            try:
                self.data, self.slices = torch.load(cached, weights_only=False)
                return
            except Exception:
                print("⚠ Stale cache — rebuilding …")
        self.process()

    @property
    def raw_file_names(self):       return []
    @property
    def processed_file_names(self): return ['gnss_dataset_v3.pt']
    def download(self):             pass

    def process(self):
        data_list = []
        T = config.TEMPORAL_WINDOW

        print("🏗  Building temporal graph dataset …")
        print(f"   Window  : {T} snapshots × {config.SNAP_STRIDE_SEC} s "
              f"= {T * config.SNAP_STRIDE_SEC:.1f} s")
        print(f"   Features: node={config.NODE_FEATURE_DIM}  "
              f"edge={config.EDGE_FEATURE_DIM}")
        print(f"   Train files : {sorted(config.TRAIN_FILES)}")
        print(f"   Test  files : {sorted(config.TEST_FILES)}")
        print(f"   Val ratio   : {config.VAL_RATIO:.0%} of each train file")

        for filename, file_label in config.DATA_FILES.items():
            csv_name = filename.replace('.bin', '_features.csv')
            csv_path = os.path.join(config.DATA_PROC_DIR, csv_name)

            if not os.path.exists(csv_path):
                print(f"   ⚠ Missing CSV: {csv_name}  (run batch_extract.py)")
                continue

            # Determine split role for this file
            if filename in config.TRAIN_FILES:
                role = 'trainval'   # will be split further by time
            elif filename in config.TEST_FILES:
                role = 'test'
            else:
                # File in DATA_FILES but neither TRAIN nor TEST → skip silently
                continue

            onset   = config.ATTACK_ONSET_SEC.get(filename, float('inf'))
            scenario = filename.replace('.bin', '')
            print(f"   📦  {filename:<25} label={file_label}  "
                  f"onset={onset}s  [{role.upper()}]")

            df = pd.read_csv(csv_path).sort_values('Time')
            times = df['Time'].unique()

            # [G1] Time-based train/val split for TRAIN_FILES
            # The LAST VAL_RATIO fraction of the recording becomes val.
            if role == 'trainval':
                cutoff_idx = int(len(times) * (1.0 - config.VAL_RATIO))
                train_times = set(times[:cutoff_idx])
                val_times   = set(times[cutoff_idx:])
            else:
                train_times = set()
                val_times   = set()

            # Build single-epoch graphs
            epoch_graphs = {}
            for t in times:
                g = _build_epoch_graph(df[df['Time'] == t])
                epoch_graphs[t] = g

            # Slide window
            for start in tqdm(range(len(times) - T + 1),
                               desc=f"   {scenario}", leave=False):
                window_times = times[start: start + T]
                window = [epoch_graphs[t] for t in window_times]

                none_count = sum(1 for g in window if g is None)
                if none_count > T * 0.2:
                    continue

                filled = []
                last_valid = None
                for g in window:
                    if g is not None:
                        last_valid = g
                        filled.append(g)
                    elif last_valid is not None:
                        filled.append(last_valid)
                    else:
                        break
                if len(filled) < T:
                    continue

                # Label: spoofed only if window ends past onset
                t_end = window_times[-1]
                label = 1 if (file_label == 1 and t_end >= onset) else 0

                # [G1] Assign split
                t_start = window_times[0]
                if role == 'test':
                    split = 'test'
                elif t_start in val_times:
                    split = 'val'
                else:
                    split = 'train'

                sample = Data(
                    y         = torch.tensor([label], dtype=torch.long),
                    split     = split,
                    scenario  = scenario,
                    timestamp = float(t_start),
                )
                for t_idx, g in enumerate(filled):
                    setattr(sample, f'nx_{t_idx}', g.x)
                    setattr(sample, f'ei_{t_idx}', g.edge_index)
                    setattr(sample, f'ea_{t_idx}', g.edge_attr)
                sample.window_len = T
                data_list.append(sample)

        if not data_list:
            raise RuntimeError(
                f"No windows built. Check CSV files in {config.DATA_PROC_DIR}")

        train_list = [d for d in data_list if d.split == 'train']
        val_list   = [d for d in data_list if d.split == 'val']
        test_list  = [d for d in data_list if d.split == 'test']

        random.shuffle(train_list)
        random.shuffle(val_list)
        data_list = train_list + val_list + test_list

        def _counts(lst):
            c = sum(1 for d in lst if d.y.item() == 0)
            s = sum(1 for d in lst if d.y.item() == 1)
            return c, s

        tc, ts = _counts(train_list)
        vc, vs = _counts(val_list)
        ec, es = _counts(test_list)

        print(f"\n💾  Saving {len(data_list)} windows …")
        print(f"   Train : clean={tc}  spoof={ts}  total={len(train_list)}")
        print(f"   Val   : clean={vc}  spoof={vs}  total={len(val_list)}")
        print(f"   Test  : clean={ec}  spoof={es}  total={len(test_list)}")

        if ts > 0:
            r = tc / ts
            if r > 2.0:
                print(f"   ⚠ Train imbalance {r:.1f}:1 — Focal Loss recommended")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("✅  Dataset saved.")