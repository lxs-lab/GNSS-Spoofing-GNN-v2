"""
graph_builder.py  —  CSV → PyTorch Geometric temporal graph dataset (v2.1)
==========================================================================
Changes from v2.0:
  • Time-resolved labels: uses config.ATTACK_ONSET_SEC to label each window
    based on whether the attack has actually started, not file-level labels.
  • Safe attribute names: nx_{t}, ei_{t}, ea_{t} to avoid PyG collation bugs.
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

def _norm_diff_cn0(v):       return v / config.DIFF_CN0_STD
def _norm_diff_doppler(v):   return np.tanh(v / config.DIFF_DOP_SCALE)
def _norm_diff_code_phase(v):return v / config.DIFF_CP_STD
def _norm_diff_dop_rate(v):  return np.tanh(v / config.DIFF_DOP_RATE_SCALE)


# ── Single-epoch graph construction ──────────────────────────────────────────

def _build_epoch_graph(group: pd.DataFrame) -> Data | None:
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
        _norm_cn0_rate(cn0_rate), _norm_dop_rate(dop_rate), _norm_peak_ratio(peak_ratio),
    ], axis=1).astype(np.float32)

    N = len(cn0)
    src_list, dst_list, edge_attrs = [], [], []
    for i in range(N):
        for j in range(N):
            if i == j: continue
            src_list.append(i); dst_list.append(j)
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
    def raw_file_names(self):   return []
    @property
    def processed_file_names(self): return ['gnss_temporal_dataset_v2.1.pt']
    def download(self): pass

    def process(self):
        data_list = []
        T = config.TEMPORAL_WINDOW

        print("🏗  Building temporal graph dataset …")
        print(f"   Window length : {T} snapshots × {config.SNAP_STRIDE_SEC} s = "
              f"{T * config.SNAP_STRIDE_SEC:.1f} s")
        print(f"   Node features : {config.NODE_FEATURE_DIM}  "
              f"Edge features : {config.EDGE_FEATURE_DIM}")
        print(f"   Train files   : {sorted(config.TRAIN_FILES)}")

        for filename, file_label in config.DATA_FILES.items():
            csv_name = filename.replace('.bin', '_features.csv')
            csv_path = os.path.join(config.DATA_PROC_DIR, csv_name)

            if not os.path.exists(csv_path):
                print(f"   ⚠ Missing CSV: {csv_name}  "
                      f"(run batch_extract.py first)")
                continue

            is_train = filename in config.TRAIN_FILES
            split    = 'TRAIN' if is_train else 'TEST'
            scenario = filename.replace('.bin', '')
            onset    = config.ATTACK_ONSET_SEC.get(filename, float('inf'))
            print(f"   📦  {filename:<25} file_label={file_label}  "
                  f"onset={onset}s  [{split}]")

            df = pd.read_csv(csv_path).sort_values('Time')

            # ── 新增：切掉欺骗文件的前 100 秒（接收机锁定阶段，无实际攻击）──
            ATTACK_START_SEC = 100.0
            if file_label == 1:
                df = df[df['Time'] >= ATTACK_START_SEC].reset_index(drop=True)

            times = df['Time'].unique()

            # Build single-epoch graphs
            epoch_graphs: list[Data | None] = []
            for t in times:
                g = _build_epoch_graph(df[df['Time'] == t])
                epoch_graphs.append(g)

            # Slide window
            for start in tqdm(range(len(epoch_graphs) - T + 1),
                               desc=f"   {scenario}", leave=False):
                window = epoch_graphs[start: start + T]

                none_count = sum(1 for g in window if g is None)
                if none_count > T * 0.2:
                    continue

                filled: list[Data] = []
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

                # ── Time-resolved label ──────────────────────────────────
                # The window spans [t_start, t_start + (T-1)*stride].
                # Label = 1 (spoofed) only if the LAST snapshot in the
                # window is past the attack onset AND the file is an attack.
                t_start = times[start]
                t_end   = times[min(start + T - 1, len(times) - 1)]
                if file_label == 1 and t_end >= onset:
                    label = 1
                else:
                    label = 0

                sample = Data(
                    y          = torch.tensor([label], dtype=torch.long),
                    train_mask = is_train,
                    scenario   = scenario,
                    timestamp  = float(t_start),
                )

                # Safe attribute names (avoid PyG edge_index* pattern)
                for t_idx, g in enumerate(filled):
                    setattr(sample, f'nx_{t_idx}',  g.x)
                    setattr(sample, f'ei_{t_idx}',  g.edge_index)
                    setattr(sample, f'ea_{t_idx}',  g.edge_attr)
                sample.window_len = T

                data_list.append(sample)

        if not data_list:
            raise RuntimeError(
                "No graph windows were built. "
                f"Check that CSV files exist in {config.DATA_PROC_DIR}")

        train_list = [d for d in data_list if d.train_mask]
        test_list  = [d for d in data_list if not d.train_mask]
        random.shuffle(train_list)
        random.shuffle(test_list)
        data_list = train_list + test_list

        n_tc = sum(1 for d in train_list if d.y.item() == 0)
        n_ts = sum(1 for d in train_list if d.y.item() == 1)
        n_ec = sum(1 for d in test_list  if d.y.item() == 0)
        n_es = sum(1 for d in test_list  if d.y.item() == 1)

        print(f"\n💾  Saving {len(data_list)} windows  "
              f"(train={len(train_list)}, test={len(test_list)}) …")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("✅  Dataset saved.")
        print(f"   Train: clean={n_tc}  spoof={n_ts}")
        print(f"   Test : clean={n_ec}  spoof={n_es}")
        if n_tc > 0 and n_ts > 0:
            ratio = n_tc / n_ts
            if ratio > 2.0:
                print(f"   ⚠ Class imbalance ratio = {ratio:.1f}:1")