"""
graph_builder.py  —  CSV → PyTorch Geometric temporal graph dataset
====================================================================
Each graph sample represents a TEMPORAL WINDOW of T consecutive snapshots
(T = config.TEMPORAL_WINDOW, default 10) centred on a single epoch.

Graph topology (per snapshot within the window):
  • Nodes  : visible satellites, each with a 6-D feature vector
  • Edges  : fully-connected directed graph (i → j) with 4-D edge attributes
             derived from inter-satellite single differences

Cross-snapshot temporal information is **not** fused at the graph level here;
instead the model receives a list of T graphs per sample and uses an LSTM
to capture temporal evolution (see model.py).

Label assignment:
  • label = 1 if ANY snapshot in the window is marked as spoofed in DATA_FILES
  • label = 0 otherwise
  (This is conservative: even one spoofed snapshot taints the window.)

Output: InMemoryDataset stored at config.DATASET_DIR
"""

import os
import random
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

def _norm_cn0(v: np.ndarray) -> np.ndarray:
    """Z-score: (x - μ) / σ  using empirical constants from cleanStatic."""
    return (v - config.CN0_MEAN) / config.CN0_STD

def _norm_doppler(v: np.ndarray) -> np.ndarray:
    """tanh compression: maps ±∞ Hz → (−1, +1)."""
    return np.tanh(v / config.DOP_SCALE)

def _norm_code_phase(v: np.ndarray) -> np.ndarray:
    """Already in [0, 1); centre at 0 for the model."""
    return v - 0.5   # → [−0.5, +0.5)

def _norm_cn0_rate(v: np.ndarray) -> np.ndarray:
    """Clip to ±3σ then scale to (−1, +1)."""
    return np.clip(v / config.CN0_RATE_SCALE, -1.0, 1.0)

def _norm_dop_rate(v: np.ndarray) -> np.ndarray:
    return np.tanh(v / config.DOP_RATE_SCALE)

def _norm_peak_ratio(v: np.ndarray) -> np.ndarray:
    return np.tanh(v / config.PEAK_RATIO_SCALE)

# Edge normalisation
def _norm_diff_cn0(v: np.ndarray) -> np.ndarray:
    return v / config.DIFF_CN0_STD

def _norm_diff_doppler(v: np.ndarray) -> np.ndarray:
    return np.tanh(v / config.DIFF_DOP_SCALE)

def _norm_diff_code_phase(v: np.ndarray) -> np.ndarray:
    return v / config.DIFF_CP_STD

def _norm_diff_dop_rate(v: np.ndarray) -> np.ndarray:
    return np.tanh(v / config.DIFF_DOP_RATE_SCALE)


# ── Single-epoch graph construction ──────────────────────────────────────────

def _build_epoch_graph(group: pd.DataFrame) -> Data | None:
    """
    Given a DataFrame `group` for one time epoch (all visible satellites),
    construct a single-epoch PyG Data object.

    Node features  [N, 6]:
      [CN0_norm, Dop_norm, CP_norm, CN0rate_norm, Doprate_norm, PeakRatio_norm]

    Edge features  [E, 4]:
      [dCN0, dDop, dCP, dDopRate]  for all directed pairs (i → j), i ≠ j

    Returns None if fewer than 4 satellites are visible (graph too sparse).
    """
    if len(group) < 4:
        return None

    # ── Node features ───────────────────────────────────────────────────────
    cn0        = group['CN0_dBHz'].values.astype(np.float32)
    doppler    = group['Doppler_Hz'].values.astype(np.float32)
    code_phase = group['CodePhase_chips'].values.astype(np.float32)
    cn0_rate   = group['CN0_rate'].values.astype(np.float32)
    dop_rate   = group['Dop_rate'].values.astype(np.float32)
    peak_ratio = group['PeakRatio'].values.astype(np.float32)

    x = np.stack([
        _norm_cn0(cn0),
        _norm_doppler(doppler),
        _norm_code_phase(code_phase),
        _norm_cn0_rate(cn0_rate),
        _norm_dop_rate(dop_rate),
        _norm_peak_ratio(peak_ratio),
    ], axis=1).astype(np.float32)   # (N, 6)

    N = len(cn0)

    # ── Edge features (fully-connected directed) ────────────────────────────
    src_list, dst_list = [], []
    edge_attrs = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            src_list.append(i)
            dst_list.append(j)

            # Directed single differences (i minus j)
            d_cn0      = _norm_diff_cn0(cn0[i]        - cn0[j])
            d_dop      = _norm_diff_doppler(doppler[i] - doppler[j])
            d_cp       = _norm_diff_code_phase(code_phase[i] - code_phase[j])
            d_dop_rate = _norm_diff_dop_rate(dop_rate[i]     - dop_rate[j])
            edge_attrs.append([d_cn0, d_dop, d_cp, d_dop_rate])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attrs,           dtype=torch.float)

    return Data(
        x          = torch.tensor(x,          dtype=torch.float),
        edge_index = edge_index,
        edge_attr  = edge_attr,
    )


# ── Dataset class ─────────────────────────────────────────────────────────────

class GNSSGraphDataset(InMemoryDataset):
    """
    InMemoryDataset where each sample is a list of T single-epoch graphs
    (a temporal window) packed into a single Data object with extra
    attributes:
      • data.graph_seq  : list of T Data objects (the temporal window)
      • data.y          : graph-level binary label (0/1)
      • data.train_mask : True if this window belongs to the training split
      • data.scenario   : source filename stem (e.g. 'ds4')

    The reason for storing the sequence as a Python list (not a batched graph)
    is that PyG's InMemory collation handles heterogeneous sizes gracefully
    through the `follow_batch` mechanism; the DataLoader is configured in
    the training script.
    """

    def __init__(self, root=None, transform=None, pre_transform=None):
        root = root or config.DATASET_DIR
        super().__init__(root, transform, pre_transform)
        # Load cached dataset or build from scratch
        cached = self.processed_paths[0]
        if os.path.exists(cached):
            try:
                self.data, self.slices = torch.load(cached, weights_only=False)
                return
            except Exception:
                print("⚠ Stale cache — rebuilding …")
        self.process()

    @property
    def raw_file_names(self):
        return []   # raw data lives outside the PyG directory structure

    @property
    def processed_file_names(self):
        return ['gnss_temporal_dataset_v2.pt']

    def download(self):
        pass   # nothing to download

    # ── Processing ──────────────────────────────────────────────────────────

    def process(self):
        data_list = []
        T = config.TEMPORAL_WINDOW    # window length in snapshots

        print("🏗  Building temporal graph dataset …")
        print(f"   Window length : {T} snapshots × {config.SNAP_STRIDE_SEC} s = "
              f"{T * config.SNAP_STRIDE_SEC:.1f} s")
        print(f"   Node features : {config.NODE_FEATURE_DIM}  "
              f"Edge features : {config.EDGE_FEATURE_DIM}")
        print(f"   Train files   : {sorted(config.TRAIN_FILES)}")

        for filename, label in config.DATA_FILES.items():
            csv_name = filename.replace('.bin', '_features.csv')
            csv_path = os.path.join(config.DATA_PROC_DIR, csv_name)

            if not os.path.exists(csv_path):
                print(f"   ⚠ Missing CSV: {csv_name}  "
                      f"(run 1_batch_extract.py first)")
                continue

            is_train = filename in config.TRAIN_FILES
            split    = 'TRAIN' if is_train else 'TEST'
            scenario = filename.replace('.bin', '')
            print(f"   📦  {filename:<25} label={label}  [{split}]")

            df = pd.read_csv(csv_path)

            # Sort by time, then group by epoch
            df = df.sort_values('Time')
            times = df['Time'].unique()

            # Build single-epoch graphs for every time step
            epoch_graphs: list[Data | None] = []
            for t in times:
                g = _build_epoch_graph(df[df['Time'] == t])
                epoch_graphs.append(g)  # None for sparse epochs

            # Slide a window of length T over the time axis
            for start in tqdm(range(len(epoch_graphs) - T + 1),
                               desc=f"   {scenario}", leave=False):
                window = epoch_graphs[start: start + T]

                # Skip windows that have too many missing epochs (> 20%)
                none_count = sum(1 for g in window if g is None)
                if none_count > T * 0.2:
                    continue

                # Fill None gaps with the previous valid graph (forward-fill)
                filled: list[Data] = []
                last_valid = None
                for g in window:
                    if g is not None:
                        last_valid = g
                        filled.append(g)
                    elif last_valid is not None:
                        filled.append(last_valid)
                    else:
                        break   # no valid graph yet; skip this window start
                if len(filled) < T:
                    continue

                # Pack the window into a single Data object.
                # We store the sequence as a list attribute; the model unpacks it.
                sample = Data(
                    y          = torch.tensor([label], dtype=torch.long),
                    train_mask = is_train,
                    scenario   = scenario,
                    timestamp  = float(times[start]),
                )
                # Store T serialised graphs as a flat tensor for efficient I/O.
                # Shape: (T, max_nodes_in_window, NODE_FEATURE_DIM)
                # — but node counts differ, so we store each graph's x, edge_index,
                #   edge_attr separately using numbered attributes.
                for t_idx, g in enumerate(filled):
                    setattr(sample, f'x_{t_idx}',          g.x)
                    setattr(sample, f'edge_index_{t_idx}', g.edge_index)
                    setattr(sample, f'edge_attr_{t_idx}',  g.edge_attr)
                sample.window_len = T

                data_list.append(sample)

        if not data_list:
            raise RuntimeError(
                "No graph windows were built. "
                f"Check that CSV files exist in {config.DATA_PROC_DIR}"
            )

        # ── Independent shuffle within each split ──────────────────────────
        train_list = [d for d in data_list if d.train_mask]
        test_list  = [d for d in data_list if not d.train_mask]
        random.shuffle(train_list)
        random.shuffle(test_list)
        data_list = train_list + test_list

        print(f"\n💾  Saving {len(data_list)} windows  "
              f"(train={len(train_list)}, test={len(test_list)}) …")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("✅  Dataset saved.")

        # ── Statistics ──────────────────────────────────────────────────────
        tr_labels = [d.y.item() for d in train_list]
        te_labels = [d.y.item() for d in test_list]
        print(f"   Train: clean={tr_labels.count(0)}  spoof={tr_labels.count(1)}")
        print(f"   Test : clean={te_labels.count(0)}  spoof={te_labels.count(1)}")
        if tr_labels.count(1) > 0:
            ratio = tr_labels.count(0) / tr_labels.count(1)
            if ratio > 2.5:
                print(f"   ⚠ Class imbalance ratio = {ratio:.1f}:1 "
                      "— class weights will be applied in training.")
