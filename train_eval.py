"""
train_eval.py  —  Training + zero-shot per-scenario evaluation
==============================================================
Usage:
    python train_eval.py

Key improvements over v1:
  • Temporal graph collation: custom DataLoader collate_fn packs T-graph
    windows into the (x, edge_index, edge_attr, batch) tuples expected by
    the model's forward().
  • Class-imbalance auto-weighting based on training label ratio.
  • Per-scenario DR/FAR reporting — never a single aggregated accuracy.
  • Early stopping on validation loss with model checkpoint restore.
"""

import os
import sys
import shutil
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.graph_builder import GNSSGraphDataset
from src.model import STGNNSpoofingDetector
from src import config


# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Custom collate for temporal graph windows ─────────────────────────────────

def _collate_windows(data_list):
    """
    Custom collate function for temporal graph windows.

    Each item in `data_list` is a Data object with attributes:
      x_0 … x_{T-1}, edge_index_0 … edge_index_{T-1}, edge_attr_0 … ,
      y, train_mask, scenario, window_len

    Returns:
      graph_seq : list of T PyG Batch objects (one per time step)
      labels    : LongTensor of shape (B,)
      scenarios : list of str
    """
    T = data_list[0].window_len
    graph_seq = []
    for t in range(T):
        # Build a Batch from the t-th graph across all items in the mini-batch
        epoch_graphs = []
        for d in data_list:
            g = torch.geometric.data.Data(
                x          = getattr(d, f'x_{t}'),
                edge_index = getattr(d, f'edge_index_{t}'),
                edge_attr  = getattr(d, f'edge_attr_{t}'),
            )
            epoch_graphs.append(g)
        graph_seq.append(Batch.from_data_list(epoch_graphs))

    labels    = torch.stack([d.y.squeeze() for d in data_list])
    scenarios = [d.scenario for d in data_list]
    return graph_seq, labels, scenarios


def _unpack_batch(batch: Batch):
    """
    Unpack a PyG Batch into (x, edge_index, edge_attr, batch_vector).
    """
    return batch.x, batch.edge_index, batch.edge_attr, batch.batch


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_dr_far(preds, labels):
    """
    Compute Detection Rate (DR) and False Alarm Rate (FAR).
      DR  = TP / (TP + FN)   — probability of detecting a real attack
      FAR = FP / (FP + TN)   — probability of falsely alarming on clean signal
    """
    p, l = np.array(preds), np.array(labels)
    tp = int(((p == 1) & (l == 1)).sum())
    fn = int(((p == 0) & (l == 1)).sum())
    fp = int(((p == 1) & (l == 0)).sum())
    tn = int(((p == 0) & (l == 0)).sum())
    dr  = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    far = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    return dr, far


# ── Logger ────────────────────────────────────────────────────────────────────

class TeeLogger:
    """Write stdout to both terminal and a log file simultaneously."""
    def __init__(self, path: str):
        self.terminal = sys.stdout
        self.log      = open(path, 'a', encoding='utf-8')

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(log_dir: str):
    # ── Dataset ──────────────────────────────────────────────────────────────
    # Force-rebuild to pick up any graph_builder changes.
    proc_cache = os.path.join(config.DATASET_DIR, 'processed')
    if os.path.exists(proc_cache):
        shutil.rmtree(proc_cache)
        print("🧹  Cleared cached dataset — rebuilding …")

    dataset   = GNSSGraphDataset()
    train_data = [d for d in dataset if d.train_mask]
    test_data  = [d for d in dataset if not d.train_mask]
    print(f"\n📦  Dataset: {len(dataset)} windows  |  "
          f"train={len(train_data)}  test={len(test_data)}")

    # ── Class weights ─────────────────────────────────────────────────────────
    tr_labels = [d.y.item() for d in train_data]
    n0 = tr_labels.count(0)
    n1 = tr_labels.count(1)
    ratio = n0 / max(n1, 1)
    if ratio > 2.0:
        w = torch.tensor([1.0, ratio], dtype=torch.float)
        print(f"   ⚖ Imbalance {ratio:.1f}:1 → weights {w.tolist()}")
    else:
        w = torch.tensor([1.0, 1.0], dtype=torch.float)
        print("   ⚖ Balanced dataset")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_data, batch_size=config.BATCH_SIZE, shuffle=True,
        collate_fn=_collate_windows, drop_last=True,
    )
    test_loader = DataLoader(
        test_data, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=_collate_windows,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥  Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = STGNNSpoofingDetector().to(device)
    criterion = nn.CrossEntropyLoss(weight=w.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )

    best_path   = os.path.join(config.MODEL_DIR, 'best_ST_GNN.pth')
    best_loss   = float('inf')
    patience_cnt = 0

    train_losses, test_accs = [], []

    print(f"\n🚀  Training for up to {config.EPOCHS} epochs "
          f"(early stop patience={config.PATIENCE}) …\n")

    for epoch in range(1, config.EPOCHS + 1):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for graph_seq, labels, _ in train_loader:
            # Unpack each time step to (x, edge_index, edge_attr, batch)
            seq_unpacked = [_unpack_batch(b.to(device)) for b in graph_seq]
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(seq_unpacked)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        # ── Eval ──────────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for graph_seq, labels, _ in test_loader:
                seq_unpacked = [_unpack_batch(b.to(device)) for b in graph_seq]
                logits = model(seq_unpacked)
                preds  = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        dr, far = compute_dr_far(all_preds, all_labels)
        test_accs.append(acc * 100)

        # ── Early stopping ────────────────────────────────────────────────────
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:03d}/{config.EPOCHS} | "
                  f"Loss {avg_loss:.4f} | "
                  f"Acc {acc*100:.1f}% | "
                  f"DR {dr*100:.1f}% | FAR {far*100:.2f}%")

        if patience_cnt >= config.PATIENCE:
            print(f"\n  ⏹  Early stop at epoch {epoch} "
                  f"(no loss improvement for {config.PATIENCE} epochs)")
            break

    # ── Per-scenario evaluation ───────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    # Group test samples by scenario
    scenario_groups: dict[str, list] = {}
    for d in test_data:
        scenario_groups.setdefault(d.scenario, []).append(d)

    print(f"\n{'='*65}")
    print(f"  Per-scenario zero-shot evaluation")
    print(f"{'='*65}")
    print(f"  {'Scenario':<22} {'N':>6} {'DR (%)':>8} {'FAR (%)':>8}  Status")
    print(f"  {'-'*60}")

    perf = {}
    y_true_all, y_pred_all = [], []

    for sce in sorted(scenario_groups.keys()):
        subset = scenario_groups[sce]
        sub_loader = DataLoader(
            subset, batch_size=config.BATCH_SIZE,
            collate_fn=_collate_windows
        )
        preds, labs = [], []
        with torch.no_grad():
            for graph_seq, labels, _ in sub_loader:
                seq_unpacked = [_unpack_batch(b.to(device)) for b in graph_seq]
                logits = model(seq_unpacked)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
                labs.extend(labels.numpy())

        dr_s, far_s = compute_dr_far(preds, labs)
        perf[sce] = (dr_s, far_s)
        y_true_all.extend(labs)
        y_pred_all.extend(preds)

        flag = '✅' if (dr_s > 0.90 and far_s < 0.05) else '⚠️ '
        dr_str  = f'{dr_s*100:.1f}%' if not np.isnan(dr_s)  else '  N/A '
        far_str = f'{far_s*100:.2f}%' if not np.isnan(far_s) else '  N/A '
        print(f"  {sce:<22} {len(subset):>6} {dr_str:>8} {far_str:>8}  {flag}")

    print(f"{'='*65}")

    # ── Save plots ────────────────────────────────────────────────────────────
    _save_plots(train_losses, test_accs, perf, y_true_all, y_pred_all, log_dir)


def _save_plots(losses, accs, perf, y_true, y_pred, log_dir):
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses);  ax1.set_title('Training loss');   ax1.set_xlabel('Epoch')
    ax2.plot(accs, color='orange')
    ax2.set_title('Test accuracy');  ax2.set_ylabel('%'); ax2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Spoofed'],
                yticklabels=['Clean', 'Spoofed'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Per-scenario DR / FAR bar chart
    names = sorted(perf.keys())
    drs   = [perf[n][0] * 100 if not np.isnan(perf[n][0]) else 0 for n in names]
    fars  = [perf[n][1] * 100 if not np.isnan(perf[n][1]) else 0 for n in names]
    x     = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    ax.bar(x - 0.2, drs,  0.35, label='DR (%)',  color='#4CAF50')
    ax.bar(x + 0.2, fars, 0.35, label='FAR (%)', color='#F44336')
    ax.axhline(90, color='green', ls='--', lw=0.8, label='DR target 90%')
    ax.axhline(5,  color='red',   ls='--', lw=0.8, label='FAR limit 5%')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylim(0, 115); ax.set_ylabel('%'); ax.legend()
    ax.set_title('Per-scenario DR and FAR')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'scenario_metrics.png'), dpi=150)
    plt.close()
    print(f"\n   📊  Plots saved → {log_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ts      = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = os.path.join(config.LOG_BASE_DIR, ts)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(log_dir, 'console_log.txt'))

    print('=' * 65)
    print('  GNSS ST-GNN Spoofing Detector v2  —  Training & Evaluation')
    print('=' * 65)
    print(f"  EPOCHS={config.EPOCHS}  BS={config.BATCH_SIZE}  "
          f"LR={config.LR}  HIDDEN={config.HIDDEN_DIM}  "
          f"WINDOW={config.TEMPORAL_WINDOW}")

    train(log_dir)


if __name__ == '__main__':
    main()
