"""
train_eval.py  —  Training + Evaluation + Baseline + Ablation (v2.1)
=====================================================================
Changes from v2.0:
  • All prior bugs fixed (Data import, attribute names, DataLoader)
  • Baseline MLP comparison (config.RUN_BASELINE_MLP)
  • Ablation: GNN-only mode (config.ABLATION_NO_TEMPORAL)
  • Attention weight visualization on last test batch
  • Time-resolved labels (via graph_builder v2.1)
"""

import os, sys, shutil, random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader        # ★ Use PyTorch native DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.graph_builder import GNSSGraphDataset
from src.model import STGNNSpoofingDetector, BaselineMLP
from src import config

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)


# ── Collate & Unpack ──────────────────────────────────────────────────────────

def _collate_windows(data_list):
    T = data_list[0].window_len
    graph_seq = []
    for t in range(T):
        epoch_graphs = []
        for d in data_list:
            g = Data(
                x          = getattr(d, f'nx_{t}'),
                edge_index = getattr(d, f'ei_{t}'),
                edge_attr  = getattr(d, f'ea_{t}'),
            )
            epoch_graphs.append(g)
        graph_seq.append(Batch.from_data_list(epoch_graphs))
    labels    = torch.stack([d.y.squeeze() for d in data_list])
    scenarios = [d.scenario for d in data_list]
    return graph_seq, labels, scenarios


def _unpack_batch(batch):
    return batch.x, batch.edge_index, batch.edge_attr, batch.batch


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_dr_far(preds, labels):
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
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'a', encoding='utf-8')
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()


# ── Training a single model ───────────────────────────────────────────────────

def _train_and_eval(model, train_loader, test_loader, test_data,
                    device, log_dir, model_name='ST-GNN'):
    """Train `model`, evaluate per-scenario, return (perf_dict, losses, accs)."""

    # Class weights
    tr_labels = []
    for d in train_loader.dataset:
        tr_labels.append(d.y.item())
    n0, n1 = tr_labels.count(0), tr_labels.count(1)
    ratio = n0 / max(n1, 1)
    w = torch.tensor([1.0, ratio], dtype=torch.float) if ratio > 2 else torch.ones(2)
    print(f"   [{model_name}] ⚖ Imbalance {ratio:.1f}:1 → weights {w.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=w.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    best_path = os.path.join(config.MODEL_DIR, f'best_{model_name}.pth')
    best_loss = float('inf')
    patience_cnt = 0
    train_losses, test_accs = [], []

    print(f"\n🚀  [{model_name}] Training for up to {config.EPOCHS} epochs …\n")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for graph_seq, labels, _ in train_loader:
            seq_unpacked = [_unpack_batch(b.to(device)) for b in graph_seq]
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(seq_unpacked)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for graph_seq, labels, _ in test_loader:
                seq_unpacked = [_unpack_batch(b.to(device)) for b in graph_seq]
                preds = model(seq_unpacked).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds); all_labels.extend(labels.numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        dr, far = compute_dr_far(all_preds, all_labels)
        test_accs.append(acc * 100)

        if avg_loss < best_loss:
            best_loss = avg_loss; patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{model_name}] Epoch {epoch:03d}/{config.EPOCHS} | "
                  f"Loss {avg_loss:.4f} | Acc {acc*100:.1f}% | "
                  f"DR {dr*100:.1f}% | FAR {far*100:.2f}%")

        if patience_cnt >= config.PATIENCE:
            print(f"\n  [{model_name}] ⏹  Early stop at epoch {epoch}")
            break

    # ── Per-scenario evaluation ───────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    scenario_groups = {}
    for d in test_data:
        scenario_groups.setdefault(d.scenario, []).append(d)

    print(f"\n{'='*65}")
    print(f"  [{model_name}] Per-scenario zero-shot evaluation")
    print(f"{'='*65}")
    print(f"  {'Scenario':<22} {'N':>6} {'DR (%)':>8} {'FAR (%)':>8}  Status")
    print(f"  {'-'*60}")

    perf = {}
    y_true_all, y_pred_all = [], []

    for sce in sorted(scenario_groups.keys()):
        subset = scenario_groups[sce]
        sub_loader = DataLoader(subset, batch_size=config.BATCH_SIZE,
                                 collate_fn=_collate_windows)
        preds, labs = [], []
        with torch.no_grad():
            for graph_seq, labels, _ in sub_loader:
                seq_unpacked = [_unpack_batch(b.to(device)) for b in graph_seq]
                preds.extend(model(seq_unpacked).argmax(dim=1).cpu().numpy())
                labs.extend(labels.numpy())

        dr_s, far_s = compute_dr_far(preds, labs)
        perf[sce] = (dr_s, far_s)
        y_true_all.extend(labs); y_pred_all.extend(preds)

        flag = '✅' if (dr_s > 0.90 and far_s < 0.05) else '⚠️ '
        dr_str  = f'{dr_s*100:.1f}%' if not np.isnan(dr_s) else '  N/A '
        far_str = f'{far_s*100:.2f}%' if not np.isnan(far_s) else '  N/A '
        print(f"  {sce:<22} {len(subset):>6} {dr_str:>8} {far_str:>8}  {flag}")

    print(f"{'='*65}")
    return perf, train_losses, test_accs, y_true_all, y_pred_all


# ── Attention Visualization ───────────────────────────────────────────────────

def _save_attention_plot(model, test_data, device, log_dir):
    """Visualize attention weights from the last snapshot of a spoofed sample."""
    # Find a spoofed sample
    spoof_samples = [d for d in test_data if d.y.item() == 1]
    if not spoof_samples:
        print("   ⚠ No spoofed samples for attention visualization")
        return

    sample = spoof_samples[0]
    model.eval()

    # Build a single-sample batch
    T = sample.window_len
    graph_seq = []
    for t in range(T):
        g = Data(
            x=getattr(sample, f'nx_{t}'),
            edge_index=getattr(sample, f'ei_{t}'),
            edge_attr=getattr(sample, f'ea_{t}'),
        )
        graph_seq.append(Batch.from_data_list([g]))

    seq_unpacked = [_unpack_batch(b.to(device)) for b in graph_seq]

    with torch.no_grad():
        logits, attn_info = model.forward_with_attention(seq_unpacked)

    if attn_info is None:
        return

    alpha = attn_info['alpha'].cpu().numpy().flatten()
    ei    = attn_info['edge_index'].cpu().numpy()
    n_nodes = getattr(sample, f'nx_{T-1}').shape[0]

    # Build attention matrix
    attn_matrix = np.zeros((n_nodes, n_nodes))
    for idx in range(ei.shape[1]):
        src, dst = ei[0, idx], ei[1, idx]
        if src < n_nodes and dst < n_nodes:
            attn_matrix[src, dst] = alpha[idx] if idx < len(alpha) else 0

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(attn_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                ax=ax, square=True,
                xticklabels=[f'Sat{i}' for i in range(n_nodes)],
                yticklabels=[f'Sat{i}' for i in range(n_nodes)])
    ax.set_title(f'GNN Attention Weights (scenario: {sample.scenario}, '
                 f't={sample.timestamp:.1f}s)')
    ax.set_xlabel('Target satellite')
    ax.set_ylabel('Source satellite')
    plt.tight_layout()
    path = os.path.join(log_dir, 'attention_heatmap.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   🔍  Attention heatmap saved → {path}")


# ── Plot Utilities ────────────────────────────────────────────────────────────

def _save_plots(losses, accs, perf, y_true, y_pred, log_dir, prefix=''):
    tag = f'{prefix}_' if prefix else ''

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(losses);  ax1.set_title(f'{prefix} Training loss'); ax1.set_xlabel('Epoch')
    ax2.plot(accs, color='orange')
    ax2.set_title(f'{prefix} Test accuracy'); ax2.set_ylabel('%'); ax2.set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{tag}training_curves.png'), dpi=150)
    plt.close()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Clean', 'Spoofed'],
                yticklabels=['Clean', 'Spoofed'])
    plt.title(f'{prefix} Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{tag}confusion_matrix.png'), dpi=150)
    plt.close()

    names = sorted(perf.keys())
    drs  = [perf[n][0]*100 if not np.isnan(perf[n][0]) else 0 for n in names]
    fars = [perf[n][1]*100 if not np.isnan(perf[n][1]) else 0 for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.2), 5))
    ax.bar(x - 0.2, drs, 0.35, label='DR (%)', color='#4CAF50')
    ax.bar(x + 0.2, fars, 0.35, label='FAR (%)', color='#F44336')
    ax.axhline(90, color='green', ls='--', lw=0.8, label='DR target 90%')
    ax.axhline(5, color='red', ls='--', lw=0.8, label='FAR limit 5%')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylim(0, 115); ax.set_ylabel('%'); ax.legend()
    ax.set_title(f'{prefix} Per-scenario DR and FAR')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{tag}scenario_metrics.png'), dpi=150)
    plt.close()
    print(f"   📊  [{prefix}] Plots saved → {log_dir}/")


def _save_comparison_plot(perf_stgnn, perf_baseline, log_dir):
    """Side-by-side DR comparison: ST-GNN vs Baseline MLP."""
    scenarios = sorted(set(perf_stgnn.keys()) & set(perf_baseline.keys()))
    if not scenarios:
        return

    dr_gnn  = [perf_stgnn[s][0]*100 if not np.isnan(perf_stgnn[s][0]) else 0
               for s in scenarios]
    dr_base = [perf_baseline[s][0]*100 if not np.isnan(perf_baseline[s][0]) else 0
               for s in scenarios]

    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(max(8, len(scenarios)*1.2), 5))
    ax.bar(x - 0.2, dr_gnn,  0.35, label='ST-GNN (ours)', color='#2196F3')
    ax.bar(x + 0.2, dr_base, 0.35, label='Baseline MLP',  color='#9E9E9E')
    ax.axhline(90, color='green', ls='--', lw=0.8, label='DR target 90%')
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=30, ha='right')
    ax.set_ylim(0, 115); ax.set_ylabel('Detection Rate (%)')
    ax.legend(); ax.set_title('ST-GNN vs Baseline MLP — Detection Rate Comparison')
    plt.tight_layout()
    path = os.path.join(log_dir, 'comparison_dr.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"   📊  Comparison plot saved → {path}")


# ── Main Training Pipeline ────────────────────────────────────────────────────

def train(log_dir):
    # ── Dataset ──────────────────────────────────────────────────────────────
    proc_cache = os.path.join(config.DATASET_DIR, 'processed')
    if os.path.exists(proc_cache):
        shutil.rmtree(proc_cache)
        print("🧹  Cleared cached dataset — rebuilding …")

    dataset    = GNSSGraphDataset()
    train_data = [d for d in dataset if d.train_mask]
    test_data  = [d for d in dataset if not d.train_mask]
    print(f"\n📦  Dataset: {len(dataset)} windows  |  "
          f"train={len(train_data)}  test={len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE,
                               shuffle=True, collate_fn=_collate_windows,
                               drop_last=True)
    test_loader  = DataLoader(test_data, batch_size=config.BATCH_SIZE,
                               shuffle=False, collate_fn=_collate_windows)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   🖥  Device: {device}")

    # ── 1. Main model: ST-GNN ────────────────────────────────────────────────
    abl_tag = 'GNN-only' if config.ABLATION_NO_TEMPORAL else 'ST-GNN'
    model = STGNNSpoofingDetector().to(device)
    perf_main, losses, accs, yt, yp = _train_and_eval(
        model, train_loader, test_loader, test_data, device, log_dir, abl_tag)
    _save_plots(losses, accs, perf_main, yt, yp, log_dir, prefix=abl_tag)

    # Attention visualization
    _save_attention_plot(model, test_data, device, log_dir)

    # ── 2. Baseline MLP (optional) ───────────────────────────────────────────
    perf_baseline = None
    if config.RUN_BASELINE_MLP:
        print("\n" + "─" * 65)
        print("  Running Baseline MLP for comparison …")
        print("─" * 65)
        baseline = BaselineMLP().to(device)
        perf_baseline, bl_losses, bl_accs, bl_yt, bl_yp = _train_and_eval(
            baseline, train_loader, test_loader, test_data, device, log_dir,
            'Baseline-MLP')
        _save_plots(bl_losses, bl_accs, perf_baseline, bl_yt, bl_yp,
                    log_dir, prefix='Baseline-MLP')

        # Comparison plot
        _save_comparison_plot(perf_main, perf_baseline, log_dir)

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  SUMMARY")
    print(f"{'='*65}")
    all_scenarios = sorted(perf_main.keys())
    header = f"  {'Scenario':<18} {'DR-'+abl_tag:>12} {'FAR-'+abl_tag:>12}"
    if perf_baseline:
        header += f" {'DR-MLP':>10} {'FAR-MLP':>10}"
    print(header)
    print(f"  {'-'*len(header)}")
    for s in all_scenarios:
        dr_m, far_m = perf_main[s]
        line = (f"  {s:<18} {dr_m*100:>11.1f}% {far_m*100:>11.2f}%")
        if perf_baseline and s in perf_baseline:
            dr_b, far_b = perf_baseline[s]
            line += f" {dr_b*100:>9.1f}% {far_b*100:>9.2f}%"
        print(line)
    print(f"{'='*65}")


def main():
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = os.path.join(config.LOG_BASE_DIR, ts)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(log_dir, 'console_log.txt'))

    print('=' * 65)
    print('  GNSS ST-GNN Spoofing Detector v2.1')
    print('=' * 65)
    abl = 'GNN-only (no LSTM)' if config.ABLATION_NO_TEMPORAL else 'ST-GNN (full)'
    print(f"  Mode: {abl}")
    print(f"  Baseline MLP: {'ON' if config.RUN_BASELINE_MLP else 'OFF'}")
    print(f"  EPOCHS={config.EPOCHS}  BS={config.BATCH_SIZE}  "
          f"LR={config.LR}  HIDDEN={config.HIDDEN_DIM}  "
          f"WINDOW={config.TEMPORAL_WINDOW}")

    train(log_dir)


if __name__ == '__main__':
    main()