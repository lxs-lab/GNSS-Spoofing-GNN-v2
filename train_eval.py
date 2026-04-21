"""
train_eval.py  —  Training + Validation + Evaluation (v3.1)
==========================================================
FIX from v3.0:
  [F1] Early stopping now uses F2-score (DR weighted 2x over precision)
       instead of raw DR. This prevents the model from saving a checkpoint
       where DR=100% but FAR=100% (all-spoofed trivial predictor).
       Formula: F2 = 5 * DR * (1-FAR) / (4*DR + (1-FAR))
       A checkpoint is only saved when FAR < FAR_SAVE_LIMIT (default 0.40).
       This means "full alarm" (FAR=100%) will never be saved even if DR=100%.
  [F2] Threshold search now requires FAR < 0.15 (tightened from 0.10)
       and skips all thresholds where the model outputs constant predictions.
  [F3] Epoch log now prints F2-score alongside DR/FAR for transparency.
"""

import os, sys, shutil, random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from src.graph_builder import GNSSGraphDataset
from src.model import STGNNSpoofingDetector, BaselineMLP
from src import config

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Maximum FAR allowed when saving a checkpoint.
# Epoch 1 where DR=100%+FAR=100% will be rejected by this guard.
FAR_SAVE_LIMIT = 0.40


# ── Collate & Unpack ──────────────────────────────────────────────────────────

def _collate_windows(data_list):
    T = data_list[0].window_len
    graph_seq = []
    for t in range(T):
        epoch_graphs = [
            Data(
                x          = getattr(d, f'nx_{t}'),
                edge_index = getattr(d, f'ei_{t}'),
                edge_attr  = getattr(d, f'ea_{t}'),
            )
            for d in data_list
        ]
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


def f2_score(dr, far):
    """
    F2-score: weights DR twice as heavily as (1-FAR).
    Returns 0.0 if either metric is NaN or FAR >= FAR_SAVE_LIMIT.
    This ensures the trivial all-spoofed predictor (DR=1, FAR=1) scores 0.
    """
    if np.isnan(dr) or np.isnan(far):
        return 0.0
    if far >= FAR_SAVE_LIMIT:   # [F1] reject high-FAR checkpoints
        return 0.0
    precision_proxy = 1.0 - far   # treat (1-FAR) as precision proxy
    denom = 4.0 * dr + precision_proxy + 1e-9
    return 5.0 * dr * precision_proxy / denom


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets,
                              weight=self.weight, reduction='none')
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Prediction helper ─────────────────────────────────────────────────────────

def _predict(model, seq, threshold):
    logits = model(seq)
    probs  = torch.softmax(logits, dim=1)[:, 1]
    return (probs >= threshold).long().cpu().numpy(), probs.cpu().numpy()


# ── Threshold search on validation set ───────────────────────────────────────

def _find_best_threshold(model, val_loader, device) -> float:
    """
    Sweep thresholds on val set. Pick the one with best F2-score
    subject to FAR < 0.15. Falls back to config.SPOOF_THRESHOLD if
    nothing satisfies the constraint.
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for graph_seq, labels, _ in val_loader:
            seq = [_unpack_batch(b.to(device)) for b in graph_seq]
            probs = torch.softmax(model(seq), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_thr = config.SPOOF_THRESHOLD
    best_f2  = -1.0

    for thr in np.arange(0.05, 0.55, 0.025):
        preds = (all_probs >= thr).astype(int)
        # [F2] Skip if model outputs a constant prediction
        if len(np.unique(preds)) < 2:
            continue
        dr, far = compute_dr_far(preds, all_labels)
        if np.isnan(dr) or np.isnan(far) or far > 0.15:
            continue
        score = f2_score(dr, far)
        if score > best_f2:
            best_f2  = score
            best_thr = float(thr)

    return best_thr


# ── Logger ────────────────────────────────────────────────────────────────────

class TeeLogger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, 'a', encoding='utf-8')
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()


# ── Core training function ────────────────────────────────────────────────────

def _train_and_eval(model, train_loader, val_loader, test_loader,
                    test_data, device, log_dir, model_name='ST-GNN'):

    # Class weights
    tr_labels = [d.y.item() for d in train_loader.dataset]
    n0, n1    = tr_labels.count(0), tr_labels.count(1)
    ratio     = n0 / max(n1, 1)
    w = (torch.tensor([1.0, min(ratio, 10.0)], dtype=torch.float)
         if ratio > 1.5 else torch.ones(2))
    print(f"   [{model_name}] train imbalance {ratio:.1f}:1 → "
          f"weights {[round(x,2) for x in w.tolist()]}")

    criterion = FocalLoss(gamma=2.0, weight=w.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    best_path    = os.path.join(config.MODEL_DIR, f'best_{model_name}.pth')
    # [F1] Use F2-score (not raw DR) as the checkpoint criterion
    best_f2      = 0.0
    patience_cnt = 0
    train_losses, val_f2s = [], []

    print(f"\n🚀  [{model_name}] Training for up to {config.EPOCHS} epochs …")
    print(f"   Early stop: F2-score  patience={config.PATIENCE}")
    print(f"   FAR save limit: {FAR_SAVE_LIMIT:.0%}  "
          f"(epoch with FAR>={FAR_SAVE_LIMIT:.0%} will not be saved)\n")

    for epoch in range(1, config.EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for graph_seq, labels, _ in train_loader:
            seq    = [_unpack_batch(b.to(device)) for b in graph_seq]
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seq), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step()

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        val_preds, val_labels = [], []
        thr = config.SPOOF_THRESHOLD
        with torch.no_grad():
            for graph_seq, labels, _ in val_loader:
                seq = [_unpack_batch(b.to(device)) for b in graph_seq]
                preds, _ = _predict(model, seq, thr)
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())

        val_dr,  val_far = compute_dr_far(val_preds, val_labels)
        # [F1] Compute F2; epochs with FAR >= FAR_SAVE_LIMIT score 0
        score = f2_score(val_dr, val_far)
        val_f2s.append(score)

        # [F1] Save checkpoint only when F2 improves
        if score > best_f2:
            best_f2 = score
            patience_cnt = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_cnt += 1

        # Print every 5 epochs (more frequent than before for visibility)
        if epoch % 1 == 0 or epoch == 1:
            vdr_s  = f'{val_dr*100:.1f}%'  if not np.isnan(val_dr)  else 'N/A'
            vfar_s = f'{val_far*100:.2f}%' if not np.isnan(val_far) else 'N/A'
            # [F3] Print F2-score
            saved  = ' *saved*' if patience_cnt == 0 else ''
            print(f"  [{model_name}] Epoch {epoch:03d}/{config.EPOCHS} | "
                  f"Loss {avg_loss:.4f} | "
                  f"Val DR {vdr_s} | Val FAR {vfar_s} | "
                  f"F2={score:.3f}{saved}")

        if patience_cnt >= config.PATIENCE:
            print(f"\n  [{model_name}] ⏹  Early stop at epoch {epoch} "
                  f"(F2 did not improve for {config.PATIENCE} epochs)")
            break

    # ── Load best checkpoint ──────────────────────────────────────────────
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
        print(f"  [{model_name}] Best val F2 = {best_f2:.3f}  "
              f"(checkpoint loaded)")
    else:
        print(f"  [{model_name}] WARNING: no valid checkpoint saved "
              f"(all epochs had FAR >= {FAR_SAVE_LIMIT:.0%}). "
              f"Using final weights.")

    model.eval()

    # ── Threshold search on val set ───────────────────────────────────────
    best_thr = _find_best_threshold(model, val_loader, device)
    print(f"  [{model_name}] Val-tuned threshold = {best_thr:.3f}  "
          f"(config default was {config.SPOOF_THRESHOLD})")

    # ── Per-scenario evaluation on test set ───────────────────────────────
    scenario_groups = {}
    for d in test_data:
        scenario_groups.setdefault(d.scenario, []).append(d)

    print(f"\n{'='*65}")
    print(f"  [{model_name}] Per-scenario zero-shot evaluation "
          f"(threshold={best_thr:.3f})")
    print(f"{'='*65}")
    print(f"  {'Scenario':<22} {'N':>6} {'DR (%)':>8} {'FAR (%)':>8}  Status")
    print(f"  {'-'*60}")

    perf = {}
    y_true_all, y_pred_all = [], []

    for sce in sorted(scenario_groups.keys()):
        sub_loader = DataLoader(
            scenario_groups[sce],
            batch_size=config.BATCH_SIZE,
            collate_fn=_collate_windows)
        preds, labs = [], []
        with torch.no_grad():
            for graph_seq, labels, _ in sub_loader:
                seq = [_unpack_batch(b.to(device)) for b in graph_seq]
                p, _ = _predict(model, seq, best_thr)
                preds.extend(p)
                labs.extend(labels.numpy())

        dr_s, far_s = compute_dr_far(preds, labs)
        perf[sce] = (dr_s, far_s)
        y_true_all.extend(labs); y_pred_all.extend(preds)

        ok = (not np.isnan(dr_s) and dr_s > 0.90 and
              not np.isnan(far_s) and far_s < 0.05)
        flag    = '✅' if ok else '⚠️ '
        dr_str  = f'{dr_s*100:.1f}%'  if not np.isnan(dr_s)  else '  N/A '
        far_str = f'{far_s*100:.2f}%' if not np.isnan(far_s) else '  N/A '
        print(f"  {sce:<22} {len(scenario_groups[sce]):>6} "
              f"{dr_str:>8} {far_str:>8}  {flag}")

    print(f"{'='*65}")
    return perf, train_losses, val_f2s, y_true_all, y_pred_all, best_thr


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _save_plots(train_losses, val_f2s, perf, y_true, y_pred,
                log_dir, prefix='ST-GNN'):
    tag = prefix.replace(' ', '_') + '_'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label='Train loss')
    ax1.set_title(f'{prefix} Training loss')
    ax1.set_xlabel('Epoch'); ax1.legend()

    ax2.plot(val_f2s, color='steelblue', label='Val F2-score')
    ax2.set_ylim(0, 1.05)
    ax2.set_title(f'{prefix} Val F2-score')
    ax2.set_xlabel('Epoch'); ax2.legend()

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
    drs   = [perf[n][0] * 100 if not np.isnan(perf[n][0]) else 0 for n in names]
    fars  = [perf[n][1] * 100 if not np.isnan(perf[n][1]) else 0 for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
    ax.bar(x - 0.2, drs,  0.35, label='DR (%)',  color='#4CAF50')
    ax.bar(x + 0.2, fars, 0.35, label='FAR (%)', color='#F44336')
    ax.axhline(90, color='green', ls='--', lw=0.8, label='DR target 90%')
    ax.axhline(5,  color='red',   ls='--', lw=0.8, label='FAR limit 5%')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylim(0, 115); ax.set_ylabel('%'); ax.legend()
    ax.set_title(f'{prefix} Per-scenario DR and FAR')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, f'{tag}scenario_metrics.png'), dpi=150)
    plt.close()
    print(f"   📊  [{prefix}] Plots saved → {log_dir}/")


def _save_comparison_plot(perf_stgnn, perf_baseline, log_dir):
    scenarios = sorted(set(perf_stgnn.keys()) & set(perf_baseline.keys()))
    if not scenarios:
        return
    dr_gnn  = [perf_stgnn[s][0]*100   if not np.isnan(perf_stgnn[s][0])   else 0
               for s in scenarios]
    dr_base = [perf_baseline[s][0]*100 if not np.isnan(perf_baseline[s][0]) else 0
               for s in scenarios]
    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.2), 5))
    ax.bar(x - 0.2, dr_gnn,  0.35, label='ST-GNN (ours)', color='#2196F3')
    ax.bar(x + 0.2, dr_base, 0.35, label='Baseline MLP',  color='#9E9E9E')
    ax.axhline(90, color='green', ls='--', lw=0.8, label='DR target 90%')
    ax.set_xticks(x); ax.set_xticklabels(scenarios, rotation=30, ha='right')
    ax.set_ylim(0, 115); ax.set_ylabel('Detection Rate (%)')
    ax.legend(); ax.set_title('ST-GNN vs Baseline MLP — DR Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'comparison_dr.png'), dpi=150)
    plt.close()
    print(f"   📊  Comparison plot saved")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def train(log_dir):
    proc_cache = os.path.join(config.DATASET_DIR, 'processed')
    if os.path.exists(proc_cache):
        shutil.rmtree(proc_cache)
        print("🧹  Cleared cached dataset — rebuilding …\n")

    dataset    = GNSSGraphDataset()
    train_data = [d for d in dataset if d.split == 'train']
    val_data   = [d for d in dataset if d.split == 'val']
    test_data  = [d for d in dataset if d.split == 'test']

    print(f"\n📦  Dataset: {len(dataset)} windows  |  "
          f"train={len(train_data)}  val={len(val_data)}  "
          f"test={len(test_data)}")

    mk = dict(batch_size=config.BATCH_SIZE, collate_fn=_collate_windows)
    train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **mk)
    val_loader   = DataLoader(val_data,   shuffle=False, **mk)
    test_loader  = DataLoader(test_data,  shuffle=False, **mk)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")

    # ── ST-GNN ───────────────────────────────────────────────────────────
    abl_tag = 'GNN-only' if config.ABLATION_NO_TEMPORAL else 'ST-GNN'
    model   = STGNNSpoofingDetector().to(device)
    perf_main, losses, val_f2s, yt, yp, thr_main = _train_and_eval(
        model, train_loader, val_loader, test_loader,
        test_data, device, log_dir, abl_tag)
    _save_plots(losses, val_f2s, perf_main, yt, yp, log_dir, prefix=abl_tag)

    # ── Baseline MLP (optional) ───────────────────────────────────────────
    perf_baseline = None
    if config.RUN_BASELINE_MLP:
        print("\n" + "─" * 65)
        print("  Running Baseline MLP for comparison …")
        print("─" * 65)
        baseline = BaselineMLP().to(device)
        perf_baseline, bl_losses, bl_f2s, bl_yt, bl_yp, _ = _train_and_eval(
            baseline, train_loader, val_loader, test_loader,
            test_data, device, log_dir, 'Baseline-MLP')
        _save_plots(bl_losses, bl_f2s, perf_baseline, bl_yt, bl_yp,
                    log_dir, prefix='Baseline-MLP')
        _save_comparison_plot(perf_main, perf_baseline, log_dir)

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL SUMMARY  (ST-GNN threshold={thr_main:.3f})")
    print(f"{'='*65}")
    header = f"  {'Scenario':<18} {'DR-'+abl_tag:>12} {'FAR-'+abl_tag:>12}"
    if perf_baseline:
        header += f" {'DR-MLP':>10} {'FAR-MLP':>10}"
    print(header)
    print(f"  {'-'*(len(header)-2)}")
    for s in sorted(perf_main.keys()):
        dr_m, far_m = perf_main[s]
        line = f"  {s:<18} {dr_m*100:>11.1f}% {far_m*100:>11.2f}%"
        if perf_baseline and s in perf_baseline:
            dr_b, far_b = perf_baseline[s]
            line += f" {dr_b*100:>9.1f}% {far_b*100:>9.2f}%"
        print(line)
    print(f"{'='*65}")


def main():
    ts      = datetime.now().strftime('%Y%m%d_%H%M')
    log_dir = os.path.join(config.LOG_BASE_DIR, ts)
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = TeeLogger(os.path.join(log_dir, 'console_log.txt'))

    print('=' * 65)
    print('  GNSS ST-GNN Spoofing Detector v3.1')
    print('=' * 65)
    abl = 'GNN-only (no LSTM)' if config.ABLATION_NO_TEMPORAL else 'ST-GNN (full)'
    print(f"  Mode: {abl}")
    print(f"  Baseline MLP : {'ON' if config.RUN_BASELINE_MLP else 'OFF'}")
    print(f"  EPOCHS={config.EPOCHS}  BS={config.BATCH_SIZE}  "
          f"LR={config.LR}  HIDDEN={config.HIDDEN_DIM}  "
          f"WINDOW={config.TEMPORAL_WINDOW}  "
          f"THRESHOLD={config.SPOOF_THRESHOLD}")
    print(f"  FAR save limit = {FAR_SAVE_LIMIT:.0%}")
    train(log_dir)


if __name__ == '__main__':
    main()