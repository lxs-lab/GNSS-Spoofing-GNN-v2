"""
model.py  —  Physics-Aware Spatio-Temporal GNN + Baseline Models (v2.1)
========================================================================
Changes from v2.0:
  • STGNNSpoofingDetector supports ABLATION_NO_TEMPORAL: skips LSTM,
    uses only the last frame's GNN embedding for classification.
  • New BaselineMLP: per-channel feature MLP (no graph structure) for
    fair comparison — proves the value of the graph topology.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool

try:
    from src import config
except ImportError:
    from .. import config


# ═══════════════════════════════════════════════════════════════════════════════
#  Spatial GNN Encoder (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class SpatialGNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, edge_dim, heads=4, dropout=0.3):
        super().__init__()
        self.bn_node = BatchNorm(in_channels)
        self.bn_edge = BatchNorm(edge_dim)

        self.conv1 = TransformerConv(
            in_channels, hidden_dim, heads=heads, dropout=dropout,
            edge_dim=edge_dim, beta=True, concat=True)
        self.bn1 = BatchNorm(hidden_dim * heads)

        self.conv2 = TransformerConv(
            hidden_dim * heads, hidden_dim, heads=1, dropout=dropout,
            edge_dim=edge_dim, beta=True, concat=False)
        self.bn2 = BatchNorm(hidden_dim)
        self.dropout_p = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x         = self.bn_node(x)
        edge_attr = self.bn_edge(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x); x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x); x = F.elu(x)

        x_mean = global_mean_pool(x, batch)
        x_max  = global_max_pool(x, batch)
        return torch.cat([x_mean, x_max], dim=1)

    def forward_with_attention(self, x, edge_index, edge_attr, batch):
        x         = self.bn_node(x)
        edge_attr = self.bn_edge(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x); x = F.elu(x)

        out, (att_edge_index, att_alpha) = self.conv2(
            x, edge_index, edge_attr, return_attention_weights=True)
        out = self.bn2(out); out = F.elu(out)

        x_mean = global_mean_pool(out, batch)
        x_max  = global_max_pool(out, batch)
        emb = torch.cat([x_mean, x_max], dim=1)
        return emb, {'edge_index': att_edge_index, 'alpha': att_alpha}


# ═══════════════════════════════════════════════════════════════════════════════
#  Full ST-GNN Model (with ablation support)
# ═══════════════════════════════════════════════════════════════════════════════

class STGNNSpoofingDetector(nn.Module):
    """
    ABLATION_NO_TEMPORAL = False (default):
      GNN encodes each of T snapshots → BiLSTM → classifier
    ABLATION_NO_TEMPORAL = True:
      GNN encodes ONLY the last snapshot → classifier (no temporal module)
    """

    def __init__(self):
        super().__init__()
        self.no_temporal = config.ABLATION_NO_TEMPORAL

        self.spatial_enc = SpatialGNNEncoder(
            in_channels=config.NODE_FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            edge_dim=config.EDGE_FEATURE_DIM,
            heads=config.GNN_HEADS,
            dropout=config.DROPOUT,
        )
        spatial_out_dim = config.HIDDEN_DIM * 2  # mean + max

        if not self.no_temporal:
            self.lstm = nn.LSTM(
                input_size=spatial_out_dim,
                hidden_size=config.LSTM_HIDDEN,
                num_layers=config.LSTM_LAYERS,
                batch_first=True,
                bidirectional=True,
                dropout=config.DROPOUT if config.LSTM_LAYERS > 1 else 0.0,
            )
            clf_input_dim = config.LSTM_HIDDEN * 2
        else:
            self.lstm = None
            clf_input_dim = spatial_out_dim

        self.clf = nn.Sequential(
            nn.Linear(clf_input_dim, config.HIDDEN_DIM),
            nn.ELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 2),
        )

    def forward(self, graph_seq):
        if self.no_temporal:
            # Only use the last snapshot
            x, edge_index, edge_attr, batch = graph_seq[-1]
            emb = self.spatial_enc(x, edge_index, edge_attr, batch)
            return self.clf(emb)

        # Full spatio-temporal path
        embs = []
        for x, edge_index, edge_attr, batch in graph_seq:
            embs.append(self.spatial_enc(x, edge_index, edge_attr, batch))
        seq_emb = torch.stack(embs, dim=1)  # (B, T, spatial_out)

        lstm_out, _ = self.lstm(seq_emb)
        temporal_emb = lstm_out[:, -1, :]    # last time step
        return self.clf(temporal_emb)

    def forward_with_attention(self, graph_seq):
        embs = []
        attn_info = None
        for t_idx, (x, edge_index, edge_attr, batch) in enumerate(graph_seq):
            if t_idx < len(graph_seq) - 1:
                embs.append(self.spatial_enc(x, edge_index, edge_attr, batch))
            else:
                emb, attn_info = self.spatial_enc.forward_with_attention(
                    x, edge_index, edge_attr, batch)
                embs.append(emb)

        if self.no_temporal:
            return self.clf(embs[-1]), attn_info

        seq_emb = torch.stack(embs, dim=1)
        lstm_out, _ = self.lstm(seq_emb)
        return self.clf(lstm_out[:, -1, :]), attn_info


# ═══════════════════════════════════════════════════════════════════════════════
#  Baseline MLP (no graph structure — per-channel features only)
# ═══════════════════════════════════════════════════════════════════════════════

class BaselineMLP(nn.Module):
    """
    Simple baseline that uses the same 6 features but ignores graph structure.

    Input: graph_seq (same format as ST-GNN for fair comparison)
    Strategy:
      1. For each time step, compute the MEAN of all satellite features → (6,)
      2. Concatenate T mean vectors → (T * 6,)
      3. Pass through 3-layer MLP → 2-class logits

    This baseline cannot exploit inter-satellite relationships (no edges),
    so any performance gap vs. the GNN proves the graph structure's value.
    """

    def __init__(self):
        super().__init__()
        input_dim = config.TEMPORAL_WINDOW * config.NODE_FEATURE_DIM  # 10 * 6 = 60
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(64, 2),
        )

    def forward(self, graph_seq):
        means = []
        for x, edge_index, edge_attr, batch in graph_seq:
            # Per-graph mean: average across all nodes within each graph in batch
            # global_mean_pool gives (B, 6)
            means.append(global_mean_pool(x, batch))
        # (B, T, 6) → (B, T*6)
        feat = torch.cat(means, dim=1)
        return self.net(feat)