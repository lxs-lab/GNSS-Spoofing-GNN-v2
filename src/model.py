"""
model.py  —  Physics-Aware Spatio-Temporal Graph Neural Network
================================================================
Architecture overview:
  1. Spatial encoder   : 2-layer Graph Transformer (TransformerConv) with edge attrs
                         compresses each epoch's constellation topology into a
                         fixed-size vector via dual mean+max pooling.
  2. Temporal encoder  : 2-layer bidirectional LSTM processes the sequence of T
                         spatial embeddings to capture temporal evolution.
  3. Classifier head   : 2-layer MLP with dropout → 2-class logits.

Key design decisions:
  • Bidirectional LSTM: traction attacks create a directional trend; bidir LSTM
    can detect both the onset and the continuation within the window.
  • Dual pooling (mean + max): mean captures the average constellation health;
    max isolates the single most anomalous satellite.
  • edge_dim = config.EDGE_FEATURE_DIM (4): model is feature-dimension-aware.
  • Return raw logits: CrossEntropyLoss handles log-softmax internally.
  • forward_with_attention(): returns GNN attention weights for XAI.
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


class SpatialGNNEncoder(nn.Module):
    """
    Two-layer Graph Transformer that encodes a single-epoch constellation
    graph into a single embedding vector.

    Input:  x          (N, NODE_FEATURE_DIM)
            edge_index (2, E)
            edge_attr  (E, EDGE_FEATURE_DIM)
            batch      (N,)  — batch vector from DataLoader
    Output: embedding  (B, hidden_dim * 2)  where B = batch size
    """

    def __init__(self, in_channels: int, hidden_dim: int, edge_dim: int,
                 heads: int = 4, dropout: float = 0.3):
        super().__init__()

        # Input batch-normalisation stabilises training with heterogeneous
        # node counts across windows.
        self.bn_node = BatchNorm(in_channels)
        self.bn_edge = BatchNorm(edge_dim)

        # Layer 1: multi-head attention, output = hidden_dim × heads
        self.conv1 = TransformerConv(
            in_channels   = in_channels,
            out_channels  = hidden_dim,
            heads         = heads,
            dropout       = dropout,
            edge_dim      = edge_dim,
            beta          = True,   # residual gating (beta learnt from data)
            concat        = True,
        )
        self.bn1 = BatchNorm(hidden_dim * heads)

        # Layer 2: single-head projection back to hidden_dim
        self.conv2 = TransformerConv(
            in_channels   = hidden_dim * heads,
            out_channels  = hidden_dim,
            heads         = 1,
            dropout       = dropout,
            edge_dim      = edge_dim,
            beta          = True,
            concat        = False,
        )
        self.bn2 = BatchNorm(hidden_dim)

        self.dropout_p = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x         = self.bn_node(x)
        edge_attr = self.bn_edge(edge_attr)

        # Layer 1
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)

        # Dual readout: mean captures global state, max captures worst anomaly
        x_mean = global_mean_pool(x, batch)   # (B, hidden_dim)
        x_max  = global_max_pool(x, batch)    # (B, hidden_dim)
        return torch.cat([x_mean, x_max], dim=1)  # (B, hidden_dim * 2)

    def forward_with_attention(self, x, edge_index, edge_attr, batch):
        """Like forward(), but also returns layer-2 attention weights."""
        x         = self.bn_node(x)
        edge_attr = self.bn_edge(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)

        # Return attention weights from the final layer only
        out, (att_edge_index, att_alpha) = self.conv2(
            x, edge_index, edge_attr, return_attention_weights=True
        )
        out = self.bn2(out)
        out = F.elu(out)

        x_mean = global_mean_pool(out, batch)
        x_max  = global_max_pool(out, batch)
        emb = torch.cat([x_mean, x_max], dim=1)
        return emb, {'edge_index': att_edge_index, 'alpha': att_alpha}


class STGNNSpoofingDetector(nn.Module):
    """
    Full spatio-temporal model for GNSS spoofing detection.

    Forward input:
      graph_seq : list of T (x, edge_index, edge_attr, batch) tuples
                  (one per snapshot in the temporal window)
    Forward output:
      logits    : (B, 2)  raw unnormalised class scores
    """

    def __init__(self):
        super().__init__()

        # ── Spatial encoder ────────────────────────────────────────────────
        self.spatial_enc = SpatialGNNEncoder(
            in_channels = config.NODE_FEATURE_DIM,
            hidden_dim  = config.HIDDEN_DIM,
            edge_dim    = config.EDGE_FEATURE_DIM,
            heads       = config.GNN_HEADS,
            dropout     = config.DROPOUT,
        )
        spatial_out_dim = config.HIDDEN_DIM * 2   # mean + max concat

        # ── Temporal encoder ───────────────────────────────────────────────
        # Bidirectional LSTM processes the T-length sequence of spatial embeddings.
        # Each spatial embedding is spatial_out_dim = HIDDEN_DIM * 2.
        self.lstm = nn.LSTM(
            input_size    = spatial_out_dim,
            hidden_size   = config.LSTM_HIDDEN,
            num_layers    = config.LSTM_LAYERS,
            batch_first   = True,
            bidirectional = True,
            dropout       = config.DROPOUT if config.LSTM_LAYERS > 1 else 0.0,
        )
        lstm_out_dim = config.LSTM_HIDDEN * 2   # bidirectional concat

        # ── Classifier head ────────────────────────────────────────────────
        self.clf = nn.Sequential(
            nn.Linear(lstm_out_dim, config.HIDDEN_DIM),
            nn.ELU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, 2),   # 2-class logits
        )

    def _encode_sequence(self, graph_seq):
        """
        Encode a temporal sequence of graphs.

        graph_seq : list of T tuples (x, edge_index, edge_attr, batch)
        Returns: embeddings of shape (B, T, spatial_out_dim)
        """
        embs = []
        for x, edge_index, edge_attr, batch in graph_seq:
            emb = self.spatial_enc(x, edge_index, edge_attr, batch)
            embs.append(emb)   # each emb is (B, spatial_out_dim)
        # Stack along the time dimension → (B, T, spatial_out_dim)
        return torch.stack(embs, dim=1)

    def forward(self, graph_seq):
        """
        graph_seq : list of T tuples (x, edge_index, edge_attr, batch)
        Returns: logits (B, 2)
        """
        # (B, T, spatial_out_dim)
        seq_emb = self._encode_sequence(graph_seq)

        # LSTM temporal encoding: take the last time-step output as summary
        # lstm_out shape: (B, T, lstm_hidden * 2)
        lstm_out, _ = self.lstm(seq_emb)
        # Use the LAST time step as the sequence summary
        # (B, lstm_hidden * 2)
        temporal_emb = lstm_out[:, -1, :]

        logits = self.clf(temporal_emb)  # (B, 2)
        return logits

    def forward_with_attention(self, graph_seq):
        """
        Inference-only variant that also returns attention weights from the
        last snapshot in the window (most relevant for current-state XAI).

        Returns: (logits, attention_info)
        """
        # Encode all but the last snapshot normally
        embs = []
        for t_idx, (x, edge_index, edge_attr, batch) in enumerate(graph_seq):
            if t_idx < len(graph_seq) - 1:
                emb = self.spatial_enc(x, edge_index, edge_attr, batch)
            else:
                # Last snapshot: collect attention weights
                emb, attn_info = self.spatial_enc.forward_with_attention(
                    x, edge_index, edge_attr, batch
                )
            embs.append(emb)

        seq_emb   = torch.stack(embs, dim=1)
        lstm_out, _ = self.lstm(seq_emb)
        temporal_emb = lstm_out[:, -1, :]
        logits       = self.clf(temporal_emb)
        return logits, attn_info
