# src/models/gat_model_focal_loss_v2.py
#
# v3 changes:
#   IN_FEATURES: 9 → 13  (matches new node feature count in graph_builder.py)
#   Everything else unchanged

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# ── Configuration ─────────────────────────────────────────────────────────────

IN_FEATURES   = 13    # per-node input features (7 stat + 2 freq + 4 extended)
HIDDEN_DIM    = 64    # hidden dimension for GAT layers
NUM_HEADS     = 4     # attention heads in GATConv
DROPOUT       = 0.3   # dropout rate
ALPHA         = 0.0   # alpha=0.0 → classify only (optimal from ablation)


# ── GAT Model ─────────────────────────────────────────────────────────────────

class GraphSenseGAT(nn.Module):
    """
    Dual-head Graph Attention Network for IoT anomaly detection.

    Architecture:
        GATConv(13 → 64, heads=4)         # Layer 1: multi-head attention
        GATConv(256 → 64, heads=1)        # Layer 2: single-head refinement
        GlobalMeanPool                    # Graph-level embedding (64,)
            ├── Reconstruction head       # Decoder MLP → (8 × 13,) → MSE loss
            └── Classification head      # MLP → sigmoid → BCE/Focal loss

    KEY NOVELTY: graph structure (edge_index, edge_weight) is recomputed per
    window from Pearson correlations — dynamic graphs. GAT attention then
    further weights these edges per node. Two levels of adaptive weighting.

    v3: IN_FEATURES increased from 9 → 13 to include zcr, rms,
        autocorr_lag1, iqr. Reconstruction head output updated accordingly.
    """

    def __init__(
        self,
        in_features : int   = IN_FEATURES,
        hidden_dim  : int   = HIDDEN_DIM,
        num_heads   : int   = NUM_HEADS,
        dropout     : float = DROPOUT,
    ):
        super().__init__()

        self.dropout    = dropout
        self.in_features = in_features

        # ── GAT Layer 1 ───────────────────────────────────────────────────
        # in_features=13, out per head=hidden_dim, heads=4
        # Output shape per node: num_heads * hidden_dim = 4 * 64 = 256
        self.gat1 = GATConv(
            in_channels  = in_features,
            out_channels = hidden_dim,
            heads        = num_heads,
            dropout      = dropout,
            edge_dim     = 1,    # scalar edge weight (correlation value)
        )

        # ── GAT Layer 2 ───────────────────────────────────────────────────
        # in = 256 (concatenated heads from layer 1), out = hidden_dim
        # heads=1 → no concatenation → output shape per node: hidden_dim=64
        self.gat2 = GATConv(
            in_channels  = hidden_dim * num_heads,
            out_channels = hidden_dim,
            heads        = 1,
            dropout      = dropout,
            edge_dim     = 1,
        )

        # ── Reconstruction Head ───────────────────────────────────────────
        # Takes graph embedding (64,) → reconstructs all node features
        # Target shape: (n_nodes * in_features,) = (8 * 13,) = (104,)
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 8 * in_features),   # 8 nodes × 13 features
        )

        # ── Classification Head ───────────────────────────────────────────
        # Takes graph embedding (64,) → binary anomaly logit
        self.classify_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),      # single logit (no sigmoid — handled in loss)
        )

    def forward(self, x, edge_index, edge_weight, batch):
        """
        Parameters
        ----------
        x           : Tensor (total_nodes, 13)    — node features
        edge_index  : Tensor (2, total_edges)     — edge connectivity
        edge_weight : Tensor (total_edges,)       — correlation weights
        batch       : Tensor (total_nodes,)       — graph assignment per node

        Returns
        -------
        recon_out   : Tensor (batch_size, 104)    — reconstructed node features
        classify_out: Tensor (batch_size, 1)      — anomaly logits (pre-sigmoid)
        """

        # edge_weight must be shape (n_edges, 1) for GATConv edge_dim=1
        ew = edge_weight.unsqueeze(-1)

        # ── GAT Layer 1 ───────────────────────────────────────────────────
        x = self.gat1(x, edge_index, edge_attr=ew)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── GAT Layer 2 ───────────────────────────────────────────────────
        x = self.gat2(x, edge_index, edge_attr=ew)
        x = F.elu(x)

        # ── Graph-level pooling ───────────────────────────────────────────
        # Collapses (total_nodes, 64) → (batch_size, 64)
        graph_embed = global_mean_pool(x, batch)

        # ── Dual heads ────────────────────────────────────────────────────
        recon_out    = self.recon_head(graph_embed)
        classify_out = self.classify_head(graph_embed)

        return recon_out, classify_out


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification on imbalanced datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma=1.0, pos_weight=n_normal/n_anomaly * 0.5 — tuned from v2 runs.
    """

    def __init__(self, gamma: float = 1.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs    = torch.sigmoid(logits)
        probs    = probs.clamp(min=1e-7, max=1 - 1e-7)

        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p_t          = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t      = self.pos_weight * targets + (1 - targets)
        focal_loss   = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()


# ── Joint Loss ────────────────────────────────────────────────────────────────

class DualLoss(nn.Module):
    """
    L_total = alpha * L_recon + (1 - alpha) * L_classify
    alpha=0.0 → classify only (optimal from ablation study)
    """

    def __init__(
        self,
        alpha      : float = ALPHA,
        pos_weight : float = 1.0,
        gamma      : float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()
        self.focal = FocalLoss(gamma=gamma, pos_weight=pos_weight)

    def forward(
        self,
        recon_out    : torch.Tensor,   # (batch_size, 104)
        classify_out : torch.Tensor,   # (batch_size, 1)
        node_targets : torch.Tensor,   # (batch_size, 104)
        labels       : torch.Tensor,   # (batch_size,)
    ):
        l_recon    = self.mse(recon_out, node_targets)
        l_classify = self.focal(classify_out.squeeze(-1), labels.float())
        l_total    = self.alpha * l_recon + (1 - self.alpha) * l_classify
        return l_total, l_recon, l_classify


# ── Sanity Check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch_size = 2
    n_nodes    = 8
    n_edges    = 10

    x           = torch.randn(batch_size * n_nodes, IN_FEATURES)
    edge_index  = torch.randint(0, n_nodes, (2, n_edges))
    edge_weight = torch.rand(n_edges)
    batch       = torch.repeat_interleave(torch.arange(batch_size), n_nodes)
    labels      = torch.tensor([0, 1], dtype=torch.float)

    model   = GraphSenseGAT()
    loss_fn = DualLoss(alpha=0.0, pos_weight=1.14, gamma=1.0)

    recon, classify = model(x, edge_index, edge_weight, batch)

    node_targets = x.view(batch_size, -1)   # (2, 104)
    loss, l_r, l_c = loss_fn(recon, classify, node_targets, labels)

    print(f"recon shape    : {recon.shape}")       # (2, 104)
    print(f"classify shape : {classify.shape}")    # (2, 1)
    print(f"loss           : {loss.item():.4f}")
    print(f"  L_recon      : {l_r.item():.4f}")
    print(f"  L_classify   : {l_c.item():.4f}")
    print("✅ GAT model v3 sanity check passed.")
