# src/models/gat_model_focal_loss_v2.py
#
# v4 changes:
#   hidden_dim: 64 → 128
#   Added gat3 layer (3rd GATConv) with residual connection from gat2 output
#   Classification + reconstruction heads updated for 128-dim input
#   IN_FEATURES stays 13 — no pipeline changes needed

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# ── Configuration ─────────────────────────────────────────────────────────────

IN_FEATURES   = 13    # per-node input features (7 stat + 2 freq + 4 extended)
HIDDEN_DIM    = 128   # hidden dimension — increased from 64
NUM_HEADS     = 4     # attention heads in GATConv layer 1
DROPOUT       = 0.3   # dropout rate
ALPHA         = 0.0   # alpha=0.0 → classify only (optimal from ablation)


# ── GAT Model ─────────────────────────────────────────────────────────────────

class GraphSenseGAT(nn.Module):
    """
    Dual-head Graph Attention Network for IoT anomaly detection.

    Architecture:
        GATConv(13 → 128, heads=4)        # Layer 1: multi-head attention
                                           # output: 4*128 = 512 per node
        GATConv(512 → 128, heads=1)        # Layer 2: single-head refinement
                                           # output: 128 per node
        GATConv(128 → 128, heads=1)        # Layer 3: deeper refinement
          + residual from layer 2 output   # skip connection: x3 = gat3(x2) + x2
        GlobalMeanPool                     # Graph-level embedding (128,)
            ├── Reconstruction head        # MLP → (8 × 13 = 104,)
            └── Classification head        # MLP → sigmoid → anomaly logit

    WHY RESIDUAL:
        On small graphs (8 nodes), stacking GAT layers risks oversmoothing —
        all node embeddings converge to the same value.
        The residual connection (x3 = gat3(x2) + x2) lets the 3rd layer
        learn incremental refinements rather than a full transformation,
        preserving the discriminative signal from layer 2.

    WHY hidden_dim=128:
        Larger embedding space gives the attention mechanism more room to
        separate normal vs anomalous sensor interaction patterns.
        The graph embedding fed to the classifier is now 128-dim vs 64-dim.
    """

    def __init__(
        self,
        in_features : int   = IN_FEATURES,
        hidden_dim  : int   = HIDDEN_DIM,
        num_heads   : int   = NUM_HEADS,
        dropout     : float = DROPOUT,
    ):
        super().__init__()

        self.dropout     = dropout
        self.hidden_dim  = hidden_dim
        self.in_features = in_features

        # ── GAT Layer 1 ───────────────────────────────────────────────────
        # in=13, out per head=128, heads=4
        # Output per node: 4 * 128 = 512
        self.gat1 = GATConv(
            in_channels  = in_features,
            out_channels = hidden_dim,
            heads        = num_heads,
            dropout      = dropout,
            edge_dim     = 1,
        )

        # ── GAT Layer 2 ───────────────────────────────────────────────────
        # in=512, out=128, heads=1
        # Output per node: 128
        self.gat2 = GATConv(
            in_channels  = hidden_dim * num_heads,  # 512
            out_channels = hidden_dim,               # 128
            heads        = 1,
            dropout      = dropout,
            edge_dim     = 1,
        )

        # ── GAT Layer 3 + Residual ─────────────────────────────────────────
        # in=128, out=128, heads=1
        # Output per node: 128
        # Residual: x3 = ELU(gat3(x2)) + x2
        # Both gat3 output and x2 are 128-dim — dimensions match, no projection needed
        self.gat3 = GATConv(
            in_channels  = hidden_dim,   # 128
            out_channels = hidden_dim,   # 128
            heads        = 1,
            dropout      = dropout,
            edge_dim     = 1,
        )

        # ── Reconstruction Head ───────────────────────────────────────────
        # graph_embed (128,) → (8 * 13,) = (104,)
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),   # 128 → 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 8 * in_features),  # 256 → 104
        )

        # ── Classification Head ───────────────────────────────────────────
        # graph_embed (128,) → scalar logit
        self.classify_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),    # 128 → 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
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
        ew = edge_weight.unsqueeze(-1)

        # ── GAT Layer 1 ───────────────────────────────────────────────────
        x = self.gat1(x, edge_index, edge_attr=ew)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── GAT Layer 2 ───────────────────────────────────────────────────
        x = self.gat2(x, edge_index, edge_attr=ew)
        x = F.elu(x)
        x2 = x   # save for residual — shape (total_nodes, 128)

        # ── GAT Layer 3 + Residual ─────────────────────────────────────────
        x = self.gat3(x2, edge_index, edge_attr=ew)
        x = F.elu(x)
        x = x + x2   # residual: adds layer2 output back in
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ── Graph-level pooling ───────────────────────────────────────────
        graph_embed = global_mean_pool(x, batch)   # (batch_size, 128)

        # ── Dual heads ────────────────────────────────────────────────────
        recon_out    = self.recon_head(graph_embed)
        classify_out = self.classify_head(graph_embed)

        return recon_out, classify_out


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=1.0, pos_weight tuned from training split.
    """

    def __init__(self, gamma: float = 1.0, pos_weight: float = 1.0):
        super().__init__()
        self.gamma      = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs        = torch.sigmoid(logits).clamp(min=1e-7, max=1 - 1e-7)
        bce_loss     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t          = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t      = self.pos_weight * targets + (1 - targets)
        return (alpha_t * focal_weight * bce_loss).mean()


# ── Joint Loss ────────────────────────────────────────────────────────────────

class DualLoss(nn.Module):
    """
    L_total = alpha * L_recon + (1 - alpha) * L_classify
    alpha=0.0 → classify only (optimal from ablation)
    """

    def __init__(self, alpha: float = ALPHA, pos_weight: float = 1.0, gamma: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.mse   = nn.MSELoss()
        self.focal = FocalLoss(gamma=gamma, pos_weight=pos_weight)

    def forward(self, recon_out, classify_out, node_targets, labels):
        l_recon    = self.mse(recon_out, node_targets)
        l_classify = self.focal(classify_out.squeeze(-1), labels.float())
        l_total    = self.alpha * l_recon + (1 - self.alpha) * l_classify
        return l_total, l_recon, l_classify


# ── Sanity Check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch_size  = 2
    n_nodes     = 8
    n_edges     = 10

    x           = torch.randn(batch_size * n_nodes, IN_FEATURES)
    edge_index  = torch.randint(0, n_nodes, (2, n_edges))
    edge_weight = torch.rand(n_edges)
    batch       = torch.repeat_interleave(torch.arange(batch_size), n_nodes)
    labels      = torch.tensor([0, 1], dtype=torch.float)

    model   = GraphSenseGAT()
    loss_fn = DualLoss(alpha=0.0, pos_weight=1.14, gamma=1.0)

    recon, classify = model(x, edge_index, edge_weight, batch)
    node_targets    = x.view(batch_size, -1)   # (2, 104)
    loss, l_r, l_c  = loss_fn(recon, classify, node_targets, labels)

    print(f"recon shape    : {recon.shape}")       # (2, 104)
    print(f"classify shape : {classify.shape}")    # (2, 1)
    print(f"loss           : {loss.item():.4f}")
    print(f"  L_recon      : {l_r.item():.4f}")
    print(f"  L_classify   : {l_c.item():.4f}")
    print("✅ GAT model v4 sanity check passed.")
