# src/models/gat_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool


# ── Configuration ─────────────────────────────────────────────────────────────

IN_FEATURES   = 9     # per-node input features (7 stat + 2 freq)
HIDDEN_DIM    = 64    # hidden dimension for GAT layers
NUM_HEADS     = 4     # attention heads in GATConv
DROPOUT       = 0.3   # dropout rate
ALPHA         = 0.5   # joint loss weight: α*L_recon + (1-α)*L_classify


# ── GAT Model ─────────────────────────────────────────────────────────────────

class GraphSenseGAT(nn.Module):
    """
    Dual-head Graph Attention Network for IoT anomaly detection.

    Architecture:
        GATConv(9 → 64, heads=4)         # Layer 1: multi-head attention
        GATConv(256 → 64, heads=1)        # Layer 2: single-head refinement
        GlobalMeanPool                    # Graph-level embedding (64,)
            ├── Reconstruction head       # Decoder MLP → (8 × 9,) → MSE loss
            └── Classification head      # Linear → sigmoid → BCE loss

    The KEY NOVELTY is that the graph structure (edge_index, edge_weight)
    is different for every window — dynamic correlation-based edges.
    The GAT attention mechanism then FURTHER weights these edges per node.
    So you get two levels of adaptive weighting — correlation + attention.
    """

    def __init__(
        self,
        in_features : int = IN_FEATURES,
        hidden_dim  : int = HIDDEN_DIM,
        num_heads   : int = NUM_HEADS,
        dropout     : float = DROPOUT,
    ):
        super().__init__()

        self.dropout = dropout

        # ── GAT Layer 1 ───────────────────────────────────────────────────
        # in_features=9, out per head=hidden_dim, heads=4
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
        # Target shape: (n_nodes * in_features,) = (8 * 9,) = (72,)
        # We reconstruct flattened node features and compare to input
        self.recon_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 8 * in_features),   # 8 nodes × 9 features
        )

        # ── Classification Head ───────────────────────────────────────────
        # Takes graph embedding (64,) → binary anomaly probability
        self.classify_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),      # single logit
        )

    def forward(self, x, edge_index, edge_weight, batch):
        """
        Parameters
        ----------
        x           : Tensor (total_nodes, 9)     — node features
        edge_index  : Tensor (2, total_edges)     — edge connectivity
        edge_weight : Tensor (total_edges,)       — correlation weights
        batch       : Tensor (total_nodes,)       — graph assignment per node

        Returns
        -------
        recon_out   : Tensor (batch_size, 72)     — reconstructed node features
        classify_out: Tensor (batch_size, 1)      — anomaly logits
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


# ── Joint Loss ────────────────────────────────────────────────────────────────

class DualLoss(nn.Module):
    """
    Joint loss combining reconstruction (unsupervised) and
    classification (supervised) objectives.

    L_total = alpha * L_recon + (1 - alpha) * L_classify

    alpha=1.0  → pure reconstruction (unsupervised ablation)
    alpha=0.0  → pure classification (supervised ablation)
    alpha=0.5  → balanced dual head (default, main model)
    """

    def __init__(self, alpha: float = ALPHA):
        super().__init__()
        self.alpha   = alpha
        self.mse     = nn.MSELoss()
        self.bce     = nn.BCEWithLogitsLoss()

    def forward(
        self,
        recon_out    : torch.Tensor,   # (batch_size, 72)
        classify_out : torch.Tensor,   # (batch_size, 1)
        node_targets : torch.Tensor,   # (batch_size, 72) — flattened input
        labels       : torch.Tensor,   # (batch_size,)    — 0/1
    ):
        l_recon    = self.mse(recon_out, node_targets)
        l_classify = self.bce(classify_out.squeeze(-1), labels.float())
        l_total    = self.alpha * l_recon + (1 - self.alpha) * l_classify
        return l_total, l_recon, l_classify


# ── Sanity Check (run this cell on Kaggle to verify model loads) ──────────────

if __name__ == "__main__":
    # Fake a single batch of 2 graphs, each with 8 nodes
    batch_size = 2
    n_nodes    = 8
    n_edges    = 10   # arbitrary

    x           = torch.randn(batch_size * n_nodes, IN_FEATURES)
    edge_index  = torch.randint(0, n_nodes, (2, n_edges))
    edge_weight = torch.rand(n_edges)
    batch       = torch.repeat_interleave(torch.arange(batch_size), n_nodes)
    labels      = torch.tensor([0, 1], dtype=torch.float)

    model    = GraphSenseGAT()
    loss_fn  = DualLoss(alpha=ALPHA)

    recon, classify = model(x, edge_index, edge_weight, batch)

    node_targets = x.view(batch_size, -1)   # (2, 72)
    loss, l_r, l_c = loss_fn(recon, classify, node_targets, labels)

    print(f"recon shape    : {recon.shape}")       # (2, 72)
    print(f"classify shape : {classify.shape}")    # (2, 1)
    print(f"loss           : {loss.item():.4f}")
    print(f"  L_recon      : {l_r.item():.4f}")
    print(f"  L_classify   : {l_c.item():.4f}")
    print("✅ GAT model sanity check passed.")