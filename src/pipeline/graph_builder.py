# src/pipeline/graph_builder.py
#
# v3 changes:
#   node_feature_suffixes expanded from 9 → 13 entries
#   Added: _zcr, _rms, _autocorr_lag1, _iqr
#   Print statement updated to reflect new shape (8 x 13)

import numpy as np
import pandas as pd
from typing import Tuple, List
from src.pipeline.loader import SENSOR_COLUMNS


# ── Configuration ─────────────────────────────────────────────────────────────

CORRELATION_THRESHOLD = 0.3   # minimum absolute correlation to draw an edge
META_COLS             = ["file_id", "window_start", "scenario"]


# ── Core Graph Builder ────────────────────────────────────────────────────────

def build_adjacency_matrix(
    window_features : pd.Series,
    threshold       : float = CORRELATION_THRESHOLD,
) -> np.ndarray:
    """
    Builds a dynamic adjacency matrix for a single window using
    precomputed pairwise Pearson correlations stored in feature_df.

    This is the KEY NOVELTY — edges are not fixed. They are recomputed
    per window based on sensor correlations observed in that window.
    A high correlation between two sensors means they share an edge.

    Parameters
    ----------
    window_features : One row of feature_df (a pd.Series)
    threshold       : Minimum |correlation| to create an edge

    Returns
    -------
    adj : np.ndarray of shape (n_sensors, n_sensors)
          Weighted adjacency matrix. adj[i][j] = correlation weight,
          0 if below threshold.
    """
    n = len(SENSOR_COLUMNS)
    adj = np.zeros((n, n), dtype=np.float32)

    # Self-loops — every sensor attends to itself
    np.fill_diagonal(adj, 1.0)

    for i in range(n):
        for j in range(i + 1, n):
            key = f"corr_{SENSOR_COLUMNS[i]}__{SENSOR_COLUMNS[j]}"

            if key not in window_features.index:
                continue

            corr_val = float(window_features[key])

            if np.isnan(corr_val):
                continue

            # Only draw edge if correlation exceeds threshold
            if abs(corr_val) >= threshold:
                adj[i][j] = corr_val
                adj[j][i] = corr_val   # undirected graph — symmetric

    return adj


def build_graph_dataset(
    feature_df : pd.DataFrame,
    labels     : np.ndarray,
    threshold  : float = CORRELATION_THRESHOLD,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Converts the full feature DataFrame into a list of graph objects.
    Each window becomes one graph with:
        - Node features : 13 features per sensor (7 stat + 2 freq + 4 extended)
        - Adjacency matrix : dynamic, correlation-based
        - Label : 0 or 1

    Parameters
    ----------
    feature_df : DataFrame from create_windows()
    labels     : np.ndarray of shape (n_windows,)
    threshold  : Correlation threshold for edge creation

    Returns
    -------
    node_features_list : List of np.ndarray, each shape (n_sensors, 13)
    adj_list           : List of np.ndarray, each shape (n_sensors, n_sensors)
    labels             : np.ndarray of shape (n_windows,)
    """
    node_features_list = []
    adj_list           = []

    # Features per sensor node: 7 statistical + 2 frequency + 4 extended = 13
    # ORDER MATTERS — must match IN_FEATURES=13 in gat_model_focal_loss_v2.py
    # Do NOT reorder these without updating IN_FEATURES
    node_feature_suffixes = [
        # Original 9
        "_mean",
        "_std",
        "_min",
        "_max",
        "_skew",
        "_kurtosis",
        "_slope",
        "_dominant_freq",
        "_spectral_energy",
        # New 4
        "_zcr",           # zero/mean-crossing rate — oscillation frequency
        "_rms",           # root mean square energy — absolute magnitude
        "_autocorr_lag1", # lag-1 autocorrelation — temporal predictability
        "_iqr",           # interquartile range — robust spread
    ]

    for idx, row in feature_df.iterrows():
        # ── Build node feature matrix ──────────────────────────────────────
        # Shape: (n_sensors=8, n_node_features=13)
        node_feats = np.zeros(
            (len(SENSOR_COLUMNS), len(node_feature_suffixes)),
            dtype=np.float32
        )

        for s_idx, sensor in enumerate(SENSOR_COLUMNS):
            for f_idx, suffix in enumerate(node_feature_suffixes):
                col = f"{sensor}{suffix}"
                if col in row.index:
                    val = row[col]
                    node_feats[s_idx, f_idx] = 0.0 if np.isnan(val) else float(val)

        # ── Build adjacency matrix ─────────────────────────────────────────
        adj = build_adjacency_matrix(row, threshold=threshold)

        node_features_list.append(node_feats)
        adj_list.append(adj)

    print(f"[graph] Built {len(node_features_list):,} graphs")
    print(f"[graph] Node feature shape : (8 sensors x 13 features)")
    print(f"[graph] Adj matrix shape   : (8 x 8), threshold={threshold}")
    print(f"[graph] Avg edges per graph: {_avg_edges(adj_list):.2f} "
          f"(excl. self-loops)")

    return node_features_list, adj_list, labels


# ── Utility ───────────────────────────────────────────────────────────────────

def _avg_edges(adj_list: List[np.ndarray]) -> float:
    """Compute average number of non-self-loop edges across all graphs."""
    total = 0
    for adj in adj_list:
        mask  = np.triu(np.ones_like(adj, dtype=bool), k=1)
        total += int((adj[mask] != 0).sum())
    return total / max(len(adj_list), 1)


def adjacency_to_edge_index(adj: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert adjacency matrix to PyG-style edge_index and edge_weight.
    Used when building PyTorch Geometric Data objects on Kaggle.

    Returns
    -------
    edge_index  : np.ndarray of shape (2, n_edges) — [source, target] pairs
    edge_weight : np.ndarray of shape (n_edges,)   — correlation weights
    """
    rows, cols  = np.where(adj != 0)
    edge_index  = np.stack([rows, cols], axis=0)
    edge_weight = adj[rows, cols]
    return edge_index, edge_weight


def graph_stats(adj_list: List[np.ndarray]) -> pd.DataFrame:
    """
    Returns a summary DataFrame of graph statistics across all windows.
    Useful for EDA and verifying graph quality.
    """
    records = []
    for adj in adj_list:
        mask        = np.triu(np.ones_like(adj, dtype=bool), k=1)
        edges       = adj[mask]
        n_edges     = int((edges != 0).sum())
        avg_weight  = float(np.abs(edges[edges != 0]).mean()) if n_edges > 0 else 0.0
        records.append({"n_edges": n_edges, "avg_edge_weight": avg_weight})

    return pd.DataFrame(records)
