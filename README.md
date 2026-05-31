# GraphSense: Sensor-Aware Anomaly Detection on Multivariate IoT Streams using Adaptive Graph Neural Networks

## Overview

GraphSense is a Graph Attention Network (GAT) based anomaly detection system for multivariate IoT sensor streams. The core idea is simple but effective - instead of treating sensors as independent signals, we model them as a dynamic graph where edges are recomputed per time window based on observed Pearson correlations between sensors. A GAT then performs attention-weighted message passing over this dynamic graph to detect anomalies.

**Dataset:** SKAB (Skoltech Anomaly Benchmark) - 35 CSV files, 46,806 rows, 8 sensors, 4 scenarios (`anomaly-free`, `valve1`, `valve2`, `other`)

**Task:** Binary anomaly detection (normal vs anomaly) on sliding windows of sensor data.

---

## Key Results

| Metric | IsolationForest | GraphSense GAT | Improvement |
|----------|----------|----------|----------|
| ROC-AUC | 0.6048 | **0.8110** | +0.206 (+34%) |
| F1 | 0.4282 | **0.6157** | +0.188 (+44%) |
| Anomaly Recall | 0.43 | **0.77** | +34pp |
| Anomaly Precision | 0.42 | **0.51** | +0.09 |

Evaluated on the same stratified test split (15% of 4,595 windows = 690 samples, 210 anomalous).

---

## The Novelty

Most GNN-based anomaly detection methods use a **fixed** graph topology — edges are determined once at dataset construction time. GraphSense uses **dynamic, per-window graphs**:

- For each sliding window, pairwise Pearson correlations are computed across all 8 sensors
- An edge is drawn between two sensors only if `|correlation| ≥ threshold`
- The edge weight equals the correlation value
- This means the graph structure changes every window — capturing transient inter-sensor dependencies that static graphs miss

The GAT then applies a second level of adaptive weighting via attention — so you get correlation-based edges AND attention-weighted message passing on top.

---

## Ablation Study

**Alpha sweep** (controls reconstruction vs classification loss weight, threshold=0.1):

| Configuration | F1 |
|---|---|
| Recon only (α=1.0) | 0.4667 |
| Dual head (α=0.5) | 0.5611 |
| Classify only (α=0.0) | **0.5793** |

Classification-only objective performs best. Reconstruction loss adds no benefit when labels are available.

**Threshold sweep** (controls edge density, α=0.0):

| Threshold | Avg Edges | F1 |
|---|---|---|
| 0.1 | 18.27 | **0.6108** |
| 0.2 | 10.54 | 0.5484 |
| 0.3 | 5.51 | 0.5901 |
| 0.5 | 1.51 | 0.5914 |

Denser graphs (lower threshold) outperform sparse ones — more inter-sensor connections preserve more anomaly signal.

---

## Architecture

```text
Input: node_features (8 nodes × 9 features), edge_index, edge_weight
↓
[GATConv Layer 1] — 4-head attention, hidden_dim=64 → (8 nodes × 256)
↓
[GATConv Layer 2] — 1-head refinement → (8 nodes × 64)
↓
[Global Mean Pool] → graph embedding (64,)
↓
[Classification Head] → sigmoid → P(anomaly)
```

Node features per sensor (9 total): mean, std, min, max, skewness, kurtosis, slope, dominant frequency, spectral energy — computed over a 60-second sliding window with 50% overlap.

---

## Project Structure

```text
graphsense/
├── src/
│   ├── pipeline/
│   │   ├── loader.py          # Load and tag SKAB CSVs
│   │   ├── features.py        # Sliding window feature extraction
│   │   └── graph_builder.py   # Dynamic adjacency matrix construction
│   ├── models/
│   │   ├── baselines.py       # Isolation Forest baseline
│   │   └── gat_model.py       # GraphSenseGAT + DualLoss
│   └── evaluation/
├── notebooks/
│   └── kaggle_training.py     # Full training pipeline (run on Kaggle)
│   └── anomaly-gat-version1.ipynb
│   └── anomaly-train-gat_Version_2_thresh_0.1.ipynb
│   └── anomaly-train-gat_version_3_fix_cell4b.ipynb
├── tests/
│   └── test_pipeline.py       # Unit tests for all pipeline steps
├── requirements.txt
└── README.md
└── ablation_study_from_version1_ipynb.txt
└── result_isolationForest_GraphBuilder_from_version1_ipynb.txt
```

---

## How to Run

### Local (pipeline only — no GPU required)

```bash
git clone https://github.com/Tathagata-030915/graphsense.git
cd graphsense
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python tests/test_pipeline.py
```

### Kaggle (training — GPU required)

1. Upload SKAB dataset to Kaggle as a dataset
2. Clone this repo in a Kaggle notebook
3. Run `notebooks/kaggle_training.py` cell by cell
4. Enable GPU accelerator in notebook settings

Dataset used: [SKAB — Skoltech Anomaly Benchmark](https://github.com/waico/SKAB)

---

## Requirements

```text
pandas
numpy
scikit-learn
scipy
pyarrow
torch          # Kaggle only
torch-geometric # Kaggle only
wandb          # Kaggle only
```

---

## Optimal Hyperparameters

| Parameter | Value |
|---|---|
| Correlation threshold | 0.1 |
| Hidden dim | 64 |
| Attention heads | 4 |
| Dropout | 0.3 |
| Loss alpha | 0.0 (classify only) |
| Batch size | 32 |
| Epochs | 50 |
| Learning rate | 1e-3 |
| Classification threshold | 0.35 |

