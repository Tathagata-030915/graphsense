# ============================================================
# GraphSense — Kaggle Training Notebook v2
# Improvements: Focal loss + WeightedRandomSampler + corrected
#               accuracy tracking threshold (0.35 not 0.5)
#
# Run cells in order. Each cell is marked with its number.
# Do NOT skip cells — each depends on the previous.
# ============================================================


# ============================================================
# CELL 1 — Install dependencies
# ============================================================
import subprocess
subprocess.run([
    "pip", "install", "--quiet",
    "torch-geometric", "pyarrow", "wandb"
], check=True)
print("✅ Dependencies installed.")


# ============================================================
# CELL 2 — Clone repo + set PYTHONPATH
# ============================================================
import sys
import os

REPO_URL = "https://github.com/Tathagata-030915/graphsense.git"
REPO_DIR = "/kaggle/working/graphsense"

if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print("✅ Repo cloned.")
else:
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
    print("✅ Repo pulled (already exists).")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

print(f"PYTHONPATH includes: {REPO_DIR}")


# ============================================================
# CELL 3 — Symlink data + run pipeline
# ============================================================
import subprocess
from pathlib import Path

RAW_DATA_SRC = "/kaggle/input/datasets/tathagataghosh03/anomaly-detection-gat/data/raw"
RAW_DATA_DST = "/kaggle/working/graphsense/data/raw"

os.makedirs("/kaggle/working/graphsense/data", exist_ok=True)

if not os.path.exists(RAW_DATA_DST):
    os.symlink(RAW_DATA_SRC, RAW_DATA_DST)
    print(f"✅ Symlink created: {RAW_DATA_DST} → {RAW_DATA_SRC}")
else:
    print("✅ Symlink already exists.")

from src.pipeline.loader import load_skab
from src.pipeline.features import create_windows
from src.pipeline.graph_builder import build_graph_dataset

df                                       = load_skab(data_dir=Path(RAW_DATA_DST))
feature_df, labels_out                   = create_windows(df)
node_features_list, adj_list, labels_out = build_graph_dataset(
    feature_df, labels_out, threshold=0.1
)

print(f"\n✅ Pipeline complete.")
print(f"   Graphs : {len(node_features_list):,}")
print(f"   Labels : {labels_out.sum()} anomalous / {(labels_out==0).sum()} normal")


# ============================================================
# CELL 4 — Build PyG dataset
# ============================================================
import torch
import numpy as np
from torch_geometric.data import Data

def build_pyg_dataset(node_features_list, adj_list, labels):
    from src.pipeline.graph_builder import adjacency_to_edge_index
    data_list = []
    for nf, adj, label in zip(node_features_list, adj_list, labels):
        edge_index, edge_weight = adjacency_to_edge_index(adj)
        data = Data(
            x           = torch.tensor(nf, dtype=torch.float),
            edge_index  = torch.tensor(edge_index, dtype=torch.long),
            edge_weight = torch.tensor(edge_weight, dtype=torch.float),
            y           = torch.tensor(label, dtype=torch.long),
        )
        data_list.append(data)
    return data_list

pyg_dataset = build_pyg_dataset(node_features_list, adj_list, labels_out)
print(f"✅ PyG dataset built: {len(pyg_dataset)} graphs")
print(f"   Sample: {pyg_dataset[0]}")


# ============================================================
# CELL 4b — Normalize node features
# ============================================================
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Flatten all node features to (n_graphs, 72)
all_node_feats = np.stack([
    pyg_dataset[i].x.numpy().flatten() for i in range(len(pyg_dataset))
])

# Split indices — fit scaler on train only to avoid data leakage
indices = np.arange(len(pyg_dataset))
train_idx, temp_idx = train_test_split(
    indices, test_size=0.30, random_state=42, stratify=labels_out
)
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, random_state=42, stratify=labels_out[temp_idx]
)

scaler = StandardScaler()
scaler.fit(all_node_feats[train_idx])

for i, data in enumerate(pyg_dataset):
    raw    = data.x.numpy().flatten().reshape(1, -1)
    data.x = torch.tensor(scaler.transform(raw).reshape(8, 9), dtype=torch.float)

# Verify normalization worked
normalized_check = np.stack([
    pyg_dataset[i].x.numpy().flatten() for i in train_idx
])
print(f"✅ Node features normalized.")
print(f"   Mean of train features (should be ~0): {normalized_check.mean():.4f}")
sample_normalized = pyg_dataset[train_idx[0]].x.numpy()
print(f"   Sample node feature range after norm: "
      f"[{sample_normalized.min():.2f}, {sample_normalized.max():.2f}]")


# ============================================================
# CELL 5 — Build DataLoaders with WeightedRandomSampler
# ============================================================
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler

BATCH_SIZE = 32

train_data = [pyg_dataset[i] for i in train_idx]
val_data   = [pyg_dataset[i] for i in val_idx]
test_data  = [pyg_dataset[i] for i in test_idx]

# ── Compute class weights for the training set ───────────────────────────────
# Purpose: oversample anomalies so each batch sees ~equal class representation.
# This does NOT change the dataset — it changes how often each sample is drawn.
train_labels  = np.array([pyg_dataset[i].y.item() for i in train_idx])
n_normal      = (train_labels == 0).sum()
n_anomaly     = (train_labels == 1).sum()
weight_normal = 1.0 / n_normal
weight_anomaly= 1.0 / n_anomaly

# Assign weight to each training sample
sample_weights = np.where(train_labels == 1, weight_anomaly, weight_normal)
sample_weights = torch.tensor(sample_weights, dtype=torch.float)

# WeightedRandomSampler draws len(train_data) samples per epoch
# with replacement, weighted by sample_weights
sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(train_data),
    replacement = True,
)

# IMPORTANT: shuffle=False when using a sampler — sampler handles ordering
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                          sampler=sampler, shuffle=False)
val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

# Compute pos_weight for focal loss: n_normal / n_anomaly
# This tells the loss how much more to penalize missing an anomaly
pos_weight = n_normal / n_anomaly

print(f"✅ Split complete.")
print(f"   Train : {len(train_data):,} | Val : {len(val_data):,} | Test : {len(test_data):,}")
print(f"   Train labels — Normal: {n_normal} | Anomaly: {n_anomaly}")
print(f"   pos_weight for focal loss: {pos_weight:.4f}")


# ============================================================
# CELL 6 — DEVICE + run_epoch
# ============================================================
from src.models.gat_model_focal_loss_v2 import GraphSenseGAT, DualLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSIFY_THRESHOLD = 0.35   # tuned from ablation — use consistently everywhere
print(f"Device: {DEVICE}")
print(f"Classification threshold: {CLASSIFY_THRESHOLD}")

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = recon_loss = class_loss = 0.0
    correct = total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch        = batch.to(DEVICE)
            node_targets = batch.x.view(batch.num_graphs, -1)
            recon, classify = model(
                batch.x, batch.edge_index, batch.edge_weight, batch.batch
            )
            loss, l_r, l_c = loss_fn(
                recon, classify, node_targets, batch.y.float()
            )
            if train:
                optim.zero_grad()
                loss.backward()
                # Gradient clipping — prevents exploding gradients with focal loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()

            total_loss += loss.item()
            recon_loss += l_r.item()
            class_loss += l_c.item()

            # FIX: use tuned threshold (0.35), not default 0.5
            # Previous version used 0.5 here — logged accuracy was misleading
            preds    = (torch.sigmoid(classify.squeeze()) > CLASSIFY_THRESHOLD).long()
            correct += (preds == batch.y).sum().item()
            total   += batch.num_graphs

    n = len(loader)
    return total_loss/n, recon_loss/n, class_loss/n, correct/total


# ============================================================
# CELL 7 — Train
# ============================================================
import wandb
import matplotlib.pyplot as plt

wandb.init(
    project = "graphsense",
    name    = "gat-focal-loss-weighted-sampler",
    config  = {
        "hidden_dim"       : 64,
        "num_heads"        : 4,
        "dropout"          : 0.3,
        "alpha"            : 0.0,
        "threshold"        : 0.1,
        "batch_size"       : BATCH_SIZE,
        "epochs"           : 75,
        "lr"               : 1e-3,
        "focal_gamma"      : 2.0,
        "pos_weight"       : pos_weight,
        "classify_threshold": CLASSIFY_THRESHOLD,
        "sampler"          : "WeightedRandomSampler",
        "grad_clip"        : 1.0,
    }
)

# pos_weight computed in Cell 5 from training split
model   = GraphSenseGAT().to(DEVICE)
loss_fn = DualLoss(alpha=0.0, pos_weight=pos_weight, gamma=2.0)
optim   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, mode="min", patience=7, factor=0.5
)

EPOCHS        = 75
best_val_loss = float("inf")
history       = {"train_loss": [], "val_loss": [],
                 "train_acc":  [], "val_acc":  []}

for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_r, tr_c, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_r, vl_c, vl_acc = run_epoch(val_loader,   train=False)

    scheduler.step(vl_loss)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    wandb.log({
        "epoch"      : epoch,
        "train/loss" : tr_loss,
        "train/acc"  : tr_acc,
        "val/loss"   : vl_loss,
        "val/acc"    : vl_acc,
        "lr"         : optim.param_groups[0]["lr"],
    })

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | "
              f"Train Loss {tr_loss:.4f} Acc {tr_acc:.3f} | "
              f"Val Loss {vl_loss:.4f} Acc {vl_acc:.3f}")

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        torch.save(model.state_dict(), "/kaggle/working/best_model.pt")

print(f"\n✅ Training complete. Best val loss: {best_val_loss:.4f}")
wandb.finish()

# Plot training curves
epochs_range = range(1, EPOCHS + 1)
fig, axes    = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "GraphSense GAT v2 — Focal Loss + Weighted Sampler (thresh=0.1, α=0.0)",
    fontsize=13
)

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", linewidth=2)
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",   linewidth=2)
axes[0].set_title("Loss per Epoch")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], label="Train Acc", linewidth=2)
axes[1].plot(epochs_range, history["val_acc"],   label="Val Acc",   linewidth=2)
axes[1].axhline(y=0.696, color="red", linestyle="--",
                alpha=0.6, label="Majority baseline (0.696)")
axes[1].set_title("Accuracy per Epoch (thresh=0.35)")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/kaggle/working/training_curves_v2.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Training curves saved.")


# ============================================================
# CELL 8 — Evaluation on test set
# ============================================================
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

model.load_state_dict(torch.load("/kaggle/working/best_model.pt"))
model.eval()

all_preds  = []
all_probs  = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(DEVICE)
        _, classify = model(
            batch.x, batch.edge_index, batch.edge_weight, batch.batch
        )
        probs  = torch.sigmoid(classify.squeeze()).cpu().numpy()
        preds  = (probs > CLASSIFY_THRESHOLD).astype(int)
        labels = batch.y.cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

all_probs  = np.array(all_probs)
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

print("=" * 60)
print("TEST SET RESULTS — GraphSense GAT v2")
print("Focal Loss + WeightedRandomSampler (thresh=0.1, α=0.0)")
print("=" * 60)
print(classification_report(all_labels, all_preds,
      target_names=["Normal", "Anomaly"]))
print(f"ROC-AUC : {roc_auc_score(all_labels, all_probs):.4f}")
print(f"F1      : {f1_score(all_labels, all_preds):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")

# ROC + PR curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("GraphSense GAT v2 — Test Set Curves", fontsize=13)

fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc_score   = roc_auc_score(all_labels, all_probs)
axes[0].plot(fpr, tpr, linewidth=2, label=f"GAT v2 (AUC = {auc_score:.4f})")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

prec, rec, _ = precision_recall_curve(all_labels, all_probs)
axes[1].plot(rec, prec, linewidth=2, label="GAT v2")
axes[1].axhline(y=all_labels.mean(), color="red", linestyle="--",
                alpha=0.6, label=f"Baseline (anomaly rate={all_labels.mean():.2f})")
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/kaggle/working/roc_pr_curves_v2.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ ROC + PR curves saved.")


# ============================================================
# CELL 9 — Isolation Forest baseline (same test split)
# ============================================================
from sklearn.ensemble import IsolationForest

X_train = all_node_feats[train_idx]
X_test  = all_node_feats[test_idx]
y_test  = np.array([pyg_dataset[i].y.item() for i in test_idx])
y_train = np.array([pyg_dataset[i].y.item() for i in train_idx])

X_train_norm = scaler.transform(X_train)
X_test_norm  = scaler.transform(X_test)

contamination = y_train.mean()
print(f"Contamination (anomaly rate in train): {contamination:.4f}")

iso = IsolationForest(
    n_estimators  = 200,
    contamination = contamination,
    random_state  = 42,
    n_jobs        = -1,
)
iso.fit(X_train_norm)

iso_preds  = (iso.predict(X_test_norm) == -1).astype(int)
iso_scores = -iso.score_samples(X_test_norm)

print("=" * 55)
print("ISOLATION FOREST BASELINE — Same Test Split")
print("=" * 55)
print(classification_report(y_test, iso_preds,
      target_names=["Normal", "Anomaly"]))
print(f"ROC-AUC : {roc_auc_score(y_test, iso_scores):.4f}")
print(f"F1      : {f1_score(y_test, iso_preds):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, iso_preds)}")

print("\n--- HEAD TO HEAD (v1 vs v2 vs IsoForest) ---")
print(f"{'Metric':<12} {'IsoForest':>12} {'GAT v1':>12} {'GAT v2':>12}")
print("-" * 50)
gat_auc = roc_auc_score(all_labels, all_probs)
gat_f1  = f1_score(all_labels, all_preds)
# v1 numbers for reference (from knowledge transfer doc)
print(f"{'ROC-AUC':<12} {roc_auc_score(y_test, iso_scores):>12.4f} {'0.8110':>12} {gat_auc:>12.4f}")
print(f"{'F1':<12} {f1_score(y_test, iso_preds):>12.4f} {'0.6157':>12} {gat_f1:>12.4f}")


# ============================================================
# CELL 10 — Ablation study (unchanged, uses original DualLoss)
# ============================================================

def train_and_eval(alpha, threshold, epochs=30):
    from sklearn.preprocessing import StandardScaler

    nfl, al, lo = build_graph_dataset(feature_df, labels_out, threshold=threshold)
    ds          = build_pyg_dataset(nfl, al, lo)

    all_feats = np.stack([ds[i].x.numpy().flatten() for i in range(len(ds))])
    sc = StandardScaler()
    sc.fit(all_feats[train_idx])
    for i, data in enumerate(ds):
        raw    = data.x.numpy().flatten().reshape(1, -1)
        data.x = torch.tensor(sc.transform(raw).reshape(8, 9), dtype=torch.float)

    assert len(ds) > train_idx.max(), "Index out of range"
    assert len(ds) > test_idx.max(),  "Index out of range"

    # Compute pos_weight for this split
    tr_labels  = np.array([ds[i].y.item() for i in train_idx])
    pw         = (tr_labels == 0).sum() / max((tr_labels == 1).sum(), 1)

    # Weighted sampler for ablation too
    sw = np.where(tr_labels == 1, 1.0 / (tr_labels==1).sum(),
                                  1.0 / (tr_labels==0).sum())
    sw = torch.tensor(sw, dtype=torch.float)
    sa = WeightedRandomSampler(sw, len(train_idx), replacement=True)

    tr_l = DataLoader([ds[i] for i in train_idx], batch_size=BATCH_SIZE,
                      sampler=sa, shuffle=False)
    te_l = DataLoader([ds[i] for i in test_idx],  batch_size=BATCH_SIZE, shuffle=False)

    m  = GraphSenseGAT().to(DEVICE)
    lf = DualLoss(alpha=alpha, pos_weight=pw, gamma=2.0)
    op = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)

    m.train()
    for _ in range(epochs):
        for batch in tr_l:
            batch = batch.to(DEVICE)
            nt    = batch.x.view(batch.num_graphs, -1)
            r, c  = m(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            loss, _, _ = lf(r, c, nt, batch.y.float())
            op.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
            op.step()

    m.eval()
    probs_all, labs_all = [], []
    with torch.no_grad():
        for batch in te_l:
            batch = batch.to(DEVICE)
            _, c  = m(batch.x, batch.edge_index, batch.edge_weight, batch.batch)
            probs_all.extend(torch.sigmoid(c.squeeze()).cpu().numpy())
            labs_all.extend(batch.y.cpu().numpy())

    preds = (np.array(probs_all) > CLASSIFY_THRESHOLD).astype(int)
    return f1_score(np.array(labs_all), preds)


print("Running ablation study... (~10-15 min)")
print("=" * 55)

alpha_results = {}
for alpha in [1.0, 0.0, 0.5]:
    label = {1.0: "Recon only", 0.0: "Classify only", 0.5: "Dual head"}[alpha]
    f1    = train_and_eval(alpha=alpha, threshold=0.1)
    alpha_results[label] = f1
    print(f"  {label:<15} | F1 = {f1:.4f}")

print()

thresh_results = {}
for thresh in [0.1, 0.2, 0.3, 0.5]:
    f1 = train_and_eval(alpha=0.0, threshold=thresh)
    thresh_results[f"thresh={thresh}"] = f1
    print(f"  threshold={thresh}     | F1 = {f1:.4f}")

print("\n✅ Ablation study complete.")
