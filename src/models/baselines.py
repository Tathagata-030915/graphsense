# src/models/baselines.py

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle


# ── Constants ─────────────────────────────────────────────────────────────────

META_COLS = ["file_id", "window_start", "scenario"]

SENSOR_COLS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]


# ── Shared Utilities ──────────────────────────────────────────────────────────

def get_feature_matrix(feature_df: pd.DataFrame) -> np.ndarray:
    """Strip metadata columns and return pure feature matrix."""
    feature_cols = [c for c in feature_df.columns if c not in META_COLS]
    return feature_df[feature_cols].values.astype(np.float32)


# ── Baseline 1: Isolation Forest ──────────────────────────────────────────────

class IsolationForestBaseline:
    """
    Isolation Forest for unsupervised anomaly detection.

    Trained ONLY on normal (label=0) windows to simulate real deployment
    where labeled anomaly data is unavailable upfront.
    """

    def __init__(self, contamination: float = 0.30, random_state: int = 42):
        self.contamination = contamination
        self.scaler        = StandardScaler()
        self.model         = IsolationForest(
            n_estimators  = 200,
            contamination = contamination,
            random_state  = random_state,
            n_jobs        = -1,
        )
        self.fitted = False

    def fit(self, feature_df: pd.DataFrame, labels: np.ndarray) -> None:
        X = get_feature_matrix(feature_df)

        # Train ONLY on normal windows
        normal_mask = labels == 0
        X_normal    = X[normal_mask]

        X_scaled = self.scaler.fit_transform(X_normal)
        self.model.fit(X_scaled)
        self.fitted = True

        print(f"[IsoForest] Trained on {X_normal.shape[0]:,} normal windows "
              f"| {X_normal.shape[1]} features")

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Returns binary predictions: 1=anomaly, 0=normal."""
        assert self.fitted, "Call fit() before predict()"
        X        = get_feature_matrix(feature_df)
        X_scaled = self.scaler.transform(X)
        # IsolationForest returns -1=anomaly, 1=normal — flip to 0/1
        raw_preds = self.model.predict(X_scaled)
        return (raw_preds == -1).astype(np.int32)

    def anomaly_scores(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Returns anomaly scores — higher means more anomalous."""
        assert self.fitted, "Call fit() before anomaly_scores()"
        X        = get_feature_matrix(feature_df)
        X_scaled = self.scaler.transform(X)
        return -self.model.score_samples(X_scaled)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model}, f)
        print(f"[IsoForest] Saved to {path}")

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.model  = data["model"]
        self.fitted = True