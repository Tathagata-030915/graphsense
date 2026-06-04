# src/pipeline/features.py
#
# v3 changes:
#   Added 4 new per-sensor features: zcr, rms, autocorr_lag1, iqr
#   Node features per sensor: 9 → 13
#   IN_FEATURES in gat_model_focal_loss_v2.py must be updated to 13

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft
from typing import Tuple
from pathlib import Path

from src.pipeline.loader import SENSOR_COLUMNS


# ── Configuration ─────────────────────────────────────────────────────────────

WINDOW_SIZE = 30      # 30 time steps per window (~30 seconds in SKAB)
STEP_SIZE   = 10      # slide forward 10 steps (67% overlap)


# ── Statistical Features ──────────────────────────────────────────────────────

def _statistical_features(window: np.ndarray, col: str) -> dict:
    """
    Compute 7 statistical features for a single sensor window.
    window: 1D array of shape (WINDOW_SIZE,)
    """
    slope, _, _, _, _ = stats.linregress(np.arange(len(window)), window)

    return {
        f"{col}_mean"     : np.mean(window),
        f"{col}_std"      : np.std(window),
        f"{col}_min"      : np.min(window),
        f"{col}_max"      : np.max(window),
        f"{col}_skew"     : stats.skew(window),
        f"{col}_kurtosis" : stats.kurtosis(window),
        f"{col}_slope"    : slope,
    }


# ── Frequency Features ────────────────────────────────────────────────────────

def _frequency_features(window: np.ndarray, col: str) -> dict:
    """
    Compute 2 FFT-based features for a single sensor window.
    Captures periodic patterns that statistical features miss.
    """
    fft_vals  = np.abs(fft(window))
    fft_freqs = fft_vals[:len(fft_vals) // 2]  # take positive frequencies only

    dominant_freq    = np.argmax(fft_freqs[1:]) + 1  # skip DC component
    spectral_energy  = np.sum(fft_freqs ** 2)

    return {
        f"{col}_dominant_freq"   : dominant_freq,
        f"{col}_spectral_energy" : spectral_energy,
    }


# ── NEW: Extended Features ─────────────────────────────────────────────────────

def _extended_features(window: np.ndarray, col: str) -> dict:
    """
    Compute 4 additional features per sensor window.

    Features added:
    ──────────────────────────────────────────────────────────────
    zcr (zero-crossing rate):
        How often the signal crosses its mean.
        High ZCR → rapid oscillation (often seen in vibration anomalies).
        We use mean-crossing not zero-crossing because sensor signals
        are not zero-centered — mean-crossing is more meaningful.

    rms (root mean square):
        sqrt(mean(x^2)) — captures signal energy/magnitude.
        Differs from std because it captures absolute magnitude,
        not just spread. A shifted anomaly (constant offset) has
        high RMS but similar std to normal — std misses it, RMS catches it.

    autocorr_lag1 (autocorrelation at lag 1):
        Pearson correlation of the signal with itself shifted by 1 step.
        Normal sensor readings are often autocorrelated (smooth trends).
        Anomalies often break this autocorrelation suddenly.
        Range: [-1, 1]. Values near 0 = unpredictable/chaotic signal.

    iqr (interquartile range):
        Q75 - Q25. Robust spread measure unlike std/range which
        are sensitive to outliers. Captures the "typical" spread
        of the middle 50% of readings. Anomalies often inflate IQR
        even when std is not extreme.
    ──────────────────────────────────────────────────────────────
    """
    # Zero-crossing rate (mean-crossing)
    mean_centered = window - np.mean(window)
    zcr = float(np.sum(np.diff(np.sign(mean_centered)) != 0)) / len(window)

    # RMS energy
    rms = float(np.sqrt(np.mean(window ** 2)))

    # Autocorrelation at lag 1
    if len(window) > 1 and np.std(window) > 0:
        # Pearson correlation between x[t] and x[t+1]
        autocorr_lag1 = float(np.corrcoef(window[:-1], window[1:])[0, 1])
        if np.isnan(autocorr_lag1):
            autocorr_lag1 = 0.0
    else:
        autocorr_lag1 = 0.0

    # IQR
    iqr = float(np.percentile(window, 75) - np.percentile(window, 25))

    return {
        f"{col}_zcr"           : zcr,
        f"{col}_rms"           : rms,
        f"{col}_autocorr_lag1" : autocorr_lag1,
        f"{col}_iqr"           : iqr,
    }


# ── Correlation Matrix ────────────────────────────────────────────────────────

def _correlation_features(window_df: pd.DataFrame) -> dict:
    """
    Compute pairwise Pearson correlations between all sensor pairs.
    These become the edge weights in our graph later.
    Returns upper triangle only to avoid redundancy.
    """
    corr_matrix = window_df[SENSOR_COLUMNS].corr(method="pearson")
    features = {}

    sensors = SENSOR_COLUMNS
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            key = f"corr_{sensors[i]}__{sensors[j]}"
            val = corr_matrix.iloc[i, j]
            features[key] = 0.0 if np.isnan(val) else val

    return features


# ── Label Extraction ──────────────────────────────────────────────────────────

def _extract_label(window_df: pd.DataFrame) -> int:
    """
    A window is anomalous if ANY row in it is labeled anomalous.
    This is conservative — flags the whole window if anomaly exists anywhere.
    """
    return int(window_df["anomaly"].any())


# ── Main Windowing Function ───────────────────────────────────────────────────

def create_windows(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Slides a window over the combined DataFrame (per file_id to avoid
    bleeding across different experiment runs) and extracts features.

    Parameters
    ----------
    df          : Combined DataFrame from load_skab()
    window_size : Number of timesteps per window
    step_size   : Stride of the sliding window

    Returns
    -------
    feature_df  : DataFrame of shape (n_windows, n_features)
    labels      : np.ndarray of shape (n_windows,) with 0/1 anomaly labels
    """
    all_features = []
    all_labels   = []

    file_ids = df["file_id"].unique()

    for file_id in file_ids:
        file_df = df[df["file_id"] == file_id].reset_index(drop=True)

        # Skip files too short for even one window
        if len(file_df) < window_size:
            print(f"[features] Skipping {file_id}: only {len(file_df)} rows "
                  f"(need {window_size})")
            continue

        n_windows = 0
        for start in range(0, len(file_df) - window_size + 1, step_size):
            end        = start + window_size
            window_df  = file_df.iloc[start:end]

            features = {}

            # Per-sensor statistical + frequency + extended features
            for col in SENSOR_COLUMNS:
                window_vals = window_df[col].values.astype(np.float64)

                # Handle edge case: constant signal (std=0 breaks some stats)
                if np.std(window_vals) == 0:
                    window_vals = window_vals + np.random.normal(0, 1e-10, len(window_vals))

                features.update(_statistical_features(window_vals, col))
                features.update(_frequency_features(window_vals, col))
                features.update(_extended_features(window_vals, col))  # NEW

            # Cross-sensor correlation features (used for edges, not node features)
            features.update(_correlation_features(window_df))

            # Metadata (not used in model, useful for debugging)
            features["file_id"]      = file_id
            features["window_start"] = start
            features["scenario"]     = window_df["scenario"].iloc[0]

            all_features.append(features)
            all_labels.append(_extract_label(window_df))
            n_windows += 1

    feature_df = pd.DataFrame(all_features)
    labels     = np.array(all_labels, dtype=np.int32)

    # Separate metadata from model features before returning
    meta_cols    = ["file_id", "window_start", "scenario"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    print(f"\n[features] Windows created : {len(feature_df):,}")
    print(f"[features] Feature columns : {len(feature_cols)}")
    print(f"[features] Anomaly rate    : {labels.mean()*100:.2f}%")
    print(f"[features] Meta columns    : {meta_cols}")
    print(f"[features] Node features per sensor: 13 (was 9)")

    return feature_df, labels


# ── Save / Load Processed Data ────────────────────────────────────────────────

def save_processed(
    feature_df : pd.DataFrame,
    labels     : np.ndarray,
    out_dir    : Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_dir / "features.parquet", index=False)
    np.save(out_dir / "labels.npy", labels)
    print(f"[features] Saved to {out_dir}")


def load_processed(out_dir: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    feature_df = pd.read_parquet(out_dir / "features.parquet")
    labels     = np.load(out_dir / "labels.npy")
    print(f"[features] Loaded {len(feature_df):,} windows from {out_dir}")
    return feature_df, labels
