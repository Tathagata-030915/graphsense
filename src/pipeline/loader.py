# src/pipeline/loader.py

import os
import pandas as pd
from pathlib import Path
from typing import Optional


# ── Constants ────────────────────────────────────────────────────────────────

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SENSOR_COLUMNS = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]

LABEL_COLUMNS = ["anomaly", "changepoint"]


# ── Core Loader ───────────────────────────────────────────────────────────────

def load_skab(data_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Walks data/raw/, loads every CSV file, tags each row with its
    scenario and filename, and returns a single combined DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        datetime, 8 sensor cols, anomaly, changepoint, scenario, file_id
    """
    data_dir = data_dir or RAW_DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Make sure SKAB CSVs are placed under data/raw/"
        )

    all_frames = []

    # Walk every subdirectory
    for scenario_dir in sorted(data_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue

        scenario_name = scenario_dir.name  # e.g. 'valve1', 'anomaly-free'

        for csv_file in sorted(scenario_dir.glob("*.csv")):
            df = _load_single_file(csv_file, scenario_name)
            if df is not None:
                all_frames.append(df)

    if not all_frames:
        raise ValueError(
            f"No CSV files found under {data_dir}. "
            f"Check your folder structure."
        )

    combined = pd.concat(all_frames, ignore_index=True)

    print(f"[loader] Loaded {len(all_frames)} files | "
          f"{len(combined):,} rows | "
          f"Scenarios: {combined['scenario'].unique().tolist()}")

    return combined


def _load_single_file(path: Path, scenario: str) -> Optional[pd.DataFrame]:
    """
    Loads a single SKAB CSV file and standardizes its format.
    Returns None if the file is malformed or empty.
    """
    try:
        df = pd.read_csv(path, sep=";", index_col="datetime", parse_dates=True)
        df.index.name = "datetime"
        df = df.reset_index()

        # Validate expected columns exist
        missing_sensors = [c for c in SENSOR_COLUMNS if c not in df.columns]
        if missing_sensors:
            print(f"[loader] WARNING: {path.name} missing columns: {missing_sensors}")
            return None

        # anomaly-free files have no label columns — add them as zeros
        for col in LABEL_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # Tag metadata
        df["scenario"] = scenario
        df["file_id"] = f"{scenario}/{path.stem}"

        # Enforce column order
        ordered_cols = ["datetime"] + SENSOR_COLUMNS + LABEL_COLUMNS + ["scenario", "file_id"]
        df = df[ordered_cols]

        # Drop rows where all sensor values are NaN
        df = df.dropna(subset=SENSOR_COLUMNS, how="all")

        return df

    except Exception as e:
        print(f"[loader] ERROR reading {path}: {e}")
        return None


# ── Helper Utilities ──────────────────────────────────────────────────────────

def get_scenario(df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Filter combined DataFrame to a single scenario."""
    return df[df["scenario"] == scenario].reset_index(drop=True)


def summary(df: pd.DataFrame) -> None:
    """Print a clean summary of the loaded dataset."""
    print("\n" + "="*55)
    print("SKAB DATASET SUMMARY")
    print("="*55)
    print(f"  Total rows       : {len(df):,}")
    print(f"  Total files      : {df['file_id'].nunique()}")
    print(f"  Date range       : {df['datetime'].min()} → {df['datetime'].max()}")
    print(f"  Anomaly rate     : {df['anomaly'].mean()*100:.2f}%")
    print(f"  Changepoint rate : {df['changepoint'].mean()*100:.2f}%")
    print("\n  Rows per scenario:")
    scenario_counts = df.groupby("scenario")["anomaly"].agg(["count", "sum"])
    scenario_counts.columns = ["total_rows", "anomaly_rows"]
    scenario_counts["anomaly_%"] = (scenario_counts["anomaly_rows"] / scenario_counts["total_rows"] * 100).round(2)
    print(scenario_counts.to_string())
    print("="*55 + "\n")