# tests/test_pipeline.py

import sys
from pathlib import Path

# Make sure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.loader import load_skab, summary, get_scenario, SENSOR_COLUMNS


def test_loader():
    df = load_skab()

    # Shape checks
    assert len(df) > 0, "DataFrame is empty"
    assert len(df.columns) == 13, f"Expected 13 columns, got {len(df.columns)}"

    # Column checks
    for col in SENSOR_COLUMNS:
        assert col in df.columns, f"Missing sensor column: {col}"
    assert "anomaly" in df.columns
    assert "scenario" in df.columns

    # Scenario checks
    scenarios = df["scenario"].unique().tolist()
    assert "anomaly-free" in scenarios, "anomaly-free scenario missing"
    assert "valve1" in scenarios, "valve1 scenario missing"

    # Label sanity
    assert df["anomaly"].isin([0, 1]).all(), "anomaly column has values other than 0/1"
    assert df["anomaly-free" == df["scenario"]]["anomaly"].sum() == 0, \
        "anomaly-free rows should have zero anomaly labels"

    # Metadata
    assert df["file_id"].nunique() > 1, "Only one file loaded — check directory structure"

    summary(df)

    # Scenario filter test
    v1 = get_scenario(df, "valve1")
    assert (v1["scenario"] == "valve1").all()

    print("\n✅ All tests passed.")


from src.pipeline.features import create_windows, SENSOR_COLUMNS, WINDOW_SIZE


def test_features():
    from src.pipeline.loader import load_skab
    df = load_skab()

    feature_df, labels = create_windows(df)

    # Shape checks
    assert len(feature_df) == len(labels), "Feature/label length mismatch"
    assert len(feature_df) > 0, "No windows created"

    # Each sensor contributes 9 features (7 stat + 2 freq)
    # 8 sensors * 9 = 72 sensor features
    # Correlation pairs: 8*(8-1)/2 = 28
    # Total model features = 72 + 28 = 100
    meta_cols    = ["file_id", "window_start", "scenario"]
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]
    assert len(feature_cols) == 100, f"Expected 100 features, got {len(feature_cols)}"

    # Label sanity
    assert set(labels).issubset({0, 1}), "Labels must be binary"
    assert labels.mean() > 0, "No anomalies found — label extraction broken"

    print(f"\n✅ Feature test passed. Shape: {feature_df.shape}, "
          f"Anomaly rate: {labels.mean()*100:.2f}%")


if __name__ == "__main__":
    test_loader()
    test_features()