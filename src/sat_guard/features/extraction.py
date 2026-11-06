
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def extract_segment_features(
    segment_df: pd.DataFrame,
) -> dict:
    """
    Extract numerical features from one telemetry segment.

    Only measurements and sampling metadata are used
    to calculate predictive features.
    """

    ordered = segment_df.sort_values("timestamp")

    values = ordered["value"].to_numpy(dtype=float)
    timestamps = ordered["timestamp"]

    if len(values) < 2:
        raise ValueError("A segment needs at least two samples.")

    sampling = float(ordered["sampling"].iloc[0])

    if sampling <= 0:
        raise ValueError("Sampling interval must be positive.")

    # Calculate elapsed time between measurements.
    intervals = timestamps.diff().dt.total_seconds().to_numpy()

    # Identify consecutive measurements without large gaps.
    valid_pairs = (
        np.isfinite(intervals[1:])
        & (intervals[1:] > 0)
        & (intervals[1:] <= sampling + 1e-6)
    )

    differences = np.diff(values)

    # Only use differences between consecutive observations.
    consecutive_diffs = differences[valid_pairs]

    # Analyze the magnitude of consecutive changes.
    abs_diffs = np.abs(consecutive_diffs)

    max_abs_diff = (
        float(np.max(abs_diffs))
        if abs_diffs.size > 0
        else 0.0
    )

    p95_abs_diff = (
        float(np.percentile(abs_diffs, 95))
        if abs_diffs.size > 0
        else 0.0
    )

    diff_std = (
        float(np.std(consecutive_diffs))
        if consecutive_diffs.size > 0
        else 0.0
    )

    
    mean_abs_diff = (
        float(np.mean(np.abs(consecutive_diffs)))
        if consecutive_diffs.size > 0
        else 0.0
    )

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "iqr": float(
            np.percentile(values, 75)
            - np.percentile(values, 25)
        ),
        "skewness": (
            float(skew(values))
            if np.std(values) > 1e-15 and len(values) >= 3
            else 0.0
        ),
        "kurtosis": (
            float(kurtosis(values))
            if np.std(values) > 1e-15 and len(values) >= 4
            else 0.0
        ),
        "mean_abs_diff": mean_abs_diff,
        "gap_count": int(
            np.sum(intervals[1:] > sampling + 1e-6)
        ),
        "segment_length": len(values),
        "max_abs_diff": max_abs_diff,
        "p95_abs_diff": p95_abs_diff,
        "diff_std": diff_std,
    }


def extract_features(
    df: pd.DataFrame,
    channel: str = "CADC0872",
) -> pd.DataFrame:
    """
    Extract one feature vector per segment for a channel.
    """

    channel_df = df[df["channel"] == channel]

    if channel_df.empty:
        raise ValueError(f"No data for channel: {channel}")

    records = []

    for segment_id, group in channel_df.groupby("segment"):
        features = extract_segment_features(group)

        features["segment"] = segment_id
        features["channel"] = channel

        # Metadata used for evaluation, not model features.
        features["anomaly"] = int(group["anomaly"].iloc[0])
        features["train"] = int(group["train"].iloc[0])

        records.append(features)

    return pd.DataFrame(records)