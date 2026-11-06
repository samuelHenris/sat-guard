
import pandas as pd

from sat_guard.data.validation import validate_telemetry


def summarize_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one summary record per telemetry segment.
    """

    return (
        df.groupby("segment")
        .agg(
            channel=("channel", "first"),
            anomaly=("anomaly", "first"),
            train=("train", "first"),
            sampling=("sampling", "first"),
            measurements=("value", "size"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            min_value=("value", "min"),
            max_value=("value", "max"),
            mean_value=("value", "mean"),
            std_value=("value", "std"),
        )
        .reset_index()
    )


def summarize_channels(
    segments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize segment labels and train/test distribution
    for each telemetry channel.
    """

    return (
        segments.groupby("channel")
        .agg(
            total_segments=("segment", "count"),
            anomalous_segments=("anomaly", "sum"),
            training_segments=(
                "train", lambda x: (x == 1).sum()
            ),
            testing_segments=(
                "train", lambda x: (x == 0).sum()
            ),
            avg_segment_length=("measurements", "mean"),
            min_segment_length=("measurements", "min"),
            max_segment_length=("measurements", "max"),
        )
        .assign(
            normal_segments=lambda x:
                x["total_segments"] - x["anomalous_segments"]
        )
        .reset_index()
    )


def analyze_timestamp_gaps(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect intervals exceeding a segment's nominal
    sampling interval.

    Returns one record per segment containing gaps.
    """

    ordered = df.sort_values(
        ["segment", "timestamp"]
    ).copy()

    ordered["interval_seconds"] = (
        ordered.groupby("segment")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # Use a small tolerance for floating-point comparisons.
    tolerance = 1e-6

    gaps = ordered[
        ordered["interval_seconds"]
        > ordered["sampling"] + tolerance
    ].copy()

    if gaps.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "channel",
                "gap_count",
                "largest_gap_seconds",
            ]
        )

    return (
        gaps.groupby(["segment", "channel"])
        .agg(
            gap_count=("interval_seconds", "count"),
            largest_gap_seconds=("interval_seconds", "max"),
        )
        .reset_index()
    )


def run_diagnostics(df: pd.DataFrame) -> dict:
    """
    Validate and summarize the telemetry dataset.
    Does not modify the source DataFrame.
    """

    validation = validate_telemetry(df)

    segments = summarize_segments(df)
    channels = summarize_channels(segments)
    gaps = analyze_timestamp_gaps(df)

    return {
        "validation": validation,
        "segments": segments,
        "channels": channels,
        "gaps": gaps,
    }
