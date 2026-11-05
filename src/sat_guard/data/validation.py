
import numpy as np
import pandas as pd


def validate_telemetry(df: pd.DataFrame) -> dict:
    """
    Validate the structure and consistency of OPS-SAT telemetry.

    Raises ValueError for invalid data.
    Returns a summary of the validated dataset.
    """

    if df.empty:
        raise ValueError("The telemetry dataset is empty.")

    required_columns = {
        "channel", "timestamp", "value", "label",
        "sampling", "anomaly", "segment", "train",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # 1. Check missing values
    if df[list(required_columns)].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    # 2. Validate sensor measurements
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError("Sensor values must be numeric.")

    if not np.isfinite(df["value"].to_numpy()).all():
        raise ValueError("Sensor values contain infinity or NaN.")

    # 3. Validate binary labels
    for column in ("anomaly", "train"):
        if not df[column].isin([0, 1]).all():
            raise ValueError(
                f"Column '{column}' must contain only 0 or 1."
            )

    # 4. Validate timestamps
    if not pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    ):
        raise ValueError(
            "Timestamps must be parsed as datetimes."
        )

    # 5. Validate sampling metadata
    if not pd.api.types.is_numeric_dtype(df["sampling"]):
        raise ValueError("Sampling must be numeric.")

    if not np.isfinite(df["sampling"].to_numpy()).all():
        raise ValueError("Sampling contains non-finite values.")

    if (df["sampling"] <= 0).any():
        raise ValueError("Sampling intervals must be positive.")

    # 6. Check segment consistency
    segment_groups = df.groupby("segment", sort=False)

    for column in ("channel", "anomaly", "train", "sampling"):
        inconsistent = segment_groups[column].nunique()

        if (inconsistent > 1).any():
            raise ValueError(
                f"Conflicting '{column}' values within segments."
            )

    # 7. Detect duplicate timestamps within segments
    duplicates = df.duplicated(
        subset=["segment", "timestamp"]
    )

    if duplicates.any():
        raise ValueError(
            f"Found {duplicates.sum()} duplicate "
            "segment/timestamp combinations."
        )

    return {
        "measurements": len(df),
        "segments": df["segment"].nunique(),
        "channels": df["channel"].nunique(),
        "anomalous_segments": int(
            segment_groups["anomaly"].first().sum()
        ),
        "normal_segments": int(
            (segment_groups["anomaly"].first() == 0).sum()
        ),
    }