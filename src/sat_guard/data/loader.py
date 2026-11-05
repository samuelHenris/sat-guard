from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "channel",
    "timestamp",
    "value",
    "label",
    "sampling",
    "anomaly",
    "segment",
    "train",
}


def load_telemetry(file_path: str | Path) -> pd.DataFrame:
    """
    Load ESA OPS-SAT telemetry from a CSV file.

    Parameters
    ----------
    file_path:
        Path to the original telemetry dataset.

    Returns
    -------
    pd.DataFrame
        Validated column structure with parsed timestamps.
    """

    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Telemetry dataset not found: {path}"
        )

    df = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="raise",
        utc=True,
    )

    return df