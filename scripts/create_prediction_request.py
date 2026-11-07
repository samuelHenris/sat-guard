
import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/segments.csv")
OUTPUT_PATH = Path("data/processed/sample_request.json")


def main():
    df = pd.read_csv(DATA_PATH)

    segment = (
        df[
            (df["channel"] == "CADC0872")
            & (df["segment"] == 5)
        ]
        .sort_values("timestamp")
    )

    if segment.empty:
        raise ValueError("Segment 1 not found")

    payload = {
        "channel": "CADC0872",
        "measurements": [
            {
                "timestamp": pd.Timestamp(
                    row.timestamp
                ).isoformat(),
                "value": float(row.value),
                "sampling": float(row.sampling),
            }
            for row in segment.itertuples()
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2)
    )

    print("Saved:", OUTPUT_PATH)
    print("Measurements:", len(segment))
    print(
        "Actual ESA anomaly label:",
        int(segment["anomaly"].iloc[0]),
    )


if __name__ == "__main__":
    main()