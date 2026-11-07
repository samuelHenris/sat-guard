
import argparse
import time

import pandas as pd
import httpx

from sat_guard.data.loader import load_telemetry


DATA_PATH = "data/raw/segments.csv"
API_URL = "http://127.0.0.1:8000/api/v1/predict"
CHANNEL = "CADC0872"


def load_segments(limit: int, label: int | None = None):
    """Load completed segments in chronological order."""

    df = load_telemetry(DATA_PATH)

    df = df[df["channel"] == CHANNEL].copy()

    # Start with training examples to avoid repeatedly
    # inspecting the official held-out test labels.
    df = df[df["train"] == 1]

    if label is not None:
        df = df[df["anomaly"] == label]

    segment_order = (
        df.groupby("segment")["timestamp"]
        .min()
        .sort_values()
        .head(limit)
        .index
    )

    for segment_id in segment_order:
        segment = (
            df[df["segment"] == segment_id]
            .sort_values("timestamp")
        )

        yield segment_id, segment


def create_payload(segment: pd.DataFrame):
    """Convert one completed segment into an API request."""

    return {
        "channel": CHANNEL,
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


def send_prediction(client: httpx.Client, payload: dict):
    """Submit telemetry and fail on unsuccessful HTTP responses."""

    response = client.post(
        API_URL,
        json=payload,
    )

    response.raise_for_status()

    return response.json()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional ground-truth label filter",
    )

    args = parser.parse_args()

    if args.limit < 1 or args.delay < 0:
        parser.error(
            "--limit must be positive and "
            "--delay must be nonnegative"
        )

    print("\nSAT-GUARD — Historical Replay")
    print("--------------------------------")

    with httpx.Client(timeout=30.0) as client:
        for segment_id, segment in load_segments(
            args.limit,
            label=args.label,
        ):
            payload = create_payload(segment)

            start = time.perf_counter()

            result = send_prediction(client, payload)

            latency_ms = (
                time.perf_counter() - start
            ) * 1000


            print(
                f"Segment {segment_id} | "
                f"Actual: {int(segment['anomaly'].iloc[0])} | "
                f"Predicted: {result['prediction']} | "
                f"Score: {result['anomaly_probability']:.3f} | "
                f"Latency: {latency_ms:.1f} ms"
            )

            time.sleep(args.delay)

    print("\nReplay completed.")


if __name__ == "__main__":
    main()