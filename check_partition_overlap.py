
import pandas as pd

from sat_guard.data.loader import load_telemetry


def main():
    df = load_telemetry("data/raw/segments.csv")

    # Analyze the selected magnetometer channel.
    df = df[df["channel"] == "CADC0872"]

    segments = (
        df.groupby("segment")
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            train=("train", "first"),
        )
        .reset_index()
        .sort_values("start")
    )

    training = segments[segments["train"] == 1]
    testing = segments[segments["train"] == 0]

    overlapping_pairs = []

    for train_row in training.itertuples():
        overlaps = testing[
            (testing["start"] <= train_row.end)
            & (testing["end"] >= train_row.start)
        ]

        for test_row in overlaps.itertuples():
            overlapping_pairs.append(
                (train_row.segment, test_row.segment)
            )

    print("\nSAT-GUARD — Partition Overlap")
    print("--------------------------------")
    print("Training segments:", len(training))
    print("Testing segments:", len(testing))
    print("Overlapping pairs:", len(overlapping_pairs))
    print("Examples:", overlapping_pairs[:10])


if __name__ == "__main__":
    main()