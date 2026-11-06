
from pathlib import Path

import matplotlib.pyplot as plt

from sat_guard.data.loader import load_telemetry


DATA_PATH = Path("data/raw/segments.csv")
OUTPUT_DIR = Path("reports/figures")

# Selected from our cross-validation error analysis
SEGMENTS = [10, 15, 57]


def plot_segment(df, segment_id):
    segment = (
        df[df["segment"] == segment_id]
        .sort_values("timestamp")
    )

    if segment.empty:
        raise ValueError(f"Segment {segment_id} not found")

    channel = segment["channel"].iloc[0]
    label = int(segment["anomaly"].iloc[0])

    label_text = "Anomalous" if label == 1 else "Normal"

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        segment["timestamp"],
        segment["value"],
        linewidth=1.2,
    )

    ax.set_title(
        f"OPS-SAT | Segment {segment_id} | "
        f"{channel} | {label_text}"
    )

    ax.set_xlabel("Recorded timestamp (UTC)")
    ax.set_ylabel("Sensor measurement (original units)")
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    path = OUTPUT_DIR / f"segment_{segment_id}.png"

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"Saved: {path}")


def main():
    df = load_telemetry(DATA_PATH)

    for segment_id in SEGMENTS:
        plot_segment(df, segment_id)


if __name__ == "__main__":
    main()