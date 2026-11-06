
from pathlib import Path

from sat_guard.data.loader import load_telemetry
from sat_guard.data.diagnostics import run_diagnostics


DATA_PATH = Path("data/raw/segments.csv")
OUTPUT_DIR = Path("reports/diagnostics")


def main():
    print("SAT-GUARD — Telemetry Diagnostics")
    print("--------------------------------")

    df = load_telemetry(DATA_PATH)
    results = run_diagnostics(df)

    print("\nDataset validation:")
    for key, value in results["validation"].items():
        print(f"{key}: {value}")

    print("\nChannel summary:")
    print(results["channels"].to_string(index=False))

    print("\nSegment length statistics:")
    print(
        results["segments"]["measurements"].describe()
    )

    print("\nTimestamp gap summary:")
    print(
        "Segments with gaps:",
        len(results["gaps"])
    )

    print("\nTop 10 segments by largest gap:")
    print(
        results["gaps"]
        .sort_values(
            "largest_gap_seconds",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in ("segments", "channels", "gaps"):
        results[name].to_csv(
            OUTPUT_DIR / f"{name}.csv",
            index=False,
        )

    print(f"\nReports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
