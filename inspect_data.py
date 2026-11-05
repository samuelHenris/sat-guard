
from pprint import pprint

from sat_guard.data.loader import load_telemetry
from sat_guard.data.validation import validate_telemetry


def main():
    df = load_telemetry("data/raw/segments.csv")

    summary = validate_telemetry(df)

    print("\nSAT-GUARD — Dataset Validation")
    print("--------------------------------")
    pprint(summary)

    print("\nMeasurements by channel:")
    print(df.groupby("channel").size())

    print("\nSegments by channel:")
    print(
        df.groupby("channel")["segment"].nunique()
    )


if __name__ == "__main__":
    main()
