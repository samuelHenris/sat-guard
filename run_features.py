
from pathlib import Path

from sat_guard.data.loader import load_telemetry
from sat_guard.data.validation import validate_telemetry
from sat_guard.features.extraction import extract_features


def main():
    df = load_telemetry("data/raw/segments.csv")

    validate_telemetry(df)

    features = extract_features(df, channel="CADC0872")

    output = Path("data/processed/cadc0872_features.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    features.to_csv(output, index=False)

    print("SAT-GUARD — Feature Extraction")
    print("------------------------------")
    print("Feature matrix:", features.shape)
    print("Training segments:", (features["train"] == 1).sum())
    print("Testing segments:", (features["train"] == 0).sum())
    print("\nFirst five feature vectors:")
    print(features.head().to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()