
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sat_guard.data.loader import load_telemetry


RAW_PATH = "data/raw/segments.csv"
FEATURE_PATH = "data/processed/cadc0872_features.csv"

ORIGINAL = [
    "mean", "std", "min", "max", "median",
    "iqr", "skewness", "kurtosis",
    "mean_abs_diff", "gap_count", "segment_length",
]

ENHANCED = ORIGINAL + [
    "max_abs_diff", "p95_abs_diff", "diff_std",
]


def main():
    raw = load_telemetry(RAW_PATH)
    features = pd.read_csv(FEATURE_PATH)

    # Find the recorded start and end of each segment.
    times = (
        raw[raw["channel"] == "CADC0872"]
        .groupby("segment")
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
        )
        .reset_index()
    )

    data = features.merge(
        times,
        on="segment",
        validate="one_to_one",
    )

    # Do not use official test segments.
    data = (
        data[data["train"] == 1]
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    split = int(len(data) * 0.70)

    training = data.iloc[:split].copy()
    validation = data.iloc[split:].copy()

    # Remove training segments overlapping the
    # later validation period.
    cutoff = validation["start_time"].min()

    training = training[
        training["end_time"] < cutoff
    ]

    print("\nSAT-GUARD — Chronological Validation")
    print("-----------------------------------")

    print("Training segments:", len(training))
    print("Validation segments:", len(validation))

    print("\nTraining label distribution:")
    print(training["anomaly"].value_counts())

    print("\nValidation label distribution:")
    print(validation["anomaly"].value_counts())

    if (
        training["anomaly"].nunique() < 2
        or validation["anomaly"].nunique() < 2
    ):
        raise ValueError(
            "Chronological split lacks both classes. "
            "Inspect the timestamp and label distribution."
        )

    for name, columns in {
        "Original": ORIGINAL,
        "Enhanced": ENHANCED,
    }.items():

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        model.fit(
            training[columns],
            training["anomaly"],
        )

        predictions = model.predict(
            validation[columns]
        )

        tn, fp, fn, tp = confusion_matrix(
            validation["anomaly"],
            predictions,
            labels=[0, 1],
        ).ravel()

        print(f"\n{name} model")
        print("-" * 30)

        print(
            "Precision:",
            precision_score(
                validation["anomaly"],
                predictions,
                zero_division=0,
            ),
        )

        print(
            "Recall:",
            recall_score(
                validation["anomaly"],
                predictions,
                zero_division=0,
            ),
        )

        print(
            "F1:",
            f1_score(
                validation["anomaly"],
                predictions,
                zero_division=0,
            ),
        )

        print(
            f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
        )


if __name__ == "__main__":
    main()