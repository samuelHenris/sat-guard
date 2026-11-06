
from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
)


DATA_PATH = Path("data/processed/cadc0872_features.csv")
REPORT_PATH = Path("reports/models/feature_comparison.csv")


ORIGINAL_FEATURES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "skewness",
    "kurtosis",
    "mean_abs_diff",
    "gap_count",
    "segment_length",
]

NEW_FEATURES = [
    "max_abs_diff",
    "p95_abs_diff",
    "diff_std",
]

EXPERIMENTS = {
    "Original (11)": ORIGINAL_FEATURES,
    "Enhanced (14)": ORIGINAL_FEATURES + NEW_FEATURES,
}


def create_model():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def main():
    df = pd.read_csv(DATA_PATH)

    # Only use the official training partition.
    train_df = df[df["train"] == 1].copy()

    y = train_df["anomaly"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for name, columns in EXPERIMENTS.items():
        X = train_df[columns]

        scores = cross_validate(
            create_model(),
            X,
            y,
            cv=cv,
            scoring={
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
            },
            n_jobs=1,
        )

        results.append({
            "experiment": name,
            "features": len(columns),
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1_mean": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
        })

    report = pd.DataFrame(results)

    print("\nSAT-GUARD — Temporal Feature Comparison")
    print("--------------------------------------")
    print(report.to_string(index=False))

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report.to_csv(REPORT_PATH, index=False)

    print(f"\nSaved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()