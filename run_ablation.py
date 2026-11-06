
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

DATA_PATH = Path("data/processed/cadc0872_features.csv")
from sat_guard.models.training import FEATURE_COLUMNS


REPORT_DIR = Path("reports/models")

EXPERIMENTS = {
    "Full model": FEATURE_COLUMNS,
    "Without segment length": [
        col for col in FEATURE_COLUMNS
        if col != "segment_length"
    ],
    "Signal features only": [
        col for col in FEATURE_COLUMNS
        if col not in ("segment_length", "gap_count")
    ],
}


def create_model():
    """Create a fresh model for each cross-validation fold."""

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

    # Use training data ONLY for model comparison.
    train_df = df[df["train"] == 1].copy()

    y_train = train_df["anomaly"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []

    for name, columns in EXPERIMENTS.items():

        X_train = train_df[columns]

        scores = cross_validate(
            create_model(),
            X_train,
            y_train,
            cv=cv,
            scoring={
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
            },
            n_jobs=1,
        )

        result = {
            "experiment": name,
            "features": len(columns),
            "precision_mean": scores[
                "test_precision"
            ].mean(),
            "recall_mean": scores[
                "test_recall"
            ].mean(),
            "f1_mean": scores["test_f1"].mean(),
            "f1_std": scores["test_f1"].std(),
        }

        results.append(result)

    report = pd.DataFrame(results)

    print("\nSAT-GUARD — Feature Ablation")
    print("--------------------------------")
    print(report.to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report.to_csv(
        REPORT_DIR / "ablation.csv",
        index=False,
    )


if __name__ == "__main__":
    main()