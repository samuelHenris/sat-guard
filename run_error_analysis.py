
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
)

from sat_guard.models.training import FEATURE_COLUMNS


DATA_PATH = Path("data/processed/cadc0872_features.csv")
REPORT_DIR = Path("reports/models/error_analysis")

EXPERIMENTS = {
    "full": FEATURE_COLUMNS,
    "signal_only": [
        col for col in FEATURE_COLUMNS
        if col not in ("segment_length", "gap_count")
    ],
}


def main():
    df = pd.read_csv(DATA_PATH)

    # Keep official test data out of this analysis.
    train_df = df[df["train"] == 1].copy()

    y = train_df["anomaly"]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for name, columns in EXPERIMENTS.items():
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        predictions = cross_val_predict(
            model,
            train_df[columns],
            y,
            cv=cv,
            method="predict",
        )

        results = train_df[
            ["segment", "channel", "anomaly",
             "segment_length", "gap_count", "kurtosis"]
        ].copy()

        results["prediction"] = predictions

        results["outcome"] = "correct"

        results.loc[
            (results["anomaly"] == 0)
            & (results["prediction"] == 1),
            "outcome",
        ] = "false_positive"

        results.loc[
            (results["anomaly"] == 1)
            & (results["prediction"] == 0),
            "outcome",
        ] = "false_negative"

        errors = results[
            results["outcome"] != "correct"
        ]

        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        print(f"\n{name.upper()}")
        print("-" * 35)
        print(f"True negatives:  {tn}")
        print(f"False positives: {fp}")
        print(f"False negatives: {fn}")
        print(f"True positives:  {tp}")

        print("\nExample classification errors:")
        print(errors.head(10).to_string(index=False))

        results.to_csv(
            REPORT_DIR / f"{name}_predictions.csv",
            index=False,
        )

        errors.to_csv(
            REPORT_DIR / f"{name}_errors.csv",
            index=False,
        )


if __name__ == "__main__":
    main()