
from pathlib import Path

import pandas as pd

from sat_guard.models.training import (
    FEATURE_COLUMNS,
    prepare_data,
    train_models,
    evaluate_model,
)


DATA_PATH = Path(
    "data/processed/cadc0872_features.csv"
)

REPORT_DIR = Path("reports/models")


def main():
    print("\nSAT-GUARD — Model Training")
    print("--------------------------------")

    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = prepare_data(df)

    print(f"Training segments: {len(X_train)}")
    print(f"Testing segments: {len(X_test)}")

    models = train_models(X_train, y_train)

    results = []

    for name, model in models.items():
        metrics = evaluate_model(
            model, X_test, y_test
        )

        print(f"\n{name}")
        print("-" * 30)

        for metric, value in metrics.items():
            print(f"{metric}: {value}")

        results.append({
            "model": name,
            **metrics,
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = pd.DataFrame(results)
    report.to_csv(
        REPORT_DIR / "evaluation.csv",
        index=False,
    )

    # Feature importance from the fitted Random Forest.
    forest = models["Random Forest"]

    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": forest.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nRandom Forest feature importance:")
    print(importance.to_string(index=False))

    importance.to_csv(
        REPORT_DIR / "feature_importance.csv",
        index=False,
    )

    # Save predictions for later error analysis.
    predictions = df[df["train"] == 0][
        ["segment", "channel", "anomaly"]
    ].copy()

    predictions["baseline_prediction"] = (
        models["Statistical Baseline"].predict(X_test)
    )

    predictions["rf_prediction"] = (
        models["Random Forest"].predict(X_test)
    )

    predictions.to_csv(
        REPORT_DIR / "predictions.csv",
        index=False,
    )

    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()