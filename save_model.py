
from pathlib import Path

import pandas as pd

from sat_guard.models.persistence import (
    train_final_model,
    save_model,
    load_model,
)


FEATURE_PATH = Path(
    "data/processed/cadc0872_features.csv"
)

MODEL_PATH = Path(
    "artifacts/cadc0872_random_forest.joblib"
)


def main():
    df = pd.read_csv(FEATURE_PATH)

    artifact = train_final_model(df)
    save_model(artifact, MODEL_PATH)

    loaded = load_model(MODEL_PATH)

    print("SAT-GUARD — Model Persistence")
    print("--------------------------------")
    print("Model saved:", MODEL_PATH)
    print("Model version:", loaded["model_version"])
    print("Channel:", loaded["channel"])
    print("Features:", len(loaded["feature_columns"]))

    # Verify the loaded model can make predictions.
    sample = df.iloc[[0]][loaded["feature_columns"]]

    prediction = loaded["model"].predict(sample)

    print("Sample prediction:", int(prediction[0]))


if __name__ == "__main__":
    main()