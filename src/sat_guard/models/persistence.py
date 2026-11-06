
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sat_guard.models.training import FEATURE_COLUMNS


def train_final_model(
    features: pd.DataFrame,
) -> dict:
    """
    Train the selected model using the official
    training partition only.
    """

    training = features[features["train"] == 1].copy()

    if training.empty:
        raise ValueError("No training data available.")

    X = training[FEATURE_COLUMNS]
    y = training["anomaly"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X, y)

    return {
        "model": model,
        "feature_columns": FEATURE_COLUMNS.copy(),
        "channel": "CADC0872",
        "model_version": "0.1.0",
    }


def save_model(artifact: dict, path: str | Path):
    """Save the trained model and its metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, path)


def load_model(path: str | Path) -> dict:
    """Load a trusted SAT-GUARD model artifact."""

    return joblib.load(path)