
import joblib
import pandas as pd

from sat_guard.models.persistence import train_final_model
from sat_guard.models.training import FEATURE_COLUMNS


def test_model_persistence(tmp_path):
    """
    A model should produce identical predictions
    before and after saving and loading.
    """

    # Small synthetic training data, independent
    # of the official ESA dataset.
    rows = []

    for i in range(20):
        row = {
            feature: float(i + 1)
            for feature in FEATURE_COLUMNS
        }

        row["anomaly"] = 0 if i < 10 else 1
        row["train"] = 1
        rows.append(row)

    df = pd.DataFrame(rows)

    artifact = train_final_model(df)

    X = df[artifact["feature_columns"]]

    prediction_before = artifact["model"].predict(X)

    path = tmp_path / "test_model.joblib"

    joblib.dump(artifact, path)
    loaded = joblib.load(path)

    prediction_after = loaded["model"].predict(X)

    assert (
        prediction_before == prediction_after
    ).all()

    assert loaded["model_version"] == "0.1.0"
    assert len(loaded["feature_columns"]) == 14