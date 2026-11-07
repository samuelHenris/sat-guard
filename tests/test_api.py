
from fastapi.testclient import TestClient

import sat_guard.api.main as api_module


def test_health_and_model_info(monkeypatch):
    """
    Test API startup and metadata without relying
    on a real model file.
    """

    class FakeModel:
        pass

    fake_artifact = {
        "model": FakeModel(),
        "model_version": "0.1.0",
        "channel": "CADC0872",
        "feature_columns": ["mean", "std"],
    }

    monkeypatch.setattr(
        api_module,
        "load_model",
        lambda path: fake_artifact,
    )

    monkeypatch.setattr(
        api_module.Path,
        "is_file",
        lambda self: True,
    )

    with TestClient(api_module.app) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy",
            "model_loaded": True,
        }

        response = client.get("/api/v1/model/info")

        assert response.status_code == 200

        result = response.json()

        assert result["model_version"] == "0.1.0"
        assert result["channel"] == "CADC0872"
        assert result["model_type"] == "FakeModel"
        assert result["feature_count"] == 2
        assert result["features"] == ["mean", "std"]


from datetime import datetime, timedelta, timezone

import pytest

from sat_guard.models.training import FEATURE_COLUMNS


class FakePredictionModel:
    classes_ = [0, 1]

    def predict(self, X):
        assert list(X.columns) == FEATURE_COLUMNS
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]


@pytest.fixture
def prediction_client(monkeypatch):
    artifact = {
        "model": FakePredictionModel(),
        "model_version": "0.1.0",
        "channel": "CADC0872",
        "feature_columns": FEATURE_COLUMNS.copy(),
    }

    monkeypatch.setattr(
        api_module,
        "load_model",
        lambda path: artifact,
    )

    monkeypatch.setattr(
        api_module.Path,
        "is_file",
        lambda self: True,
    )

    with TestClient(api_module.app) as client:
        yield client


def sample_payload():
    start = datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )

    return {
        "channel": "CADC0872",
        "measurements": [
            {
                "timestamp": (
                    start + timedelta(seconds=i)
                ).isoformat(),
                "value": float(i % 3) * 0.00001,
                "sampling": 1.0,
            }
            for i in range(8)
        ],
    }


def test_successful_prediction(prediction_client):
    response = prediction_client.post(
        "/api/v1/predict",
        json=sample_payload(),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] == 1
    assert result["classification"] == "anomalous"
    assert result["anomaly_probability"] == pytest.approx(0.8)
    assert result["measurement_count"] == 8


def test_unsupported_channel(prediction_client):
    payload = sample_payload()
    payload["channel"] = "UNKNOWN"

    response = prediction_client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 400


def test_duplicate_timestamps(prediction_client):
    payload = sample_payload()

    payload["measurements"][1]["timestamp"] = (
        payload["measurements"][0]["timestamp"]
    )

    response = prediction_client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_sampling(prediction_client):
    payload = sample_payload()
    payload["measurements"][0]["sampling"] = -1.0

    response = prediction_client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422
