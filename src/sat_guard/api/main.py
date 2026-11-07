
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from sat_guard.models.persistence import load_model

import pandas as pd

from fastapi import HTTPException

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from fastapi.responses import Response

from sat_guard.api.metrics import (
    PREDICTIONS,
    ANOMALIES,
    PREDICTION_DURATION,
)

from sat_guard.features.extraction import (
    extract_segment_features,
)

from sat_guard.api.schemas import (
    PredictionRequest,
    PredictionResponse,
)

DEFAULT_MODEL_PATH = Path(
    "artifacts/cadc0872_random_forest.joblib"
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once at startup."""

    model_path = Path(
        os.getenv(
            "SAT_GUARD_MODEL_PATH",
            str(DEFAULT_MODEL_PATH),
        )
    )

    if not model_path.is_file():
        raise RuntimeError(
            f"Model artifact not found: {model_path}"
        )

    artifact = load_model(model_path)
    app.state.model_artifact = artifact

    print(
        f"SAT-GUARD model v{artifact['model_version']} loaded"
    )

    yield

    app.state.model_artifact = None


app = FastAPI(
    title="SAT-GUARD",
    description="Satellite telemetry anomaly detection API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    artifact = request.app.state.model_artifact

    return {
        "status": "healthy",
        "model_loaded": artifact is not None,
    }


@app.get("/api/v1/model/info")
def model_info(request: Request):
    artifact = request.app.state.model_artifact

    return {
        "model_version": artifact["model_version"],
        "channel": artifact["channel"],
        "model_type": type(artifact["model"]).__name__,
        "feature_count": len(artifact["feature_columns"]),
        "features": artifact["feature_columns"],
    }


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
    request: Request,
):
    """Classify a completed telemetry segment."""

    artifact = request.app.state.model_artifact

    # This model was trained on one specific channel.
    if payload.channel != artifact["channel"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported channel: {payload.channel}. "
                f"Expected {artifact['channel']}."
            ),
        )

    # Convert validated API measurements into
    # the DataFrame format used during training.
    frame = pd.DataFrame([
        {
            "timestamp": point.timestamp,
            "value": point.value,
            "sampling": point.sampling,
        }
        for point in payload.measurements
    ])


    # Measure feature extraction and model inference.
    with PREDICTION_DURATION.labels(
        channel=artifact["channel"]
    ).time():

        features = extract_segment_features(frame)

        columns = artifact["feature_columns"]

        X = pd.DataFrame(
            [{name: features[name] for name in columns}],
            columns=columns,
        )

        model = artifact["model"]

        prediction = int(model.predict(X)[0])

        probabilities = model.predict_proba(X)[0]
        anomaly_index = list(model.classes_).index(1)

        anomaly_probability = float(
            probabilities[anomaly_index]
        )

    classification = (
        "anomalous" if prediction == 1 else "normal"
    )

    # Count only completed, successful predictions.
    PREDICTIONS.labels(
        channel=artifact["channel"],
        classification=classification,
    ).inc()

    if prediction == 1:
        ANOMALIES.labels(
            channel=artifact["channel"]
        ).inc()


    return PredictionResponse(
        channel=artifact["channel"],
        model_version=artifact["model_version"],
        prediction=prediction,
        classification=classification,
        anomaly_probability=anomaly_probability,
        measurement_count=len(payload.measurements),
    )


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Expose metrics in Prometheus text format."""

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
