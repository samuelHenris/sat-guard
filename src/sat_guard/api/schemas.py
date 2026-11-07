
from datetime import datetime

from pydantic import (
    BaseModel,
    Field,
    AwareDatetime,
    model_validator,
)


class TelemetryMeasurement(BaseModel):
    timestamp: AwareDatetime
    value: float = Field(allow_inf_nan=False)
    sampling: float = Field(gt=0, allow_inf_nan=False)


class PredictionRequest(BaseModel):
    channel: str
    measurements: list[TelemetryMeasurement] = Field(
        min_length=8,
        max_length=5000,
    )

    @model_validator(mode="after")
    def validate_timestamps(self):
        timestamps = [
            point.timestamp
            for point in self.measurements
        ]

        if any(
            current <= previous
            for previous, current in zip(
                timestamps, timestamps[1:]
            )
        ):
            raise ValueError(
                "Timestamps must be strictly increasing"
            )

        return self


class PredictionResponse(BaseModel):
    channel: str
    model_version: str
    prediction: int
    classification: str
    anomaly_probability: float
    measurement_count: int