
import pandas as pd
import pytest

from sat_guard.data.validation import validate_telemetry


@pytest.fixture
def sample_telemetry():
    """A small valid dataset for unit testing."""

    return pd.DataFrame({
        "channel": ["CADC0872", "CADC0872", "CADC0872"],
        "timestamp": pd.to_datetime([
            "2022-06-01T10:00:00Z",
            "2022-06-01T10:00:01Z",
            "2022-06-01T10:00:02Z",
        ]),
        "value": [0.001, 0.002, 0.003],
        "label": ["anomaly"] * 3,
        "sampling": [1, 1, 1],
        "anomaly": [1, 1, 1],
        "segment": [1, 1, 1],
        "train": [1, 1, 1],
    })


def test_valid_data(sample_telemetry):
    result = validate_telemetry(sample_telemetry)

    assert result["measurements"] == 3
    assert result["segments"] == 1
    assert result["anomalous_segments"] == 1


def test_missing_value(sample_telemetry):
    sample_telemetry.loc[0, "value"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_telemetry(sample_telemetry)


def test_invalid_anomaly_label(sample_telemetry):
    sample_telemetry.loc[0, "anomaly"] = 2

    with pytest.raises(ValueError, match="only 0 or 1"):
        validate_telemetry(sample_telemetry)


def test_inconsistent_segment_label(sample_telemetry):
    sample_telemetry.loc[0, "anomaly"] = 0

    with pytest.raises(ValueError, match="Conflicting"):
        validate_telemetry(sample_telemetry)


def test_duplicate_timestamp(sample_telemetry):
    sample_telemetry.loc[1, "timestamp"] = (
        sample_telemetry.loc[0, "timestamp"]
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_telemetry(sample_telemetry)
