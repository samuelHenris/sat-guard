
import pandas as pd
import pytest

from sat_guard.features.extraction import (
    extract_segment_features,
)


def test_basic_features():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2022-01-01T00:00:00Z",
            "2022-01-01T00:00:01Z",
            "2022-01-01T00:00:02Z",
            "2022-01-01T00:00:03Z",
        ]),
        "value": [1.0, 2.0, 3.0, 4.0],
        "sampling": [1, 1, 1, 1],
    })

    features = extract_segment_features(df)

    assert features["mean"] == pytest.approx(2.5)
    assert features["min"] == 1.0
    assert features["max"] == 4.0
    assert features["gap_count"] == 0
    assert features["segment_length"] == 4
    assert features["mean_abs_diff"] == pytest.approx(1.0)


def test_timestamp_gap():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2022-01-01T00:00:00Z",
            "2022-01-01T00:00:01Z",
            "2022-01-01T00:00:11Z",
        ]),
        "value": [1.0, 2.0, 100.0],
        "sampling": [1, 1, 1],
    })

    features = extract_segment_features(df)

    assert features["gap_count"] == 1

    # The jump from 2 to 100 crosses a data gap.
    assert features["mean_abs_diff"] == pytest.approx(1.0)

def test_spike_features():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2022-01-01T00:00:00Z",
            "2022-01-01T00:00:01Z",
            "2022-01-01T00:00:02Z",
            "2022-01-01T00:00:03Z",
            "2022-01-01T00:00:04Z",
        ]),
        "value": [1.0, 1.1, 5.0, 1.2, 1.1],
        "sampling": [1] * 5,
    })

    features = extract_segment_features(df)

    assert features["max_abs_diff"] == pytest.approx(3.9)
    assert features["p95_abs_diff"] > 3.0
    assert features["diff_std"] > 1.0
