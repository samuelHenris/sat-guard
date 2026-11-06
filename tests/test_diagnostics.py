
import pandas as pd

from sat_guard.data.diagnostics import (
    summarize_segments,
    summarize_channels,
    analyze_timestamp_gaps,
)


def example_data():
    return pd.DataFrame({
        "segment": [1, 1, 1, 1, 2, 2],
        "channel": ["CADC0872"] * 6,
        "timestamp": pd.to_datetime([
            "2022-01-01T00:00:00Z",
            "2022-01-01T00:00:01Z",
            "2022-01-01T00:00:02Z",
            "2022-01-01T00:00:08Z",
            "2022-01-01T00:01:00Z",
            "2022-01-01T00:01:01Z",
        ]),
        "value": [1.0, 1.2, 1.1, 1.3, 2.0, 2.1],
        "anomaly": [1, 1, 1, 1, 0, 0],
        "train": [1, 1, 1, 1, 0, 0],
        "sampling": [1] * 6,
        "label": ["anomaly"] * 6,
    })


def test_segment_summary():
    result = summarize_segments(example_data())

    assert len(result) == 2
    assert result.loc[0, "measurements"] == 4
    assert result.loc[1, "measurements"] == 2


def test_channel_summary():
    segments = summarize_segments(example_data())
    result = summarize_channels(segments)

    assert result.loc[0, "total_segments"] == 2
    assert result.loc[0, "anomalous_segments"] == 1
    assert result.loc[0, "training_segments"] == 1


def test_timestamp_gaps():
    result = analyze_timestamp_gaps(example_data())

    assert len(result) == 1
    assert result.loc[0, "segment"] == 1
    assert result.loc[0, "gap_count"] == 1
    assert result.loc[0, "largest_gap_seconds"] == 6.0


def test_diagnostics_do_not_modify_input():
    df = example_data()
    original = df.copy(deep=True)

    analyze_timestamp_gaps(df)

    pd.testing.assert_frame_equal(df, original)
