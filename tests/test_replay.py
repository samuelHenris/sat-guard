
import httpx
import pandas as pd
import pytest

from scripts import replay_telemetry as replay


@pytest.fixture
def telemetry_data():
    """Small synthetic dataset with out-of-order segments."""

    rows = []

    for segment_id, start, anomaly in [
        (30, "2026-01-03T00:00:00Z", 1),
        (10, "2026-01-01T00:00:00Z", 0),
        (20, "2026-01-02T00:00:00Z", 1),
    ]:
        for i in range(8):
            rows.append({
                "segment": segment_id,
                "channel": "CADC0872",
                "timestamp": (
                    pd.Timestamp(start)
                    + pd.Timedelta(seconds=i)
                ),
                "value": float(i),
                "sampling": 1.0,
                "anomaly": anomaly,
                "train": 1,
            })

    return pd.DataFrame(rows)


def test_replay_chronological_order(
    monkeypatch, telemetry_data
):
    monkeypatch.setattr(
        replay,
        "load_telemetry",
        lambda path: telemetry_data.copy(),
    )

    segments = list(replay.load_segments(limit=3))

    ids = [segment_id for segment_id, _ in segments]

    assert ids == [10, 20, 30]


def test_replay_anomaly_filter(
    monkeypatch, telemetry_data
):
    monkeypatch.setattr(
        replay,
        "load_telemetry",
        lambda path: telemetry_data.copy(),
    )

    segments = list(
        replay.load_segments(limit=5, label=1)
    )

    ids = [segment_id for segment_id, _ in segments]

    assert ids == [20, 30]

    payload = replay.create_payload(segments[0][1])

    assert payload["channel"] == "CADC0872"
    assert len(payload["measurements"]) == 8
    assert payload["measurements"][0]["value"] == 0.0


def test_replay_http_error():
    def handler(request):
        return httpx.Response(
            status_code=503,
            json={"detail": "Service unavailable"},
        )

    transport = httpx.MockTransport(handler)

    with httpx.Client(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            replay.send_prediction(
                client,
                {"channel": "CADC0872"},
            )
