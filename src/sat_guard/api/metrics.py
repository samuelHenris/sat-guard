
from prometheus_client import Counter, Histogram


PREDICTIONS = Counter(
    "sat_guard_predictions_total",
    "Number of successful SAT-GUARD predictions",
    ["channel", "classification"],
)

ANOMALIES = Counter(
    "sat_guard_anomalies_total",
    "Number of segments classified as anomalous",
    ["channel"],
)

PREDICTION_DURATION = Histogram(
    "sat_guard_prediction_duration_seconds",
    "Time spent extracting features and predicting",
    ["channel"],
    buckets=(
        0.005, 0.01, 0.025, 0.05,
        0.075, 0.1, 0.25, 0.5, 1.0,
    ),
)