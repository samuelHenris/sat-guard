
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.base import BaseEstimator, ClassifierMixin


# Exclude identifiers, labels, and train/test metadata.
# Keep these 11 feature names in a fixed order.
FEATURE_COLUMNS = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "iqr",
    "skewness",
    "kurtosis",
    "mean_abs_diff",
    "gap_count",
    "segment_length",
    "max_abs_diff",
    "p95_abs_diff",
    "diff_std",
]


class StatisticalBaseline(ClassifierMixin, BaseEstimator):
    """
    Detect outliers using robust feature deviations
    from normal training segments.

    Thresholds are learned only from training data.
    """

    def __init__(self, quantile=0.99):
        self.quantile = quantile

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        normal = X[y == 0]

        if len(normal) == 0:
            raise ValueError("No normal training segments.")

        self.center_ = np.median(normal, axis=0)

        q75 = np.percentile(normal, 75, axis=0)
        q25 = np.percentile(normal, 25, axis=0)

        # Avoid division by zero for constant features.
        self.scale_ = q75 - q25
        self.scale_ = np.where(
            self.scale_ > 1e-12,
            self.scale_,
            1.0,
        )

        normal_scores = self.score_samples(normal)

        self.threshold_ = np.quantile(
            normal_scores, self.quantile
        )

        return self

    def score_samples(self, X):
        X = np.asarray(X, dtype=float)

        deviations = np.abs(
            (X - self.center_) / self.scale_
        )

        return deviations.max(axis=1)

    def predict(self, X):
        scores = self.score_samples(X)

        return (scores > self.threshold_).astype(int)


def prepare_data(df: pd.DataFrame, feature_columns=None):
    """
    Split feature vectors using the official ESA
    train/test partition.
    """

    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    train_df = df[df["train"] == 1].copy()
    test_df = df[df["train"] == 0].copy()

    X_train = train_df[feature_columns]
    y_train = train_df["anomaly"]

    X_test = test_df[feature_columns]
    y_test = test_df["anomaly"]

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing split is empty.")

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Fit both models using training data only.
    """

    baseline = StatisticalBaseline(quantile=0.99)
    baseline.fit(X_train, y_train)

    random_forest = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    random_forest.fit(X_train, y_train)

    return {
        "Statistical Baseline": baseline,
        "Random Forest": random_forest,
    }


def evaluate_model(model, X_test, y_test):
    """
    Evaluate predictions on held-out segments.
    """

    predictions = model.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test, predictions, zero_division=0
            )
        ),
        "false_positive_rate": (
            float(fp / (fp + tn))
            if fp + tn > 0
            else 0.0
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }