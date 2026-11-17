"""Classification metrics for ECG CVD detection.

Provides a single compute_metrics() function that returns a consistent dict of
metrics for binary classification, including medical-relevant measures such as
sensitivity (recall) and specificity.

All functions accept numpy arrays and do not require any deep learning framework.
"""
import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

#: Keys always present in the output of compute_metrics().
BASE_METRIC_KEYS: list[str] = [
    "accuracy",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
]

#: Additional keys present when y_prob is provided.
PROB_METRIC_KEYS: list[str] = [
    "roc_auc",
    "avg_precision",
    "optimal_threshold",
]


def _sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """Return (sensitivity, specificity) for binary predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sensitivity, specificity


def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return the probability threshold that maximises Youden's J (sensitivity + specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx])


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute binary classification metrics.

    Args:
        y_true: Ground-truth binary labels of shape (n,). Values must be 0 or 1.
        y_pred: Predicted binary labels of shape (n,). Values must be 0 or 1.
        y_prob: Predicted probability for the positive class, shape (n,).
            When provided, adds roc_auc, avg_precision, and optimal_threshold.

    Returns:
        Dictionary with the following keys (all values are floats in [0, 1] unless noted):

        Always present:
            accuracy    : fraction of correct predictions.
            sensitivity : true positive rate (recall for the positive class).
            specificity : true negative rate.
            f1          : F1-score for the positive class.
            mcc         : Matthews Correlation Coefficient in [-1, 1].

        When y_prob is provided:
            roc_auc           : area under the ROC curve.
            avg_precision     : area under the precision-recall curve.
            optimal_threshold : probability threshold maximising Youden's J.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    sensitivity, specificity = _sensitivity_specificity(y_true, y_pred)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
            metrics["optimal_threshold"] = _optimal_threshold(y_true, y_prob)
        except ValueError as exc:
            logger.warning("Could not compute probability-based metrics: %s", exc)

    return metrics
