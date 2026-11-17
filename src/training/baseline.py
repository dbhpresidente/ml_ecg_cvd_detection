"""Training and evaluation of sklearn baseline classifiers.

Accepts pre-extracted feature matrices (output of src.features.extract_features_batch)
and returns a fitted pipeline together with per-split metrics.

Usage:
    >>> from src.features import extract_features_batch
    >>> from src.training.baseline import train_baseline
    >>> X_train = extract_features_batch(signals_train)
    >>> X_val   = extract_features_batch(signals_val)
    >>> clf, results = train_baseline(X_train, y_train, X_val, y_val)
    >>> print(results["val"]["roc_auc"])
"""
import logging

import numpy as np
from sklearn.pipeline import Pipeline

from src.models.baseline import build_baseline
from src.models.metrics import compute_metrics

logger = logging.getLogger(__name__)


def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    name: str = "gradient_boosting",
    **kwargs,
) -> tuple[Pipeline, dict[str, dict[str, float]]]:
    """Fit a baseline classifier and evaluate on train and validation splits.

    Args:
        X_train: Feature matrix for training, shape (n_train, n_features).
        y_train: Binary labels for training, shape (n_train,).
        X_val: Feature matrix for validation, shape (n_val, n_features).
        y_val: Binary labels for validation, shape (n_val,).
        name: Baseline classifier name passed to build_baseline().
            One of ``'logistic'``, ``'random_forest'``, ``'gradient_boosting'``.
        **kwargs: Additional keyword arguments forwarded to build_baseline().

    Returns:
        Tuple of:
            - Fitted sklearn Pipeline.
            - Dict with keys ``'train'`` and ``'val'``, each mapping to a
              metrics dict from compute_metrics() (accuracy, AUC, F1, …).

    Raises:
        ValueError: If name is not a recognised baseline identifier.
    """
    clf = build_baseline(name, **kwargs)

    logger.info("Fitting %s on %d samples (%d features)...", name, len(y_train), X_train.shape[1])
    clf.fit(X_train, y_train)

    results: dict[str, dict[str, float]] = {}
    for split_name, (X, y) in [("train", (X_train, y_train)), ("val", (X_val, y_val))]:
        y_pred = clf.predict(X)
        y_prob = clf.predict_proba(X)[:, 1]
        metrics = compute_metrics(y, y_pred, y_prob)
        results[split_name] = metrics
        logger.info(
            "[%s] AUC=%.4f  F1=%.4f  sens=%.4f  spec=%.4f",
            split_name,
            metrics.get("roc_auc", float("nan")),
            metrics["f1"],
            metrics["sensitivity"],
            metrics["specificity"],
        )

    return clf, results
