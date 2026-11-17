"""Classical ML baselines for ECG CVD detection.

Provides sklearn Pipeline objects that combine feature scaling with a classifier.
All baselines accept feature vectors from src.features.extract_features_batch()
as input (shape (n_records, n_features)).

Usage:
    >>> from src.models.baseline import build_baseline
    >>> model = build_baseline("random_forest")
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>> y_prob = model.predict_proba(X_test)[:, 1]
"""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: Names of available baseline classifiers.
BASELINE_NAMES: list[str] = ["logistic", "random_forest", "gradient_boosting"]


def build_baseline(name: str, **kwargs) -> Pipeline:
    """Build a scikit-learn Pipeline for the requested baseline classifier.

    Each pipeline integrates preprocessing (scaling where appropriate) and
    a classifier configured with class-weight balancing for the imbalanced
    ECG dataset.

    Args:
        name: Classifier identifier.  One of:
            - ``"logistic"``          : L2 logistic regression with StandardScaler.
            - ``"random_forest"``     : Random forest (tree-based, scale-invariant).
            - ``"gradient_boosting"`` : Histogram-based gradient boosting
              (fast, handles large feature sets; no scaling needed).
        **kwargs: Keyword arguments forwarded to the underlying sklearn estimator.
            These override the default hyperparameters.

    Returns:
        A fitted-compatible sklearn Pipeline with steps ``[("scaler", ...), ("clf", ...)]``
        for logistic, or ``[("clf", ...)]`` for tree-based models.

    Raises:
        ValueError: If name is not one of the supported classifier identifiers.

    Examples:
        >>> clf = build_baseline("logistic", C=0.1)
        >>> clf = build_baseline("random_forest", n_estimators=200)
        >>> clf = build_baseline("gradient_boosting", max_iter=200)
    """
    if name == "logistic":
        defaults = dict(max_iter=1000, class_weight="balanced", solver="lbfgs", C=1.0)
        defaults.update(kwargs)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**defaults)),
        ])

    elif name == "random_forest":
        defaults = dict(
            n_estimators=100,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
        defaults.update(kwargs)
        return Pipeline([
            ("clf", RandomForestClassifier(**defaults)),
        ])

    elif name == "gradient_boosting":
        defaults = dict(
            max_iter=100,
            class_weight="balanced",
            random_state=42,
        )
        defaults.update(kwargs)
        return Pipeline([
            ("clf", HistGradientBoostingClassifier(**defaults)),
        ])

    else:
        raise ValueError(
            f"Unknown baseline '{name}'. "
            f"Choose one of: {BASELINE_NAMES}"
        )
