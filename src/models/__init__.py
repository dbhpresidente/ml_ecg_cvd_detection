"""ECG CVD detection models.

Public API:
    compute_metrics : Evaluate binary classification predictions.
    build_baseline  : Build a scikit-learn baseline pipeline.
    CNN1D           : 1D ResNet-style CNN (requires PyTorch).

CNN1D must be imported directly from src.models.cnn1d to avoid a hard
dependency on PyTorch for projects that only use the sklearn baselines.
"""
from src.models.baseline import build_baseline
from src.models.metrics import compute_metrics

__all__ = ["build_baseline", "compute_metrics"]
