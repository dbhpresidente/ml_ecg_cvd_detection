"""ECG feature extraction module.

Public API:
    extract_features       : Extract a feature vector from a single ECG record.
    extract_features_batch : Extract features from a batch of ECG records.
    get_feature_names      : Return ordered feature names for interpretability.
"""
from src.features.extract import extract_features, extract_features_batch, get_feature_names

__all__ = ["extract_features", "extract_features_batch", "get_feature_names"]
