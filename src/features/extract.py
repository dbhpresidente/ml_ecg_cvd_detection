"""Main feature extraction pipeline for ECG signals.

Combines time-domain, frequency-domain, and morphological features into a
single flat feature vector of 176 dimensions per record:

    - Time domain   :  8 features × 12 leads = 96
    - Frequency     :  6 features × 12 leads = 72
    - Morphological :  8 HRV features (global)
    - Total         : 176
"""
import logging

import numpy as np

from src.features.frequency_domain import (
    FREQ_FEATURE_NAMES,
    N_FREQ_FEATURES_PER_LEAD,
    compute_frequency_features,
)
from src.features.morphological import MORPH_FEATURE_NAMES, compute_morphological_features
from src.features.time_domain import (
    N_TIME_FEATURES_PER_LEAD,
    TIME_FEATURE_NAMES,
    compute_time_features,
)

logger = logging.getLogger(__name__)

#: Standard 12-lead ECG lead names (PTB-XL ordering).
LEAD_NAMES: list[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_feature_names(n_leads: int = 12) -> list[str]:
    """Return ordered feature names matching the output of extract_features.

    Args:
        n_leads: Number of ECG leads.  Defaults to 12.

    Returns:
        List of 176 feature name strings (for n_leads=12).
    """
    lead_labels = LEAD_NAMES[:n_leads]

    time_names = [
        f"time_{lead}_{feat}"
        for lead in lead_labels
        for feat in TIME_FEATURE_NAMES
    ]
    freq_names = [
        f"freq_{lead}_{feat}"
        for lead in lead_labels
        for feat in FREQ_FEATURE_NAMES
    ]

    return time_names + freq_names + MORPH_FEATURE_NAMES


def extract_features(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    """Extract a full feature vector from a single preprocessed ECG record.

    Args:
        signal: ECG signal of shape (leads, samples).
        fs: Sampling frequency in Hz.

    Returns:
        1-D float64 feature vector of length
        (n_leads * 8) + (n_leads * 6) + 8 = 176 for a 12-lead signal.
    """
    time_feats = compute_time_features(signal, fs=fs)
    freq_feats = compute_frequency_features(signal, fs=fs)
    morph_feats = compute_morphological_features(signal, fs=fs)
    return np.concatenate([time_feats, freq_feats, morph_feats])


def extract_features_batch(
    signals: np.ndarray,
    fs: int = 500,
) -> np.ndarray:
    """Extract features for a batch of preprocessed ECG records.

    Args:
        signals: Array of shape (n_records, leads, samples).
        fs: Sampling frequency in Hz.

    Returns:
        2-D float64 array of shape (n_records, n_features).
    """
    if signals.ndim != 3:
        raise ValueError(f"Expected signals of shape (n_records, leads, samples), got {signals.shape}")

    n_records = signals.shape[0]
    n_features = (
        signals.shape[1] * N_TIME_FEATURES_PER_LEAD
        + signals.shape[1] * N_FREQ_FEATURES_PER_LEAD
        + 8  # morphological
    )

    result = np.zeros((n_records, n_features), dtype=np.float64)
    for i, signal in enumerate(signals):
        if i % 500 == 0 and i > 0:
            logger.info("Feature extraction: %d/%d records processed", i, n_records)
        result[i] = extract_features(signal, fs=fs)

    return result
