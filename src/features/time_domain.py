"""Time-domain feature extraction for ECG signals.

Computes 8 statistical features per lead from the amplitude of the signal.
"""
import numpy as np
from scipy.stats import kurtosis, skew

#: Number of features extracted per lead.
N_TIME_FEATURES_PER_LEAD: int = 8

#: Ordered names of the per-lead time-domain features.
TIME_FEATURE_NAMES: list[str] = [
    "mean",
    "std",
    "min",
    "max",
    "rms",
    "skewness",
    "kurtosis",
    "range",
]


def compute_time_features(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    """Compute time-domain statistical features for a single ECG record.

    Extracts 8 amplitude statistics per lead and returns them as a flat vector.

    Args:
        signal: ECG signal of shape (leads, samples).
        fs: Sampling frequency in Hz (unused; kept for API consistency).

    Returns:
        Feature vector of shape (leads * 8,) with float64 values, ordered as:
        [lead0_mean, lead0_std, ..., lead0_range, lead1_mean, ..., leadN_range].
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected signal of shape (leads, samples), got {signal.shape}")

    signal = signal.astype(np.float64)
    mean = signal.mean(axis=-1)
    std = signal.std(axis=-1)
    minimum = signal.min(axis=-1)
    maximum = signal.max(axis=-1)
    rms = np.sqrt(np.mean(signal**2, axis=-1))
    skewness = skew(signal, axis=-1)
    kurt = kurtosis(signal, axis=-1)
    amp_range = maximum - minimum

    # Stack to (leads, 8), then flatten to (leads * 8,)
    features = np.stack([mean, std, minimum, maximum, rms, skewness, kurt, amp_range], axis=-1)
    return features.flatten()
