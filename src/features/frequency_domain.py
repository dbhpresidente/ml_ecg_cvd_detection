"""Frequency-domain feature extraction for ECG signals.

Computes 6 spectral features per lead using Welch's power spectral density estimate.

Frequency bands (Hz):
    low : 0.5–5    (baseline / low-frequency components)
    mid : 5–15     (P-wave and T-wave dominant)
    high: 15–40    (QRS complex dominant)
"""
import numpy as np
from scipy.signal import welch

#: Number of features extracted per lead.
N_FREQ_FEATURES_PER_LEAD: int = 6

#: Ordered names of the per-lead frequency-domain features.
FREQ_FEATURE_NAMES: list[str] = [
    "power_low",
    "power_mid",
    "power_high",
    "spectral_entropy",
    "dominant_freq",
    "total_power",
]

# ECG-relevant frequency bands in Hz
_BAND_LOW = (0.5, 5.0)
_BAND_MID = (5.0, 15.0)
_BAND_HIGH = (15.0, 40.0)


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    """Integrate PSD over [low, high] Hz using the trapezoidal rule."""
    mask = (freqs >= low) & (freqs <= high)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _spectral_entropy(psd: np.ndarray) -> float:
    """Shannon entropy of the normalized PSD (0 = single frequency, 1 = flat spectrum)."""
    total = psd.sum()
    if total < 1e-12:
        return 0.0
    p = psd / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)) / np.log2(len(psd)))


def compute_frequency_features(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    """Compute frequency-domain spectral features for a single ECG record.

    Uses Welch's method to estimate the PSD for each lead, then extracts
    band powers, spectral entropy, dominant frequency, and total power.

    Args:
        signal: ECG signal of shape (leads, samples).
        fs: Sampling frequency in Hz.

    Returns:
        Feature vector of shape (leads * 6,) with float64 values, ordered as:
        [lead0_power_low, ..., lead0_total_power, lead1_power_low, ..., leadN_total_power].
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected signal of shape (leads, samples), got {signal.shape}")

    n_leads = signal.shape[0]
    n_samples = signal.shape[-1]
    nperseg = min(256, n_samples // 4)

    lead_features = np.zeros((n_leads, N_FREQ_FEATURES_PER_LEAD), dtype=np.float64)

    for i in range(n_leads):
        freqs, psd = welch(signal[i], fs=fs, nperseg=nperseg)

        power_low = _band_power(freqs, psd, *_BAND_LOW)
        power_mid = _band_power(freqs, psd, *_BAND_MID)
        power_high = _band_power(freqs, psd, *_BAND_HIGH)
        sp_entropy = _spectral_entropy(psd)
        dom_freq = float(freqs[np.argmax(psd)])
        total_power = float(np.trapezoid(psd, freqs))

        lead_features[i] = [power_low, power_mid, power_high, sp_entropy, dom_freq, total_power]

    return lead_features.flatten()
