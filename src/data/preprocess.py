"""ECG signal preprocessing pipeline.

Implements standard preprocessing steps for 12-lead ECG signals:
bandpass filtering, baseline wander removal, and normalization.
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def bandpass_filter(
    signal: np.ndarray,
    fs: int = 500,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to remove baseline wander and HF noise.

    Args:
        signal: ECG signal of shape (leads, samples) or (samples,).
        fs: Sampling frequency in Hz.
        low_hz: Low cutoff frequency in Hz (removes baseline wander).
        high_hz: High cutoff frequency in Hz (removes high-frequency noise).
        order: Filter order.

    Returns:
        Filtered signal with the same shape as input.
    """
    nyq = fs / 2.0
    low = low_hz / nyq
    high = high_hz / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=-1).astype(np.float32)


def notch_filter(
    signal: np.ndarray,
    fs: int = 500,
    notch_hz: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Apply a notch filter to remove powerline interference (50 or 60 Hz).

    Args:
        signal: ECG signal of shape (leads, samples) or (samples,).
        fs: Sampling frequency in Hz.
        notch_hz: Frequency to remove (50 Hz for Europe, 60 Hz for North America).
        quality_factor: Quality factor Q. Higher Q = narrower notch.

    Returns:
        Filtered signal with the same shape as input.
    """
    b, a = iirnotch(notch_hz, quality_factor, fs)
    return filtfilt(b, a, signal, axis=-1).astype(np.float32)


def normalize(
    signal: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """Normalize ECG signal amplitude.

    Args:
        signal: ECG signal of shape (leads, samples) or (batch, leads, samples).
        method: Normalization method. One of:
            - 'zscore': zero mean, unit variance per lead.
            - 'minmax': scale to [0, 1] per lead.
            - 'robust': scale by median and IQR per lead (outlier-resistant).

    Returns:
        Normalized signal with the same shape as input.

    Raises:
        ValueError: If method is not recognized.
    """
    signal = signal.copy().astype(np.float32)
    axis = -1  # normalize along the time axis

    if method == "zscore":
        mean = signal.mean(axis=axis, keepdims=True)
        std = signal.std(axis=axis, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)  # avoid division by zero
        return (signal - mean) / std

    elif method == "minmax":
        min_val = signal.min(axis=axis, keepdims=True)
        max_val = signal.max(axis=axis, keepdims=True)
        denom = np.where((max_val - min_val) < 1e-8, 1.0, max_val - min_val)
        return (signal - min_val) / denom

    elif method == "robust":
        median = np.median(signal, axis=axis, keepdims=True)
        q75 = np.percentile(signal, 75, axis=axis, keepdims=True)
        q25 = np.percentile(signal, 25, axis=axis, keepdims=True)
        iqr = np.where((q75 - q25) < 1e-8, 1.0, q75 - q25)
        return (signal - median) / iqr

    else:
        raise ValueError(f"Unknown normalization method: '{method}'. Use 'zscore', 'minmax', or 'robust'.")


def resample(signal: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """Resample an ECG signal to a target sampling frequency.

    Args:
        signal: ECG signal of shape (leads, samples) or (samples,).
        orig_fs: Original sampling frequency in Hz.
        target_fs: Target sampling frequency in Hz.

    Returns:
        Resampled signal. Number of samples changes proportionally.
    """
    if orig_fs == target_fs:
        return signal

    from scipy.signal import resample as sp_resample

    orig_samples = signal.shape[-1]
    target_samples = int(orig_samples * target_fs / orig_fs)
    return sp_resample(signal, target_samples, axis=-1).astype(np.float32)


def preprocess(
    signal: np.ndarray,
    fs: int = 500,
    bandpass: bool = True,
    notch: bool = True,
    notch_hz: float = 50.0,
    normalization: str = "zscore",
) -> np.ndarray:
    """Full preprocessing pipeline for ECG signals.

    Applies bandpass filtering, optional notch filtering, and normalization
    in the recommended order.

    Args:
        signal: ECG signal of shape (leads, samples) or (batch, leads, samples).
        fs: Sampling frequency in Hz.
        bandpass: Whether to apply bandpass filter (0.5–40 Hz).
        notch: Whether to apply notch filter for powerline interference.
        notch_hz: Powerline frequency to remove (50 or 60 Hz).
        normalization: Normalization method ('zscore', 'minmax', 'robust').

    Returns:
        Preprocessed signal with the same shape as input and float32 dtype.
    """
    signal = signal.astype(np.float32)

    if bandpass:
        signal = bandpass_filter(signal, fs=fs)

    if notch:
        signal = notch_filter(signal, fs=fs, notch_hz=notch_hz)

    signal = normalize(signal, method=normalization)

    return signal
