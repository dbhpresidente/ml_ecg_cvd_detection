"""Morphological / HRV feature extraction for ECG signals.

Detects R-peaks on lead II and derives 8 heart rate variability (HRV) features.
If fewer than 2 R-peaks are found all features are returned as zero.

HRV features extracted:
    mean_hr_bpm  : mean heart rate in BPM
    std_hr_bpm   : standard deviation of heart rate
    mean_rr_ms   : mean RR interval in milliseconds
    sdnn_ms      : standard deviation of RR intervals (SDNN)
    rmssd_ms     : root mean square of successive RR differences (RMSSD)
    pnn50        : fraction of successive RR differences > 50 ms
    rr_cv        : coefficient of variation of RR intervals (SDNN / mean_RR)
    n_beats      : number of detected R-peaks
"""
import numpy as np
from scipy.signal import find_peaks

#: Index of Lead II in a standard 12-lead ECG array.
LEAD_II_IDX: int = 1

#: Number of HRV features extracted per record.
N_MORPH_FEATURES: int = 8

#: Ordered names of the morphological / HRV features.
MORPH_FEATURE_NAMES: list[str] = [
    "mean_hr_bpm",
    "std_hr_bpm",
    "mean_rr_ms",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "rr_cv",
    "n_beats",
]


def _detect_r_peaks(lead: np.ndarray, fs: int) -> np.ndarray:
    """Detect R-peak sample indices on a single ECG lead.

    Args:
        lead: 1-D ECG signal array of shape (samples,).
        fs: Sampling frequency in Hz.

    Returns:
        Array of R-peak indices.  May be empty if no peaks are found.
    """
    # Minimum 400 ms between peaks → max ~150 BPM
    min_distance = int(0.4 * fs)
    peaks, _ = find_peaks(lead, distance=min_distance, prominence=0.5)
    return peaks


def compute_morphological_features(signal: np.ndarray, fs: int = 500) -> np.ndarray:
    """Compute HRV features from R-peak detection on Lead II.

    Args:
        signal: ECG signal of shape (leads, samples).  Lead II is at index 1.
        fs: Sampling frequency in Hz.

    Returns:
        Feature vector of shape (8,) with float64 values in the order defined
        by MORPH_FEATURE_NAMES.  All zeros if fewer than 2 R-peaks are detected.
    """
    if signal.ndim != 2:
        raise ValueError(f"Expected signal of shape (leads, samples), got {signal.shape}")

    lead = signal[LEAD_II_IDX]
    peaks = _detect_r_peaks(lead, fs)

    zeros = np.zeros(N_MORPH_FEATURES, dtype=np.float64)

    if len(peaks) < 2:
        return zeros

    # RR intervals in samples → convert to ms
    rr_samples = np.diff(peaks).astype(np.float64)
    rr_ms = rr_samples / fs * 1000.0

    mean_rr = rr_ms.mean()
    std_rr = rr_ms.std()

    if mean_rr < 1e-6:
        return zeros

    # Heart rate in BPM: 60_000 ms / RR_ms
    hr = 60_000.0 / rr_ms
    mean_hr = hr.mean()
    std_hr = hr.std()

    # RMSSD
    successive_diff = np.diff(rr_ms)
    rmssd = float(np.sqrt(np.mean(successive_diff**2))) if len(successive_diff) > 0 else 0.0

    # pNN50: fraction of |successive differences| > 50 ms
    pnn50 = float((np.abs(successive_diff) > 50.0).mean()) if len(successive_diff) > 0 else 0.0

    rr_cv = std_rr / mean_rr

    return np.array(
        [mean_hr, std_hr, mean_rr, std_rr, rmssd, pnn50, rr_cv, float(len(peaks))],
        dtype=np.float64,
    )
