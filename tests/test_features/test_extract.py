"""Tests for the ECG feature extraction pipeline."""
import numpy as np
import pytest

from src.features.extract import extract_features, extract_features_batch, get_feature_names
from src.features.frequency_domain import compute_frequency_features
from src.features.morphological import compute_morphological_features
from src.features.time_domain import compute_time_features

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ecg_signal() -> np.ndarray:
    """Single 12-lead ECG signal: random noise at 500 Hz, 10 s."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal((12, 5000)).astype(np.float32)


@pytest.fixture
def ecg_batch() -> np.ndarray:
    """Batch of 4 ECG signals."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal((4, 12, 5000)).astype(np.float32)


@pytest.fixture
def ecg_with_peaks() -> np.ndarray:
    """Synthetic 12-lead ECG with clear R-peaks at 70 BPM for Lead II.

    R-peaks are modelled as narrow Gaussians so scipy find_peaks can detect them.
    """
    fs, duration, hr_bpm = 500, 10, 70
    n_samples = fs * duration
    signal = np.zeros((12, n_samples), dtype=np.float32)

    rr_samples = int(fs * 60 / hr_bpm)  # ~428 samples between peaks
    rng = np.random.default_rng(seed=7)
    signal += rng.standard_normal(signal.shape).astype(np.float32) * 0.05

    t = np.arange(-15, 16)
    gaussian = np.exp(-(t**2) / (2 * 4**2)).astype(np.float32)

    for lead in range(12):
        for peak_centre in range(rr_samples // 2, n_samples, rr_samples):
            lo = peak_centre - 15
            hi = peak_centre + 16
            if lo >= 0 and hi <= n_samples:
                signal[lead, lo:hi] += gaussian * (2.0 if lead == 1 else 1.5)

    return signal


# ---------------------------------------------------------------------------
# Time-domain features
# ---------------------------------------------------------------------------

class TestTimeDomainFeatures:
    def test_output_length(self, ecg_signal: np.ndarray) -> None:
        result = compute_time_features(ecg_signal)
        assert result.shape == (12 * 8,)

    def test_output_dtype(self, ecg_signal: np.ndarray) -> None:
        result = compute_time_features(ecg_signal)
        assert result.dtype == np.float64

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="leads, samples"):
            compute_time_features(np.zeros((5000,)))

    def test_min_leq_max(self, ecg_signal: np.ndarray) -> None:
        feats = compute_time_features(ecg_signal).reshape(12, 8)
        # col 2 = min, col 3 = max
        assert np.all(feats[:, 2] <= feats[:, 3])

    def test_rms_positive(self, ecg_signal: np.ndarray) -> None:
        feats = compute_time_features(ecg_signal).reshape(12, 8)
        # col 4 = rms
        assert np.all(feats[:, 4] >= 0)

    def test_fewer_leads(self) -> None:
        rng = np.random.default_rng(1)
        signal = rng.standard_normal((6, 5000)).astype(np.float32)
        result = compute_time_features(signal)
        assert result.shape == (6 * 8,)


# ---------------------------------------------------------------------------
# Frequency-domain features
# ---------------------------------------------------------------------------

class TestFrequencyDomainFeatures:
    def test_output_length(self, ecg_signal: np.ndarray) -> None:
        result = compute_frequency_features(ecg_signal)
        assert result.shape == (12 * 6,)

    def test_output_dtype(self, ecg_signal: np.ndarray) -> None:
        result = compute_frequency_features(ecg_signal)
        assert result.dtype == np.float64

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="leads, samples"):
            compute_frequency_features(np.zeros((5000,)))

    def test_band_powers_non_negative(self, ecg_signal: np.ndarray) -> None:
        feats = compute_frequency_features(ecg_signal).reshape(12, 6)
        # cols 0,1,2 = power_low, power_mid, power_high
        assert np.all(feats[:, :3] >= 0)

    def test_spectral_entropy_bounded(self, ecg_signal: np.ndarray) -> None:
        feats = compute_frequency_features(ecg_signal).reshape(12, 6)
        # col 3 = spectral_entropy, should be in [0, 1]
        assert np.all(feats[:, 3] >= 0)
        assert np.all(feats[:, 3] <= 1.0 + 1e-9)

    def test_total_power_positive(self, ecg_signal: np.ndarray) -> None:
        feats = compute_frequency_features(ecg_signal).reshape(12, 6)
        # col 5 = total_power
        assert np.all(feats[:, 5] > 0)


# ---------------------------------------------------------------------------
# Morphological / HRV features
# ---------------------------------------------------------------------------

class TestMorphologicalFeatures:
    def test_output_length(self, ecg_signal: np.ndarray) -> None:
        result = compute_morphological_features(ecg_signal)
        assert result.shape == (8,)

    def test_output_dtype(self, ecg_signal: np.ndarray) -> None:
        result = compute_morphological_features(ecg_signal)
        assert result.dtype == np.float64

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="leads, samples"):
            compute_morphological_features(np.zeros((5000,)))

    def test_no_peaks_returns_zeros(self) -> None:
        # Flat signal → no R-peaks
        flat = np.zeros((12, 5000), dtype=np.float32)
        result = compute_morphological_features(flat)
        assert np.all(result == 0.0)

    def test_heart_rate_physiological(self, ecg_with_peaks: np.ndarray) -> None:
        result = compute_morphological_features(ecg_with_peaks, fs=500)
        mean_hr = result[0]
        # Normal physiological range 40–200 BPM
        assert 40 <= mean_hr <= 200, f"Unexpected mean HR: {mean_hr}"

    def test_n_beats_positive(self, ecg_with_peaks: np.ndarray) -> None:
        result = compute_morphological_features(ecg_with_peaks, fs=500)
        n_beats = result[7]
        assert n_beats > 0

    def test_pnn50_bounded(self, ecg_with_peaks: np.ndarray) -> None:
        result = compute_morphological_features(ecg_with_peaks, fs=500)
        pnn50 = result[5]
        assert 0.0 <= pnn50 <= 1.0


# ---------------------------------------------------------------------------
# Full pipeline — extract_features / extract_features_batch
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_output_length_12_leads(self, ecg_signal: np.ndarray) -> None:
        result = extract_features(ecg_signal)
        expected = 12 * 8 + 12 * 6 + 8  # 176
        assert result.shape == (expected,)

    def test_output_dtype(self, ecg_signal: np.ndarray) -> None:
        result = extract_features(ecg_signal)
        assert result.dtype == np.float64

    def test_no_nans(self, ecg_signal: np.ndarray) -> None:
        result = extract_features(ecg_signal)
        assert not np.any(np.isnan(result))

    def test_no_infs(self, ecg_signal: np.ndarray) -> None:
        result = extract_features(ecg_signal)
        assert not np.any(np.isinf(result))


class TestExtractFeaturesBatch:
    def test_output_shape(self, ecg_batch: np.ndarray) -> None:
        result = extract_features_batch(ecg_batch)
        assert result.shape == (4, 176)

    def test_output_dtype(self, ecg_batch: np.ndarray) -> None:
        result = extract_features_batch(ecg_batch)
        assert result.dtype == np.float64

    def test_wrong_ndim_raises(self, ecg_signal: np.ndarray) -> None:
        with pytest.raises(ValueError, match="n_records, leads, samples"):
            extract_features_batch(ecg_signal)

    def test_consistent_with_single(self, ecg_batch: np.ndarray) -> None:
        batch_result = extract_features_batch(ecg_batch)
        single_result = extract_features(ecg_batch[2])
        np.testing.assert_array_equal(batch_result[2], single_result)


# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

class TestGetFeatureNames:
    def test_length_12_leads(self) -> None:
        names = get_feature_names(n_leads=12)
        assert len(names) == 176

    def test_all_strings(self) -> None:
        names = get_feature_names()
        assert all(isinstance(n, str) for n in names)

    def test_no_duplicates(self) -> None:
        names = get_feature_names()
        assert len(names) == len(set(names))

    def test_length_6_leads(self) -> None:
        names = get_feature_names(n_leads=6)
        assert len(names) == 6 * 8 + 6 * 6 + 8
