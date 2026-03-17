"""Tests for ECG preprocessing functions."""
import numpy as np
import pytest

from src.data.preprocess import bandpass_filter, normalize, notch_filter, preprocess, resample


@pytest.fixture
def ecg_signal() -> np.ndarray:
    """Single 12-lead ECG signal at 500 Hz, 10 seconds."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal((12, 5000)).astype(np.float32)


@pytest.fixture
def ecg_batch() -> np.ndarray:
    """Batch of 8 ECG signals."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal((8, 12, 5000)).astype(np.float32)


class TestBandpassFilter:
    def test_output_shape_preserved(self, ecg_signal: np.ndarray) -> None:
        result = bandpass_filter(ecg_signal, fs=500)
        assert result.shape == ecg_signal.shape

    def test_output_dtype_float32(self, ecg_signal: np.ndarray) -> None:
        result = bandpass_filter(ecg_signal, fs=500)
        assert result.dtype == np.float32

    def test_reduces_dc_component(self, ecg_signal: np.ndarray) -> None:
        # Add DC offset, filter should remove it
        signal_with_dc = ecg_signal + 5.0
        result = bandpass_filter(signal_with_dc, fs=500)
        assert abs(result.mean()) < abs(signal_with_dc.mean())


class TestNotchFilter:
    def test_output_shape_preserved(self, ecg_signal: np.ndarray) -> None:
        result = notch_filter(ecg_signal, fs=500, notch_hz=50.0)
        assert result.shape == ecg_signal.shape

    def test_output_dtype_float32(self, ecg_signal: np.ndarray) -> None:
        result = notch_filter(ecg_signal, fs=500)
        assert result.dtype == np.float32


class TestNormalize:
    @pytest.mark.parametrize("method", ["zscore", "minmax", "robust"])
    def test_output_shape_preserved(self, ecg_signal: np.ndarray, method: str) -> None:
        result = normalize(ecg_signal, method=method)
        assert result.shape == ecg_signal.shape

    def test_zscore_zero_mean(self, ecg_signal: np.ndarray) -> None:
        result = normalize(ecg_signal, method="zscore")
        assert np.allclose(result.mean(axis=-1), 0, atol=1e-5)

    def test_zscore_unit_std(self, ecg_signal: np.ndarray) -> None:
        result = normalize(ecg_signal, method="zscore")
        assert np.allclose(result.std(axis=-1), 1, atol=1e-5)

    def test_minmax_range(self, ecg_signal: np.ndarray) -> None:
        result = normalize(ecg_signal, method="minmax")
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_invalid_method_raises(self, ecg_signal: np.ndarray) -> None:
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize(ecg_signal, method="invalid")

    def test_batch_input(self, ecg_batch: np.ndarray) -> None:
        result = normalize(ecg_batch, method="zscore")
        assert result.shape == ecg_batch.shape


class TestResample:
    def test_same_rate_returns_original(self, ecg_signal: np.ndarray) -> None:
        result = resample(ecg_signal, orig_fs=500, target_fs=500)
        np.testing.assert_array_equal(result, ecg_signal)

    def test_downsample_reduces_samples(self, ecg_signal: np.ndarray) -> None:
        result = resample(ecg_signal, orig_fs=500, target_fs=100)
        assert result.shape == (12, 1000)

    def test_upsample_increases_samples(self, ecg_signal: np.ndarray) -> None:
        result = resample(ecg_signal, orig_fs=100, target_fs=500)
        assert result.shape[0] == 12
        assert result.shape[1] > ecg_signal.shape[1]


class TestPreprocess:
    def test_output_shape_preserved(self, ecg_signal: np.ndarray) -> None:
        result = preprocess(ecg_signal, fs=500)
        assert result.shape == ecg_signal.shape

    def test_output_dtype_float32(self, ecg_signal: np.ndarray) -> None:
        result = preprocess(ecg_signal, fs=500)
        assert result.dtype == np.float32

    def test_batch_input(self, ecg_batch: np.ndarray) -> None:
        result = preprocess(ecg_batch, fs=500)
        assert result.shape == ecg_batch.shape

    def test_no_filters(self, ecg_signal: np.ndarray) -> None:
        result = preprocess(ecg_signal, fs=500, bandpass=False, notch=False)
        assert result.shape == ecg_signal.shape
