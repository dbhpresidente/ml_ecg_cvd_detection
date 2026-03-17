import numpy as np
import pytest


@pytest.fixture
def sample_ecg() -> np.ndarray:
    """Generate a synthetic 12-lead ECG signal for testing."""
    rng = np.random.default_rng(seed=42)
    return rng.standard_normal((12, 5000))  # 12-lead, 10s at 500Hz


@pytest.fixture
def sample_labels() -> list[int]:
    return [0, 1, 0, 1, 0]


@pytest.fixture
def sample_batch() -> tuple[np.ndarray, np.ndarray]:
    """Generate a small batch of ECG signals with labels."""
    rng = np.random.default_rng(seed=42)
    signals = rng.standard_normal((8, 12, 5000))
    labels = rng.integers(0, 2, size=8)
    return signals, labels
