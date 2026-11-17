"""Tests for src/models: metrics, baselines, and CNN1D."""
import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.models.baseline import BASELINE_NAMES, build_baseline
from src.models.metrics import BASE_METRIC_KEYS, PROB_METRIC_KEYS, compute_metrics

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=42)


@pytest.fixture
def binary_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perfect predictions: y_pred == y_true, y_prob near 0/1."""
    y_true = RNG.integers(0, 2, size=100)
    y_pred = y_true.copy()
    y_prob = y_true.astype(float) * 0.9 + (1 - y_true) * 0.1
    return y_true, y_pred, y_prob


@pytest.fixture
def noisy_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Imperfect but non-trivial predictions."""
    y_true = RNG.integers(0, 2, size=200)
    noise = RNG.integers(0, 2, size=200)
    y_pred = np.where(noise < 1, y_true, 1 - y_true)  # ~50% correct
    y_prob = RNG.uniform(0, 1, size=200)
    return y_true, y_pred, y_prob


@pytest.fixture
def feature_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Small feature matrix (n_samples, n_features) with binary labels."""
    X = RNG.standard_normal((120, 176))  # 176 = feature dim from src/features
    y = RNG.integers(0, 2, size=120)
    return X, y


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_base_keys_always_present(self, binary_labels) -> None:
        y_true, y_pred, _ = binary_labels
        result = compute_metrics(y_true, y_pred)
        for key in BASE_METRIC_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_prob_keys_absent_without_y_prob(self, binary_labels) -> None:
        y_true, y_pred, _ = binary_labels
        result = compute_metrics(y_true, y_pred)
        for key in PROB_METRIC_KEYS:
            assert key not in result

    def test_prob_keys_present_with_y_prob(self, binary_labels) -> None:
        y_true, y_pred, y_prob = binary_labels
        result = compute_metrics(y_true, y_pred, y_prob)
        for key in PROB_METRIC_KEYS:
            assert key in result

    def test_perfect_predictions_accuracy_one(self, binary_labels) -> None:
        y_true, y_pred, _ = binary_labels
        result = compute_metrics(y_true, y_pred)
        assert result["accuracy"] == pytest.approx(1.0)

    def test_perfect_predictions_auc_one(self, binary_labels) -> None:
        y_true, y_pred, y_prob = binary_labels
        result = compute_metrics(y_true, y_pred, y_prob)
        assert result["roc_auc"] == pytest.approx(1.0, abs=0.01)

    def test_all_values_are_floats(self, noisy_labels) -> None:
        y_true, y_pred, y_prob = noisy_labels
        result = compute_metrics(y_true, y_pred, y_prob)
        for key, val in result.items():
            assert isinstance(val, float), f"Expected float for '{key}', got {type(val)}"

    def test_sensitivity_in_range(self, noisy_labels) -> None:
        y_true, y_pred, _ = noisy_labels
        result = compute_metrics(y_true, y_pred)
        assert 0.0 <= result["sensitivity"] <= 1.0

    def test_specificity_in_range(self, noisy_labels) -> None:
        y_true, y_pred, _ = noisy_labels
        result = compute_metrics(y_true, y_pred)
        assert 0.0 <= result["specificity"] <= 1.0

    def test_mcc_in_range(self, noisy_labels) -> None:
        y_true, y_pred, _ = noisy_labels
        result = compute_metrics(y_true, y_pred)
        assert -1.0 <= result["mcc"] <= 1.0

    def test_optimal_threshold_in_zero_one(self, binary_labels) -> None:
        y_true, y_pred, y_prob = binary_labels
        result = compute_metrics(y_true, y_pred, y_prob)
        assert 0.0 <= result["optimal_threshold"] <= 1.0

    def test_all_same_class_does_not_crash(self) -> None:
        y_true = np.zeros(50, dtype=int)
        y_pred = np.zeros(50, dtype=int)
        result = compute_metrics(y_true, y_pred)
        assert "accuracy" in result


# ---------------------------------------------------------------------------
# build_baseline
# ---------------------------------------------------------------------------

class TestBuildBaseline:
    @pytest.mark.parametrize("name", BASELINE_NAMES)
    def test_returns_pipeline(self, name: str) -> None:
        clf = build_baseline(name)
        assert isinstance(clf, Pipeline)

    def test_invalid_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown baseline"):
            build_baseline("svm")

    @pytest.mark.parametrize("name", BASELINE_NAMES)
    def test_fit_predict(self, name: str, feature_dataset) -> None:
        X, y = feature_dataset
        split = 80
        clf = build_baseline(name)
        clf.fit(X[:split], y[:split])
        y_pred = clf.predict(X[split:])
        assert y_pred.shape == (len(X) - split,)
        assert set(y_pred).issubset({0, 1})

    @pytest.mark.parametrize("name", BASELINE_NAMES)
    def test_predict_proba(self, name: str, feature_dataset) -> None:
        X, y = feature_dataset
        clf = build_baseline(name)
        clf.fit(X[:80], y[:80])
        proba = clf.predict_proba(X[80:])
        assert proba.shape == (len(X) - 80, 2)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_logistic_has_scaler(self) -> None:
        clf = build_baseline("logistic")
        assert "scaler" in clf.named_steps

    def test_tree_models_no_scaler(self) -> None:
        for name in ("random_forest", "gradient_boosting"):
            clf = build_baseline(name)
            assert "scaler" not in clf.named_steps

    def test_kwargs_forwarded(self) -> None:
        clf = build_baseline("logistic", C=0.01)
        assert clf.named_steps["clf"].C == 0.01


# ---------------------------------------------------------------------------
# CNN1D (skipped if PyTorch is not installed)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="PyTorch not installed")


class TestCNN1D:
    @pytest.fixture
    def model(self):
        from src.models.cnn1d import CNN1D
        return CNN1D(n_leads=12, n_classes=2, dropout=0.0)

    def test_output_shape_default(self, model) -> None:
        x = torch.randn(4, 12, 5000)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_output_shape_batch_1(self, model) -> None:
        x = torch.randn(1, 12, 5000)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_output_shape_low_res(self) -> None:
        from src.models.cnn1d import CNN1D
        model = CNN1D(n_leads=12, n_classes=2)
        x = torch.randn(4, 12, 1000)  # 100 Hz × 10 s
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2)

    def test_no_nan_in_output(self, model) -> None:
        x = torch.randn(8, 12, 5000)
        with torch.no_grad():
            out = model(x)
        assert not torch.isnan(out).any()

    def test_count_parameters_positive(self, model) -> None:
        assert model.count_parameters() > 0

    def test_custom_n_classes(self) -> None:
        from src.models.cnn1d import CNN1D
        model = CNN1D(n_leads=12, n_classes=5)
        x = torch.randn(2, 12, 5000)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 5)

    def test_custom_n_leads(self) -> None:
        from src.models.cnn1d import CNN1D
        model = CNN1D(n_leads=1, n_classes=2)
        x = torch.randn(2, 1, 5000)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_gradients_flow(self, model) -> None:
        x = torch.randn(2, 12, 5000, requires_grad=False)
        out = model(x)
        loss = out.sum()
        loss.backward()
        # At least one parameter should have a gradient
        assert any(p.grad is not None for p in model.parameters())
