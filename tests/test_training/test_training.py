"""Tests for src/training: baseline trainer and PyTorch Trainer."""
from pathlib import Path

import numpy as np
import pytest

from src.training.baseline import train_baseline

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=0)

N_TRAIN, N_VAL = 80, 20
N_FEATURES = 176  # matches src/features output
# Short signals for speed: 1 second at 500 Hz (not 10 s)
N_LEADS, N_SAMPLES = 12, 500


@pytest.fixture
def feature_data():
    """Small pre-extracted feature matrices with binary labels."""
    X_train = RNG.standard_normal((N_TRAIN, N_FEATURES)).astype(np.float32)
    y_train = RNG.integers(0, 2, size=N_TRAIN)
    X_val = RNG.standard_normal((N_VAL, N_FEATURES)).astype(np.float32)
    y_val = RNG.integers(0, 2, size=N_VAL)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def signal_data():
    """Small raw ECG signal arrays with binary labels (short signals for speed)."""
    X_train = RNG.standard_normal((N_TRAIN, N_LEADS, N_SAMPLES)).astype(np.float32)
    y_train = RNG.integers(0, 2, size=N_TRAIN)
    X_val = RNG.standard_normal((N_VAL, N_LEADS, N_SAMPLES)).astype(np.float32)
    y_val = RNG.integers(0, 2, size=N_VAL)
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# train_baseline
# ---------------------------------------------------------------------------

class TestTrainBaseline:
    @pytest.mark.parametrize("name", ["logistic", "random_forest", "gradient_boosting"])
    def test_returns_pipeline_and_results(self, name, feature_data) -> None:
        X_train, y_train, X_val, y_val = feature_data
        clf, results = train_baseline(X_train, y_train, X_val, y_val, name=name)
        assert clf is not None
        assert "train" in results
        assert "val" in results

    def test_results_contain_auc(self, feature_data) -> None:
        X_train, y_train, X_val, y_val = feature_data
        _, results = train_baseline(X_train, y_train, X_val, y_val)
        assert "roc_auc" in results["val"]
        assert "roc_auc" in results["train"]

    def test_auc_in_range(self, feature_data) -> None:
        X_train, y_train, X_val, y_val = feature_data
        _, results = train_baseline(X_train, y_train, X_val, y_val)
        assert 0.0 <= results["val"]["roc_auc"] <= 1.0

    def test_invalid_name_raises(self, feature_data) -> None:
        X_train, y_train, X_val, y_val = feature_data
        with pytest.raises(ValueError, match="Unknown baseline"):
            train_baseline(X_train, y_train, X_val, y_val, name="svm")

    def test_kwargs_forwarded(self, feature_data) -> None:
        X_train, y_train, X_val, y_val = feature_data
        clf, _ = train_baseline(X_train, y_train, X_val, y_val, name="logistic", C=0.01)
        assert clf.named_steps["clf"].C == 0.01


# ---------------------------------------------------------------------------
# TrainerConfig
# ---------------------------------------------------------------------------

class TestTrainerConfig:
    def test_defaults(self) -> None:
        from src.training.trainer import TrainerConfig
        cfg = TrainerConfig()
        assert cfg.epochs == 50
        assert cfg.batch_size == 32
        assert cfg.early_stopping_patience == 10
        assert cfg.device == "auto"

    def test_from_dict_known_keys(self) -> None:
        from src.training.trainer import TrainerConfig
        cfg = TrainerConfig.from_dict({"epochs": 10, "batch_size": 16, "unknown_key": "ignored"})
        assert cfg.epochs == 10
        assert cfg.batch_size == 16

    def test_from_dict_checkpoint_dir_as_path(self) -> None:
        from src.training.trainer import TrainerConfig
        cfg = TrainerConfig.from_dict({"checkpoint_dir": "my/checkpoints"})
        assert isinstance(cfg.checkpoint_dir, Path)

    def test_from_dict_empty(self) -> None:
        from src.training.trainer import TrainerConfig
        cfg = TrainerConfig.from_dict({})
        assert cfg.epochs == 50  # default unchanged


# ---------------------------------------------------------------------------
# Trainer (skipped if PyTorch not available)
# ---------------------------------------------------------------------------

torch = pytest.importorskip("torch", reason="PyTorch not installed")


class TestTrainer:
    @pytest.fixture
    def small_config(self, tmp_path):
        from src.training.trainer import TrainerConfig
        return TrainerConfig(
            epochs=2,
            batch_size=8,
            early_stopping_patience=5,
            checkpoint_dir=tmp_path / "ckpts",
            device="cpu",
        )

    @pytest.fixture
    def model(self):
        from src.models.cnn1d import CNN1D
        return CNN1D(n_leads=N_LEADS, n_classes=2, dropout=0.0)

    @pytest.fixture
    def trainer(self, model, small_config):
        from src.training.trainer import Trainer
        return Trainer(model, small_config)

    # -- fit --

    def test_fit_returns_history(self, trainer, signal_data) -> None:
        X_train, y_train, X_val, y_val = signal_data
        history = trainer.fit(X_train, y_train, X_val, y_val)
        assert isinstance(history, list)
        assert len(history) == 2  # 2 epochs

    def test_history_has_required_keys(self, trainer, signal_data) -> None:
        X_train, y_train, X_val, y_val = signal_data
        history = trainer.fit(X_train, y_train, X_val, y_val)
        for record in history:
            assert "epoch" in record
            assert "train_loss" in record
            assert "val_loss" in record
            assert "val_f1" in record

    def test_train_loss_is_positive(self, trainer, signal_data) -> None:
        X_train, y_train, X_val, y_val = signal_data
        history = trainer.fit(X_train, y_train, X_val, y_val)
        assert all(r["train_loss"] > 0 for r in history)

    def test_checkpoint_created(self, trainer, signal_data, small_config) -> None:
        X_train, y_train, X_val, y_val = signal_data
        trainer.fit(X_train, y_train, X_val, y_val)
        assert (small_config.checkpoint_dir / "best.pt").exists()

    # -- evaluate --

    def test_evaluate_returns_metrics(self, trainer, signal_data) -> None:
        X_train, y_train, X_val, y_val = signal_data
        trainer.fit(X_train, y_train, X_val, y_val)
        metrics = trainer.evaluate(X_val, y_val)
        assert "accuracy" in metrics
        assert "sensitivity" in metrics
        assert "roc_auc" in metrics

    def test_evaluate_metrics_in_range(self, trainer, signal_data) -> None:
        X_train, y_train, X_val, y_val = signal_data
        trainer.fit(X_train, y_train, X_val, y_val)
        metrics = trainer.evaluate(X_val, y_val)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    # -- save / load checkpoint --

    def test_save_and_load_checkpoint(self, trainer, signal_data, tmp_path) -> None:
        X_train, y_train, X_val, y_val = signal_data
        trainer.fit(X_train, y_train, X_val, y_val)

        ckpt_path = tmp_path / "test.pt"
        trainer.save_checkpoint(ckpt_path, epoch=1, metrics={"roc_auc": 0.75})

        # Corrupt weights, then reload
        for p in trainer.model.parameters():
            p.data.fill_(0.0)

        meta = trainer.load_checkpoint(ckpt_path)
        assert meta["epoch"] == 1
        assert meta["metrics"]["roc_auc"] == pytest.approx(0.75)

        # Model should not be all zeros after loading
        total_abs = sum(p.data.abs().sum().item() for p in trainer.model.parameters())
        assert total_abs > 0.0

    # -- early stopping --

    def test_early_stopping_stops_training(self, model, tmp_path, signal_data) -> None:
        from src.training.trainer import Trainer, TrainerConfig
        cfg = TrainerConfig(
            epochs=100,
            batch_size=8,
            early_stopping_patience=2,  # tight patience
            checkpoint_dir=tmp_path / "ckpts",
            device="cpu",
        )
        trainer = Trainer(model, cfg)
        X_train, y_train, X_val, y_val = signal_data
        history = trainer.fit(X_train, y_train, X_val, y_val)
        # Must have stopped before all 100 epochs
        assert len(history) < 100

    # -- device resolution --

    def test_resolve_device_cpu(self) -> None:
        from src.training.trainer import Trainer
        device = Trainer._resolve_device("cpu")
        assert device == torch.device("cpu")

    def test_resolve_device_auto_returns_torch_device(self) -> None:
        from src.training.trainer import Trainer
        device = Trainer._resolve_device("auto")
        assert isinstance(device, torch.device)
