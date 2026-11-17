"""PyTorch training loop for CNN1D ECG classification.

Requires PyTorch.  Install with:  pip install torch

Features:
    - AdamW optimiser with configurable LR and weight decay.
    - Class-balanced CrossEntropyLoss (handles CVD/NORM imbalance).
    - ReduceLROnPlateau scheduler on validation loss.
    - Early stopping on validation ROC-AUC.
    - Best-checkpoint saving / loading.
    - Auto device selection (CUDA → MPS → CPU).

Usage:
    >>> from src.models.cnn1d import CNN1D
    >>> from src.training.trainer import Trainer, TrainerConfig
    >>> model = CNN1D()
    >>> cfg = TrainerConfig(epochs=50, batch_size=32, checkpoint_dir=Path("checkpoints"))
    >>> trainer = Trainer(model, cfg)
    >>> history = trainer.fit(X_train, y_train, X_val, y_val)
    >>> test_metrics = trainer.evaluate(X_test, y_test)
"""
import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for Trainer.  Install it with:  pip install torch"
    ) from exc

from src.models.metrics import compute_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Hyperparameters and paths for the CNN training loop.

    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Mini-batch size for DataLoader.
        learning_rate: Initial learning rate for AdamW.
        weight_decay: L2 regularisation coefficient for AdamW.
        early_stopping_patience: Stop training if val AUC does not improve
            for this many consecutive epochs.
        lr_scheduler_patience: Reduce LR if val loss does not improve for
            this many epochs (ReduceLROnPlateau).
        lr_scheduler_factor: Factor by which LR is reduced on plateau.
        checkpoint_dir: Directory for saving checkpoints.
        device: Compute device.  ``'auto'`` selects CUDA → MPS → CPU.
    """

    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    device: str = "auto"

    @classmethod
    def from_dict(cls, d: dict) -> "TrainerConfig":
        """Build a TrainerConfig from a plain dict (e.g., parsed from YAML).

        Unknown keys are silently ignored so the full config dict can be passed.

        Args:
            d: Dictionary with optional keys matching TrainerConfig fields.

        Returns:
            TrainerConfig instance.
        """
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        if "checkpoint_dir" in filtered:
            filtered["checkpoint_dir"] = Path(filtered["checkpoint_dir"])
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Internal PyTorch Dataset wrapper
# ---------------------------------------------------------------------------

class _ECGDataset(Dataset):
    """Thin PyTorch Dataset wrapping numpy signal and label arrays."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray) -> None:
        self.signals = torch.from_numpy(np.asarray(signals, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        return self.signals[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Training loop for CNN1D on ECG data.

    Args:
        model: Instantiated CNN1D (or any nn.Module with the same interface).
        config: TrainerConfig.  Uses defaults if not provided.

    Example:
        >>> model = CNN1D(n_leads=12, n_classes=2)
        >>> trainer = Trainer(model, TrainerConfig(epochs=30))
        >>> history = trainer.fit(X_train, y_train, X_val, y_val)
        >>> metrics = trainer.evaluate(X_test, y_test)
    """

    def __init__(self, model: nn.Module, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.device = self._resolve_device(self.config.device)
        self.model = model.to(self.device)
        logger.info("Trainer — device: %s  params: %s",
                    self.device, _fmt_params(model))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict[str, float]]:
        """Train with early stopping and save the best checkpoint.

        Args:
            X_train: Training signals, shape (n, leads, samples).
            y_train: Training binary labels, shape (n,).
            X_val: Validation signals, shape (n, leads, samples).
            y_val: Validation binary labels, shape (n,).

        Returns:
            History: list of one dict per epoch with keys
            ``epoch``, ``train_loss``, ``val_loss``, and all val metrics
            prefixed with ``val_`` (e.g. ``val_roc_auc``, ``val_f1``).
        """
        cfg = self.config
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            _ECGDataset(X_train, y_train),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            _ECGDataset(X_val, y_val),
            batch_size=cfg.batch_size,
            shuffle=False,
        )

        criterion = self._build_criterion(y_train)
        optimizer = self._build_optimizer()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=cfg.lr_scheduler_patience,
            factor=cfg.lr_scheduler_factor,
        )

        best_val_auc: float = -1.0
        patience_counter: int = 0
        history: list[dict[str, float]] = []

        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            val_loss, y_true, y_pred, y_prob = self._eval_epoch(val_loader, criterion)

            val_metrics = compute_metrics(y_true, y_pred, y_prob)
            val_auc = val_metrics.get("roc_auc", 0.0)

            scheduler.step(val_loss)

            epoch_record: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(epoch_record)

            logger.info(
                "Epoch %3d/%d  train=%.4f  val=%.4f  auc=%.4f  f1=%.4f  "
                "sens=%.4f  spec=%.4f",
                epoch, cfg.epochs,
                train_loss, val_loss, val_auc, val_metrics["f1"],
                val_metrics["sensitivity"], val_metrics["specificity"],
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                self.save_checkpoint(
                    cfg.checkpoint_dir / "best.pt", epoch, val_metrics
                )
                logger.info("  -> New best val_auc=%.4f  checkpoint saved", best_val_auc)
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    logger.info(
                        "Early stopping at epoch %d  (patience=%d, best_auc=%.4f)",
                        epoch, cfg.early_stopping_patience, best_val_auc,
                    )
                    break

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """Evaluate the model on a held-out split.

        Args:
            X: Signals, shape (n, leads, samples).
            y: Binary labels, shape (n,).

        Returns:
            Metrics dict from compute_metrics() (accuracy, AUC, F1, …).
        """
        loader = DataLoader(
            _ECGDataset(X, y),
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        criterion = nn.CrossEntropyLoss()
        _, y_true, y_pred, y_prob = self._eval_epoch(loader, criterion)
        return compute_metrics(y_true, y_pred, y_prob)

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save model weights and metadata to a .pt file.

        Args:
            path: Destination file path.
            epoch: Current epoch (stored as metadata).
            metrics: Validation metrics at this point (stored as metadata).
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "metrics": metrics,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict[str, object]:
        """Load model weights from a checkpoint file.

        Args:
            path: Path to a .pt checkpoint produced by save_checkpoint().

        Returns:
            Metadata dict with keys ``epoch`` and ``metrics``
            (model weights are loaded in-place into self.model).
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(
            "Loaded checkpoint '%s'  epoch=%d  val_auc=%.4f",
            path,
            checkpoint["epoch"],
            checkpoint["metrics"].get("roc_auc", float("nan")),
        )
        return {"epoch": checkpoint["epoch"], "metrics": checkpoint["metrics"]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> "torch.device":
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _build_optimizer(self) -> "torch.optim.Optimizer":
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_criterion(self, y_train: np.ndarray) -> "nn.CrossEntropyLoss":
        """CrossEntropyLoss with inverse-frequency class weights."""
        n_neg = max(int((y_train == 0).sum()), 1)
        n_pos = max(int((y_train == 1).sum()), 1)
        total = n_neg + n_pos
        weight = torch.tensor(
            [total / (2.0 * n_neg), total / (2.0 * n_pos)],
            device=self.device,
            dtype=torch.float32,
        )
        return nn.CrossEntropyLoss(weight=weight)

    def _train_epoch(
        self,
        loader: "DataLoader",
        criterion: "nn.CrossEntropyLoss",
        optimizer: "torch.optim.Optimizer",
    ) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0

        for signals, labels in loader:
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            loss = criterion(self.model(signals), labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

        return total_loss / len(loader.dataset)

    def _eval_epoch(
        self,
        loader: "DataLoader",
        criterion: "nn.CrossEntropyLoss",
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Run one evaluation epoch. Returns (avg_loss, y_true, y_pred, y_prob)."""
        self.model.eval()
        total_loss = 0.0
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for signals, labels in loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(signals)
                total_loss += criterion(logits, labels).item() * len(labels)

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds)
                all_probs.append(probs)

        avg_loss = total_loss / len(loader.dataset)
        return (
            avg_loss,
            np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs),
        )


def _fmt_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{n:,}"
