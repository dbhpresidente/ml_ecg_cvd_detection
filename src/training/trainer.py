"""PyTorch training loop for ECG classification (binary and multi-label).

Requires PyTorch.  Install with:  pip install torch

Features:
    - Binary mode   : CrossEntropyLoss, AUC-based early stopping.
    - Multilabel mode: BCEWithLogitsLoss, macro-AUC early stopping.
    - AdamW optimiser with configurable LR and weight decay.
    - Cosine annealing scheduler with optional linear warmup.
    - ReduceLROnPlateau (legacy / binary mode default).
    - AMP (fp16) with GradScaler for GPU training.
    - Gradient clipping (max_norm).
    - Best-checkpoint saving / loading.
    - Auto device selection (CUDA → MPS → CPU).

Usage (multilabel / xresnet1d):
    >>> from src.models.xresnet1d import xresnet1d101
    >>> from src.training.trainer import Trainer, TrainerConfig
    >>> model = xresnet1d101()
    >>> cfg = TrainerConfig(
    ...     mode="multilabel", num_classes=5,
    ...     scheduler_name="cosine", warmup_epochs=5,
    ...     use_amp=True, gradient_clip_val=1.0,
    ...     epochs=100, batch_size=32,
    ...     checkpoint_dir=Path("checkpoints"),
    ... )
    >>> trainer = Trainer(model, cfg)
    >>> history = trainer.fit(X_train, y_train, X_val, y_val)
"""
import dataclasses
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

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

# Superclass names (must match SUPERCLASSES order in loader.py)
_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """Hyperparameters and settings for the training loop.

    Attributes:
        mode: ``"binary"`` (CrossEntropyLoss, int64 labels) or
              ``"multilabel"`` (BCEWithLogitsLoss, float32 labels shape (n,C)).
        num_classes: Number of output classes (required for multilabel metrics).
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial learning rate for AdamW.
        weight_decay: L2 regularisation for AdamW.
        early_stopping_patience: Stop if val AUC does not improve this many epochs.
        use_amp: Enable automatic mixed precision (fp16). Requires CUDA.
        gradient_clip_val: Max gradient norm (0 = disabled).
        scheduler_name: ``"cosine"`` (cosine annealing + warmup),
                        ``"plateau"`` (ReduceLROnPlateau on val loss), or
                        ``"onecycle"`` (OneCycleLR — stepped per batch, fastai default).
        warmup_epochs: Linear LR warmup epochs (cosine scheduler only).
        lr_scheduler_patience: ReduceLROnPlateau patience (plateau scheduler only).
        lr_scheduler_factor: ReduceLROnPlateau factor (plateau scheduler only).
        lr_onecycle_pct_start: Fraction of training for LR ramp-up (onecycle only).
        lr_onecycle_div_factor: initial_lr = max_lr / div_factor (onecycle only).
        augment: Enable signal augmentation on the training set (amplitude
            scaling, Gaussian noise, lead dropout). Applied only during training.
        checkpoint_dir: Directory for saving checkpoints.
        device: ``"auto"`` selects CUDA → MPS → CPU.
    """
    mode: str = "binary"
    num_classes: int = 2
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    use_amp: bool = False
    gradient_clip_val: float = 0.0
    scheduler_name: str = "plateau"
    warmup_epochs: int = 0
    lr_scheduler_patience: int = 3
    lr_scheduler_factor: float = 0.5
    lr_onecycle_pct_start: float = 0.3
    lr_onecycle_div_factor: float = 25.0
    augment: bool = False
    use_pos_weight: bool = False
    label_smoothing: float = 0.0
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    device: str = "auto"
    resume_from: Path | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "TrainerConfig":
        """Build from a plain dict (unknown keys are silently ignored)."""
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        if "checkpoint_dir" in filtered:
            filtered["checkpoint_dir"] = Path(filtered["checkpoint_dir"])
        if "resume_from" in filtered and filtered["resume_from"] is not None:
            filtered["resume_from"] = Path(filtered["resume_from"])
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Internal PyTorch Dataset
# ---------------------------------------------------------------------------

class _ECGDataset(Dataset):
    """Thin PyTorch Dataset wrapping numpy signal and label arrays.

    Supports both int64 (binary) and float32 (multilabel) label arrays.
    When ``augment=True``, applies on-the-fly signal augmentation during training:
        - Amplitude scaling  : multiply each lead by U(0.75, 1.25)
        - Gaussian noise     : add N(0, 0.05) noise per sample
        - Lead dropout       : zero out 1–2 random leads with p=0.3
    """

    def __init__(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        self.signals = torch.from_numpy(np.asarray(signals, dtype=np.float32))
        if labels.dtype == np.float32 or labels.ndim == 2:
            self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        else:
            self.labels = torch.from_numpy(np.asarray(labels, dtype=np.int64))
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        signal = self.signals[idx].clone()
        if self.augment:
            signal = _augment_signal(signal)
        return signal, self.labels[idx]


def _augment_signal(x: "torch.Tensor") -> "torch.Tensor":
    """Apply random augmentations to a single ECG signal (leads, samples).

    Follows the Strodthoff/NeuroSyd PTB-XL benchmark augmentation protocol:
    - Amplitude scaling: one scalar per sample (not per lead) from U(0.8, 1.2).
      Per-lead scaling distorts inter-lead amplitude ratios that encode cardiac
      axis information — use a single scalar applied to all leads uniformly.
    - Gaussian noise: σ=0.01 absolute (signals are z-score normalised, O(1)).
      Previous σ=0.05·σ_lead was 5× too aggressive, suppressing P/T-wave detail.

    Args:
        x: Float32 tensor of shape (n_leads, samples).

    Returns:
        Augmented tensor of the same shape.
    """
    # Amplitude scaling: single scalar for the whole sample — preserves inter-lead ratios
    scale = torch.empty(1).uniform_(0.8, 1.2)
    x = x * scale

    # Gaussian noise: σ=0.01 absolute (Strodthoff benchmark protocol)
    x = x + torch.randn_like(x) * 0.01

    return x


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Training loop for ECG classification models.

    Args:
        model: Any nn.Module with signature forward(x) → logits.
        config: TrainerConfig.  Uses defaults if not provided.
    """

    def __init__(self, model: nn.Module, config: "TrainerConfig | None" = None) -> None:
        self.config = config or TrainerConfig()
        self.device = self._resolve_device(self.config.device)
        self.model = model.to(self.device)
        self._scaler: "torch.cuda.amp.GradScaler | None" = None
        if self.config.use_amp and self.device.type == "cuda":
            self._scaler = torch.amp.GradScaler("cuda")
            logger.info("AMP (fp16) enabled")
        logger.info(
            "Trainer — mode: %s  device: %s  params: %s",
            self.config.mode, self.device, _fmt_params(model),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict]:
        """Train with early stopping and save the best checkpoint.

        Args:
            X_train: Signals, shape (n, leads, samples).
            y_train: Binary → (n,) int64.  Multilabel → (n, C) float32.
            X_val:   Validation signals.
            y_val:   Validation labels.

        Returns:
            History: list of dicts with epoch metrics.
        """
        cfg = self.config
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            _ECGDataset(X_train, y_train, augment=cfg.augment),
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
            num_workers=0,
        )
        val_loader = DataLoader(
            _ECGDataset(X_val, y_val),
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=(self.device.type == "cuda"),
            num_workers=0,
        )

        criterion = self._build_criterion(y_train)
        optimizer = self._build_optimizer()

        # OneCycleLR must be built after DataLoader (needs steps_per_epoch)
        if cfg.scheduler_name == "onecycle":
            total_steps = cfg.epochs * len(train_loader)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.learning_rate,
                total_steps=total_steps,
                pct_start=cfg.lr_onecycle_pct_start,
                div_factor=cfg.lr_onecycle_div_factor,
                final_div_factor=1e4,
                anneal_strategy="cos",
            )
            batch_scheduler = scheduler
            logger.info(
                "OneCycleLR: max_lr=%.2e  total_steps=%d  pct_start=%.2f",
                cfg.learning_rate, total_steps, cfg.lr_onecycle_pct_start,
            )
        else:
            scheduler = self._build_scheduler(optimizer)
            batch_scheduler = None

        best_val_auc: float = -1.0
        patience_counter: int = 0
        history: list[dict] = []
        start_epoch: int = 1

        # --- Resume from checkpoint -------------------------------------------
        if cfg.resume_from is not None and cfg.resume_from.exists():
            ckpt = torch.load(cfg.resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_auc = ckpt.get("best_val_auc", ckpt["metrics"].get(
                "macro_auc", ckpt["metrics"].get("roc_auc", -1.0)
            ))
            patience_counter = ckpt.get("patience_counter", 0)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                target = batch_scheduler if batch_scheduler is not None else scheduler
                target.load_state_dict(ckpt["scheduler_state_dict"])
            elif cfg.scheduler_name == "onecycle":
                # Partial resume (best.pt only has model weights): rebuild OneCycleLR
                # for remaining epochs so the lr cycle is correctly scoped.
                remaining_epochs = cfg.epochs - ckpt["epoch"]
                remaining_steps = remaining_epochs * len(train_loader)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=cfg.learning_rate,
                    total_steps=remaining_steps,
                    pct_start=cfg.lr_onecycle_pct_start,
                    div_factor=cfg.lr_onecycle_div_factor,
                    final_div_factor=1e4,
                    anneal_strategy="cos",
                )
                batch_scheduler = scheduler
                logger.info(
                    "OneCycleLR rebuilt for resume: max_lr=%.2e  remaining_steps=%d",
                    cfg.learning_rate, remaining_steps,
                )
            logger.info(
                "Resumed from '%s'  start_epoch=%d  best_auc=%.4f  patience=%d",
                cfg.resume_from, start_epoch, best_val_auc, patience_counter,
            )
        # ----------------------------------------------------------------------

        last_ckpt_path = cfg.checkpoint_dir / "last.pt"

        for epoch in range(start_epoch, cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader, criterion, optimizer, batch_scheduler)
            val_loss, y_true, y_prob = self._eval_epoch(val_loader, criterion)

            if cfg.mode == "multilabel":
                val_metrics = _compute_multilabel_metrics(y_true, y_prob)
            else:
                y_pred = (y_prob >= 0.5).astype(int)
                val_metrics = compute_metrics(y_true, y_pred, y_prob)

            val_auc = val_metrics.get("macro_auc", val_metrics.get("roc_auc", 0.0))

            # Scheduler step (plateau/cosine only — onecycle steps per batch)
            if cfg.scheduler_name == "plateau":
                scheduler.step(val_loss)
            elif cfg.scheduler_name != "onecycle":
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_record: dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(epoch_record)

            if cfg.mode == "multilabel":
                per_class = val_metrics.get("per_class_auc", {})
                class_str = "  ".join(
                    f"{c}:{per_class.get(c, 0):.3f}" for c in _SUPERCLASSES
                )
                logger.info(
                    "Epoch %3d/%d  loss=%.4f  macro_auc=%.4f  lr=%.2e | %s",
                    epoch, cfg.epochs, train_loss, val_auc, current_lr, class_str,
                )
            else:
                logger.info(
                    "Epoch %3d/%d  train=%.4f  val=%.4f  auc=%.4f  f1=%.4f  lr=%.2e",
                    epoch, cfg.epochs, train_loss, val_loss,
                    val_auc, val_metrics.get("f1", float("nan")), current_lr,
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
                    # Save last.pt before stopping so resume is possible
                    self._save_full_checkpoint(
                        last_ckpt_path, epoch, val_metrics, optimizer,
                        batch_scheduler or scheduler, best_val_auc, patience_counter,
                    )
                    break

            # Save last.pt for resumability
            self._save_full_checkpoint(
                last_ckpt_path, epoch, val_metrics, optimizer,
                batch_scheduler or scheduler, best_val_auc, patience_counter,
            )

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the model on a held-out split.

        Args:
            X: Signals, shape (n, leads, samples).
            y: Labels matching the configured mode.

        Returns:
            Metrics dict.
        """
        loader = DataLoader(
            _ECGDataset(X, y),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )
        criterion = self._build_criterion(y)
        _, y_true, y_prob = self._eval_epoch(loader, criterion)

        if self.config.mode == "multilabel":
            return _compute_multilabel_metrics(y_true, y_prob)
        y_pred = (y_prob >= 0.5).astype(int)
        return compute_metrics(y_true, y_pred, y_prob)

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict) -> None:
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict(), "metrics": metrics},
            path,
        )

    def _save_full_checkpoint(
        self,
        path: Path,
        epoch: int,
        metrics: dict,
        optimizer: "torch.optim.Optimizer",
        scheduler: object,
        best_val_auc: float,
        patience_counter: int,
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": metrics,
                "best_val_auc": best_val_auc,
                "patience_counter": patience_counter,
            },
            path,
        )

    def load_checkpoint(self, path: Path) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        auc = checkpoint["metrics"].get(
            "macro_auc", checkpoint["metrics"].get("roc_auc", float("nan"))
        )
        logger.info(
            "Loaded checkpoint '%s'  epoch=%d  val_auc=%.4f",
            path, checkpoint["epoch"], auc,
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
        # SSM parameters (eigenvalues, timescales, B/C matrices) must NOT receive
        # weight decay — wd shrinks A_real/log_dt toward zero, pinning all eigenvalues
        # at -ln(2) and all timescales at dt=1, destroying learned kernel diversity.
        # This follows Gu et al. 2022 (S4D) reference implementation practice.
        _SSM_PARAM_KEYWORDS = ("A_real", "A_imag", "B_real", "B_imag",
                               "C_real", "C_imag", "log_dt")
        ssm_params, other_params = [], []
        for name, p in self.model.named_parameters():
            if any(k in name for k in _SSM_PARAM_KEYWORDS):
                ssm_params.append(p)
            else:
                other_params.append(p)
        if ssm_params:
            return torch.optim.AdamW(
                [
                    {"params": other_params, "weight_decay": self.config.weight_decay},
                    {"params": ssm_params,   "weight_decay": 0.0},
                ],
                lr=self.config.learning_rate,
            )
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_criterion(self, y_train: np.ndarray) -> nn.Module:
        if self.config.mode == "multilabel":
            pos_weight = None
            if self.config.use_pos_weight:
                n_pos = y_train.sum(axis=0).clip(min=1)
                n_neg = len(y_train) - n_pos
                pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=self.device)
                logger.info(
                    "pos_weight per class: %s",
                    {c: f"{w:.3f}" for c, w in zip(_SUPERCLASSES, pos_weight.tolist())},
                )
            eps = self.config.label_smoothing
            if eps > 0.0:
                # Label smoothing for multilabel BCE: soft targets y → y*(1-ε) + ε/2
                # Reduces overconfidence and improves calibration on imbalanced classes.
                logger.info("Label smoothing enabled: ε=%.3f", eps)
                base_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
                smooth = eps

                class _SmoothedBCE(nn.Module):
                    def forward(self, logits, targets):
                        soft = targets * (1.0 - smooth) + 0.5 * smooth
                        return base_loss(logits, soft).mean()

                return _SmoothedBCE()
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # Binary: inverse-frequency class weights
        n_neg = max(int((y_train == 0).sum()), 1)
        n_pos = max(int((y_train == 1).sum()), 1)
        total = n_neg + n_pos
        weight = torch.tensor(
            [total / (2.0 * n_neg), total / (2.0 * n_pos)],
            device=self.device, dtype=torch.float32,
        )
        return nn.CrossEntropyLoss(weight=weight)

    def _build_scheduler(
        self, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        cfg = self.config
        if cfg.scheduler_name == "cosine":
            return _cosine_schedule_with_warmup(optimizer, cfg.warmup_epochs, cfg.epochs)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=cfg.lr_scheduler_patience,
            factor=cfg.lr_scheduler_factor,
        )

    def _train_epoch(
        self,
        loader: "DataLoader",
        criterion: nn.Module,
        optimizer: "torch.optim.Optimizer",
        batch_scheduler: "torch.optim.lr_scheduler.LRScheduler | None" = None,
    ) -> float:
        self.model.train()
        total_loss = 0.0
        cfg = self.config
        amp_enabled = self._scaler is not None

        for signals, labels in loader:
            signals = signals.to(self.device, non_blocking=True)
            labels  = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = criterion(self.model(signals), labels)
                self._scaler.scale(loss).backward()
                if cfg.gradient_clip_val > 0:
                    self._scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip_val)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                loss = criterion(self.model(signals), labels)
                loss.backward()
                if cfg.gradient_clip_val > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip_val)
                optimizer.step()

            if batch_scheduler is not None:
                batch_scheduler.step()

            total_loss += loss.item() * len(labels)

        return total_loss / len(loader.dataset)

    def _eval_epoch(
        self,
        loader: "DataLoader",
        criterion: nn.Module,
    ) -> "tuple[float, np.ndarray, np.ndarray]":
        """Returns (avg_loss, y_true, y_prob).

        y_prob shape: (n,) for binary, (n, C) for multilabel.
        """
        self.model.eval()
        total_loss = 0.0
        all_labels, all_probs = [], []

        with torch.no_grad():
            for signals, labels in loader:
                signals = signals.to(self.device, non_blocking=True)
                labels  = labels.to(self.device, non_blocking=True)

                logits = self.model(signals)
                total_loss += criterion(logits, labels).item() * len(labels)

                if self.config.mode == "multilabel":
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs)

        return (
            total_loss / len(loader.dataset),
            np.concatenate(all_labels),
            np.concatenate(all_probs),
        )


# ---------------------------------------------------------------------------
# Multilabel metrics helper
# ---------------------------------------------------------------------------

def _compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """Compute macro-AUC and per-class AUC for 5-class multilabel output.

    Args:
        y_true: Ground-truth array, shape (n, 5), float32.
        y_prob: Sigmoid probabilities, shape (n, 5), float32.

    Returns:
        Dict with ``macro_auc`` and ``per_class_auc`` (dict keyed by class name).
    """
    n_classes = y_true.shape[1]
    class_names = _SUPERCLASSES[:n_classes]
    per_class: dict[str, float] = {}

    aucs = []
    for i, name in enumerate(class_names):
        col_true = y_true[:, i]
        # Skip classes with only one label value present
        if len(np.unique(col_true)) < 2:
            logger.warning("Class %s has only one unique label — skipping AUC", name)
            continue
        auc = float(roc_auc_score(col_true, y_prob[:, i]))
        per_class[name] = auc
        aucs.append(auc)

    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    return {"macro_auc": macro_auc, "per_class_auc": per_class}


# ---------------------------------------------------------------------------
# Cosine scheduler with linear warmup
# ---------------------------------------------------------------------------

def _cosine_schedule_with_warmup(
    optimizer: "torch.optim.Optimizer",
    warmup_epochs: int,
    total_epochs: int,
    min_lr_factor: float = 0.01,
) -> "torch.optim.lr_scheduler.LambdaLR":
    """LambdaLR with linear warmup followed by cosine annealing.

    Args:
        optimizer:      Wrapped optimizer.
        warmup_epochs:  Number of linear warmup epochs.
        total_epochs:   Total training epochs.
        min_lr_factor:  LR floor as a fraction of base LR (default 1%).
    """
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return max(min_lr_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fmt_params(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{n:,}"
