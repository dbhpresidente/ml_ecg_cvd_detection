"""ECG training pipelines.

Public API:
    train_baseline : Fit and evaluate a sklearn baseline on extracted features.
    Trainer        : PyTorch training loop for CNN1D (requires torch).
    TrainerConfig  : Dataclass with CNN training hyperparameters.
"""
from src.training.baseline import train_baseline
from src.training.trainer import Trainer, TrainerConfig

__all__ = ["train_baseline", "Trainer", "TrainerConfig"]
