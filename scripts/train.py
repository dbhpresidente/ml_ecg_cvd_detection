"""Training entry point for ECG CVD detection.

Supports both sklearn baselines (using extracted features) and the CNN1D model
(end-to-end on raw signals), selected via model.architecture in the config.

Usage:
    # CNN (default config)
    python scripts/train.py

    # Baseline with custom data dir
    python scripts/train.py --config configs/baseline.yaml --data-dir data/raw/ptb-xl

    # Override seed and run name
    python scripts/train.py --seed 7 --run-name experiment_01
"""
import argparse
import json
import logging
import pickle
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("pyyaml is required: pip install pyyaml") from exc
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
        force=True,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    logger.info("Saved: %s", path)


def _print_metrics(metrics: dict[str, float], split: str) -> None:
    logger.info(
        "[%s]  AUC=%.4f  F1=%.4f  Acc=%.4f  Sens=%.4f  Spec=%.4f  MCC=%.4f",
        split,
        metrics.get("roc_auc", float("nan")),
        metrics.get("f1", float("nan")),
        metrics.get("accuracy", float("nan")),
        metrics.get("sensitivity", float("nan")),
        metrics.get("specificity", float("nan")),
        metrics.get("mcc", float("nan")),
    )


# ---------------------------------------------------------------------------
# Training routines
# ---------------------------------------------------------------------------

def _train_baseline(cfg: dict, dataset, run_dir: Path) -> None:
    """Extract features and fit a sklearn baseline pipeline."""
    from src.features import extract_features_batch
    from src.training.baseline import train_baseline

    architecture = cfg["model"]["architecture"]
    fs = cfg["data"]["sampling_rate"]

    logger.info("=== Baseline: %s ===", architecture)

    logger.info("Loading train split...")
    X_train_raw, y_train = dataset.get_split("train")
    logger.info("Loading val split...")
    X_val_raw, y_val = dataset.get_split("val")

    logger.info("Extracting features (train)...")
    X_train = extract_features_batch(X_train_raw, fs=fs)
    logger.info("Extracting features (val)...")
    X_val = extract_features_batch(X_val_raw, fs=fs)

    clf, results = train_baseline(X_train, y_train, X_val, y_val, name=architecture)

    _print_metrics(results["train"], "train")
    _print_metrics(results["val"], "val")

    model_path = run_dir / "model.pkl"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info("Model saved: %s", model_path)

    _save_json(results, run_dir / "metrics.json")


def _train_cnn(cfg: dict, dataset, run_dir: Path, checkpoint_dir: Path) -> None:
    """Train the CNN1D model with the PyTorch Trainer."""
    try:
        from src.models.cnn1d import CNN1D
        from src.training.trainer import Trainer, TrainerConfig
    except ImportError as exc:
        raise SystemExit("PyTorch is required for CNN training: pip install torch") from exc

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = CNN1D(
        n_leads=data_cfg["leads"],
        n_classes=model_cfg["num_classes"],
        dropout=model_cfg.get("dropout", 0.3),
    )
    logger.info("CNN1D: %s parameters", f"{model.count_parameters():,}")

    trainer_cfg = TrainerConfig.from_dict({
        **cfg.get("training", {}),
        "checkpoint_dir": checkpoint_dir,
        "device": "auto",
    })

    logger.info("=== CNN1D training ===")
    logger.info("Loading train split...")
    X_train, y_train = dataset.get_split("train")
    logger.info("Loading val split...")
    X_val, y_val = dataset.get_split("val")

    trainer = Trainer(model, trainer_cfg)
    history = trainer.fit(X_train, y_train, X_val, y_val)

    _save_json({"history": history}, run_dir / "history.json")

    # Evaluate best checkpoint on val
    best_ckpt = checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt)
        val_metrics = trainer.evaluate(X_val, y_val)
        _print_metrics(val_metrics, "val (best ckpt)")
        _save_json(val_metrics, run_dir / "metrics_val.json")


def _train_xresnet(cfg: dict, dataset, run_dir: Path, checkpoint_dir: Path) -> None:
    """Train XResNet1D-50 or XResNet1D-101 in multilabel mode."""
    try:
        from src.models.xresnet1d import xresnet1d50, xresnet1d101
        from src.training.trainer import Trainer, TrainerConfig
    except ImportError as exc:
        raise SystemExit("PyTorch is required for XResNet training: pip install torch") from exc

    model_cfg = cfg["model"]
    architecture = model_cfg["architecture"]

    factory = {"xresnet1d50": xresnet1d50, "xresnet1d101": xresnet1d101}[architecture]
    model = factory(
        n_leads=cfg["data"].get("leads", 12),
        n_classes=model_cfg["num_classes"],
        kernel_size=model_cfg.get("kernel_size", 5),
        dropout=model_cfg.get("dropout", 0.5),
    )
    logger.info("%s: %s parameters", architecture, f"{model.count_parameters():,}")

    trainer_cfg = TrainerConfig.from_dict({
        **cfg.get("training", {}),
        "checkpoint_dir": checkpoint_dir,
        "device": "auto",
    })

    logger.info("=== %s training ===", architecture)
    logger.info("Loading train split...")
    X_train, y_train = dataset.get_split("train")
    logger.info("Train: %d records  labels shape: %s", len(X_train), y_train.shape)
    logger.info("Loading val split...")
    X_val, y_val = dataset.get_split("val")

    trainer = Trainer(model, trainer_cfg)
    history = trainer.fit(X_train, y_train, X_val, y_val)

    _save_json({"history": history}, run_dir / "history.json")

    best_ckpt = checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt)
        val_metrics = trainer.evaluate(X_val, y_val)
        logger.info(
            "[val (best ckpt)]  macro_auc=%.4f | %s",
            val_metrics.get("macro_auc", float("nan")),
            "  ".join(
                f"{c}:{val_metrics.get('per_class_auc', {}).get(c, float('nan')):.3f}"
                for c in ["NORM", "MI", "STTC", "CD", "HYP"]
            ),
        )
        _save_json(val_metrics, run_dir / "metrics_val.json")


def _train_s4d(cfg: dict, dataset, run_dir: Path, checkpoint_dir: Path, resume_from: Path | None = None) -> None:
    """Train S4D (Diagonal State Space Model) in multilabel mode."""
    try:
        from src.models.s4d import S4DModel
        from src.training.trainer import Trainer, TrainerConfig
    except ImportError as exc:
        raise SystemExit("PyTorch is required for S4D training: pip install torch") from exc

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = S4DModel(
        n_leads=data_cfg.get("leads", 12),
        n_classes=model_cfg["num_classes"],
        d_model=model_cfg.get("d_model", 256),
        d_state=model_cfg.get("d_state", 32),
        n_layers=model_cfg.get("n_layers", 6),
        dropout=model_cfg.get("dropout", 0.1),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", True),
    )
    logger.info("S4D: %s parameters", f"{model.count_parameters():,}")

    trainer_cfg = TrainerConfig.from_dict({
        **cfg.get("training", {}),
        "checkpoint_dir": checkpoint_dir,
        "device": "auto",
        **({"resume_from": resume_from} if resume_from is not None else {}),
    })

    logger.info("=== S4D training ===")
    logger.info("Loading train split...")
    X_train, y_train = dataset.get_split("train")
    logger.info("Train: %d records  labels shape: %s", len(X_train), y_train.shape)
    logger.info("Loading val split...")
    X_val, y_val = dataset.get_split("val")

    trainer = Trainer(model, trainer_cfg)
    history = trainer.fit(X_train, y_train, X_val, y_val)

    _save_json({"history": history}, run_dir / "history.json")

    best_ckpt = checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        trainer.load_checkpoint(best_ckpt)
        val_metrics = trainer.evaluate(X_val, y_val)
        logger.info(
            "[val (best ckpt)]  macro_auc=%.4f | %s",
            val_metrics.get("macro_auc", float("nan")),
            "  ".join(
                f"{c}:{val_metrics.get('per_class_auc', {}).get(c, float('nan')):.3f}"
                for c in ["NORM", "MI", "STTC", "CD", "HYP"]
            ),
        )
        _save_json(val_metrics, run_dir / "metrics_val.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ECG CVD detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Run name (overrides experiment.name in config).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (overrides experiment.seed in config).")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="PTB-XL root directory (overrides data.raw_dir in config).")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Resume training from this checkpoint (last.pt or best.pt).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = _load_config(args.config)

    run_name = args.run_name or cfg["experiment"]["name"]
    seed = args.seed if args.seed is not None else cfg["experiment"].get("seed", 42)
    data_dir = args.data_dir or Path(cfg["data"]["raw_dir"])
    output_dir = Path(cfg["logging"]["output_dir"])
    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"]) / run_name
    run_dir = output_dir / run_name
    architecture = cfg["model"]["architecture"]

    _setup_logging(run_dir / "train.log")
    logger.info("Run: %s  |  architecture: %s  |  seed: %d", run_name, architecture, seed)
    logger.info("Config: %s", args.config)

    _set_seed(seed)

    # Load dataset
    from src.data.dataset import PTBXLDataset
    from src.models.baseline import BASELINE_NAMES

    logger.info("Loading PTB-XL metadata from %s ...", data_dir)
    try:
        dataset = PTBXLDataset(
            data_dir=data_dir,
            mode=cfg["data"].get("mode", "binary"),
            positive_classes=cfg["data"].get("positive_classes", ["MI", "STTC", "CD", "HYP"]),
            sampling_rate=cfg["data"]["sampling_rate"],
            min_confidence=cfg["data"].get("min_confidence", 100.0),
            normalization=cfg["data"].get("normalization", "zscore"),
            bandpass=cfg["data"].get("bandpass", True),
            notch=cfg["data"].get("notch", True),
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"PTB-XL data not found: {exc}\n"
            "Download it first with:  python scripts/download_data.py"
        ) from exc

    logger.info("Dataset loaded. Class distribution: %s", dataset.class_distribution())
    report = dataset.validate_splits()
    logger.info("Split validation: %s", report)

    xresnet_architectures = {"xresnet1d50", "xresnet1d101"}

    if architecture in BASELINE_NAMES:
        _train_baseline(cfg, dataset, run_dir)
    elif architecture == "cnn_1d":
        _train_cnn(cfg, dataset, run_dir, checkpoint_dir)
    elif architecture in xresnet_architectures:
        _train_xresnet(cfg, dataset, run_dir, checkpoint_dir)
    elif architecture == "s4d":
        _train_s4d(cfg, dataset, run_dir, checkpoint_dir, resume_from=args.resume)
    else:
        raise SystemExit(
            f"Unknown architecture: '{architecture}'. "
            f"Choose from: cnn_1d, xresnet1d50, xresnet1d101, s4d, {', '.join(BASELINE_NAMES)}"
        )

    logger.info("Done. Outputs saved to %s", run_dir)


if __name__ == "__main__":
    main()
