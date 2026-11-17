"""Evaluation entry point for ECG CVD detection.

Loads a trained model (sklearn .pkl or PyTorch .pt checkpoint) and evaluates
it on the requested dataset split, printing and saving classification metrics.

Usage:
    # Evaluate CNN on test split
    python scripts/evaluate.py --checkpoint checkpoints/cnn_baseline/best.pt

    # Evaluate baseline on val split
    python scripts/evaluate.py --config configs/baseline.yaml \\
        --checkpoint outputs/rf_baseline/model.pkl --split val

    # Override data directory
    python scripts/evaluate.py --checkpoint outputs/rf_baseline/model.pkl \\
        --data-dir data/raw/ptb-xl
"""
import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with train.py)
# ---------------------------------------------------------------------------

def _load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("pyyaml is required: pip install pyyaml") from exc
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
    logger.info("Metrics saved: %s", path)


def _print_metrics_table(metrics: dict[str, float], split: str, architecture: str) -> None:
    """Print a formatted metrics table to stdout."""
    separator = "-" * 52
    print(separator)
    print(f"  Model : {architecture}")
    print(f"  Split : {split}")
    print(separator)
    ordered = [
        ("ROC-AUC",     metrics.get("roc_auc")),
        ("Avg Prec.",   metrics.get("avg_precision")),
        ("F1",          metrics.get("f1")),
        ("Accuracy",    metrics.get("accuracy")),
        ("Sensitivity", metrics.get("sensitivity")),
        ("Specificity", metrics.get("specificity")),
        ("MCC",         metrics.get("mcc")),
        ("Opt. Thresh", metrics.get("optimal_threshold")),
    ]
    for name, value in ordered:
        if value is not None:
            print(f"  {name:<14} {value:.4f}")
    print(separator)


# ---------------------------------------------------------------------------
# Evaluation routines
# ---------------------------------------------------------------------------

def _evaluate_baseline(
    cfg: dict,
    checkpoint: Path,
    split: str,
    dataset,
    output_path: Path,
) -> None:
    """Load a pickled sklearn pipeline, extract features, and evaluate."""
    from src.features import extract_features_batch
    from src.models.metrics import compute_metrics

    architecture = cfg["model"]["architecture"]
    fs = cfg["data"]["sampling_rate"]

    logger.info("Loading model from %s ...", checkpoint)
    with open(checkpoint, "rb") as f:
        clf = pickle.load(f)

    logger.info("Loading %s split...", split)
    X_raw, y_true = dataset.get_split(split)

    logger.info("Extracting features (%d records)...", len(y_true))
    X = extract_features_batch(X_raw, fs=fs)

    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    metrics = compute_metrics(y_true, y_pred, y_prob)

    _print_metrics_table(metrics, split, architecture)
    _save_json(metrics, output_path)


def _evaluate_cnn(
    cfg: dict,
    checkpoint: Path,
    split: str,
    dataset,
    output_path: Path,
) -> None:
    """Rebuild CNN1D, load checkpoint weights, and evaluate."""
    try:
        from src.models.cnn1d import CNN1D
        from src.training.trainer import Trainer, TrainerConfig
    except ImportError as exc:
        raise SystemExit("PyTorch is required: pip install torch") from exc

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = CNN1D(
        n_leads=data_cfg["leads"],
        n_classes=model_cfg["num_classes"],
        dropout=model_cfg.get("dropout", 0.3),
    )

    trainer = Trainer(model, TrainerConfig(device="auto"))
    trainer.load_checkpoint(checkpoint)

    logger.info("Loading %s split...", split)
    X, y = dataset.get_split(split)

    logger.info("Evaluating on %d records...", len(y))
    metrics = trainer.evaluate(X, y)

    _print_metrics_table(metrics, split, "cnn_1d")
    _save_json(metrics, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ECG CVD detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default="configs/default.yaml",
                        help="Path to YAML config file.")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to .pt (CNN) or .pkl (baseline) checkpoint.")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="Dataset split to evaluate.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="PTB-XL root directory (overrides data.raw_dir in config).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save metrics JSON (default: next to checkpoint).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging()

    cfg = _load_config(args.config)
    architecture = cfg["model"]["architecture"]
    data_dir = args.data_dir or Path(cfg["data"]["raw_dir"])

    output_path = args.output or args.checkpoint.parent / f"metrics_{args.split}.json"

    logger.info("Checkpoint  : %s", args.checkpoint)
    logger.info("Architecture: %s", architecture)
    logger.info("Split       : %s", args.split)
    logger.info("Data dir    : %s", data_dir)

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    from src.data.dataset import PTBXLDataset
    from src.models.baseline import BASELINE_NAMES

    logger.info("Loading PTB-XL metadata...")
    try:
        dataset = PTBXLDataset(
            data_dir=data_dir,
            positive_classes=cfg["data"].get("positive_classes", ["MI", "STTC", "CD", "HYP"]),
            sampling_rate=cfg["data"]["sampling_rate"],
            min_confidence=cfg["data"].get("min_confidence", 100.0),
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"PTB-XL data not found: {exc}\n"
            "Download it first with:  python scripts/download_data.py"
        ) from exc

    if architecture in BASELINE_NAMES:
        _evaluate_baseline(cfg, args.checkpoint, args.split, dataset, output_path)
    elif architecture == "cnn_1d":
        _evaluate_cnn(cfg, args.checkpoint, args.split, dataset, output_path)
    else:
        raise SystemExit(
            f"Unknown architecture: '{architecture}'. "
            f"Choose from: cnn_1d, {', '.join(BASELINE_NAMES)}"
        )


if __name__ == "__main__":
    main()
