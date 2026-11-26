"""Training script for ECGFounder LoRA fine-tuning on PTB-XL.

Downloads the pretrained ECGFounder checkpoint from HuggingFace Hub on first
run, injects LoRA adapters into SE modules, and fine-tunes on PTB-XL 5-class
multilabel classification using the same Trainer as other models.

Usage:
    python -m scripts.train_ecgfounder

    # Custom config
    python -m scripts.train_ecgfounder --config configs/ecgfounder_lora_v1.yaml

    # Resume
    python -m scripts.train_ecgfounder \\
        --config configs/ecgfounder_lora_v1.yaml \\
        --resume checkpoints/ecgfounder_lora_v1/last.pt

    # Pre-downloaded checkpoint (skip HF download)
    python -m scripts.train_ecgfounder \\
        --checkpoint checkpoints/ecgfounder/12_lead_ECGFounder.pth
"""
import argparse
import json
import logging
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


def _setup_logging(log_path: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
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
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    logger.info("Saved: %s", path)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(cfg: dict, args: argparse.Namespace) -> None:
    try:
        from src.models.ecgfounder import build_ecgfounder_lora
        from src.training.trainer import Trainer, TrainerConfig
    except ImportError as exc:
        raise SystemExit(f"Import error: {exc}") from exc

    exp_name     = cfg["experiment"]["name"]
    model_cfg    = cfg["model"]
    data_cfg     = cfg["data"]
    train_cfg    = cfg["training"]
    output_dir   = Path(cfg["logging"]["output_dir"]) / exp_name
    ckpt_dir     = Path(cfg["logging"]["checkpoint_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Seed ---
    seed = cfg["experiment"].get("seed", 42)
    _set_seed(seed)
    logger.info("Run: %s  |  seed: %d", exp_name, seed)
    logger.info("Config: %s", args.config)

    # --- Dataset ---
    logger.info("Loading PTB-XL metadata from %s ...", data_cfg["raw_dir"])
    try:
        from src.data.dataset import PTBXLDataset
    except ImportError as exc:
        raise SystemExit(f"Cannot import dataset: {exc}") from exc

    try:
        dataset = PTBXLDataset(
            data_dir=data_cfg["raw_dir"],
            mode=data_cfg["mode"],
            sampling_rate=data_cfg["sampling_rate"],
            normalization=data_cfg.get("normalization", "zscore"),
            bandpass=data_cfg.get("bandpass", False),
            notch=data_cfg.get("notch", False),
            min_confidence=data_cfg.get("min_confidence", 0.0),
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"PTB-XL data not found: {exc}\n"
            "Download it first with:  python scripts/download_data.py"
        ) from exc

    split_info = dataset.validate_splits()
    logger.info("Split validation: %s", split_info)

    # --- Model ---
    checkpoint = args.checkpoint or model_cfg.get("checkpoint") or None
    if checkpoint is not None:
        checkpoint = Path(checkpoint)

    logger.info("Building ECGFounder + LoRA (rank=%d, alpha=%.1f)...",
                model_cfg.get("lora_rank", 8),
                model_cfg.get("lora_alpha", 16.0))

    model = build_ecgfounder_lora(
        n_classes=model_cfg["num_classes"],
        rank=model_cfg.get("lora_rank", 8),
        alpha=model_cfg.get("lora_alpha", 16.0),
        unfreeze_last_n_stages=model_cfg.get("unfreeze_last_n_stages", 0),
        checkpoint=checkpoint,
        hf_cache_dir=Path(model_cfg.get("hf_cache_dir", "checkpoints/ecgfounder")),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "ECGFounder LoRA: total=%s  trainable=%s  (%.1f%%)",
        f"{total:,}", f"{trainable:,}", 100.0 * trainable / total,
    )

    # --- Trainer config ---
    trainer_cfg = TrainerConfig.from_dict({
        **train_cfg,
        "checkpoint_dir": ckpt_dir,
        "device": "auto",
        "resume_from": args.resume,
    })

    # --- Load data ---
    logger.info("=== ECGFounder LoRA fine-tuning ===")
    logger.info("Loading train split...")
    X_train, y_train = dataset.get_split("train")
    logger.info("Train: %d records  labels shape: %s", len(y_train), y_train.shape)

    logger.info("Loading val split...")
    X_val, y_val = dataset.get_split("val")
    logger.info("Val:   %d records", len(y_val))

    # --- Train ---
    trainer = Trainer(model, trainer_cfg)
    history = trainer.fit(X_train, y_train, X_val, y_val)
    _save_json(history, output_dir / "history.json")

    # --- Evaluate best checkpoint ---
    logger.info("Loading best checkpoint for final evaluation...")
    trainer.load_checkpoint(ckpt_dir / "best.pt")

    for split_name, X_ev, y_ev in [("val", X_val, y_val)]:
        metrics = trainer.evaluate(X_ev, y_ev)
        logger.info(
            "[%s (best ckpt)]  macro_auc=%.4f | %s",
            split_name, metrics["macro_auc"],
            "  ".join(
                f"{c}:{metrics['per_class_auc'].get(c, 0):.3f}"
                for c in ["NORM", "MI", "STTC", "CD", "HYP"]
            ),
        )
        _save_json(metrics, output_dir / f"metrics_{split_name}.json")

    logger.info("Done. Outputs saved to %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune ECGFounder with LoRA on PTB-XL multilabel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path,
                   default=Path("configs/ecgfounder_lora_v1.yaml"),
                   help="YAML config file.")
    p.add_argument("--resume", type=Path, default=None,
                   help="Path to last.pt checkpoint for resuming training.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to pretrained ECGFounder .pth (overrides config).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    cfg = _load_config(args.config)

    exp_name   = cfg["experiment"]["name"]
    output_dir = Path(cfg["logging"]["output_dir"]) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir / f"{exp_name}.log")

    _train(cfg, args)


if __name__ == "__main__":
    main()
