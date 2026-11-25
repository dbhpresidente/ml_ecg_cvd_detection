"""Ensemble inference: xresnet1d50_v6 + S4D_v8.

Loads two trained checkpoints, runs inference separately, averages sigmoid
probabilities, and reports per-model and ensemble macro-AUC on val or test.

Usage:
    python scripts/ensemble.py

    # Custom checkpoints or split
    python scripts/ensemble.py \\
        --xresnet checkpoints/xresnet1d50_v6/best.pt \\
        --s4d     checkpoints/s4d_v8/best.pt \\
        --split   test \\
        --output  outputs/ensemble_v1/metrics_test.json

    # Weighted averaging (xresnet weight=0.6, s4d weight=0.4)
    python scripts/ensemble.py --weights 0.6 0.4
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

_DEFAULT_XRESNET_CKPT = Path("checkpoints/xresnet1d50_v6/best.pt")
_DEFAULT_S4D_CKPT     = Path("checkpoints/s4d_v8/best.pt")
_DEFAULT_DATA_DIR     = Path("data/raw/ptb-xl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
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
    logger.info("Metrics saved → %s", path)


def _compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Macro-AUC + per-class AUC for 5-class multilabel output."""
    from sklearn.metrics import roc_auc_score

    per_class: dict[str, float] = {}
    aucs: list[float] = []
    for i, name in enumerate(SUPERCLASSES):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            logger.warning("Class %s has only one label value — skipping AUC", name)
            continue
        auc = float(roc_auc_score(col, y_prob[:, i]))
        per_class[name] = auc
        aucs.append(auc)

    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    return {"macro_auc": macro_auc, "per_class_auc": per_class}


def _print_metrics(metrics: dict, label: str) -> None:
    per = metrics.get("per_class_auc", {})
    class_str = "  ".join(f"{c}:{per.get(c, float('nan')):.4f}" for c in SUPERCLASSES)
    print(f"  {label:<20}  macro_auc={metrics['macro_auc']:.4f}  | {class_str}")


# ---------------------------------------------------------------------------
# Inference: get raw sigmoid probabilities from one model
# ---------------------------------------------------------------------------

def _get_probs(
    model: "torch.nn.Module",
    X: np.ndarray,
    batch_size: int = 32,
    device: "torch.device | None" = None,
) -> np.ndarray:
    """Run inference and return sigmoid probabilities (n, C)."""
    import torch

    model.eval()
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(
                np.asarray(X[start : start + batch_size], dtype=np.float32)
            ).to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_xresnet(checkpoint: Path, device: "torch.device") -> "torch.nn.Module":
    """Instantiate xresnet1d50 and load weights from checkpoint."""
    import torch
    from src.models.xresnet1d import xresnet1d50

    model = xresnet1d50(n_leads=12, n_classes=5, kernel_size=5, dropout=0.5)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    epoch = ckpt.get("epoch", "?")
    auc   = ckpt.get("metrics", {}).get("macro_auc", float("nan"))
    logger.info("xresnet1d50_v6  checkpoint loaded  epoch=%s  val_macro_auc=%.4f", epoch, auc)
    return model


def _load_s4d(checkpoint: Path, device: "torch.device") -> "torch.nn.Module":
    """Instantiate S4DModel (v8 config) and load weights from checkpoint."""
    import torch
    from src.models.s4d import S4DModel

    model = S4DModel(
        n_leads=12,
        n_classes=5,
        d_model=128,
        d_state=8,
        n_layers=4,
        dropout=0.1,
        gradient_checkpointing=False,
    )
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    epoch = ckpt.get("epoch", "?")
    auc   = ckpt.get("metrics", {}).get("macro_auc", float("nan"))
    logger.info("s4d_v8          checkpoint loaded  epoch=%s  val_macro_auc=%.4f", epoch, auc)
    return model


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def _load_split(
    data_dir: Path,
    split: str,
    bandpass_xresnet: bool,
    bandpass_s4d: bool,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Load the requested split twice — once per preprocessing config.

    xresnet_v6 was trained with bandpass=True, s4d_v8 with bandpass=False.
    Returns (X_xresnet, X_s4d, y).
    """
    from src.data.dataset import PTBXLDataset

    logger.info("Loading PTB-XL split='%s' for xresnet (bandpass=%s)...", split, bandpass_xresnet)
    ds_xresnet = PTBXLDataset(
        data_dir=data_dir,
        mode="multilabel",
        sampling_rate=500,
        normalization="zscore",
        bandpass=bandpass_xresnet,
        notch=bandpass_xresnet,       # notch mirrors bandpass for each model
        min_confidence=0.0,
    )
    X_xresnet, y = ds_xresnet.get_split(split)
    logger.info("  → %d records loaded", len(y))

    logger.info("Loading PTB-XL split='%s' for s4d (bandpass=%s)...", split, bandpass_s4d)
    ds_s4d = PTBXLDataset(
        data_dir=data_dir,
        mode="multilabel",
        sampling_rate=500,
        normalization="zscore",
        bandpass=bandpass_s4d,
        notch=bandpass_s4d,
        min_confidence=0.0,
    )
    X_s4d, y_s4d = ds_s4d.get_split(split)

    # Sanity check: both datasets must yield the same labels
    assert np.array_equal(y, y_s4d), (
        "Label mismatch between xresnet and s4d dataset splits — "
        "check that both use the same PTB-XL root and min_confidence."
    )

    return X_xresnet, X_s4d, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ensemble xresnet1d50_v6 + S4D_v8 for PTB-XL multilabel ECG classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--xresnet", type=Path, default=_DEFAULT_XRESNET_CKPT,
                   help="Path to xresnet1d50_v6 best.pt checkpoint.")
    p.add_argument("--s4d", type=Path, default=_DEFAULT_S4D_CKPT,
                   help="Path to s4d_v8 best.pt checkpoint.")
    p.add_argument("--split", choices=["val", "test"], default="test",
                   help="Dataset split to evaluate.")
    p.add_argument("--data-dir", type=Path, default=_DEFAULT_DATA_DIR,
                   help="PTB-XL root directory.")
    p.add_argument("--weights", type=float, nargs=2, default=None,
                   metavar=("W_XRESNET", "W_S4D"),
                   help="Averaging weights for [xresnet, s4d]. Must sum to 1. "
                        "Default: uniform (0.5, 0.5).")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Inference batch size.")
    p.add_argument("--output", type=Path, default=None,
                   help="Path to save metrics JSON. "
                        "Default: outputs/ensemble_xresnet_v6_s4d_v8/metrics_<split>.json")
    p.add_argument("--device", type=str, default="auto",
                   help="Device: 'auto', 'cuda', 'cpu'.")
    return p.parse_args()


def _resolve_device(device: str) -> "torch.device":
    import torch
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import torch

    args = parse_args()
    _setup_logging()

    # Validate checkpoints
    for label, path in [("xresnet", args.xresnet), ("s4d", args.s4d)]:
        if not path.exists():
            raise SystemExit(f"{label} checkpoint not found: {path}")

    # Averaging weights
    if args.weights is not None:
        w_xresnet, w_s4d = args.weights
        if abs(w_xresnet + w_s4d - 1.0) > 1e-6:
            raise SystemExit(
                f"Weights must sum to 1.0, got {w_xresnet} + {w_s4d} = {w_xresnet + w_s4d}"
            )
    else:
        w_xresnet, w_s4d = 0.5, 0.5

    output_path = args.output or Path(
        f"outputs/ensemble_xresnet_v6_s4d_v8/metrics_{args.split}.json"
    )

    device = _resolve_device(args.device)
    logger.info("Device        : %s", device)
    logger.info("Split         : %s", args.split)
    logger.info("Weights       : xresnet=%.2f  s4d=%.2f", w_xresnet, w_s4d)
    logger.info("xresnet ckpt  : %s", args.xresnet)
    logger.info("s4d ckpt      : %s", args.s4d)

    # ---- Load data -------------------------------------------------------
    # xresnet_v6: bandpass=True (trained with bandpass)
    # s4d_v8:     bandpass=False (trained without bandpass)
    X_xresnet, X_s4d, y_true = _load_split(
        data_dir=args.data_dir,
        split=args.split,
        bandpass_xresnet=True,
        bandpass_s4d=False,
    )

    # ---- Load models & run inference ------------------------------------
    model_xresnet = _load_xresnet(args.xresnet, device)
    model_s4d     = _load_s4d(args.s4d, device)

    logger.info("Running inference: xresnet1d50_v6...")
    probs_xresnet = _get_probs(model_xresnet, X_xresnet, args.batch_size, device)

    logger.info("Running inference: s4d_v8...")
    probs_s4d = _get_probs(model_s4d, X_s4d, args.batch_size, device)

    # ---- Ensemble --------------------------------------------------------
    probs_ensemble = w_xresnet * probs_xresnet + w_s4d * probs_s4d

    # ---- Metrics ---------------------------------------------------------
    metrics_xresnet  = _compute_multilabel_metrics(y_true, probs_xresnet)
    metrics_s4d      = _compute_multilabel_metrics(y_true, probs_s4d)
    metrics_ensemble = _compute_multilabel_metrics(y_true, probs_ensemble)

    sep = "-" * 72
    print(sep)
    print(f"  Ensemble evaluation — split={args.split}")
    print(sep)
    _print_metrics(metrics_xresnet,  "xresnet1d50_v6")
    _print_metrics(metrics_s4d,      "s4d_v8")
    print(sep)
    _print_metrics(metrics_ensemble, "ENSEMBLE")
    print(sep)

    # ---- Improvement summary --------------------------------------------
    gain_vs_xresnet = metrics_ensemble["macro_auc"] - metrics_xresnet["macro_auc"]
    gain_vs_s4d     = metrics_ensemble["macro_auc"] - metrics_s4d["macro_auc"]
    print(f"  Ensemble gain vs xresnet : {gain_vs_xresnet:+.4f}")
    print(f"  Ensemble gain vs s4d     : {gain_vs_s4d:+.4f}")
    print(sep)

    # ---- Save JSON -------------------------------------------------------
    results = {
        "split": args.split,
        "weights": {"xresnet": w_xresnet, "s4d": w_s4d},
        "checkpoints": {
            "xresnet": str(args.xresnet),
            "s4d": str(args.s4d),
        },
        "xresnet1d50_v6": metrics_xresnet,
        "s4d_v8": metrics_s4d,
        "ensemble": metrics_ensemble,
        "gain_vs_xresnet": gain_vs_xresnet,
        "gain_vs_s4d": gain_vs_s4d,
    }
    _save_json(results, output_path)


if __name__ == "__main__":
    main()
