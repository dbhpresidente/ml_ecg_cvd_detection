"""3-model ensemble: ECGFounder_lora_v2 + xresnet1d50_v6 + xresnet1d101_v7.1.

Runs a weight sweep on val, then evaluates the best combination on test.

Usage:
    python -m scripts.ensemble3
    python -m scripts.ensemble3 --weights 0.4 0.35 0.25   # ef / xresnet50 / xresnet101
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

_CKPT_EF   = Path("checkpoints/ecgfounder_lora_v2/best.pt")
_CKPT_X50  = Path("checkpoints/xresnet1d50_v6/best.pt")
_CKPT_X101 = Path("checkpoints/xresnet_v7.1/best.pt")
_DATA_DIR  = Path("data/raw/ptb-xl")


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
    logger.info("Saved -> %s", path)


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    per_class: dict[str, float] = {}
    aucs: list[float] = []
    for i, name in enumerate(SUPERCLASSES):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            continue
        a = float(roc_auc_score(col, y_prob[:, i]))
        per_class[name] = a
        aucs.append(a)
    return {"macro_auc": float(np.mean(aucs)), "per_class_auc": per_class}


def _print_metrics(metrics: dict, label: str) -> None:
    per = metrics.get("per_class_auc", {})
    class_str = "  ".join(f"{c}:{per.get(c, float('nan')):.4f}" for c in SUPERCLASSES)
    print(f"  {label:<26}  macro_auc={metrics['macro_auc']:.4f}  | {class_str}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _get_probs(model, X: np.ndarray, batch_size: int, device) -> np.ndarray:
    import torch
    model.eval()
    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(
                np.asarray(X[start: start + batch_size], dtype=np.float32)
            ).to(device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_xresnet50(ckpt: Path, device):
    import torch
    from src.models.xresnet1d import xresnet1d50
    model = xresnet1d50(n_leads=12, n_classes=5, kernel_size=5, dropout=0.5)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    logger.info("xresnet1d50_v6 loaded  ep=%s  val_auc=%.4f",
                state.get("epoch", "?"), state.get("metrics", {}).get("macro_auc", float("nan")))
    return model


def _load_xresnet101(ckpt: Path, device):
    import torch
    from src.models.xresnet1d import xresnet1d101
    model = xresnet1d101(n_leads=12, n_classes=5, kernel_size=5, dropout=0.7)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    logger.info("xresnet1d101_v7.1 loaded  ep=%s  val_auc=%.4f",
                state.get("epoch", "?"), state.get("metrics", {}).get("macro_auc", float("nan")))
    return model


def _load_ecgfounder(ckpt: Path, device):
    import torch
    from src.models.ecgfounder import build_ecgfounder_lora
    model = build_ecgfounder_lora(
        n_classes=5, rank=16, alpha=32.0,
        unfreeze_last_n_stages=0,
        checkpoint=None,
        hf_cache_dir=Path("checkpoints/ecgfounder"),
    )
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    logger.info("ECGFounder_lora_v2 loaded  ep=%s  val_auc=%.4f",
                state.get("epoch", "?"), state.get("metrics", {}).get("macro_auc", float("nan")))
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _load_dataset(data_dir: Path, split: str, bandpass: bool) -> tuple:
    from src.data.dataset import PTBXLDataset
    ds = PTBXLDataset(
        data_dir=data_dir, mode="multilabel", sampling_rate=500,
        normalization="zscore", bandpass=bandpass, notch=bandpass,
        min_confidence=0.0,
    )
    X, y = ds.get_split(split)
    logger.info("Loaded split='%s' bandpass=%s -> %d records", split, bandpass, len(y))
    return X, y


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="3-model ensemble: ECGFounder_v2 + xresnet50_v6 + xresnet101_v7.1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ef",    type=Path, default=_CKPT_EF)
    p.add_argument("--x50",   type=Path, default=_CKPT_X50)
    p.add_argument("--x101",  type=Path, default=_CKPT_X101)
    p.add_argument("--data-dir", type=Path, default=_DATA_DIR)
    p.add_argument("--weights", type=float, nargs=3, default=None,
                   metavar=("W_EF", "W_X50", "W_X101"),
                   help="Fixed weights. If omitted, run a val sweep then evaluate test.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def _resolve_device(device: str):
    import torch
    if device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _setup_logging()

    for label, path in [("ecgfounder", args.ef), ("xresnet50", args.x50), ("xresnet101", args.x101)]:
        if not path.exists():
            raise SystemExit(f"{label} checkpoint not found: {path}")

    device = _resolve_device(args.device)
    logger.info("Device: %s", device)

    # ---- Load data (bandpass=True for xresnet; bandpass=False for ECGFounder) ----
    logger.info("=== Loading val data ===")
    X_bp_val,   y_val   = _load_dataset(args.data_dir, "val",  bandpass=True)
    X_nobp_val, y_val2  = _load_dataset(args.data_dir, "val",  bandpass=False)
    assert np.array_equal(y_val, y_val2), "Label mismatch in val sets"

    logger.info("=== Loading test data ===")
    X_bp_test,   y_test   = _load_dataset(args.data_dir, "test", bandpass=True)
    X_nobp_test, y_test2  = _load_dataset(args.data_dir, "test", bandpass=False)
    assert np.array_equal(y_test, y_test2), "Label mismatch in test sets"

    # ---- Load models ----
    logger.info("=== Loading models ===")
    m_ef   = _load_ecgfounder(args.ef, device)
    m_x50  = _load_xresnet50(args.x50, device)
    m_x101 = _load_xresnet101(args.x101, device)

    # ---- Inference ----
    logger.info("=== Inference ===")
    p_ef_val   = _get_probs(m_ef,   X_nobp_val,  args.batch_size, device)
    p_x50_val  = _get_probs(m_x50,  X_bp_val,    args.batch_size, device)
    p_x101_val = _get_probs(m_x101, X_bp_val,    args.batch_size, device)

    p_ef_test   = _get_probs(m_ef,   X_nobp_test, args.batch_size, device)
    p_x50_test  = _get_probs(m_x50,  X_bp_test,   args.batch_size, device)
    p_x101_test = _get_probs(m_x101, X_bp_test,   args.batch_size, device)

    sep = "-" * 80

    # ---- Individual model metrics ----
    m_ef_val   = _auc(y_val,  p_ef_val)
    m_x50_val  = _auc(y_val,  p_x50_val)
    m_x101_val = _auc(y_val,  p_x101_val)
    m_ef_test   = _auc(y_test, p_ef_test)
    m_x50_test  = _auc(y_test, p_x50_test)
    m_x101_test = _auc(y_test, p_x101_test)

    print(sep)
    print("  Individual models (val):")
    _print_metrics(m_ef_val,   "ECGFounder_lora_v2")
    _print_metrics(m_x50_val,  "xresnet1d50_v6")
    _print_metrics(m_x101_val, "xresnet1d101_v7.1")
    print(sep)

    if args.weights is not None:
        # Fixed weights mode
        w_ef, w_x50, w_x101 = args.weights
        total = w_ef + w_x50 + w_x101
        w_ef, w_x50, w_x101 = w_ef / total, w_x50 / total, w_x101 / total
        p_ens_val  = w_ef * p_ef_val  + w_x50 * p_x50_val  + w_x101 * p_x101_val
        p_ens_test = w_ef * p_ef_test + w_x50 * p_x50_test + w_x101 * p_x101_test
        m_ens_val  = _auc(y_val,  p_ens_val)
        m_ens_test = _auc(y_test, p_ens_test)
        print(f"  Weights: ef={w_ef:.3f}  x50={w_x50:.3f}  x101={w_x101:.3f}")
        _print_metrics(m_ens_val,  "ENSEMBLE (val)")
        _print_metrics(m_ens_test, "ENSEMBLE (test)")
        return

    # ---- Weight sweep on val ----
    # Sweep ef weight from 0.2 to 0.6; split remainder evenly between x50 and x101
    # Also try varying the x50/x101 split
    print("  Weight sweep on val:")
    print(f"  {'w_ef':>5} {'w_x50':>6} {'w_x101':>7}  val_macro_auc")
    print("  " + "-" * 44)

    best_val_auc = 0.0
    best_weights = (1/3, 1/3, 1/3)

    # Grid: w_ef in {0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50}
    # remainder split as w_x50 : w_x101 in {50/50, 60/40, 55/45}
    candidates = []
    for w_ef in [0.20, 0.25, 0.30, 0.33, 0.35, 0.40, 0.45, 0.50]:
        rem = 1.0 - w_ef
        for ratio in [0.40, 0.45, 0.50, 0.55, 0.60]:
            w_x50  = rem * ratio
            w_x101 = rem * (1.0 - ratio)
            candidates.append((w_ef, w_x50, w_x101))

    # Also try uniform
    candidates.append((1/3, 1/3, 1/3))

    for w_ef, w_x50, w_x101 in candidates:
        p_ens = w_ef * p_ef_val + w_x50 * p_x50_val + w_x101 * p_x101_val
        m_ens = _auc(y_val, p_ens)
        auc   = m_ens["macro_auc"]
        print(f"  {w_ef:5.2f} {w_x50:6.2f} {w_x101:7.2f}   {auc:.4f}")
        if auc > best_val_auc:
            best_val_auc = auc
            best_weights = (w_ef, w_x50, w_x101)

    print(sep)
    w_ef, w_x50, w_x101 = best_weights
    print(f"  Best val weights: ef={w_ef:.3f}  x50={w_x50:.3f}  x101={w_x101:.3f}  val={best_val_auc:.4f}")

    # ---- Final evaluation on test ----
    p_best_val  = w_ef * p_ef_val  + w_x50 * p_x50_val  + w_x101 * p_x101_val
    p_best_test = w_ef * p_ef_test + w_x50 * p_x50_test + w_x101 * p_x101_test
    m_best_val  = _auc(y_val,  p_best_val)
    m_best_test = _auc(y_test, p_best_test)

    # Reference 2-model ensemble
    p_2m_val  = 0.5 * p_ef_val  + 0.5 * p_x50_val
    p_2m_test = 0.5 * p_ef_test + 0.5 * p_x50_test
    m_2m_val  = _auc(y_val,  p_2m_val)
    m_2m_test = _auc(y_test, p_2m_test)

    print(sep)
    print("  FINAL RESULTS:")
    _print_metrics(m_ef_test,   "ECGFounder_lora_v2")
    _print_metrics(m_x50_test,  "xresnet1d50_v6")
    _print_metrics(m_x101_test, "xresnet1d101_v7.1")
    print("  " + "-" * 60)
    _print_metrics(m_2m_val,  "2-model ref (val)")
    _print_metrics(m_2m_test, "2-model ref (test)")
    print("  " + "-" * 60)
    _print_metrics(m_best_val,  "3-model ensemble (val)")
    _print_metrics(m_best_test, "3-model ensemble (test)")
    print(sep)
    gain = m_best_test["macro_auc"] - m_2m_test["macro_auc"]
    print(f"  Gain vs 2-model ensemble: {gain:+.4f}")
    print(sep)

    # ---- Save ----
    out = {
        "best_weights": {"ef": w_ef, "x50": w_x50, "x101": w_x101},
        "individual_val":  {"ecgfounder": m_ef_val, "xresnet50": m_x50_val, "xresnet101": m_x101_val},
        "individual_test": {"ecgfounder": m_ef_test, "xresnet50": m_x50_test, "xresnet101": m_x101_test},
        "ensemble_2model_val":  m_2m_val,
        "ensemble_2model_test": m_2m_test,
        "ensemble_3model_val":  m_best_val,
        "ensemble_3model_test": m_best_test,
        "gain_vs_2model_test":  gain,
    }
    _save_json(out, Path("outputs/ensemble_3model/metrics.json"))


if __name__ == "__main__":
    main()
