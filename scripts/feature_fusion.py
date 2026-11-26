"""Late-fusion experiment: 3-model DL ensemble + PTB-XL+ clinical features.

Combines sigmoid probabilities from ECGFounder_lora_v2 + xresnet1d50_v6 +
xresnet1d101_v7.1 with 40 selected clinical features from 12sl_features.csv
and trains a meta-learner on top.

Usage:
    python -m scripts.feature_fusion
    python -m scripts.feature_fusion --no-cache   # force re-run inference
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

_DATA_DIR    = Path("data/raw/ptb-xl")
_FEATURES_CSV = Path("data/raw/ptb-xl-plus/features/12sl_features.csv")
_CKPT_EF    = Path("checkpoints/ecgfounder_lora_v2/best.pt")
_CKPT_X50   = Path("checkpoints/xresnet1d50_v6/best.pt")
_CKPT_X101  = Path("checkpoints/xresnet_v7.1/best.pt")
_OUT_DIR    = Path("outputs/feature_fusion")
_PROBS_DIR  = _OUT_DIR / "probs"

# 40/30/30 baseline ensemble weights
_BASELINE_WEIGHTS = (0.4, 0.3, 0.3)

# Clinical feature candidates (~50 hardcoded, clinically meaningful)
_CLINICAL_CANDIDATES = [
    # Global intervals and rates
    "QRS_Dur_Global", "QT_IntBazett_Global", "QT_IntFridericia_Global",
    "PR_Int_Global", "HR_Ventr_Global", "RR_Mean_Global",
    "R_AxisFrontal_Global", "T_AxisFront_Global", "P_AxisFront_Global",
    "P_Dur_Global",
    # HYP voltages
    "R_Amp_aVL", "S_Amp_V1", "R_Amp_V5", "R_Amp_V6", "S_Amp_V3",
    "R_Amp_I", "R_Amp_II",
    # ST amplitude per lead (all 12)
    "ST_Amp_I", "ST_Amp_II", "ST_Amp_III", "ST_Amp_aVR", "ST_Amp_aVL",
    "ST_Amp_aVF", "ST_Amp_V1", "ST_Amp_V2", "ST_Amp_V3", "ST_Amp_V4",
    "ST_Amp_V5", "ST_Amp_V6",
    # T amplitude per lead (all 12)
    "T_Amp_I", "T_Amp_II", "T_Amp_III", "T_Amp_aVR", "T_Amp_aVL",
    "T_Amp_aVF", "T_Amp_V1", "T_Amp_V2", "T_Amp_V3", "T_Amp_V4",
    "T_Amp_V5", "T_Amp_V6",
    # Q amplitude per key leads
    "Q_Amp_II", "Q_Amp_III", "Q_Amp_aVF", "Q_Amp_V1", "Q_Amp_V2",
    "Q_Amp_V3", "Q_Amp_V4",
    # R amplitude V1-V4 (CD/RBBB pattern)
    "R_Amp_V1", "R_Amp_V2", "R_Amp_V3", "R_Amp_V4",
]

# Derived feature names (always kept if columns they depend on exist)
_DERIVED_FEATURES = ["Sokolow_Lyon", "Cornell"]

# Number of features to keep after AUROC selection
_TOP_K = 40


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

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
    logger.info("Saved -> %s", path)


def _resolve_device(device_str: str):
    import torch
    if device_str != "auto":
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute macro-AUC and per-class AUC."""
    from sklearn.metrics import roc_auc_score
    per_class: dict[str, float] = {}
    aucs: list[float] = []
    for i, name in enumerate(SUPERCLASSES):
        col = y_true[:, i]
        if len(np.unique(col)) < 2:
            logger.warning("Class %s has only one unique value — skipping AUC.", name)
            continue
        a = float(roc_auc_score(col, y_prob[:, i]))
        per_class[name] = a
        aucs.append(a)
    return {"macro_auc": float(np.mean(aucs)), "per_class_auc": per_class}


def _print_metrics(metrics: dict, label: str) -> None:
    per = metrics.get("per_class_auc", {})
    class_str = "  ".join(f"{c}:{per.get(c, float('nan')):.4f}" for c in SUPERCLASSES)
    print(f"  {label:<32}  macro_auc={metrics['macro_auc']:.4f}  | {class_str}")


def _get_probs(model, X: np.ndarray, batch_size: int, device) -> np.ndarray:
    """Run forward pass on X and return sigmoid probabilities (N, 5)."""
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
    logger.info(
        "xresnet1d50_v6 loaded  ep=%s  val_auc=%.4f",
        state.get("epoch", "?"),
        state.get("metrics", {}).get("macro_auc", float("nan")),
    )
    return model


def _load_xresnet101(ckpt: Path, device):
    import torch
    from src.models.xresnet1d import xresnet1d101
    model = xresnet1d101(n_leads=12, n_classes=5, kernel_size=5, dropout=0.7)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    logger.info(
        "xresnet1d101_v7.1 loaded  ep=%s  val_auc=%.4f",
        state.get("epoch", "?"),
        state.get("metrics", {}).get("macro_auc", float("nan")),
    )
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
    logger.info(
        "ECGFounder_lora_v2 loaded  ep=%s  val_auc=%.4f",
        state.get("epoch", "?"),
        state.get("metrics", {}).get("macro_auc", float("nan")),
    )
    return model


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_dataset_with_ids(data_dir: Path, split: str, bandpass: bool) -> tuple:
    """Load split, return (X, y, ecg_ids).

    ecg_ids is a numpy array of int64 ecg_ids in the same order as X/y.
    """
    from src.data.dataset import PTBXLDataset
    ds = PTBXLDataset(
        data_dir=data_dir, mode="multilabel", sampling_rate=500,
        normalization="zscore", bandpass=bandpass, notch=bandpass,
        min_confidence=0.0,
    )
    mask = ds._get_fold_mask(split)
    subset = ds._records_df[mask]
    ecg_ids = subset.index.values.astype(np.int64)
    X, y = ds.get_split(split)
    logger.info(
        "Loaded split='%s' bandpass=%s -> %d records", split, bandpass, len(y)
    )
    return X, y, ecg_ids


# ---------------------------------------------------------------------------
# Probability cache
# ---------------------------------------------------------------------------

def _probs_cache_path(split: str, model_name: str) -> Path:
    return _PROBS_DIR / f"{model_name}_{split}.npy"


def _labels_cache_path(split: str) -> Path:
    return _PROBS_DIR / f"labels_{split}.npy"


def _ids_cache_path(split: str) -> Path:
    return _PROBS_DIR / f"ecg_ids_{split}.npy"


def _all_cache_exists(splits: list[str], model_names: list[str]) -> bool:
    for split in splits:
        for name in model_names:
            if not _probs_cache_path(split, name).exists():
                return False
        if not _labels_cache_path(split).exists():
            return False
        if not _ids_cache_path(split).exists():
            return False
    return True


def _run_inference(batch_size: int, device) -> dict:
    """Run all three models on train/val/test, return nested dict of probs/labels/ids."""
    logger.info("=== Running DL model inference ===")
    _PROBS_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]

    # Load bandpass=True datasets first (xresnet50 + xresnet101)
    logger.info("Loading bandpass=True datasets...")
    data_bp = {}
    for split in splits:
        X, y, ids = _load_dataset_with_ids(_DATA_DIR, split, bandpass=True)
        data_bp[split] = (X, y, ids)

    # Load bandpass=False datasets (ECGFounder)
    logger.info("Loading bandpass=False datasets...")
    data_nobp = {}
    for split in splits:
        X, y, ids = _load_dataset_with_ids(_DATA_DIR, split, bandpass=False)
        data_nobp[split] = (X, y, ids)

    # Assert label equality between bandpass variants
    for split in splits:
        y_bp   = data_bp[split][1]
        y_nobp = data_nobp[split][1]
        assert np.array_equal(y_bp, y_nobp), (
            f"Label mismatch between bandpass=True/False for split '{split}'"
        )
        assert np.array_equal(data_bp[split][2], data_nobp[split][2]), (
            f"ecg_id mismatch between bandpass=True/False for split '{split}'"
        )
    logger.info("Label/ID equality verified across bandpass variants.")

    # Load models
    logger.info("=== Loading models ===")
    m_ef   = _load_ecgfounder(_CKPT_EF, device)
    m_x50  = _load_xresnet50(_CKPT_X50, device)
    m_x101 = _load_xresnet101(_CKPT_X101, device)

    # Inference
    result = {}
    for split in splits:
        X_bp,   y, ids = data_bp[split]
        X_nobp, _,  _  = data_nobp[split]

        logger.info("=== Inference: split=%s ===", split)
        p_ef   = _get_probs(m_ef,   X_nobp, batch_size, device)
        p_x50  = _get_probs(m_x50,  X_bp,   batch_size, device)
        p_x101 = _get_probs(m_x101, X_bp,   batch_size, device)

        # Save to cache
        np.save(_probs_cache_path(split, "ef"),   p_ef)
        np.save(_probs_cache_path(split, "x50"),  p_x50)
        np.save(_probs_cache_path(split, "x101"), p_x101)
        np.save(_labels_cache_path(split), y)
        np.save(_ids_cache_path(split), ids)
        logger.info("Saved probs cache for split='%s'", split)

        result[split] = {
            "probs_ef":   p_ef,
            "probs_x50":  p_x50,
            "probs_x101": p_x101,
            "y":          y,
            "ecg_ids":    ids,
        }

    return result


def _load_probs_from_cache() -> dict:
    """Load all probs/labels/ids from numpy cache files."""
    logger.info("Loading DL probabilities from cache...")
    result = {}
    for split in ["train", "val", "test"]:
        result[split] = {
            "probs_ef":   np.load(_probs_cache_path(split, "ef")),
            "probs_x50":  np.load(_probs_cache_path(split, "x50")),
            "probs_x101": np.load(_probs_cache_path(split, "x101")),
            "y":          np.load(_labels_cache_path(split)),
            "ecg_ids":    np.load(_ids_cache_path(split)),
        }
        logger.info(
            "Loaded cached probs for split='%s'  n=%d", split, len(result[split]["y"])
        )
    return result


# ---------------------------------------------------------------------------
# Clinical feature pipeline
# ---------------------------------------------------------------------------

def _add_derived_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add Sokolow_Lyon and Cornell derived features if source cols exist.

    Returns the modified df and a list of derived feature names that were
    actually added.
    """
    added: list[str] = []

    # Sokolow-Lyon = |S_Amp_V1| + max(R_Amp_V5, R_Amp_V6)
    if "S_Amp_V1" in df.columns and "R_Amp_V5" in df.columns and "R_Amp_V6" in df.columns:
        df = df.copy()
        df["Sokolow_Lyon"] = df["S_Amp_V1"].abs() + df[["R_Amp_V5", "R_Amp_V6"]].max(axis=1)
        added.append("Sokolow_Lyon")
        logger.info("Derived feature added: Sokolow_Lyon")
    else:
        logger.warning(
            "Sokolow_Lyon skipped — missing one of: S_Amp_V1, R_Amp_V5, R_Amp_V6"
        )

    # Cornell = R_Amp_aVL + |S_Amp_V3|
    if "R_Amp_aVL" in df.columns and "S_Amp_V3" in df.columns:
        if "Sokolow_Lyon" not in df.columns:
            df = df.copy()
        df["Cornell"] = df["R_Amp_aVL"] + df["S_Amp_V3"].abs()
        added.append("Cornell")
        logger.info("Derived feature added: Cornell")
    else:
        logger.warning("Cornell skipped — missing one of: R_Amp_aVL, S_Amp_V3")

    return df, added


def _load_clinical_features(ecg_ids_train: np.ndarray) -> tuple[pd.DataFrame, list[str]]:
    """Load 12sl_features.csv, add derived features, return full feature df.

    Also returns the list of available candidate columns (including derived).
    """
    logger.info("Loading clinical features from %s", _FEATURES_CSV)
    df = pd.read_csv(_FEATURES_CSV, index_col="ecg_id")
    logger.info("12sl_features.csv: %d rows, %d columns", len(df), df.shape[1])

    # Add derived features
    df, derived_added = _add_derived_features(df)

    # Filter candidate list to columns that actually exist in the CSV
    available = [c for c in _CLINICAL_CANDIDATES if c in df.columns]
    missing   = [c for c in _CLINICAL_CANDIDATES if c not in df.columns]
    if missing:
        logger.warning("Clinical candidates not found in CSV (%d): %s", len(missing), missing)
    logger.info("Clinical candidates available: %d / %d", len(available), len(_CLINICAL_CANDIDATES))

    # Add derived features to the available list (always keep if added)
    for feat in derived_added:
        if feat not in available:
            available.append(feat)

    return df, available, derived_added


def _build_clinical_matrix(
    df: pd.DataFrame,
    ecg_ids: np.ndarray,
    feature_cols: list[str],
    impute_medians: dict | None = None,
    scaler_mean: np.ndarray | None = None,
    scaler_std: np.ndarray | None = None,
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    """Build clinical feature matrix for a set of ecg_ids.

    If impute_medians / scaler_mean / scaler_std are None, compute from this
    data (train set). Otherwise use provided stats (val/test set).

    Returns:
        X_clin:        (N, n_features) float32 array
        impute_medians: dict {col: median} computed on this data (returned only
                        for train; pass in for val/test)
        scaler_mean:   (n_features,) mean computed on this data
        scaler_std:    (n_features,) std computed on this data
    """
    subset = df.loc[ecg_ids, feature_cols].copy()

    is_train = impute_medians is None

    # Impute missing values
    if is_train:
        impute_medians = {}
        for col in feature_cols:
            med = float(subset[col].median())
            impute_medians[col] = med
            subset[col] = subset[col].fillna(med)
    else:
        for col in feature_cols:
            subset[col] = subset[col].fillna(impute_medians[col])

    X = subset.values.astype(np.float32)

    # Standardize
    if is_train:
        scaler_mean = X.mean(axis=0)
        scaler_std  = X.std(axis=0)
        scaler_std  = np.where(scaler_std == 0, 1.0, scaler_std)  # avoid div-by-zero

    X_scaled = (X - scaler_mean) / scaler_std
    return X_scaled.astype(np.float32), impute_medians, scaler_mean, scaler_std


def _auroc_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
    derived_features: list[str],
    top_k: int = 40,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Select top_k features by mean AUROC across 5 classes on train set.

    Always keeps derived_features in the selection.

    Returns:
        selected_cols: list of selected feature names
        ranked:        list of (feature_name, mean_auroc) sorted descending
    """
    from sklearn.metrics import roc_auc_score

    n_features = X_train.shape[1]
    mean_aurocs = np.zeros(n_features)

    for j in range(n_features):
        aucs = []
        for i in range(5):
            col = y_train[:, i]
            if len(np.unique(col)) < 2:
                continue
            try:
                a = roc_auc_score(col, X_train[:, j])
                # Flip if below 0.5 (feature may be inversely informative)
                if a < 0.5:
                    a = 1.0 - a
                aucs.append(a)
            except Exception:
                pass
        mean_aurocs[j] = float(np.mean(aucs)) if aucs else 0.5

    ranked_idx = np.argsort(mean_aurocs)[::-1]
    ranked = [(feature_cols[j], float(mean_aurocs[j])) for j in ranked_idx]

    # Always include derived features; fill remaining slots from top ranked
    derived_set = set(derived_features)
    selected = [f for f in derived_features if f in feature_cols]
    selected_set = set(selected)
    remaining_slots = top_k - len(selected)

    for feat, _ in ranked:
        if remaining_slots <= 0:
            break
        if feat not in selected_set:
            selected.append(feat)
            selected_set.add(feat)
            remaining_slots -= 1

    logger.info(
        "Feature selection: kept %d / %d features (top-%d by AUROC, derived always kept: %s)",
        len(selected), n_features, top_k, [f for f in derived_features if f in feature_cols],
    )
    return selected, ranked


# ---------------------------------------------------------------------------
# Meta-learner
# ---------------------------------------------------------------------------

def _build_meta_input(
    probs_ef:   np.ndarray,
    probs_x50:  np.ndarray,
    probs_x101: np.ndarray,
    clin:       np.ndarray | None,
) -> np.ndarray:
    """Stack DL probs and optional clinical features into meta-learner input."""
    parts = [probs_ef, probs_x50, probs_x101]
    if clin is not None:
        parts.append(clin)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _build_stacking_input(
    ensemble_probs: np.ndarray,
    clin:           np.ndarray | None,
) -> np.ndarray:
    """Stack pre-computed ensemble probs (N,5) and optional clinical features.

    This is the 'stacking-on-ensemble' variant: the meta-learner receives the
    already-averaged ensemble output as its DL signal (5 features instead of
    15), which guarantees it starts from the optimal weighted average.
    """
    parts = [ensemble_probs]
    if clin is not None:
        parts.append(clin)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _train_lr(X_train, y_train, C: float) -> list:
    """Train 5 binary LogisticRegression classifiers, return list of fitted models."""
    from sklearn.linear_model import LogisticRegression
    clfs = []
    for i in range(5):
        clf = LogisticRegression(C=C, max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y_train[:, i])
        clfs.append(clf)
    return clfs


def _train_mlp(X_train, y_train) -> list:
    """Train 5 binary MLPClassifiers, return list of fitted models."""
    from sklearn.neural_network import MLPClassifier
    clfs = []
    for i in range(5):
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        )
        clf.fit(X_train, y_train[:, i])
        clfs.append(clf)
    return clfs


def _predict_meta(clfs: list, X: np.ndarray) -> np.ndarray:
    """Return probability matrix (N, 5) from list of binary classifiers."""
    probs = np.stack(
        [clf.predict_proba(X)[:, 1] for clf in clfs], axis=1
    )
    return probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Late-fusion: 3-model DL ensemble + PTB-XL+ clinical features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Force re-run of DL inference even if cache files exist.",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device",    type=str, default="auto")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _setup_logging()

    sep = "=" * 80

    # ------------------------------------------------------------------
    # 1. DL probabilities (cached or fresh inference)
    # ------------------------------------------------------------------
    model_names = ["ef", "x50", "x101"]
    splits      = ["train", "val", "test"]

    use_cache = (not args.no_cache) and _all_cache_exists(splits, model_names)

    if use_cache:
        logger.info("Cache found — loading DL probs from disk.")
        dl_data = _load_probs_from_cache()
    else:
        if args.no_cache:
            logger.info("--no-cache flag set — re-running inference.")
        else:
            logger.info("Cache incomplete — running inference.")
        for label, path in [
            ("ecgfounder", _CKPT_EF),
            ("xresnet50",  _CKPT_X50),
            ("xresnet101", _CKPT_X101),
        ]:
            if not path.exists():
                raise SystemExit(f"{label} checkpoint not found: {path}")
        device = _resolve_device(args.device)
        logger.info("Device: %s", device)
        dl_data = _run_inference(args.batch_size, device)

    # Convenience aliases
    p_ef_tr,   p_x50_tr,   p_x101_tr   = (dl_data["train"]["probs_ef"],
                                            dl_data["train"]["probs_x50"],
                                            dl_data["train"]["probs_x101"])
    p_ef_val,  p_x50_val,  p_x101_val  = (dl_data["val"]["probs_ef"],
                                            dl_data["val"]["probs_x50"],
                                            dl_data["val"]["probs_x101"])
    p_ef_te,   p_x50_te,   p_x101_te   = (dl_data["test"]["probs_ef"],
                                            dl_data["test"]["probs_x50"],
                                            dl_data["test"]["probs_x101"])
    y_train = dl_data["train"]["y"]
    y_val   = dl_data["val"]["y"]
    y_test  = dl_data["test"]["y"]
    ids_train = dl_data["train"]["ecg_ids"]
    ids_val   = dl_data["val"]["ecg_ids"]
    ids_test  = dl_data["test"]["ecg_ids"]

    logger.info(
        "Split sizes — train=%d  val=%d  test=%d",
        len(y_train), len(y_val), len(y_test),
    )

    # ------------------------------------------------------------------
    # 2. Baseline ensemble (40/30/30 simple average)
    # ------------------------------------------------------------------
    w_ef, w_x50, w_x101 = _BASELINE_WEIGHTS
    p_ens_val  = w_ef * p_ef_val  + w_x50 * p_x50_val  + w_x101 * p_x101_val
    p_ens_test = w_ef * p_ef_te   + w_x50 * p_x50_te   + w_x101 * p_x101_te

    m_ens_val  = _auc(y_val,  p_ens_val)
    m_ens_test = _auc(y_test, p_ens_test)

    print(sep)
    print("  BASELINE — 3-model ensemble (40/30/30 probability average):")
    _print_metrics(m_ens_val,  "Ensemble val")
    _print_metrics(m_ens_test, "Ensemble test")
    print(sep)

    # ------------------------------------------------------------------
    # 3. Clinical feature pipeline
    # ------------------------------------------------------------------
    logger.info("=== Clinical feature pipeline ===")
    df_clin, available_cols, derived_added = _load_clinical_features(ids_train)

    # Verify coverage
    all_ids = np.concatenate([ids_train, ids_val, ids_test])
    missing_ids = set(all_ids) - set(df_clin.index)
    if missing_ids:
        logger.warning(
            "Clinical features missing for %d ecg_ids — will be fully imputed.",
            len(missing_ids),
        )

    # Build initial clinical matrix on train (compute impute/scaler stats)
    X_clin_tr_raw, impute_medians, scaler_mean, scaler_std = _build_clinical_matrix(
        df_clin, ids_train, available_cols,
    )
    logger.info(
        "Initial clinical matrix: train shape = %s", X_clin_tr_raw.shape
    )

    # AUROC-based feature selection on train
    selected_cols, ranked_all = _auroc_feature_selection(
        X_clin_tr_raw, y_train, available_cols, derived_added, top_k=_TOP_K
    )

    # Print top-10 features by AUROC
    print(sep)
    print("  Top-10 clinical features by mean AUROC (train):")
    for feat, auroc in ranked_all[:10]:
        marker = " [derived]" if feat in _DERIVED_FEATURES else ""
        print(f"    {feat:<30}  mean_auroc={auroc:.4f}{marker}")
    print(sep)

    # Rebuild matrices with selected columns only
    X_clin_tr, impute_medians, scaler_mean, scaler_std = _build_clinical_matrix(
        df_clin, ids_train, selected_cols,
    )
    X_clin_val, _, _, _ = _build_clinical_matrix(
        df_clin, ids_val, selected_cols,
        impute_medians=impute_medians,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    X_clin_te, _, _, _ = _build_clinical_matrix(
        df_clin, ids_test, selected_cols,
        impute_medians=impute_medians,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    logger.info(
        "Final clinical matrices — train=%s  val=%s  test=%s",
        X_clin_tr.shape, X_clin_val.shape, X_clin_te.shape,
    )

    # ------------------------------------------------------------------
    # 4. Build meta-learner input matrices
    # ------------------------------------------------------------------
    X_meta_tr  = _build_meta_input(p_ef_tr,  p_x50_tr,  p_x101_tr,  X_clin_tr)
    X_meta_val = _build_meta_input(p_ef_val, p_x50_val, p_x101_val, X_clin_val)
    X_meta_te  = _build_meta_input(p_ef_te,  p_x50_te,  p_x101_te,  X_clin_te)
    logger.info(
        "Meta-input shapes — train=%s  val=%s  test=%s",
        X_meta_tr.shape, X_meta_val.shape, X_meta_te.shape,
    )

    # ------------------------------------------------------------------
    # 5. Meta-learner variants
    # ------------------------------------------------------------------
    logger.info("=== Training meta-learners ===")

    variants = {
        "LR (C=1.0)":    lambda: _train_lr(X_meta_tr, y_train, C=1.0),
        "LR_strong (C=0.1)": lambda: _train_lr(X_meta_tr, y_train, C=0.1),
        "MLP (64,32)":   lambda: _train_mlp(X_meta_tr, y_train),
    }

    val_results: dict[str, dict] = {}
    trained_clfs: dict[str, list] = {}

    print(sep)
    print("  Meta-learner validation results:")
    for name, train_fn in variants.items():
        logger.info("Training %s ...", name)
        clfs = train_fn()
        p_val_meta = _predict_meta(clfs, X_meta_val)
        m_val = _auc(y_val, p_val_meta)
        val_results[name]  = m_val
        trained_clfs[name] = clfs
        _print_metrics(m_val, name)

    print(sep)

    # Pick best by val macro-AUC
    best_name = max(val_results, key=lambda k: val_results[k]["macro_auc"])
    best_clfs = trained_clfs[best_name]
    logger.info("Best meta-learner on val: %s  macro_auc=%.4f",
                best_name, val_results[best_name]["macro_auc"])

    # ------------------------------------------------------------------
    # 6. Test evaluation with best meta-learner
    # ------------------------------------------------------------------
    p_test_best = _predict_meta(best_clfs, X_meta_te)
    m_test_best = _auc(y_test, p_test_best)

    print(sep)
    print(f"  Best meta-learner: {best_name}")
    print("  TEST RESULTS (best meta-learner):")
    _print_metrics(m_test_best, f"{best_name} [test]")
    print("  Per-class AUC breakdown:")
    for cls, auc in m_test_best["per_class_auc"].items():
        print(f"    {cls:<6}  {auc:.4f}")
    print(sep)

    # ------------------------------------------------------------------
    # 7. Stacking-on-ensemble: ensemble probs (5) + clinical (40) → LR
    #    Uses the pre-averaged 40/30/30 ensemble as the DL signal so the
    #    meta-learner starts from the optimal base and only learns the
    #    clinical adjustment.
    # ------------------------------------------------------------------
    logger.info("=== Stacking-on-ensemble experiment ===")

    p_ens_tr = w_ef * p_ef_tr + w_x50 * p_x50_tr + w_x101 * p_x101_tr

    X_stack_tr  = _build_stacking_input(p_ens_tr,  X_clin_tr)
    X_stack_val = _build_stacking_input(p_ens_val, X_clin_val)
    X_stack_te  = _build_stacking_input(p_ens_test, X_clin_te)
    logger.info(
        "Stacking-on-ensemble shapes — train=%s  val=%s  test=%s",
        X_stack_tr.shape, X_stack_val.shape, X_stack_te.shape,
    )

    stack_variants = {
        "Stack LR (C=1.0)":     lambda: _train_lr(X_stack_tr, y_train, C=1.0),
        "Stack LR_strong (C=0.1)": lambda: _train_lr(X_stack_tr, y_train, C=0.1),
        "Stack LR_tight (C=0.01)": lambda: _train_lr(X_stack_tr, y_train, C=0.01),
    }

    stack_val_results: dict[str, dict] = {}
    stack_trained_clfs: dict[str, list] = {}

    print(sep)
    print("  Stacking-on-ensemble (5 ensemble probs + 40 clinical) — validation:")
    for name, train_fn in stack_variants.items():
        logger.info("Training %s ...", name)
        clfs = train_fn()
        p_v = _predict_meta(clfs, X_stack_val)
        m_v = _auc(y_val, p_v)
        stack_val_results[name]  = m_v
        stack_trained_clfs[name] = clfs
        _print_metrics(m_v, name)
    print(sep)

    best_stack_name = max(stack_val_results, key=lambda k: stack_val_results[k]["macro_auc"])
    best_stack_clfs = stack_trained_clfs[best_stack_name]
    logger.info(
        "Best stacking model on val: %s  macro_auc=%.4f",
        best_stack_name, stack_val_results[best_stack_name]["macro_auc"],
    )

    p_stack_te  = _predict_meta(best_stack_clfs, X_stack_te)
    m_stack_te  = _auc(y_test, p_stack_te)

    print(sep)
    print(f"  STACKING-ON-ENSEMBLE — Best: {best_stack_name}")
    print("  TEST RESULTS:")
    _print_metrics(m_stack_te, f"{best_stack_name} [test]")
    print("  Per-class AUC breakdown:")
    for cls, auc in m_stack_te["per_class_auc"].items():
        print(f"    {cls:<6}  {auc:.4f}")
    gain_stack_vs_ens = m_stack_te["macro_auc"] - m_ens_test["macro_auc"]
    print(f"  Gain (stacking vs baseline ensemble):  {gain_stack_vs_ens:+.4f}")
    print(sep)

    # Also run stacking with ensemble-only (no clinical) as reference point
    X_ens_only_tr  = _build_stacking_input(p_ens_tr,    None)
    X_ens_only_val = _build_stacking_input(p_ens_val,   None)
    X_ens_only_te  = _build_stacking_input(p_ens_test,  None)
    clfs_ens_only  = _train_lr(X_ens_only_tr, y_train, C=0.01)
    p_ens_only_val = _predict_meta(clfs_ens_only, X_ens_only_val)
    p_ens_only_te  = _predict_meta(clfs_ens_only, X_ens_only_te)
    m_ens_only_val = _auc(y_val,  p_ens_only_val)
    m_ens_only_te  = _auc(y_test, p_ens_only_te)

    # ------------------------------------------------------------------
    # 8. Ablation study
    # ------------------------------------------------------------------
    logger.info("=== Ablation study ===")

    # a. DL probs only (no clinical features) → 15 features
    X_dl_tr  = _build_meta_input(p_ef_tr,  p_x50_tr,  p_x101_tr,  None)
    X_dl_val = _build_meta_input(p_ef_val, p_x50_val, p_x101_val, None)
    X_dl_te  = _build_meta_input(p_ef_te,  p_x50_te,  p_x101_te,  None)

    clfs_dl_only = _train_lr(X_dl_tr, y_train, C=0.1)
    p_dl_val = _predict_meta(clfs_dl_only, X_dl_val)
    p_dl_te  = _predict_meta(clfs_dl_only, X_dl_te)
    m_dl_val = _auc(y_val,  p_dl_val)
    m_dl_te  = _auc(y_test, p_dl_te)

    # b. Clinical features only (no DL probs)
    clfs_clin_only_val, clfs_clin_only_te = None, None
    from sklearn.linear_model import LogisticRegression
    clfs_clin = []
    for i in range(5):
        clf = LogisticRegression(C=0.1, max_iter=1000, solver="lbfgs")
        clf.fit(X_clin_tr, y_train[:, i])
        clfs_clin.append(clf)
    p_clin_val = _predict_meta(clfs_clin, X_clin_val)
    p_clin_te  = _predict_meta(clfs_clin, X_clin_te)
    m_clin_val = _auc(y_val,  p_clin_val)
    m_clin_te  = _auc(y_test, p_clin_te)

    # c. Full fusion (already computed above as best test)
    print(sep)
    print("  ABLATION (best meta-learner type on each condition):")
    print("  Condition a — DL probs only (15 features, no clinical):")
    _print_metrics(m_dl_val, "DL-only val")
    _print_metrics(m_dl_te,  "DL-only test")
    print("  Condition b — Clinical features only (no DL probs):")
    _print_metrics(m_clin_val, "Clinical-only val")
    _print_metrics(m_clin_te,  "Clinical-only test")
    print("  Condition c — DL probs + clinical features (full fusion, 55->LR):")
    _print_metrics(val_results[best_name], f"{best_name} val")
    _print_metrics(m_test_best, f"{best_name} test")
    print("  Condition d — Ensemble probs only (5->LR, C=0.01):")
    _print_metrics(m_ens_only_val, "Ens-only val")
    _print_metrics(m_ens_only_te,  "Ens-only test")
    print("  Condition e — Ensemble probs + clinical (45->LR, stacking):")
    _print_metrics(stack_val_results[best_stack_name], f"{best_stack_name} val")
    _print_metrics(m_stack_te, f"{best_stack_name} test")
    print(sep)

    # Gains
    gain_vs_ens = m_test_best["macro_auc"] - m_ens_test["macro_auc"]
    gain_vs_dl  = m_test_best["macro_auc"] - m_dl_te["macro_auc"]
    print(f"  Gain (full fusion vs baseline ensemble):   {gain_vs_ens:+.4f}")
    print(f"  Gain (full fusion vs DL-only meta):         {gain_vs_dl:+.4f}")
    print(f"  Gain (stacking-on-ens vs baseline ensemble):{gain_stack_vs_ens:+.4f}")
    print(sep)

    # ------------------------------------------------------------------
    # 8. Save metrics JSON
    # ------------------------------------------------------------------
    output = {
        "baseline_ensemble_40_30_30": {
            "val":  m_ens_val,
            "test": m_ens_test,
        },
        "meta_learner_val_results": val_results,
        "best_meta_learner": best_name,
        "best_meta_learner_test": m_test_best,
        "stacking_on_ensemble": {
            "best_variant": best_stack_name,
            "val_results":  stack_val_results,
            "test":         m_stack_te,
            "gain_vs_baseline": float(gain_stack_vs_ens),
        },
        "ablation": {
            "dl_probs_only": {
                "val":  m_dl_val,
                "test": m_dl_te,
            },
            "clinical_only": {
                "val":  m_clin_val,
                "test": m_clin_te,
            },
            "full_fusion_55": {
                "val":  val_results[best_name],
                "test": m_test_best,
            },
            "ensemble_probs_only_5": {
                "val":  m_ens_only_val,
                "test": m_ens_only_te,
            },
            "stacking_45": {
                "val":  stack_val_results[best_stack_name],
                "test": m_stack_te,
            },
        },
        "top_10_clinical_features": [
            {"feature": feat, "mean_auroc": auroc}
            for feat, auroc in ranked_all[:10]
        ],
        "selected_clinical_features": selected_cols,
        "gain_vs_baseline_ensemble": float(gain_vs_ens),
        "gain_vs_dl_only_meta":       float(gain_vs_dl),
        "gain_stacking_vs_baseline":  float(gain_stack_vs_ens),
    }

    out_path = _OUT_DIR / "metrics.json"
    _save_json(output, out_path)
    logger.info("All results saved -> %s", out_path)


if __name__ == "__main__":
    main()
