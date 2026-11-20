"""PTB-XL dataset class compatible with PyTorch and scikit-learn workflows."""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import (
    ALL_CVD_CLASSES,
    SUPERCLASSES,
    get_binary_label,
    get_multilabel,
    load_metadata,
    load_raw_signals,
)
from src.data.preprocess import preprocess

logger = logging.getLogger(__name__)


class PTBXLDataset:
    """PTB-XL ECG dataset with preprocessing and train/val/test splitting.

    Supports two modes:
        - ``"binary"``    (default): any-CVD vs. NORM, labels shape (n,) int64.
        - ``"multilabel"``:          5 superclasses, labels shape (n, 5) float32.
                                     Matches the PTB-XL benchmark protocol of
                                     Strodthoff et al. (IEEE JBHI 2021).

    Split integrity guarantees (verified by validate_splits()):
        1. Patient-level isolation: each patient_id appears in exactly one fold.
           PTB-XL was designed this way; validate_splits() asserts it.
        2. No unlabelled records in multilabel mode: records with no assigned
           superclass (all-zero label vector) are removed at init time.

    Uses the official strat_fold column (1–10) for reproducible splits.

    Example:
        >>> # Binary: any CVD vs. NORM
        >>> dataset = PTBXLDataset("data/raw/ptb-xl")
        >>> # Multi-label: 5 superclasses (NORM, MI, STTC, CD, HYP)
        >>> dataset = PTBXLDataset("data/raw/ptb-xl", mode="multilabel")
        >>> X_train, y_train = dataset.get_split("train")
        >>> X_test, y_test = dataset.get_split("test")

    Attributes:
        data_dir: Path to the PTB-XL root directory.
        sampling_rate: ECG sampling frequency in Hz.
        mode: ``"binary"`` or ``"multilabel"``.
        superclasses: Ordered list of class names (multilabel mode).
    """

    # PTB-XL official split: fold 9 = val, fold 10 = test, 1-8 = train
    TRAIN_FOLDS = list(range(1, 9))
    VAL_FOLDS = [9]
    TEST_FOLDS = [10]

    superclasses: list[str] = SUPERCLASSES

    def __init__(
        self,
        data_dir: str | Path,
        mode: str = "binary",
        positive_classes: list[str] | str = ALL_CVD_CLASSES,
        sampling_rate: int = 500,
        normalization: str = "zscore",
        bandpass: bool = True,
        notch: bool = True,
        notch_hz: float = 50.0,
        min_confidence: float = 100.0,
    ) -> None:
        """Initialize the PTB-XL dataset.

        Args:
            data_dir: Path to the PTB-XL root directory.
            mode: ``"binary"`` (default) or ``"multilabel"``.
                Binary mode filters to clear NORM/CVD records and returns (n,) int64.
                Multilabel mode excludes unlabelled records and returns (n, 5) float32.
            positive_classes: (Binary mode only) Superclass(es) for positive label.
            sampling_rate: ECG sampling rate in Hz (500 or 100).
            normalization: Signal normalization ('zscore', 'minmax', 'robust').
            bandpass: Whether to apply bandpass filter.
            notch: Whether to apply notch filter.
            notch_hz: Powerline frequency (50 or 60 Hz).
            min_confidence: Minimum annotator confidence for labels.
                Binary mode default 100.0; multilabel follows Strodthoff (use 0.0).
        """
        if mode not in ("binary", "multilabel"):
            raise ValueError(f"mode must be 'binary' or 'multilabel', got '{mode}'")

        self.data_dir = Path(data_dir)
        self.mode = mode
        self.positive_classes = (
            [positive_classes] if isinstance(positive_classes, str)
            else list(positive_classes)
        )
        self.sampling_rate = sampling_rate
        self.normalization = normalization
        self.bandpass = bandpass
        self.notch = notch
        self.notch_hz = notch_hz
        self.min_confidence = min_confidence

        self._records_df, self._scp_df = load_metadata(self.data_dir)

        if mode == "binary":
            self._records_df = self._filter_binary(self._records_df)
        else:
            self._records_df = self._filter_multilabel(self._records_df)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only records that are clearly NORM or positive_classes."""
        labels = get_binary_label(df, self._scp_df, self.positive_classes, self.min_confidence)
        norm_labels = get_binary_label(df, self._scp_df, ["NORM"], self.min_confidence)
        mask = (labels == 1) | (norm_labels == 1)
        return df[mask]

    def _filter_multilabel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with no assigned superclass (all-zero label vector).

        These records have SCP codes that do not map to any of the 5 diagnostic
        superclasses (NORM, MI, STTC, CD, HYP). Including them as negatives for
        all classes would introduce ambiguous supervision.
        """
        labels = get_multilabel(df, self._scp_df, self.min_confidence)
        has_label = labels.sum(axis=1) > 0
        n_removed = int((~has_label).sum())
        if n_removed > 0:
            logger.info(
                "Multilabel mode: removed %d records with no superclass assignment "
                "(%d -> %d records).",
                n_removed, len(df), int(has_label.sum()),
            )
        return df[has_label]

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    def _get_fold_mask(self, split: str) -> pd.Series:
        """Return a boolean mask for the requested split."""
        fold_map = {
            "train": self.TRAIN_FOLDS,
            "val": self.VAL_FOLDS,
            "test": self.TEST_FOLDS,
        }
        if split not in fold_map:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")
        return self._records_df.strat_fold.isin(fold_map[split])

    def get_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        """Load and preprocess ECG signals for a dataset split.

        Args:
            split: One of 'train', 'val', or 'test'.

        Returns:
            Tuple (X, y) where:
                X: float32 array of shape (n_records, 12, samples).
                y: Binary mode  → int64 array of shape (n_records,).
                   Multilabel   → float32 array of shape (n_records, 5),
                                  columns: NORM, MI, STTC, CD, HYP.
        """
        mask = self._get_fold_mask(split)
        subset = self._records_df[mask]

        print(f"Loading {split} split: {len(subset)} records...")
        signals = load_raw_signals(subset, self.data_dir, self.sampling_rate)

        print(f"Preprocessing {split} signals...")
        signals = preprocess(
            signals,
            fs=self.sampling_rate,
            bandpass=self.bandpass,
            notch=self.notch,
            notch_hz=self.notch_hz,
            normalization=self.normalization,
        )

        if self.mode == "multilabel":
            labels = get_multilabel(subset, self._scp_df, self.min_confidence)
        else:
            labels = get_binary_label(
                subset, self._scp_df, self.positive_classes, self.min_confidence
            )
        return signals, labels

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_splits(self) -> dict:
        """Validate patient-level isolation and label completeness across splits.

        Checks:
            1. No patient appears in more than one fold (train / val / test).
            2. No record has an all-zero label vector (multilabel mode only).

        Returns:
            Dict with keys:
                ``patient_isolation_ok`` (bool): True if no patient crosses splits.
                ``patients_in_multiple_splits`` (list): patient_ids violating isolation.
                ``total_records`` (int): records after filtering.
                ``records_per_split`` (dict): count per split name.
                ``unlabelled_records`` (int): records with all-zero labels (always 0
                    in multilabel mode because they were removed at init).

        Raises:
            AssertionError: if patient isolation is violated (should never happen
                with the official PTB-XL folds).
        """
        # 1. Patient-level isolation
        patient_folds = (
            self._records_df
            .groupby("patient_id")["strat_fold"]
            .nunique()
        )
        multi_fold_patients = patient_folds[patient_folds > 1].index.tolist()
        isolation_ok = len(multi_fold_patients) == 0

        if not isolation_ok:
            logger.error(
                "SPLIT VIOLATION: %d patient(s) appear in more than one fold: %s",
                len(multi_fold_patients), multi_fold_patients[:10],
            )
        else:
            logger.info(
                "Patient isolation OK — %d unique patients, all in a single fold.",
                self._records_df.patient_id.nunique(),
            )

        assert isolation_ok, (
            f"Patient isolation violated: {multi_fold_patients[:5]} ... "
            "Check the strat_fold column."
        )

        # 2. Records per split
        records_per_split = {
            split: int(self._get_fold_mask(split).sum())
            for split in ("train", "val", "test")
        }

        # 3. Unlabelled records (always 0 in multilabel mode after init filter)
        if self.mode == "multilabel":
            labels = get_multilabel(self._records_df, self._scp_df, self.min_confidence)
            unlabelled = int((labels.sum(axis=1) == 0).sum())
        else:
            unlabelled = 0

        logger.info(
            "Split sizes — train: %d  val: %d  test: %d",
            records_per_split["train"],
            records_per_split["val"],
            records_per_split["test"],
        )
        logger.info("Unlabelled records remaining: %d", unlabelled)

        return {
            "patient_isolation_ok": isolation_ok,
            "patients_in_multiple_splits": multi_fold_patients,
            "total_records": len(self._records_df),
            "records_per_split": records_per_split,
            "unlabelled_records": unlabelled,
        }

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def class_distribution(self, split: str | None = None) -> dict[str, int]:
        """Return class counts for a split or the full dataset.

        Binary mode: returns {negative, positive} counts.
        Multilabel mode: returns per-superclass positive counts.

        Args:
            split: One of 'train', 'val', 'test', or None for all records.

        Returns:
            Dictionary mapping class name to positive-sample count.
        """
        df = self._records_df[self._get_fold_mask(split)] if split else self._records_df

        if self.mode == "multilabel":
            labels = get_multilabel(df, self._scp_df, self.min_confidence)
            return {sc: int(labels[:, i].sum()) for i, sc in enumerate(SUPERCLASSES)}

        labels = get_binary_label(df, self._scp_df, self.positive_classes, self.min_confidence)
        return {
            "negative (NORM)": int((labels == 0).sum()),
            f"positive {self.positive_classes}": int((labels == 1).sum()),
        }
