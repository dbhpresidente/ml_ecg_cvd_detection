"""PTB-XL dataset class compatible with PyTorch and scikit-learn workflows."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import ALL_CVD_CLASSES, get_binary_label, load_metadata, load_raw_signals
from src.data.preprocess import preprocess


class PTBXLDataset:
    """PTB-XL ECG dataset with preprocessing and train/val/test splitting.

    Loads ECG signals and binary labels from PTB-XL. Uses the official
    strat_fold column (1–10) for reproducible splits.

    By default, positive_classes includes all CVD superclasses (any CVD vs. NORM).

    Example:
        >>> # Any CVD vs. NORM (default)
        >>> dataset = PTBXLDataset("data/raw/ptb-xl")
        >>> # MI only vs. NORM
        >>> dataset = PTBXLDataset("data/raw/ptb-xl", positive_classes="MI")
        >>> # MI or STTC vs. NORM
        >>> dataset = PTBXLDataset("data/raw/ptb-xl", positive_classes=["MI", "STTC"])
        >>> X_train, y_train = dataset.get_split("train")
        >>> X_test, y_test = dataset.get_split("test")

    Attributes:
        data_dir: Path to the PTB-XL root directory.
        sampling_rate: ECG sampling frequency in Hz.
        positive_classes: List of diagnostic superclasses used as positive label.
    """

    # PTB-XL official split: fold 9 = val, fold 10 = test, 1-8 = train
    TRAIN_FOLDS = list(range(1, 9))
    VAL_FOLDS = [9]
    TEST_FOLDS = [10]

    def __init__(
        self,
        data_dir: str | Path,
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
            positive_classes: Superclass(es) to use as positive label.
                A record is positive if it belongs to ANY of the listed classes.
                Valid values: 'MI', 'STTC', 'CD', 'HYP'.
                Defaults to None, which uses all four (any CVD vs. NORM).
                Examples:
                    - None                        → any CVD vs. NORM (default)
                    - "MI"                        → MI vs. NORM
                    - ["MI", "STTC"]              → MI or STTC vs. NORM
                    - ["MI", "STTC", "CD", "HYP"] → same as default
            sampling_rate: ECG sampling rate in Hz (500 or 100).
            normalization: Signal normalization method ('zscore', 'minmax', 'robust').
            bandpass: Whether to apply bandpass filter.
            notch: Whether to apply notch filter.
            notch_hz: Powerline frequency (50 or 60 Hz).
            min_confidence: Minimum annotator confidence for labels.
        """
        self.data_dir = Path(data_dir)
        self.positive_classes = (
            [positive_classes] if isinstance(positive_classes, str)
            else positive_classes
        )
        self.sampling_rate = sampling_rate
        self.normalization = normalization
        self.bandpass = bandpass
        self.notch = notch
        self.notch_hz = notch_hz
        self.min_confidence = min_confidence

        self._records_df, self._scp_df = load_metadata(self.data_dir)
        self._records_df = self._filter_binary(self._records_df)

    def _filter_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only records that are clearly NORM or positive_classes."""
        labels = get_binary_label(df, self._scp_df, self.positive_classes, self.min_confidence)
        norm_labels = get_binary_label(
            df, self._scp_df, ["NORM"], self.min_confidence
        )
        mask = (labels == 1) | (norm_labels == 1)
        return df[mask]

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
                y: int64 array of shape (n_records,) with binary labels.
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

        labels = get_binary_label(subset, self._scp_df, self.positive_classes, self.min_confidence)
        return signals, labels

    def class_distribution(self, split: str | None = None) -> dict[str, int]:
        """Return class counts for a split or the full dataset.

        Args:
            split: One of 'train', 'val', 'test', or None for all records.

        Returns:
            Dictionary with 'negative (NORM)' and 'positive' counts.
        """
        df = self._records_df[self._get_fold_mask(split)] if split else self._records_df
        labels = get_binary_label(df, self._scp_df, self.positive_classes, self.min_confidence)
        return {
            "negative (NORM)": int((labels == 0).sum()),
            f"positive {self.positive_classes}": int((labels == 1).sum()),
        }
