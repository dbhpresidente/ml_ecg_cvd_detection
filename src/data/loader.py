"""PTB-XL dataset loader.

Loads ECG records and metadata from the PTB-XL dataset.
Reference: Wagner et al. (2020), Nature Scientific Data.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

ALL_CVD_CLASSES: list[str] = ["MI", "STTC", "CD", "HYP"]


def load_metadata(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load PTB-XL metadata and SCP statement definitions.

    Args:
        data_dir: Path to the PTB-XL root directory (contains ptbxl_database.csv).

    Returns:
        Tuple of (records_df, scp_df) where records_df contains per-record metadata
        and scp_df contains the SCP diagnostic code definitions.

    Raises:
        FileNotFoundError: If ptbxl_database.csv or scp_statements.csv are not found.
    """
    db_path = data_dir / "ptbxl_database.csv"
    scp_path = data_dir / "scp_statements.csv"

    if not db_path.exists():
        raise FileNotFoundError(f"PTB-XL database file not found: {db_path}")
    if not scp_path.exists():
        raise FileNotFoundError(f"SCP statements file not found: {scp_path}")

    records_df = pd.read_csv(db_path, index_col="ecg_id")
    records_df.scp_codes = records_df.scp_codes.apply(ast.literal_eval)

    scp_df = pd.read_csv(scp_path, index_col=0)

    return records_df, scp_df


def load_signal(record_path: str, data_dir: Path, sampling_rate: int = 500) -> np.ndarray:
    """Load a single ECG record from disk.

    Args:
        record_path: Relative path to the record (from ptbxl_database.csv filename_hr
            or filename_lr columns).
        data_dir: Path to the PTB-XL root directory.
        sampling_rate: Target sampling rate in Hz. Use 500 for high-res or 100 for low-res.

    Returns:
        ECG signal as a float32 array of shape (12, samples).
    """
    full_path = str(data_dir / record_path)
    record = wfdb.rdrecord(full_path)
    signal = record.p_signal.T.astype(np.float32)  # (leads, samples)
    return signal


def load_raw_signals(
    records_df: pd.DataFrame,
    data_dir: Path,
    sampling_rate: int = 500,
) -> np.ndarray:
    """Load all ECG signals for a set of records.

    Args:
        records_df: DataFrame from load_metadata(), filtered to desired records.
        data_dir: Path to the PTB-XL root directory.
        sampling_rate: 500 for high-resolution, 100 for low-resolution records.

    Returns:
        Array of shape (n_records, 12, samples) with float32 values.
    """
    col = "filename_hr" if sampling_rate == 500 else "filename_lr"
    signals = [load_signal(row[col], data_dir, sampling_rate) for _, row in records_df.iterrows()]
    return np.stack(signals, axis=0)


def get_labels(
    records_df: pd.DataFrame,
    scp_df: pd.DataFrame,
    superclass: bool = True,
    min_confidence: float = 100.0,
) -> pd.Series:
    """Extract diagnostic labels from SCP codes.

    Args:
        records_df: DataFrame from load_metadata().
        scp_df: SCP statements DataFrame from load_metadata().
        superclass: If True, aggregate to 5 superclasses (NORM, MI, STTC, CD, HYP).
            If False, return raw SCP codes.
        min_confidence: Minimum annotator confidence (0-100) to include a label.

    Returns:
        Series indexed by ecg_id with lists of diagnostic labels.
    """
    # Filter SCP codes to diagnostic statements only
    diagnostic_scp = scp_df[scp_df.diagnostic == 1].index.tolist()

    def extract_labels(scp_codes: dict) -> list[str]:
        if superclass:
            classes = set()
            for code, confidence in scp_codes.items():
                if confidence >= min_confidence and code in scp_df.index:
                    sc = scp_df.loc[code, "diagnostic_class"]
                    if pd.notna(sc):
                        classes.add(sc)
            return sorted(classes)
        else:
            return [
                code for code, conf in scp_codes.items()
                if conf >= min_confidence and code in diagnostic_scp
            ]

    return records_df.scp_codes.apply(extract_labels)


def get_binary_label(
    records_df: pd.DataFrame,
    scp_df: pd.DataFrame,
    positive_classes: list[str] | str = ALL_CVD_CLASSES,
    min_confidence: float = 100.0,
) -> np.ndarray:
    """Extract binary labels for one or more diagnostic superclasses vs. NORM.

    A record is labeled positive (1) if it belongs to ANY of the positive_classes.
    This allows grouping multiple superclasses as a single "CVD" positive label.

    Args:
        records_df: DataFrame from load_metadata().
        scp_df: SCP statements DataFrame from load_metadata().
        positive_classes: One or more superclasses to use as positive label.
            Valid values: 'MI', 'STTC', 'CD', 'HYP'.
            Examples:
                - "MI"                        → MI vs. NORM
                - ["MI", "STTC"]              → MI or STTC vs. NORM
                - ["MI", "STTC", "CD", "HYP"] → any CVD vs. NORM
        min_confidence: Minimum annotator confidence to include a label.

    Returns:
        Binary array of shape (n_records,) where 1 = positive, 0 = NORM.
    """
    if isinstance(positive_classes, str):
        positive_classes = [positive_classes]

    positive_set = set(positive_classes)
    labels = get_labels(records_df, scp_df, superclass=True, min_confidence=min_confidence)
    binary = np.array([
        1 if positive_set & set(lbls) else 0
        for lbls in labels
    ], dtype=np.int64)
    return binary
