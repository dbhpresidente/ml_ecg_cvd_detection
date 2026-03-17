"""Download script for ECG datasets from PhysioNet."""
import argparse
from pathlib import Path

import wfdb


DATASETS = {
    "ptb-xl": {
        "physionet_name": "ptb-xl",
        "version": "1.0.3",
        "description": "PTB-XL: 21,837 12-lead ECGs, 71 diagnostic classes",
        "size_gb": 6.6,
    },
    "mitbih": {
        "physionet_name": "mitdb",
        "version": "1.0.0",
        "description": "MIT-BIH Arrhythmia: 48 two-lead 30-min recordings",
        "size_gb": 0.1,
    },
    "cpsc2018": {
        "physionet_name": "cpsc2018",
        "version": "1.0.0",
        "description": "CPSC 2018: 6,877 12-lead ECGs, 9 rhythm classes",
        "size_gb": 1.7,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ECG datasets from PhysioNet")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="ptb-xl",
        help="Dataset to download (default: ptb-xl)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory to save the dataset (default: data/raw)",
    )
    return parser.parse_args()


def download(dataset_key: str, output_dir: Path) -> None:
    """Download a PhysioNet dataset to the specified directory.

    Args:
        dataset_key: Key from the DATASETS dictionary.
        output_dir: Root directory where the dataset will be saved.
    """
    info = DATASETS[dataset_key]
    target_dir = output_dir / dataset_key

    print(f"Dataset  : {info['description']}")
    print(f"Size     : ~{info['size_gb']} GB")
    print(f"Target   : {target_dir.resolve()}")
    print()

    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"Directory {target_dir} already exists and is not empty. Skipping download.")
        return

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {dataset_key} from PhysioNet...")
    wfdb.dl_database(info["physionet_name"], dl_dir=str(target_dir))
    print(f"\nDone. Dataset saved to: {target_dir.resolve()}")


def main() -> None:
    args = parse_args()
    download(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
