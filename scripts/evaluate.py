"""Evaluation entry point for ECG CVD detection."""
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ECG CVD detection model")
    parser.add_argument("--config", type=Path, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=Path, required=False)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: implement evaluation pipeline
    print(f"Evaluating on {args.split} split")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
