"""Training entry point for ECG CVD detection."""
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ECG CVD detection model")
    parser.add_argument("--config", type=Path, default="configs/default.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: implement training pipeline
    print(f"Config: {args.config}")
    print(f"Run name: {args.run_name}")


if __name__ == "__main__":
    main()
