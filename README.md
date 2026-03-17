# ml_ecg_cvd_detection

Machine Learning for ECG-based Cardiovascular Disease Detection.

## Requirements

- Python >= 3.10
- See `requirements.txt` for full dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train
make train
# or
python scripts/train.py --config configs/default.yaml --run-name baseline

# Evaluate
make eval
# or
python scripts/evaluate.py --checkpoint checkpoints/best.pt --split test

# Run tests
make test

# Lint + type check
make lint
```

## Project Structure

```
ml_ecg_cvd_detection/
├── data/
│   ├── raw/           # Original ECG records (not tracked by git)
│   ├── processed/     # Preprocessed signals ready for training
│   └── external/      # Third-party datasets (PhysioNet, etc.)
├── notebooks/         # Exploratory analysis
├── src/
│   ├── data/          # Data loading, preprocessing, dataset classes
│   ├── models/        # Model architectures
│   ├── features/      # Feature extraction
│   ├── training/      # Training and evaluation logic
│   └── utils/         # Shared utilities and metrics
├── tests/             # Unit and integration tests
├── configs/           # Experiment configurations (YAML)
├── scripts/           # CLI entry points (train, evaluate)
├── reports/           # Metrics, figures, generated reports
└── references/        # Papers and research notes
```

## Configuration

Experiments are configured via YAML files in `configs/`. The default config is at `configs/default.yaml`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data.sampling_rate` | 500 Hz | Target ECG sampling frequency |
| `data.leads` | 12 | Number of ECG leads |
| `model.architecture` | `cnn_1d` | Model architecture |
| `training.epochs` | 50 | Maximum training epochs |
| `training.learning_rate` | 1e-3 | Initial learning rate |
