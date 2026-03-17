# ml_ecg_cvd_detection

Machine Learning for ECG-based Cardiovascular Disease Detection.

## Requirements

- Python >= 3.10
- See `requirements.txt` for full dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Data

This project uses [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) (Wagner et al., 2020) —
21,837 12-lead ECGs from 18,885 patients, annotated with 71 diagnostic SCP codes grouped into
5 superclasses: **NORM, MI, STTC, CD, HYP**.

### Download

```bash
make download-ptbxl
# or
python scripts/download_data.py --workers 8
```

Downloads ~3 GB to `data/raw/ptb-xl/` (not tracked by git). Re-running resumes any interrupted download.

### Loading the dataset

```python
from src.data.dataset import PTBXLDataset

# Any CVD vs. NORM (default)
dataset = PTBXLDataset("data/raw/ptb-xl")

# Single superclass
dataset = PTBXLDataset("data/raw/ptb-xl", positive_classes="MI")

# Custom subset of superclasses
dataset = PTBXLDataset("data/raw/ptb-xl", positive_classes=["MI", "STTC"])

X_train, y_train = dataset.get_split("train")  # shape: (N, 12, 5000)
X_val,   y_val   = dataset.get_split("val")
X_test,  y_test  = dataset.get_split("test")

print(dataset.class_distribution("train"))
```

Splits follow the official PTB-XL stratified folds: folds 1–8 = train, 9 = val, 10 = test.

### Preprocessing

The pipeline applied by default: bandpass filter (0.5–40 Hz) → notch filter (50 Hz) → z-score normalization per lead.
Configurable via `PTBXLDataset` constructor arguments (`bandpass`, `notch`, `normalization`).

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
