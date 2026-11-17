# State-of-the-Art Survey: ECG-Based Cardiovascular Disease Prediction

**Last updated:** November 2026  
**Scope:** Machine learning and deep learning for CVD detection and risk prediction from 12-lead and single-lead electrocardiograms  
**Intended use:** Background research for thesis and journal article writing. All cited sources have been verified with DOI or URL by literature search agents.

---

## Table of Contents

| File | Topic |
|------|-------|
| [01_datasets.md](01_datasets.md) | Public ECG datasets and clinical cohorts |
| [02_features_and_targets.md](02_features_and_targets.md) | Input features and prediction targets |
| [03_methods_classical_ml.md](03_methods_classical_ml.md) | Classical ML methods (LR, SVM, RF, XGBoost) |
| [04_methods_deep_learning.md](04_methods_deep_learning.md) | Deep learning methods (CNN, RNN, Transformer, SSL, foundation models) |
| [05_benchmark_results.md](05_benchmark_results.md) | Benchmark results on PTB-XL, PhysioNet challenges, CPSC |
| [06_gaps_and_opportunities.md](06_gaps_and_opportunities.md) | Research gaps, open problems, clinical translation |
| [references.md](references.md) | Full bibliography with DOIs (50+ verified sources) |

---

## Executive Summary

### Datasets
The field has converged on a small set of large public benchmarks. **PTB-XL** (Wagner et al., 2020; 21,799 records, 500 Hz, 12-lead, 71 SCP-ECG labels) is the primary benchmark for multi-label diagnostic classification. The **PhysioNet/CinC Challenges** (2017, 2020, 2021) introduced standardised multi-label evaluation across tens of thousands of recordings aggregated from multiple international sources. **CODE-15%** (345,779 records; Ribeiro et al., 2020) and **MIMIC-IV-ECG** (~800,000 records; Gow et al., 2023) are the largest publicly available corpora. All major datasets use 12-lead, 500 Hz, 10-second recordings. See [01_datasets.md](01_datasets.md).

### Features and Targets
ECG-based models use either **raw waveforms** (end-to-end DL) or **hand-crafted features** (time-domain statistics, HRV indices, frequency-band power, morphological measurements). The dominant prediction framing is **multi-label classification** of concurrent cardiac conditions; binary (CVD vs. NORM) and long-term risk prediction are also common. The five PTB-XL superclasses — NORM, MI, STTC, CD, HYP — serve as the canonical binary/multi-class target space. See [02_features_and_targets.md](02_features_and_targets.md).

### Classical ML
Random forest and XGBoost on hand-crafted ECG features achieve 84–96% accuracy on individual condition classification tasks but are outperformed by deep learning on PTB-XL's multi-label benchmark "by a large margin" (Strodthoff et al., 2021). Feature selection and interpretability remain advantages of classical approaches. See [03_methods_classical_ml.md](03_methods_classical_ml.md).

### Deep Learning
1D ResNets and Inception architectures are the state-of-the-art supervised baseline on PTB-XL (macro AUC 0.925–0.937). Self-supervised methods (CLOCS, ST-MEM) match or exceed supervised performance with 5–20× fewer labels. Foundation models (ECGFounder, ECG-FM, HuBERT-ECG) trained on millions of ECGs generalise across tasks with minimal fine-tuning. Transformer-based models show strong results on long-range temporal tasks but do not consistently outperform ResNets on standard benchmarks. See [04_methods_deep_learning.md](04_methods_deep_learning.md).

### Benchmarks
Best published results on PTB-XL diagnostic superclass: macro AUC **0.937** (xresnet1d101, Strodthoff et al., 2021); **0.933** with self-supervised ST-MEM (ICLR 2024). PhysioNet 2017 top F1: 0.831. PhysioNet 2020 top weighted score: 0.533 (Wide and Deep Transformer). All top-10 teams in PhysioNet 2020 used CNNs; ensembles consistently add 0.5–1% AUC. See [05_benchmark_results.md](05_benchmark_results.md).

### Gaps and Opportunities
Major open problems: (1) **cross-dataset generalisation** (domain shift from hardware/population differences); (2) **demographic fairness** (AUC gaps of >0.14 between young and elderly patients); (3) **explainability** (Grad-CAM + SHAP used in <52% of studies, no consensus standard); (4) **prospective clinical validation** (only 11 RCTs identified by 2023); (5) **long-term risk prediction** (SEER, AIRE are promising but underexplored); (6) **edge deployment** for wearable continuous monitoring. See [06_gaps_and_opportunities.md](06_gaps_and_opportunities.md).

---

## Citation Note

> All sources in this survey were retrieved and verified by automated literature search (November 2026). DOIs are provided as clickable links. Items marked **[UNVERIFIED]** could not be fully confirmed and should be independently verified before citing in a manuscript.
