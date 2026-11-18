# Classical Machine Learning Methods for ECG-Based CVD Prediction

## 1. Overview

Classical ML approaches for ECG-based CVD prediction require a two-stage pipeline:

1. **Feature extraction** — transform raw ECG waveforms into a fixed-length feature vector using signal processing algorithms (see [02_features_and_targets.md](02_features_and_targets.md)).
2. **Classification** — train a supervised model on the extracted features.

This paradigm is interpretable, computationally lightweight, and competitive on well-defined single-condition tasks. However, it is outperformed by deep learning on multi-label diagnostic benchmarks "by a large margin" (Strodthoff et al., 2021).

---

## 2. Feature Engineering for ECG

The quality of classical ML results depends critically on feature engineering. The standard feature set for ECG classification combines:

### 2.1 HRV-Based Features (from R–R Intervals)
R-peak detection (Pan–Tompkins algorithm or similar) followed by computation of time-domain and frequency-domain HRV statistics (see [02_features_and_targets.md §1.2](02_features_and_targets.md)). HRV features capture autonomic modulation and are strong predictors of mortality and arrhythmia risk.

### 2.2 Morphological Features
QRS duration, PR/QT intervals, ST deviation, T-wave morphology, and axis measurements — extracted via delineation algorithms that identify waveform fiducial points.

### 2.3 Spectral Features
Power spectral density in ECG-relevant frequency bands (0.5–5 Hz, 5–15 Hz, 15–40 Hz), dominant frequency, spectral entropy. Useful for AF detection (irregular dominant frequency).

### 2.4 Statistical Features
Per-lead amplitude statistics (mean, SD, min, max, RMS, skewness, kurtosis, range) computed directly from the signal. Simple but effective as a baseline feature set.

### 2.5 Wavelet Features
Discrete wavelet transform (DWT) coefficients at multiple decomposition levels. Used for simultaneous time–frequency analysis and ECG denoising before feature extraction.

---

## 3. Classification Models

### 3.1 Logistic Regression

- **Role:** Interpretable linear baseline; useful for identifying which features drive predictions.
- **Performance:** AUC ~0.72 on Framingham-style clinical features; on ECG features typically lower than tree-based methods.
- **Strengths:** Calibrated probabilities; supports L1/L2 regularisation for feature selection.
- **Limitations:** Cannot capture non-linear interactions between ECG features.

### 3.2 Support Vector Machine (SVM)

- **Variants used:** RBF kernel SVM; linear SVM.
- **Performance:** Typical accuracy ~70% on ECG classification tasks — lower than ensemble methods.
- **Notable combination:** SVM with autoencoders for unsupervised ECG feature learning, followed by SVM classification. This approach extracts latent representations before classification.
- **Limitations:** Does not scale well to very large datasets; hyperparameter tuning is expensive.

### 3.3 Random Forest

- **Performance:** Consistently achieves 84–95% accuracy across ECG classification studies. Identified as one of the top performers among classical methods.
- **Strengths:** Built-in feature importance; robust to irrelevant features; handles class imbalance via `class_weight='balanced_subsample'`.
- **Limitations:** Slower inference than linear models; can overfit on small datasets.

### 3.4 XGBoost / Gradient Boosting

- **Performance:** XGBoost achieved 96.6% accuracy after hyperparameter tuning in one comparative study, surpassing Random Forest.
- **Strengths:** State-of-the-art on tabular feature data; built-in SHAP feature importance; handles missing values.
- **Limitations:** Requires careful tuning of learning rate, depth, and regularisation; risk of overfitting.

### 3.5 Stacking Ensembles

Combining SVM, RF, and XGBoost in a meta-learner framework has been shown to improve calibration and reduce false positives for cardiovascular risk prediction compared to individual models.

---

## 4. Comparative Performance

The following summarises performance ranges reported across classical ML studies on ECG-based CVD tasks:

| Model | Task | Accuracy | Notes |
|-------|------|----------|-------|
| Logistic Regression | Framingham CVD risk | AUC ~0.72 | Clinical features only |
| SVM | Arrhythmia classification | ~70% | ECG features |
| Random Forest | CVD classification (various) | 84–95% | ECG features |
| XGBoost | CVD classification | Up to 96.6% | After tuning |
| Stacking ensemble | CVD risk prediction | > RF alone | Improved calibration |

**On PTB-XL (multi-label diagnostic task):** Strodthoff et al. (2021) report that the best classical baseline (Wavelet + NN) achieves macro AUC of 0.849, compared to 0.925 for the best deep learning architecture. Deep learning outperforms classical ML "by a large margin."

**2026 update:** A comprehensive evaluation on MIT-BIH (Yun Song et al., 2026) confirmed that CNN-LSTM hybrids now achieve F1 = 0.951 — the best single-architecture result on that benchmark — further widening the gap with classical SVM and tree-based approaches (SVM ~0.65 F1, RF ~0.84–0.90 F1). This reinforces that the classical ML ceiling on well-studied benchmarks has been reached.

> Song Y, et al. "Deep Neural Network Architectures for ECG Classification: A Comprehensive Evaluation." *arXiv:2602.17701* (2026). URL: [arxiv.org/abs/2602.17701](https://arxiv.org/abs/2602.17701) [UNVERIFIED — verify authors and DOI before citing]

---

## 5. Limitations of Classical ML for ECG

1. **Manual feature engineering required:** Features must be explicitly designed by domain experts. Subtle or unexpected patterns may be missed.
2. **Performance ceiling on multi-lead/multi-label tasks:** Classical features cannot easily capture the spatiotemporal relationships across 12 leads simultaneously.
3. **R-peak detection dependency:** Most HRV and morphological features require accurate R-peak detection as a prerequisite; errors propagate downstream.
4. **Poor generalisation across datasets:** Hand-crafted feature pipelines are sensitive to differences in sampling rates, lead configurations, and signal quality.
5. **Multi-label limitations:** Classical classifiers must be wrapped in one-vs-rest or label-powerset schemes for multi-label tasks, which scale poorly with the number of conditions.

---

## 6. When Classical ML Remains Appropriate

Classical ML still has a role in:
- **Interpretable screening tools** where feature-level explanations are required (e.g., clinician-facing risk reports).
- **Resource-constrained environments** where neural network inference is infeasible (edge devices, embedded systems).
- **Small datasets** where DL would overfit (rare conditions, limited data collection budgets).
- **Feature importance analysis** to identify which ECG characteristics drive predictions.
- **Baselines and ablation studies** in research papers to quantify the added value of deep learning.

---

## References

- Strodthoff N, Wagner P, Schaeffter T, Samek W. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE JBHI* 25(5), 1519–1528 (2021). DOI: [10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989)
- Jang J, et al. "Predicting the Risk of Cardiovascular Diseases in the Elderly Based on Clinical Data and Heart Rate Variability Using Machine Learning." *Journal of Clinical Medicine* 15(13), 5141 (2024). DOI: [10.3390/jcm15135141](https://doi.org/10.3390/jcm15135141)
- "Cardiovascular Disease (CVD) Prediction Using Machine Learning Techniques with XGBoost Feature Importance Analysis." ResearchGate, 2024. URL: [researchgate.net/publication/380296956](https://www.researchgate.net/publication/380296956)
