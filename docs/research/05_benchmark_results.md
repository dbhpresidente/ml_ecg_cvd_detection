# Benchmark Results: ECG-Based CVD Classification

## 1. PTB-XL Diagnostic Superclass Benchmark

The PTB-XL benchmark (Strodthoff et al., 2021) is the standard reference for comparing ECG classification architectures. Models are trained on folds 1–8, validated on fold 9, and tested on fold 10 (patient-independent split). Performance is reported as macro-averaged AUC across the five diagnostic superclasses (NORM, MI, STTC, CD, HYP).

### 1.1 Supervised Baselines

| Architecture | Macro AUC | Notes |
|---|---|---|
| **xresnet1d101** | **0.937** | Best single supervised model |
| inception1d | 0.931 | Inception-style 1D CNN |
| resnet1d_wang | 0.919 | Shallow ResNet baseline |
| xresnet1d50 | 0.936 | Compressed xResNet |
| lstm (2-layer) | 0.907 | LSTM baseline |
| lstm_bidir | 0.913 | Bidirectional LSTM |
| Wavelet + NN | 0.849 | Best classical ML baseline |
| **Ensemble (top-5)** | **0.939** | Model averaging |

> Source: Strodthoff N, et al. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE JBHI* 25(5), 1519–1528 (2021). DOI: [10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989)

**Key takeaway:** Deep learning outperforms the best classical ML baseline (Wavelet + NN, AUC 0.849) by 0.088 AUC points. Ensembles add ~0.002 over the best single model. ResNet variants consistently outperform LSTM-based models.

### 1.2 Self-Supervised Methods on PTB-XL

| Method | Pre-training Data | Fine-tune Labels | Macro AUC | Notes |
|---|---|---|---|---|
| ST-MEM (ViT-Base) | PTB-XL (self-supervised) | 100% labels | 0.933 | ICLR 2024 |
| ST-MEM | PTB-XL | 10% labels | 0.916 | 5% label efficiency vs full supervision |
| CLOCS | PTB-XL | 100% labels | ~0.920 | Contrastive learning |
| 3KG | PTB-XL | 100% labels | ~0.918 | Spatial-temporal augmentations |
| Supervised xresnet1d101 | — | 100% labels | 0.937 | Reference |

> ST-MEM: Reed CJ, et al. "ST-MEM: Masked ECG Modeling for Self-Supervised Cardiac Representation Learning." *ICLR* (2024). DOI: [10.48550/arXiv.2305.17295](https://doi.org/10.48550/arXiv.2305.17295)

> CLOCS: Kiyasseh D, et al. "CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients." *ICML* (2021). DOI: [10.48550/arXiv.2005.13249](https://doi.org/10.48550/arXiv.2005.13249)

**Key takeaway:** Self-supervised methods reach within 0.004 AUC of the supervised state-of-the-art while requiring dramatically fewer labels. ST-MEM with 10% labels (≈1,760 records) matches or beats most fully supervised baselines.

### 1.3 Per-Superclass Results (xresnet1d101)

| Superclass | AUC | Prevalence |
|---|---|---|
| NORM | 0.952 | 9,514 / 21,799 (43.6%) |
| MI | 0.934 | 5,469 / 21,799 (25.1%) |
| STTC | 0.918 | 5,235 / 21,799 (24.0%) |
| CD | 0.939 | 4,898 / 21,799 (22.5%) |
| HYP | 0.892 | 2,649 / 21,799 (12.1%) |

HYP (Hypertrophy) consistently achieves the lowest AUC, likely due to its lower prevalence and subtler waveform morphology.

---

## 2. PhysioNet/CinC Challenge Results

### 2.1 PhysioNet 2017 — AF Classification (Single Lead)

**Task:** 4-class (Normal / AF / Other rhythm / Noisy), scored by F1 on the held-out test set.

| Rank | Team / Method | F1 (test) | Architecture |
|---|---|---|---|
| 1 | Zihlmann et al. | **0.831** | CNN + GRU |
| 2 | Teijeiro et al. | 0.826 | Hierarchical classifier |
| 3 | Datta et al. | 0.820 | CNN |
| — | Hannun et al. (DNN) | 0.830 | 34-layer CNN (independent publication) |

> Clifford GD, et al. "AF Classification from a Short Single Lead ECG Recording: the PhysioNet/CinC Challenge 2017." *Computers in Cardiology* 44 (2017). DOI: [10.22489/CinC.2017.065-469](https://doi.org/10.22489/CinC.2017.065-469)

> Hannun AY, et al. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." *Nature Medicine* 25, 65–69 (2019). DOI: [10.1038/s41591-018-0268-3](https://doi.org/10.1038/s41591-018-0268-3)

### 2.2 PhysioNet 2020 — 12-Lead ECG Classification

**Task:** 27 scored diagnoses (SNOMED-CT), multi-label. Custom weighted accuracy metric (partial credit for clinically related misclassifications).

| Rank | Team | Weighted Score | Architecture |
|---|---|---|---|
| 1 | prna | **0.533** | Wide and Deep Transformer |
| 2 | DSAIL_SNU | 0.520 | Ensemble (CNN + LSTM) |
| 3 | CeZIS | 0.516 | CNN-based |
| 4 | BRACE | 0.515 | Ensemble |
| 5 | HeartBEAT | 0.512 | LSTM + attention |

> Perez Alday EA, et al. "Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020." *Physiological Measurement* 41(12), 124003 (2021). DOI: [10.1088/1361-6579/abc960](https://doi.org/10.1088/1361-6579/abc960)

**Notable observations:**
- All top-10 teams used neural networks; none used classical ML exclusively.
- Ensembles consistently scored higher than single models.
- The gap between rank 1 (0.533) and rank 10 (≈0.490) was modest, suggesting diminishing returns from architectural improvements under the custom metric.

### 2.3 PhysioNet 2021 — Reduced-Lead ECG Classification

**Task:** Same 30 conditions as 2020 but across 12-, 6-, 4-, 3-, and 2-lead configurations. Winning approach required handling all lead sets.

| Rank | Team | 12-lead Score | 2-lead Score |
|---|---|---|---|
| 1 | ISIBrno-AIMT | **0.604** | **0.558** |
| 2 | CeZIS | 0.587 | 0.542 |
| 3 | DSAIL_SNU | 0.579 | 0.531 |

> Reyna MA, et al. "Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/CinC Challenge 2021." *Computers in Cardiology* (2021). DOI: [10.23919/CinC53138.2021.9662687](https://doi.org/10.23919/CinC53138.2021.9662687)

**Key finding:** 2-lead performance (0.558) was 92.4% of 12-lead performance (0.604), confirming the feasibility of wearable deployment with reduced lead counts.

---

## 3. CPSC 2018 Results

**Task:** 9-class rhythm classification from 12-lead ECG.

| Method | Median F1 | Notes |
|---|---|---|
| Chen et al. CNN-BiGRU-Attention | **0.84** | Challenge winner |
| Multi-label CNN | 0.81 | Ensemble |
| SVM baseline | ~0.65 | Feature-based |

> Chen TM, Huang CH, Shih ESC, Hwang MJ. "Detection and Classification of Cardiac Arrhythmias by a Challenge-Best Deep Learning Neural Network Model." *iScience* 23(3), 100886 (2020). DOI: [10.1016/j.isci.2020.100886](https://doi.org/10.1016/j.isci.2020.100886)

---

## 4. Specific Condition Results

### 4.1 Atrial Fibrillation Detection

| Study | Dataset | Model | AUC / F1 |
|---|---|---|---|
| Hannun et al. (2019) | Holter (1-lead, Stanford) | 34-layer CNN | F1 0.830 (AF class) |
| Ribeiro et al. (2020) | CODE-15% | ResNet (6-layer) | F1 0.983 (AF) |
| Schwab et al. | CinC 2017 | LSTM + attention | AUC 0.978 |

### 4.2 Left Ventricular Dysfunction (EF ≤35%)

| Study | Dataset | Model | AUC |
|---|---|---|---|
| Attia et al. (2019) | Mayo Clinic (n=44,959) | CNN (I-EHR) | 0.93 |
| Attia et al. (2021) | Mayo Clinic prospective | CNN | Sensitivity 85.7% |
| Yao et al. (2021) | BIDMC | Transformer | 0.91 |

> Attia ZI, et al. "Screening for cardiac contractile dysfunction using an artificial intelligence-enabled electrocardiogram." *Nature Medicine* 25, 70–74 (2019). DOI: [10.1038/s41591-018-0240-2](https://doi.org/10.1038/s41591-018-0240-2)

### 4.3 Long-Term Mortality Prediction

| System | Dataset | Model | C-index / AUC |
|---|---|---|---|
| SEER | Stanford (n=311,334) | ResNet survival | AUC 0.83 (5-yr CVD mort.) |
| AIRE | BIDMC (n=1,163,401) | Discrete-time survival | C-index 0.775 |
| Raghunath et al. (2020) | Geisinger (n=253,397) | ResNet | AUC 0.85 (1-yr mortality) |

> Hughes JW, et al. "A deep learning-based electrocardiogram risk score for long term cardiovascular death and disease." *npj Digital Medicine* 6, 169 (2023). DOI: [10.1038/s41746-023-00916-6](https://doi.org/10.1038/s41746-023-00916-6)

> Radhakrishnan A, et al. "Artificial intelligence-enabled electrocardiogram for mortality and cardiovascular risk estimation." *The Lancet Digital Health* 6(11), e791–e802 (2024). DOI: [10.1016/S2589-7500(24)00172-9](https://doi.org/10.1016/S2589-7500(24)00172-9)

> Raghunath S, et al. "Prediction of mortality from 12-lead electrocardiogram voltage data using a deep neural network." *Nature Medicine* 26, 886–891 (2020). DOI: [10.1038/s41591-020-0870-z](https://doi.org/10.1038/s41591-020-0870-z)

---

## 5. Foundation Model Benchmarks

Foundation models pre-trained on millions of ECGs are increasingly evaluated on PTB-XL and clinical tasks. Results from a systematic arXiv preprint comparing ECG foundation models (2025):

| Model | Pre-training data | PTB-XL macro AUC | Parameters |
|---|---|---|---|
| ECGFounder | 10M+ ECGs (private) | ~0.930 | >100M |
| ECG-FM | 1.6M ECGs (MIMIC-IV-ECG) | ~0.925 | 90.9M |
| HuBERT-ECG | 9.1M ECGs (CODE dataset) | ~0.928 | — |
| Supervised xresnet1d101 | PTB-XL only | 0.937 | ~5M |

> Note: Foundation model results on PTB-XL vary by fine-tuning protocol and are not always directly comparable to the Strodthoff benchmark due to differences in preprocessing. See individual papers for exact experimental conditions.

> Moor M, et al. "Foundation models for generalist medical artificial intelligence." *Nature* 616, 259–265 (2023). DOI: [10.1038/s41586-023-05881-4](https://doi.org/10.1038/s41586-023-05881-4)

---

## 6. Evaluation Methodology Notes

### 6.1 Class Imbalance
PTB-XL is moderately imbalanced (HYP: 12.1% prevalence). Best practices include:
- Reporting macro-averaged AUC (treats all classes equally).
- Stratified k-fold splitting by patient ID (not record ID) to prevent data leakage.
- Using AUPRC (area under precision-recall curve) as an additional metric for rare classes.

### 6.2 Patient-Independent Splits
Using patient-level splits (as in PTB-XL folds 1–10) is mandatory. Record-level splits can artificially inflate performance by leaking the same patient's ECGs into train and test.

### 6.3 Calibration
Several studies report that neural networks on ECG data are over-confident (high confidence for wrong predictions). Platt scaling and temperature calibration are rarely applied but recommended before clinical use.

### 6.4 Metric Selection
- **Multi-label:** Macro-averaged AUC is standard (treats each label independently).
- **Challenge metrics:** PhysioNet 2020/2021 use a custom weighted accuracy that awards partial credit for clinically similar confusions — this is not interchangeable with macro AUC.
- **Binary (single condition):** AUC, sensitivity, specificity, and F1 are all commonly reported. Sensitivity/specificity at a clinical operating point (e.g., sensitivity ≥85%) is most clinically relevant.
