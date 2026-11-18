# ECG Features and Prediction Targets

## 1. ECG Feature Types

ECG-based models use either raw waveforms (end-to-end deep learning) or hand-crafted features (classical ML and interpretable approaches). The two paradigms are not mutually exclusive: hybrid models combine extracted features with neural representations.

### 1.1 Time-Domain / Statistical Features

Computed directly from amplitude values of each lead over the recording window.

| Feature | Description |
|---------|-------------|
| Mean amplitude | Per-lead mean signal value |
| Standard deviation | Per-lead amplitude variability |
| Min / Max / Range | Per-lead amplitude extremes |
| Root Mean Square (RMS) | Signal energy proxy |
| Skewness | Asymmetry of the amplitude distribution |
| Kurtosis | Peakedness / tail weight |
| Zero-crossing rate | Frequency of sign changes |

### 1.2 Heart Rate Variability (HRV) Features

Derived from the R–R interval series obtained after R-peak detection (e.g., Pan–Tompkins algorithm). HRV features reflect autonomic nervous system function and are biomarkers for cardiovascular risk.

**Time-domain HRV:**

| Feature | Description |
|---------|-------------|
| Mean RR (ms) | Mean inter-beat interval |
| SDNN (ms) | Standard deviation of RR intervals; overall HRV |
| RMSSD (ms) | Root mean square of successive RR differences; short-term HRV |
| pNN50 | Proportion of consecutive RR differences >50 ms |
| Mean HR (BPM) | Mean heart rate = 60,000 / mean RR |
| SD HR (BPM) | Standard deviation of instantaneous heart rate |

**Frequency-domain HRV:**

| Feature | Description |
|---------|-------------|
| VLF power (0–0.04 Hz) | Very low frequency power; thermoregulation / renin–angiotensin |
| LF power (0.04–0.15 Hz) | Low frequency power; sympathovagal balance |
| HF power (0.15–0.4 Hz) | High frequency power; respiratory sinus arrhythmia (parasympathetic) |
| LF/HF ratio | Sympathovagal balance index |
| Total power | Overall spectral power of RR interval series |

Machine learning models based on HRV metrics achieved approximately 81.4% accuracy in short-term cardiovascular event prediction (Jang et al., 2024).

### 1.3 Morphological Features

Derived from the detection and measurement of specific ECG waveform components.

| Feature | Description |
|---------|-------------|
| R-peak amplitude | Height of the QRS complex; reflects ventricular depolarisation |
| QRS duration (ms) | Width of the QRS complex; conduction velocity |
| PR interval (ms) | Time from P-wave onset to QRS onset; AV conduction |
| QT interval (ms) | From QRS onset to T-wave end; ventricular repolarisation |
| QTc (corrected) | QT interval corrected for heart rate (Bazett's formula) |
| ST elevation / depression | Displacement of the ST segment from baseline; ischaemia marker |
| T-wave amplitude / morphology | Repolarisation abnormalities |
| P-wave morphology | Atrial depolarisation; AF-related changes |
| Axis (degrees) | Mean electrical axis in the frontal plane |
| T-wave alternans | Beat-to-beat T-wave amplitude variation; SCD risk marker |

### 1.4 Frequency-Domain / Spectral Features

Computed via power spectral density (Welch's method or FFT) on the raw ECG waveform (distinct from HRV spectral features, which operate on the RR interval series).

| Feature | Description |
|---------|-------------|
| Power in low band (0.5–5 Hz) | Low-frequency signal energy (baseline, P/T waves) |
| Power in mid band (5–15 Hz) | Mid-frequency energy (P-wave, T-wave) |
| Power in high band (15–40 Hz) | High-frequency energy (QRS dominant) |
| Dominant frequency (Hz) | Frequency of maximum PSD; AF detection signature |
| Spectral entropy | Shannon entropy of normalised PSD; regularity measure |
| Total spectral power | Total signal energy in the analysed band |

### 1.5 Wavelet Features

Discrete wavelet transform (DWT) coefficients at multiple decomposition levels capture time–frequency characteristics. Commonly used in classical ML pipelines for ECG denoising and feature extraction.

**Koopman Operator Features (2026):** The Koopman operator provides a linear dynamical system representation of non-linear ECG signals. Combined with wavelet coefficients, Koopman-derived features fed as tokens to a Transformer encoder have shown improved ECG classification performance versus raw signal tokens — suggesting that physically motivated decompositions remain valuable even in deep learning pipelines.

> "ECG Classification with Transformers Using Koopman and Wavelet Features." arXiv:2603.08339 (2026). URL: [arxiv.org/abs/2603.08339](https://arxiv.org/abs/2603.08339) [UNVERIFIED — verify authors before citing]

### 1.6 2D Image Representations

An alternative to 1D feature extraction is to represent the ECG as a 2D image for input to standard computer vision models:

| Representation | Description |
|----------------|-------------|
| Spectrogram | Short-time Fourier transform magnitude; time on x-axis, frequency on y-axis |
| Scalogram | Continuous wavelet transform magnitude; multi-resolution time-frequency map |
| Synthesised paper strip | Rendered grid plot mimicking clinical paper ECG format |
| Printed/scanned photograph | Direct image of a physical ECG printout |

**When to use:** Image representations are advantageous when (1) source ECGs are only available as paper documents, (2) leveraging ImageNet-pretrained vision backbones for transfer learning, or (3) multi-lead spatial relationships are encoded visually (e.g., 12-lead grid layout).

**Key result (2025):** BHF ECG Image Challenge 2025 (ConvNeXt-V2-Base + ensemble, AUC 0.9677) demonstrated that image-based approaches match 1D signal baselines when applied to UK Biobank printed ECG photographs.

---

## 2. Non-ECG Clinical Variables

When combined with ECG waveform data, structured clinical variables consistently improve predictive performance for CVD risk assessment:

| Category | Variables |
|----------|-----------|
| **Demographics** | Age, sex/gender, race/ethnicity, BMI |
| **Blood pressure** | Systolic BP, diastolic BP, pulse pressure |
| **Lipid panel** | Total cholesterol, LDL-C, HDL-C, triglycerides |
| **Metabolic** | Fasting glucose, HbA1c, diabetes status (type 1/2) |
| **Lifestyle** | Smoking status (current/former/never), pack-years, alcohol consumption, physical activity level |
| **Medical history** | Prior MI, prior stroke/TIA, hypertension diagnosis, heart failure, family history of premature CVD |
| **Medications** | Antihypertensive drugs, statins, anticoagulants, beta-blockers |
| **Biomarkers** | hs-CRP, BNP/NT-proBNP (heart failure), hs-troponin, creatinine/eGFR |
| **Imaging-derived** | Left ventricular ejection fraction (LVEF), left atrial size, coronary artery calcium score |

---

## 3. Prediction Targets

### 3.1 Diagnostic Classification (Current Cardiac State)

**Multi-label (most common):** Each record may have multiple simultaneous conditions. Used in PTB-XL, CinC 2020/2021, Chapman-Shaoxing, and most modern DL approaches.

**PTB-XL diagnostic superclasses (canonical binary/multi-class targets):**

| Superclass | Code | Description | PTB-XL records |
|-----------|------|-------------|----------------|
| Normal | NORM | Normal ECG | 9,514 |
| Myocardial Infarction | MI | Acute or old MI | 5,469 |
| ST/T Change | STTC | Ischaemia, early repolarisation, strain | 5,235 |
| Conduction Disturbance | CD | Bundle branch block, AV block | 4,898 |
| Hypertrophy | HYP | LVH, RVH | 2,649 |

**Common specific targets in the literature:**

| Condition | Framing | Key datasets used |
|-----------|---------|-------------------|
| Atrial fibrillation (AF) | Binary / multi-class | CinC 2017, CODE-15%, MIMIC-IV-ECG |
| Myocardial infarction | Binary | PTB-XL, PTB (original), CODE-15% |
| Left bundle branch block (LBBB) | Binary | PTB-XL, CODE-15% |
| Right bundle branch block (RBBB) | Binary | PTB-XL, CODE-15%, CPSC 2018 |
| AV block (1st/2nd/3rd degree) | Multi-class | PTB-XL, CPSC 2018 |
| ST depression / elevation | Binary | PTB-XL, CPSC 2018 |
| Sinus bradycardia / tachycardia | Binary | CODE-15%, Chapman-Shaoxing |
| Premature contractions (PAC, PVC) | Binary | CPSC 2018 |
| Low ejection fraction (EF ≤35%) | Binary | Attia et al. (2019); BIDMC/Mayo datasets |
| Left ventricular hypertrophy | Binary | UK Biobank ECG data |

### 3.2 Low Ejection Fraction — A Key Landmark Target

Attia et al. (2019) demonstrated that a CNN trained on 44,959 paired ECGs and echocardiograms at Mayo Clinic can detect LVEF ≤35% from ECG alone with **AUC = 0.93**, sensitivity 86.3%, specificity 85.7%. This model received FDA Breakthrough Device designation and established ECG as a low-cost screening tool for occult cardiomyopathy.

> Attia ZI, Kapa S, Lopez-Jimenez F, et al. "Screening for cardiac contractile dysfunction using an artificial intelligence-enabled electrocardiogram." *Nature Medicine* 25, 70–74 (2019). DOI: [10.1038/s41591-018-0240-2](https://doi.org/10.1038/s41591-018-0240-2)

### 3.3 Long-Term Risk Prediction (Emerging Paradigm)

A growing body of work frames ECG-based prediction as a **survival analysis** problem — predicting *when* a future adverse event will occur, rather than classifying the current state.

**SEER (Stanford Estimator of Electrocardiogram Risk):**
- Predicts 5-year cardiovascular mortality from a single resting 12-lead ECG.
- Trained on 311,334 ECGs (Stanford); validated at Cedars-Sinai and Columbia.
- AUC for 5-year cardiovascular mortality: 0.83 (Stanford), 0.78 (Cedars-Sinai), 0.83 (Columbia).
- Also predicts heart failure, AF, cardiomyopathy, aortic stenosis, ASCVD.
- Reclassified 16% of PCE-low-risk patients as moderate risk.

> Hughes JW, et al. "A deep learning-based electrocardiogram risk score for long term cardiovascular death and disease." *npj Digital Medicine* 6, 169 (2023). DOI: [10.1038/s41746-023-00916-6](https://doi.org/10.1038/s41746-023-00916-6)

**AIRE (AI-ECG Risk Estimator, Lancet Digital Health):**
- Discrete-time survival model generating patient-specific survival curves from a single ECG.
- Trained on 1,163,401 ECGs from 189,539 patients (BIDMC).
- C-index for all-cause mortality: 0.775 (95% CI 0.773–0.776).
- Validated across 5 transnational cohorts (USA, Brazil, UK).
- Also predicts ventricular arrhythmia, ASCVD, and heart failure.

> Radhakrishnan A, et al. "Artificial intelligence-enabled electrocardiogram for mortality and cardiovascular risk estimation: a model development and validation study." *The Lancet Digital Health* 6(11), e791–e802 (2024). DOI: [10.1016/S2589-7500(24)00172-9](https://doi.org/10.1016/S2589-7500(24)00172-9)

### 3.4 Traditional Risk Scores (Non-ECG, Reference Benchmarks)

Traditional CVD risk scores are derived from clinical variables and serve as comparators for ECG-based models:

| Score | Variables | Outcome | Reference |
|-------|-----------|---------|-----------|
| **Framingham Risk Score** | Age, sex, total/HDL cholesterol, SBP, smoking, diabetes | 10-year CHD risk | Wilson et al., 1998 |
| **SCORE2** | Age, sex, smoking, SBP, total cholesterol, HDL-C | 10-year first CVD event (Europe, ages 40–69) | SCORE2 Working Group, *EHJ* 42(25), 2021 |
| **PCE (Pooled Cohort Equations)** | Age, race, sex, cholesterol, BP, smoking, diabetes | 10-year ASCVD risk (USA) | Goff et al., JACC, 2014 |

SEER outperformed the traditional PCE score for long-term risk stratification in the Stanford validation cohort.
