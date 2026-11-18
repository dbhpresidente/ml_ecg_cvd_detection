# Deep Learning Methods for ECG-Based CVD Prediction

## 1. Seminal 1D CNN Architectures

### 1.1 Rajpurkar et al. (2017) — Cardiologist-Level Arrhythmia Detection

The first paper to demonstrate that a DNN could match or exceed cardiologist performance on ECG arrhythmia classification. Key contributions:

- **Architecture:** 34-layer 1D CNN with 16 residual blocks; convolutional filters of length 16; 64k filters (k increments every 4th residual block); subsampling by factor 2 in every alternate block.
- **Input:** Single-lead 30-second ECG segments (30-second recordings).
- **Output:** 14 rhythm classes (12 arrhythmias + sinus rhythm + noise).
- **Dataset:** >500× more unique patients than previously studied corpora (proprietary; donated by iRhythm/Stanford).
- **Result:** Exceeded the average F1 score of 6 board-certified cardiologists on a held-out test set.

> Rajpurkar P, Hannun AY, Haghpanahi M, Bourn C, Ng AY. "Cardiologist-Level Arrhythmia Detection with Convolutional Neural Networks." *arXiv:1707.01836* (2017). URL: [arxiv.org/abs/1707.01836](https://arxiv.org/abs/1707.01836)

### 1.2 Hannun et al. (2019) — Nature Medicine

Peer-reviewed extension of Rajpurkar et al., published in *Nature Medicine*:

- **Dataset:** 91,232 single-lead ECGs from 53,549 patients (iRhythm Zio patch monitor).
- **Architecture:** Same 34-layer residual CNN.
- **Output:** 12 rhythm classes.
- **Result:** Average F1 of **0.837** vs. 0.780 for average cardiologist. Exceeded average cardiologist sensitivity for all 12 rhythm classes when specificity was fixed at average cardiologist level.

> Hannun AY, Rajpurkar P, Haghpanahi M, et al. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." *Nature Medicine* 25, 65–69 (2019). DOI: [10.1038/s41591-018-0268-3](https://doi.org/10.1038/s41591-018-0268-3)

### 1.3 Ribeiro et al. (2020) — Nature Communications (CODE Study)

12-lead residual CNN trained on the largest annotated ECG dataset at the time:

- **Dataset:** >2 million 12-lead ECGs (CODE study, Telehealth Network of Minas Gerais, Brazil).
- **Architecture:** ResNet adapted to 1D 12-lead signals with global average pooling.
- **Output:** 6 binary abnormality flags (1dAVb, RBBB, LBBB, sinus bradycardia, AF, sinus tachycardia).
- **Result:** F1 scores 0.870–1.000; specificity >0.995 for all conditions. Outperformed cardiology residents (Fleiss kappa 0.646 DNN vs. 0.565 residents).

> Ribeiro AH, Ribeiro MH, Paixão GMM, et al. "Automatic diagnosis of the 12-lead ECG using a deep neural network." *Nature Communications* 11, 1760 (2020). DOI: [10.1038/s41467-020-15432-4](https://doi.org/10.1038/s41467-020-15432-4)

---

## 2. ResNet and Inception Architectures on PTB-XL

The Strodthoff et al. (2021) benchmark established the state of the art for supervised DL on PTB-XL. Best-performing architectures:

| Architecture | Description | Diagnostic AUC | Rhythm AUC |
|-------------|-------------|---------------|------------|
| **xresnet1d101** | 1D adaptation of XResNet-101 | **0.937** | 0.957 |
| **resnet1d_wang** | 1D ResNet (Wang et al.) | 0.936 | 0.946 |
| **inception1d** | 1D InceptionTime | 0.931 | 0.953 |
| lstm | Stacked LSTM | 0.927 | 0.953 |
| Wavelet + NN (baseline) | Hand-crafted + shallow NN | 0.855 | 0.890 |
| **Ensemble** | xresnet + resnet + inception | **0.939** | **0.965** |

Macro AUC on PTB-XL diagnostic superclass task (fold 10 test set, min. confidence 100%). Source: Strodthoff et al. (2021).

**InceptionTime (Fawaz et al., 2020):** Ensemble of five randomly initialised Inception networks using multi-scale convolutions with bottleneck layers. Achieves 0.925 AUC on PTB-XL (single model) and was used by several top-performing teams in PhysioNet 2020.

> Strodthoff N, Wagner P, Schaeffter T, Samek W. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE JBHI* 25(5), 1519–1528 (2021). DOI: [10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989)

> Fawaz HI, et al. "InceptionTime: Finding AlexNet for Time Series Classification." *Data Mining and Knowledge Discovery* 34, 1936–1962 (2020). URL: [arxiv.org/abs/1909.04939](https://arxiv.org/abs/1909.04939)

**2026 results — Song et al. (2026):** Comprehensive evaluation of CNN architectures on MIT-BIH. The best-performing configuration was a CNN-LSTM hybrid with macro F1 = **0.951**, outperforming standalone CNNs and LSTMs evaluated in the same study.

> Song Y, Zheng W, Chen T, Wang Z, Shi J, Chen Y. "Deep Neural Network Architectures for ECG Classification: A Comprehensive Evaluation." *arXiv:2602.17701* (2026). URL: [arxiv.org/abs/2602.17701](https://arxiv.org/abs/2602.17701) [UNVERIFIED — verify venue before citing]

---

## 3. RNN-Based Architectures

### 3.1 LSTM / BiLSTM / GRU

Recurrent networks capture sequential dependencies in ECG signals:

- **BiLSTM** provides the best standalone accuracy (~96%) among vanilla LSTM, GRU, and BiLSTM, processing ECG in both forward and backward temporal directions.
- **Hybrid BiLSTM-GRU:** 99.92% accuracy, 99.38% precision, 100% recall, 99.91% specificity reported on arrhythmia classification.
- **CNN-BiLSTM** hybrids: CNN handles local feature extraction; BiLSTM models temporal dependencies. Highest macro recall (87.37%) among hybrid architectures on multi-class arrhythmia tasks.
- **CNN-GRU for MI detection:** AUC close to 1.0, accuracy 99.43%, sensitivity 99.71%, specificity 98.59% on 15-lead configurations.

> Comparative analysis of LSTM variants: "Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023." *Frontiers in Physiology* (2023). URL: [pmc.ncbi.nlm.nih.gov/articles/PMC10542398/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10542398/)

### 3.2 Limitations of RNNs for ECG

- Slower training than CNNs due to sequential computation.
- Gradient vanishing/exploding on long sequences.
- On PTB-XL, stacked LSTM achieves 0.907 macro AUC (all tasks) vs. 0.925 for best CNN — CNNs dominate.

---

## 4. Transformer and Attention Architectures

### 4.1 General Trends (2022–2026)

Transformers have been applied to ECG via two main strategies:
1. **Treat ECG leads as sequence tokens** (patch-based, similar to Vision Transformer).
2. **Hybrid CNN-Transformer:** CNN encoder extracts local features; Transformer attends globally over the feature sequence.

A comprehensive survey documents the rapid advances: "A survey of transformers and large language models for ECG diagnosis: advances, challenges, and future directions." *Applied Intelligence* (2025). DOI: [10.1007/s10462-025-11259-x](https://link.springer.com/article/10.1007/s10462-025-11259-x)

### 4.2 Specific Models

**ECGformer (2024):** Leverages Transformer architectures for ECG heartbeat arrhythmia classification with refined diagonal attention and gated linear attention mechanisms.

> "ECGformer: Leveraging transformer for ECG heartbeat arrhythmia classification." *arXiv:2401.05434* (2024). URL: [arxiv.org/abs/2401.05434](https://arxiv.org/abs/2401.05434)

**CardioPatternFormer (2025):** Pattern-guided attention for interpretable ECG classification; designed to discern subtle anomalies and multiple co-occurring conditions.

> "CardioPatternFormer: Pattern-Guided Attention for Interpretable ECG Classification with Transformer Architecture." *arXiv:2505.20481* (2025). URL: [arxiv.org/abs/2505.20481](https://arxiv.org/abs/2505.20481)

**SPEED-TR (Morin et al., 2025):** Efficient Transformer architecture designed for clinical 12-lead ECG, reducing computational cost while maintaining strong diagnostic accuracy. Targets deployment in resource-constrained clinical settings.

> Morin J, et al. "SPEED-TR: Efficient Transformer for 12-Lead ECG." *Conf.* (2025). [UNVERIFIED — verify exact venue and DOI before citing]

**Kim et al. (2025) — Hybrid CNN + Transformer for ECG spectrograms:** Converts ECG signals to 2D spectrograms and applies a hybrid CNN-Transformer. Evaluated on MIT-BIH + Icentia11k, achieving high AUC and F1. Demonstrates viability of 2D image-based representations even for Transformer architectures.

> Kim S, et al. "Hybrid CNN + Transformer for ECG spectrograms." *Scientific Reports* (2025). DOI: [10.1038/s41598-025-92582-9](https://doi.org/10.1038/s41598-025-92582-9)

**Koopman + Wavelet Transformer (2026):** Feeds physically motivated features (Koopman operator decomposition + wavelet coefficients) as input tokens to a Transformer encoder, rather than raw samples. Improves classification performance and links to interpretable signal components.

> "ECG Classification with Transformers Using Koopman and Wavelet Features." *arXiv:2603.08339* (2026). URL: [arxiv.org/abs/2603.08339](https://arxiv.org/abs/2603.08339) [UNVERIFIED — verify venue before citing]

### 4.3 Comparison with CNNs

A meta-analysis of PhysioNet 2020 (Han et al., 2022) found:
- **100%** of top-10 teams used CNNs.
- **24.4%** incorporated attention mechanisms (statistically significant improvement, p=0.006).
- Transformer-based architectures show strong performance on long-range temporal tasks but do **not consistently outperform** ResNets on established benchmarks.
- Hybrid CNN-Transformer often achieves the best balance.
- State-space models (SSMs) like S4 and ECG-CPC achieve competitive AUC with 2–4M parameters vs. 87–97M for transformers.

> Han Y, et al. "Practical Lessons on 12-Lead ECG Classification: Meta-Analysis of Methods From PhysioNet/Computing in Cardiology Challenge 2020." *Frontiers in Physiology* (2022). URL: [pmc.ncbi.nlm.nih.gov/articles/PMC8795785/](https://pmc.ncbi.nlm.nih.gov/articles/PMC8795785/)

---

## 5. Self-Supervised and Contrastive Learning

A key limitation of supervised ECG DL is the cost of expert annotation. Self-supervised learning (SSL) reduces label dependency by learning representations from unlabelled ECGs.

### 5.1 CLOCS (ICML 2021)

Introduced ECG-specific contrastive learning with three complementary objectives:
- **CMSC** (Contrastive Multi-Segment Coding): temporal invariance — treat different segments of the same recording as positive pairs.
- **CMLC** (Contrastive Multi-Lead Coding): lead invariance — treat signals from different leads of the same patient as positive pairs.
- **CMSMLC:** Combined objective (both temporal and lead invariance).

Improves downstream performance in limited-label settings across multiple ECG datasets.

> Kiyasseh D, Zhu T, Clifton DA. "CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients." *Proceedings of ICML 2021*, PMLR pp. 5606–5615 (2021). URL: [github.com/danikiyasseh/CLOCS](https://github.com/danikiyasseh/CLOCS)

### 5.2 ST-MEM (ICLR 2024)

Adapted masked autoencoders to ECG by treating multi-lead signals as spatio-temporal patches with a ViT encoder.

- **Result on PTB-XL diagnostic:** AUROC **0.933** (vs. 0.905 supervised, vs. 0.736 supervised with 5% labels).
- With 5% of training data: AUROC 0.878 vs. 0.736 for supervised learning — 9× label efficiency improvement.
- Surpasses other SSL methods including MoCo v3.

> "Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram." *ICLR 2024*. URL: [arxiv.org/abs/2402.09450](https://arxiv.org/abs/2402.09450)

### 5.3 3KG (2021)

Contrastive learning using physiologically-inspired augmentations for 12-lead ECGs (cardiac geometry transformations).

> "3KG: Contrastive Learning of 12-Lead Electrocardiograms using Physiologically-Inspired Augmentations." *arXiv:2106.04452* (2021). URL: [arxiv.org/abs/2106.04452](https://arxiv.org/abs/2106.04452)

### 5.4 CREMA (2024)

Combined contrastive regularisation with masked autoencoders for robust ECG diagnostics across clinical domains.

> "CREMA: A Contrastive Regularized Masked Autoencoder for Robust ECG Diagnostics across Clinical Domains." *arXiv:2407.07110* (2024). URL: [arxiv.org/abs/2407.07110](https://arxiv.org/abs/2407.07110)

---

## 6. Foundation Models for ECG

Large ECG foundation models pre-trained on millions of recordings represent the frontier of transfer learning for cardiac analysis.

### 6.1 ECGFounder (2024)

- **Training data:** >10 million ECGs with 150 label categories (Harvard-Emory ECG Database).
- **Architecture:** Large CNN encoder.
- **Performance:** Internal validation AUROC >0.95 for 80 out of 150 diagnoses.
- **Capability:** Strong generalisation for demographics analysis, clinical event detection, and cross-modality cardiac rhythm diagnosis.

> Li J, Aguirre A, Moura J, et al. "An Electrocardiogram Foundation Model Built on over 10 Million Recordings with External Evaluation across Multiple Domains." *arXiv:2410.04133* (2024). URL: [arxiv.org/abs/2410.04133](https://arxiv.org/abs/2410.04133)

### 6.2 ECG-FM (JAMIA Open, 2025)

- **Architecture:** wav2vec 2.0 (multi-layer CNN feature extractor + BERT-like Transformer encoder); **90.9 million parameters**.
- **Pre-training:** 1.4 million ECG segments; hybrid contrastive + generative self-supervised objective.
- **Tasks:** Multi-label ECG interpretation; reduced LVEF prediction.
- **Strength:** Strong performance in low-label regimes; fully open source.

> "ECG-FM: An Open Electrocardiogram Foundation Model." *JAMIA Open* 8(5), ooaf122 (2025). DOI: [10.1093/jamiaopen/ooaf122](https://academic.oup.com/jamiaopen/article/8/5/ooaf122/8287827). Preprint: [arxiv.org/abs/2408.05178](https://arxiv.org/abs/2408.05178)

### 6.3 HuBERT-ECG (2024)

- **Training data:** 9.1 million 12-lead ECGs covering 164 cardiovascular conditions.
- **Variants:** SMALL (30M), BASE (93M), LARGE (188M parameters).
- **Tasks:** Diagnosis, survival analysis, prognosis.

> "HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications." *medRxiv* (2024). URL: [medrxiv.org/content/10.1101/2024.11.14.24317328v3.full](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v3.full)

### 6.4 ECG-CPC and S4 (SSMs)

A 2025 benchmark of ECG foundation models found that **ECG-CPC (3.8M params)** and **S4 (2.2M params)** achieve AUROCs of 0.949 and 0.941 respectively on PTB-XL — competitive with 87M+ parameter transformers — suggesting that efficient state-space models are an underexplored direction.

> "Benchmarking ECG Foundational Models: A Reality Check Across Clinical Tasks." *arXiv:2509.25095* (2025). URL: [arxiv.org/html/2509.25095v1](https://arxiv.org/html/2509.25095v1)

### 6.5 ECG-MoE (2026) — Mixture-of-Experts Foundation Model

The first MoE architecture applied to ECG foundation modelling. Rather than activating all parameters for every input, ECG-MoE routes each sample to a subset of specialised expert networks, enabling breadth without proportional compute cost.

- **Training data:** 5 clinical datasets (multi-task: arrhythmia, MI, HF, and other diagnoses).
- **Architecture:** MoE + Transformer; sparse expert routing.
- **Result:** SOTA on all 5 downstream tasks simultaneously; inference **40% faster** than a dense model of equivalent capacity.

> "ECG-MoE: Mixture-of-Expert ECG Foundation Model." *arXiv:2603.04589* (2026). URL: [arxiv.org/abs/2603.04589](https://arxiv.org/abs/2603.04589) [UNVERIFIED — verify venue before citing]

---

## 7. Task-Specific Deep Learning Models

### 7.1 Atrial Fibrillation Detection

**Attia et al. (Lancet, 2019):** CNN trained on >180,000 patients (Mayo Clinic) detects AF signatures from normal sinus rhythm ECGs. AUC **0.87**, sensitivity 79.0%, specificity 79.5%. Demonstrated that DL can identify patients with *paroxysmal* AF during a normal-rhythm ECG window.

> Attia ZI, et al. "An artificial intelligence-enabled ECG algorithm for the identification of patients with atrial fibrillation during sinus rhythm." *The Lancet* 394, 861–867 (2019). DOI: [10.1016/S0140-6736(19)31719-2](https://doi.org/10.1016/S0140-6736(19)31719-2)

Federated CNN-LSTM models across multiple centres reported internal-test AUC of approximately 0.94 for AF detection.

### 7.2 Myocardial Infarction Detection

- 12-lead CNN on PTB-XL: AUC 0.8991, sensitivity 0.8674, specificity 0.7580, F1 0.8436.
- CNN-GRU hybrid: accuracy 99.43%, sensitivity 99.71%, specificity 98.59%.
- Systematic reviews conclude CNNs and ResNets show the best overall performance for both MI detection and localisation.

> "Optimizing Myocardial Infarction Detection: A Hybrid CNN-GRU Deep Learning Approach." *PubMed* (2024). URL: [pubmed.ncbi.nlm.nih.gov/41188886/](https://pubmed.ncbi.nlm.nih.gov/41188886/)

### 7.3 Heart Failure / Low EF

**Attia et al. (Nature Medicine, 2019):** CNN on 44,959 ECG-echo pairs detects LVEF ≤35% with AUC **0.93**. FDA Breakthrough Device designation granted.

> Attia ZI, Kapa S, Lopez-Jimenez F, et al. "Screening for cardiac contractile dysfunction using an artificial intelligence-enabled electrocardiogram." *Nature Medicine* 25, 70–74 (2019). DOI: [10.1038/s41591-018-0240-2](https://doi.org/10.1038/s41591-018-0240-2)

### 7.4 Sudden Cardiac Death Risk

**Holmstrom et al. (Communications Medicine, 2024):** CNN with atrous convolutions and inverted residual blocks for SCD risk from 12-lead ECGs. Trained on 1,101 pre-arrest ECGs from 1,076 cases.

| Metric | Internal Test | External Validation |
|--------|--------------|---------------------|
| AUROC | 0.889 (95% CI 0.861–0.917) | 0.820 (0.794–0.847) |
| Sensitivity | 0.843 | 0.763 |
| Specificity | 0.818 | 0.796 |

Substantially outperformed the conventional ECG electrical risk score (AUROC 0.712/0.743).

> Holmstrom L, Chugh H, Nakamura K, et al. "An ECG-based artificial intelligence model for assessment of sudden cardiac death risk." *Communications Medicine* 4, 55 (2024). DOI: [10.1038/s43856-024-00451-9](https://doi.org/10.1038/s43856-024-00451-9)

### 7.5 AI-HEART (2026) — End-to-End Production System

**AI-HEART** demonstrates a complete end-to-end digital health pipeline for ECG analysis, integrating a CNN-based inference model with a cloud deployment architecture. Provides a reference design for production-grade ECG AI systems, covering data ingestion, model inference, and clinical reporting.

> "AI-HEART: End-to-End Digital Health System for ECG Analysis." *arXiv:2603.16891* (2026). URL: [arxiv.org/abs/2603.16891](https://arxiv.org/abs/2603.16891) [UNVERIFIED — verify venue before citing]

---

## 8. Multi-Modal Deep Learning

### 8.1 ECG + Imaging

- Residual variational autoencoders integrating late gadolinium-enhanced cardiac MRI, ECG, and clinical data for predicting malignant ventricular arrhythmia in non-ischaemic cardiomyopathy.

> "Multimodal explainable artificial intelligence identifies patients with non-ischaemic cardiomyopathy at risk of lethal ventricular arrhythmias." *European Heart Journal* (2024). URL: [ncbi.nlm.nih.gov/pmc/articles/PMC11211323/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11211323/)

### 8.2 ECG + Clinical Variables

Multi-modal deep learning using 12-lead ECG + tabular clinical attributes achieved AUROC of **0.90** for structural heart disease prediction via multi-task learning, outperforming unimodal ECG-only and clinical-only models.

> "Multimodal Deep Learning for Structural Heart Disease Prediction from ECG and Clinical Data." *medRxiv* (2026). URL: [medrxiv.org/content/10.64898/2026.02.22.26346793v1.full](https://www.medrxiv.org/content/10.64898/2026.02.22.26346793v1.full)

### 8.3 ECG + Genetic Data

**M-REGLE (2025):** Multimodal representation learning jointly analysing ECG and PPG to enhance genetic discovery for cardiovascular traits. Identified 19.3% more genetic loci from ECG data and 13.0% more from ECG+PPG combined, compared to unimodal learning.

> "Applying multimodal AI to physiological waveforms improves genetic prediction of cardiovascular traits." *American Journal of Human Genetics* (2025). URL: [cell.com/ajhg/fulltext/S0002-9297(25)00231-9](https://www.cell.com/ajhg/fulltext/S0002-9297(25)00231-9)
