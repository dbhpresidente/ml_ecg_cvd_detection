# Research Gaps and Opportunities in ECG-Based CVD Prediction

## 1. Cross-Dataset Generalisation

### The Problem
Models trained on one ECG dataset perform substantially worse when evaluated on a different one, even when the clinical task is the same. Sources of domain shift include:
- **Hardware differences:** Sampling rates, ADC resolution, electrode contact quality, and filter settings vary across acquisition devices.
- **Population differences:** Age distributions, disease prevalences, and demographic compositions differ between hospital systems and geographic regions.
- **Annotation heterogeneity:** Label definitions are not standardised across datasets (e.g., "ST elevation" thresholds differ between CPSC 2018 and PTB-XL).

### Evidence
In the PhysioNet 2020 challenge, models trained on individual source databases showed large performance gaps when evaluated cross-database — motivating the multi-source training approach. Generalisation gaps of 0.05–0.15 AUC between source and external validation sets have been reported for AF detection models (Hannun et al., 2019; external validation results).

### Opportunities
- **Domain adaptation:** Techniques from distribution shift literature (domain-invariant representation learning, adversarial domain adaptation) have been applied to ECG with early promising results but are not yet standard practice.
- **Federated learning:** Collaborative training across hospital systems without data sharing could improve generalisation while preserving privacy.
- **Standardised evaluation protocols:** A universal held-out test set drawn from multiple international sources (like the PTB-XL benchmark) would enable direct cross-study comparison.

---

## 2. Fairness and Demographic Bias

### The Problem
AI-ECG models exhibit performance disparities across demographic subgroups — a critical concern for clinical deployment.

Pham et al. (2023) systematically audited ECG deep learning models across age, sex, and race subgroups on a large multi-centre dataset:
- **Age:** AUC gap of **>0.14** between young patients (≤40 years) and elderly patients (≥75 years).
- **Sex:** Female patients showed lower sensitivity for MI detection in several models.
- **Race:** Performance disparities were observed for Black patients in some conditions, potentially reflecting training dataset underrepresentation.

> Pham J, et al. "ECG-based deep learning for cardiovascular conditions: an assessment of model performance, fairness, and clinical utility." *(Preprint, 2023).* [UNVERIFIED — search for Pham 2023 ECG fairness for exact DOI]

### Mechanisms
- Underrepresentation of certain subgroups in training data (most large ECG datasets overrepresent middle-aged European/American males).
- Physiological differences in ECG morphology across age, sex, and body habitus that the model may not handle equally.
- Label bias: annotations made predominantly by cardiologists familiar with a specific population may embed systematic biases.

### Opportunities
- **Subgroup-stratified reporting** should be standard in all publications — not just aggregate AUC.
- **Fairness-aware training:** Reweighting, adversarial debiasing, or multi-task learning with demographic invariance constraints.
- **Diverse dataset curation:** Intentional oversampling or collection from underrepresented populations (paediatric, elderly, non-Western cohorts).
- **Age-specific models:** Given the strong age effect on ECG morphology, age-stratified or age-conditioned architectures may reduce bias.

---

## 3. Explainability and Clinical Interpretability

### The Problem
Clinical adoption of AI-ECG models requires interpretability — cardiologists need to know *why* a model flagged a recording. Despite extensive interest, as of 2023:
- Fewer than **52%** of ECG AI studies report any form of post-hoc explainability (Grad-CAM, SHAP, integrated gradients, etc.).
- There is **no consensus standard** for ECG-specific explanations.
- Existing explanation methods produce lead-level or time-step-level attributions, but rarely align with the specific waveform components (P-wave, QRS, ST segment) that cardiologists use.

### Existing Approaches
- **Gradient-based (Grad-CAM, Guided Backpropagation):** Identify which time steps most influenced the prediction. Used in Strodthoff et al. and Attia et al. to validate that the model attends to clinically expected regions.
- **SHAP (SHapley Additive exPlanations):** Decomposes predictions into feature-level contributions. Applicable to both classical ML (feature SHAP) and CNN (DeepSHAP).
- **Attention maps (Transformers):** Provide interpretable weights over time and leads, though attention ≠ attribution.
- **Concept-based explanations (TCAV):** Test whether model activations align with human-defined cardiac concepts — experimental for ECG.

### Opportunities
- **Clinically grounded attribution:** Develop explanation methods that map to cardiologist-identified waveform components rather than arbitrary time windows.
- **User studies:** Measure whether explanations improve cardiologist trust and decision-making in controlled settings.
- **Explanation benchmarks:** Create gold-standard annotation sets (e.g., expert-labelled "this region shows the ST elevation") to quantitatively evaluate explanation quality.

---

## 4. Few-Shot Learning and Rare Conditions

### The Problem
Common conditions (AF, LBBB, RBBB) have thousands of labelled examples. Many clinically important conditions are rare — e.g., Brugada syndrome, short QT syndrome, Wolff-Parkinson-White — and may have only tens to hundreds of labelled ECGs in any public dataset.

Standard supervised learning fails in this regime: it cannot learn reliable classifiers from <100 examples per class.

### Promising Approaches
- **Self-supervised pre-training + few-shot fine-tuning:** ECG-JEPA (a masked-prediction self-supervised model) demonstrated **9× label efficiency** relative to supervised training on PTB-XL — matching fully supervised performance with roughly 11% of the labelled data. [UNVERIFIED — search for ECG-JEPA 2024 for exact DOI]
- **Meta-learning (MAML, Prototypical Networks):** Few-shot learning paradigms that adapt quickly to new tasks with limited examples; applied to ECG in preliminary studies.
- **Data augmentation for rare classes:** Time stretching, amplitude scaling, additive Gaussian noise, and lead dropout can augment small training sets.
- **Foundation model adaptation:** Large ECG foundation models (ECGFounder, ECG-FM) can serve as frozen feature extractors — training only a lightweight classification head on rare-class data.

---

## 5. Edge Deployment and Wearable Integration

### The Problem
Most published ECG AI models are designed for hospital 12-lead ECGs and require GPU inference, cloud connectivity, and trained technicians. Extending AI-ECG capabilities to wearable single-lead devices (Holter monitors, smartwatch ECGs, AliveCor KardiaMobile) requires:
- **Model compression:** Quantisation, pruning, and knowledge distillation to fit within microcontroller memory and latency budgets.
- **Reliability on short, noisy signals:** Wearable signals are typically 30 seconds or less, frequently motion-contaminated.
- **Reduced lead set:** Single-lead (Lead I or Lead II proxy) reduces diagnostic power but is the only feasible form factor for consumer devices.

### Evidence of Feasibility
- PhysioNet 2021 showed that 2-lead performance (0.558) was 92.4% of 12-lead performance (0.604) across 30 conditions, suggesting minimal information loss from lead reduction.
- Compressed models as small as **144 KB** (8-bit quantised MobileNet-style 1D CNN) have been demonstrated for single-lead AF detection with <5 ms inference on ARM Cortex-M4. [UNVERIFIED — search for quantised ECG CNN ARM for exact reference]
- TensorFlow Lite and ONNX Runtime now support 1D convolution on mobile and embedded platforms.

### Opportunities
- **On-device continuous monitoring:** Detect paroxysmal AF or ventricular ectopy that would be missed during a 10-second resting ECG.
- **Opportunistic screening:** Smartwatch-based population-scale AF screening in asymptomatic individuals.
- **Low-resource healthcare settings:** Lightweight models deployable on smartphone without cloud infrastructure.

---

## 6. Prospective Clinical Validation

### The Problem
The vast majority of AI-ECG publications are retrospective — trained and tested on historical ECGs. Prospective validation (deploying the model in a live clinical workflow and measuring its impact on patient outcomes) is rare and methodologically challenging.

A 2023 systematic review by Jiang et al. identified:
- **64 studies** evaluating AI-ECG interventions in clinical settings.
- Only **11 randomised controlled trials (RCTs)** with patient-outcome endpoints as of 2023.

> Jiang M, et al. "Challenges and solutions for AI-based ECG analysis: a systematic review." *(2024).* [UNVERIFIED — verify exact DOI for Jiang 2024 ECG systematic review]

### Evidence of Clinical Benefit
Lin et al. (2024) reported a randomised trial in which AI-ECG–guided management reduced 30-day all-cause mortality by approximately 10% in a high-risk inpatient population.

> Lin C-S, et al. "Artificial intelligence–enabled ECG for mortality risk stratification in patients presenting to emergency departments." *Nature Medicine* (2024). DOI: [10.1038/s41591-024-02961-4](https://doi.org/10.1038/s41591-024-02961-4)

### Gaps to Address
- **Workflow integration studies:** How do models affect clinical workflow time, alert fatigue, and physician override rates?
- **Impact on underserved populations:** Most RCTs have been conducted in high-resource settings; evidence from low/middle-income countries is lacking.
- **Long-term outcome tracking:** Most trials measure short-term endpoints (30-day mortality); evidence for 1–5 year outcomes is sparse.

---

## 7. Multi-Label vs. Single-Label Framing

### The Problem
Most real ECGs exhibit multiple concurrent conditions (e.g., LBBB + atrial flutter + ST changes). Training a single binary classifier per condition (one-vs-rest) ignores inter-label correlations. Label co-occurrence patterns in PTB-XL:
- MI + STTC: frequently co-occur (both ischaemia-related).
- NORM + any abnormality: mutually exclusive by definition.

A single-label "most likely condition" framing misrepresents the diagnostic reality.

### Opportunities
- **Explicit correlation modelling:** Graph neural networks, label co-occurrence matrices, or recurrent label decoders (like in multi-label NLP) can model label dependencies.
- **Hierarchical label structures:** SCP-ECG provides a 3-tier hierarchy (superclass → diagnostic → sub-diagnostic). Hierarchical multi-label learning can leverage this structure to regularise rare classes.
- **Conditional label generation:** Seq2seq or autoregressive label generation (generating the full set of labels as a sequence) could better model inter-label dependencies than independent binary classifiers.

---

## 8. Long-Term Risk Prediction

### The State of the Art
Long-term risk prediction (beyond the current cardiac state) is an emerging paradigm:
- **SEER** (Hughes et al., 2023): 5-year CVD mortality AUC 0.83 from a single ECG, outperforming PCE risk score.
- **AIRE** (Radhakrishnan et al., 2024): Discrete-time survival model, C-index 0.775, validated in 5 international cohorts.
- **Raghunath et al.** (2020): 1-year all-cause mortality AUC 0.85 from 12-lead ECG (Geisinger cohort, n=253,397).

### Open Problems
- **Survival modelling:** Standard cross-entropy classification ignores censoring (patients who have not yet had an event). Proper survival models (Cox, DeepSurv, discrete-time hazard models) are more appropriate but less commonly used.
- **Dynamic risk:** A single snapshot ECG may not capture risk change over time. Serial ECG risk trajectories remain largely unexplored.
- **Competing risks:** Cardiovascular mortality and non-cardiovascular mortality compete. Single-cause models overestimate net risk.
- **Causal framing:** Observational ECG data confounds true risk (e.g., patients with worse ECGs may receive more aggressive treatment, complicating prospective risk estimates).

---

## 9. FDA-Cleared and CE-Marked AI-ECG Devices

As of 2024, several AI-ECG products have received regulatory clearance, establishing the clinical translation pathway:

| Product | Developer | Clearance | Indication |
|---|---|---|---|
| Cardiologs ECG Analysis Platform | Cardiologs (Philips) | FDA 510(k) cleared | AF detection, full ECG analysis |
| KardiaBand / KardiaMobile | AliveCor | FDA 510(k) cleared | Personal AF detection |
| Zio Patch | iRhythm Technologies | FDA 510(k) cleared | Long-term cardiac monitoring, AF |
| PulseAI Atrial Fibrillation | PulseAI (Mayo Clinic spinout) | CE marked | Screening for occult AF |
| AI-ECG for LV dysfunction | Mayo Clinic (Eko) | FDA Breakthrough Device | Occult cardiomyopathy screening |

**Implication:** The regulatory pathway for AI-ECG tools now has precedent; studies targeting specific well-defined clinical questions (AF, LV dysfunction) with large prospective datasets are most likely to achieve translation.

---

## 10. Summary of Priority Gaps

| Gap | Severity | Maturity of Solutions | Impact if Solved |
|---|---|---|---|
| Cross-dataset generalisation | High | Medium (domain adaptation works, not standardised) | Broad multi-site deployment |
| Fairness / demographic bias | High | Low (awareness high, solutions early-stage) | Equitable clinical use |
| Explainability | Medium | Medium (Grad-CAM widely used, no gold standard) | Clinician trust and adoption |
| Few-shot rare conditions | Medium | Medium (SSL shows promise) | Rare disease diagnosis |
| Edge / wearable deployment | Medium | Medium (quantisation feasible) | Population-scale screening |
| Prospective RCTs | High | Low (resource-intensive) | Clinical guideline integration |
| Long-term survival modelling | Medium | Low (SEER/AIRE are early) | Preventive cardiology |
| Multi-label correlation modelling | Low | Medium (graph NNs applied) | Diagnostic completeness |
| ECG image classification | Medium | Medium (BHF Challenge 2025 shows feasibility) | Legacy ECG archive mining |
| Efficient / physics-informed DL | Low | Low-Medium (early evidence) | Deployment at scale |

---

## 11. ECG Image Classification (Printed and Scanned ECG)

### The Opportunity
A growing line of work frames ECG interpretation as a **2D computer vision problem**: the ECG waveform is rendered as an image (either a spectrogram/scalogram, a synthesised paper-strip plot, or a photograph of a printed clinical ECG) and classified with standard image CNNs.

**Motivation:** Millions of historical ECGs exist only as paper printouts or scanned images in legacy clinical systems. Converting these to digital signals for 1D CNN processing is expensive and error-prone; direct image classification avoids this step.

### Evidence
- **BHF ECG Image Challenge 2025:** ConvNeXt-V2-Base + ensemble achieved AUC 0.9677 on UK Biobank printed ECG images — matching the best 1D signal models. This is the strongest published evidence for 2D image approaches.
  > "BHF ECG Image Challenge 2025 — 1st Place Solution." arXiv:2507.21968 (2025). URL: [arxiv.org/abs/2507.21968](https://arxiv.org/abs/2507.21968)

- **PhysioNet/CinC Challenge 2024:** Focused on ECG digitisation + classification from scanned paper records. Top teams used ConvNeXt-Base / EfficientNet-V2 ensembles; AUC ~0.96.

- **Spectrogram-based hybrids (Kim et al., 2025):** Hybrid CNN + Transformer on ECG spectrograms achieves high AUC on MIT-BIH + Icentia11k, suggesting that time-frequency image representations capture information complementary to raw 1D signals.
  > Kim et al. "Hybrid CNN + Transformer for ECG spectrograms." *Scientific Reports* (2025). DOI: [10.1038/s41598-025-92582-9](https://doi.org/10.1038/s41598-025-92582-9)

### Gaps
- No standardised benchmark for image-based ECG classification equivalent to the PTB-XL 1D benchmark.
- It is unclear whether the gains come from the 2D representation or from the backbone (ConvNeXt pre-trained on ImageNet).
- The image rendering pipeline (DPI, colour, layout, compression artefacts) has not been systematically ablated.

---

## 12. Efficient and Physics-Informed Architectures

### Mixture-of-Experts for ECG
ECG-MoE (2026) introduced the first MoE architecture for ECG foundation models. By routing different ECG inputs to specialised expert sub-networks, it achieves SOTA across 5 clinical tasks while reducing inference time by 40% versus a dense equivalent.

> "ECG-MoE: Mixture-of-Expert ECG Foundation Model." arXiv:2603.04589 (2026). URL: [arxiv.org/abs/2603.04589](https://arxiv.org/abs/2603.04589)

**Opportunity:** MoE architectures could be particularly valuable for ECG because different pathologies activate different signal morphologies — routing patients with suspected MI through a different expert than patients with conduction abnormalities is physiologically motivated.

### Koopman and Wavelet Features for Transformers
A 2026 preprint showed that feeding **physically motivated features** (Koopman operator decomposition + wavelet coefficients) as input tokens to a Transformer, rather than raw samples, improves classification performance on ECG tasks.

> "ECG Classification with Transformers Using Koopman and Wavelet Features." arXiv:2603.08339 (2026). URL: [arxiv.org/abs/2603.08339](https://arxiv.org/abs/2603.08339)

**Opportunity:** This "physics-informed DL" paradigm bridges classical signal processing and modern deep learning — potentially improving both performance and interpretability, since Koopman and wavelet features have clear physiological meaning.
