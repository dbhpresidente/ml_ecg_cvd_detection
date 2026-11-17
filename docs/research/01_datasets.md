# ECG Datasets for Cardiovascular Disease Prediction

## 1. Summary Table

| Dataset | Records | Patients | Leads | Hz | Duration | Labels / Conditions | License | DOI / URL |
|---------|---------|----------|-------|----|----------|---------------------|---------|-----------|
| **PTB-XL** | 21,799 | 18,869 | 12 | 500 | 10 s | 71 SCP-ECG statements; 5 superclasses | CC BY 4.0 | [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6) |
| **CinC 2017** | 12,186 | — | 1 (Lead I) | 300 | 9–61 s | 4-class (Normal/AF/Other/Noisy) | Open | [10.13026/d3hm-sf11](https://physionet.org/content/challenge-2017/1.0.0/) |
| **CinC 2020** | 66,361 | — | 12 | 257–1000 | Variable | 27 SNOMED-CT codes (multi-label) | Partial | [10.13026/dvyd-kd57](https://physionet.org/content/challenge-2020/1.0.2/) |
| **CinC 2021** | 88,253 | — | 12 | 257–1000 | Variable | 30 SNOMED-CT codes (multi-label) | Partial | [10.13026/jvs0-mc58](https://physionet.org/content/challenge-2021/1.0.3/) |
| **MIMIC-IV-ECG** | ~800,000 | ~160,000 | 12 | 500 | 10 s | Free-text reports + machine measurements | Credentialed | [10.13026/4nqg-sb35](https://physionet.org/content/mimic-iv-ecg/1.0/) |
| **CODE-15%** | 345,779 | 233,770 | 12 | 400 | 7–10 s | 6 abnormalities + normal | CC BY 4.0 | [10.5281/zenodo.4916206](https://zenodo.org/records/4916206) |
| **Chapman-Shaoxing** | 10,646 | 10,646 | 12 | 500 | 10 s | 11 rhythms + 67 conditions | Open | [10.1038/s41597-020-0386-x](https://doi.org/10.1038/s41597-020-0386-x) |
| **CPSC 2018** | 6,877 | — | 12 | 500 | 6–60 s | 9 rhythm classes | Open | Via PhysioNet (CinC 2020) |
| **Georgia** | 10,344 | — | 12 | 500 | 5–10 s | 67 SNOMED-CT codes | Open | Via PhysioNet (CinC 2020) |
| **Ningbo** | ~34,905 | — | 12 | 500 | Variable | Multi-label, standardised statements | Open | [10.1038/s41597-022-01403-5](https://doi.org/10.1038/s41597-022-01403-5) |

---

## 2. Dataset Details

### 2.1 PTB-XL

PTB-XL is the most widely adopted public benchmark for 12-lead ECG classification.

- **Source:** Physikalisch-Technische Bundesanstalt (PTB), Berlin; clinical recordings from 1989–1996.
- **Records:** 21,799 ECG records from 18,869 unique patients (52% male, 48% female; ages 0–95).
- **Leads:** 12 standard leads.
- **Sampling rate:** 500 Hz (also provided downsampled to 100 Hz for low-resolution experiments).
- **Duration:** 10 seconds per recording.
- **Labels:** 71 ECG statements conforming to the SCP-ECG standard, organised into five diagnostic superclasses:
  - NORM (Normal): 9,514 records
  - MI (Myocardial Infarction): 5,469 records
  - STTC (ST/T Change): 5,235 records
  - CD (Conduction Disturbance): 4,898 records
  - HYP (Hypertrophy): 2,649 records
- **Annotation:** Multi-label; annotated by up to two cardiologists with likelihood scores (0–100%). The standard benchmark filters to records with ≥100% annotator confidence for reproducibility.
- **Splits:** Official stratified 10-fold split (patient-independent): folds 1–8 train, fold 9 validation, fold 10 test.
- **Extended version:** PTB-XL+ adds waveform-level feature annotations (Strodthoff et al., 2023; [10.1038/s41597-023-02153-8](https://doi.org/10.1038/s41597-023-02153-8)).

**Primary citation:**
> Wagner P, Strodthoff N, Bousseljot RD, Samek W, Schaeffter T. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7, 154 (2020). DOI: [10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6)

**Benchmark citation:**
> Strodthoff N, Wagner P, Schaeffter T, Samek W. "Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL." *IEEE Journal of Biomedical and Health Informatics* 25(5), 1519–1528 (2021). DOI: [10.1109/JBHI.2020.3022989](https://doi.org/10.1109/JBHI.2020.3022989)

---

### 2.2 PhysioNet/Computing in Cardiology Challenges

#### CinC Challenge 2017 — AF Classification from Single-Lead ECG

- **Records:** 12,186 total (8,528 public training, 3,658 hidden test).
- **Lead:** Single-lead (Lead I), recorded with AliveCor KardiaMobile device.
- **Sampling rate:** 300 Hz.
- **Duration:** 9–61 seconds.
- **Labels:** 4-class — Normal sinus rhythm, Atrial fibrillation, Other rhythm, Noisy.
- **Clinical relevance:** Established the standard for AF detection from wearable, short-duration ECGs.

> Clifford GD, Liu C, Moody B, et al. "AF Classification from a Short Single Lead ECG Recording: the PhysioNet/CinC Challenge 2017." *Computers in Cardiology* 44 (2017). DOI: [10.22489/CinC.2017.065-469](https://doi.org/10.22489/CinC.2017.065-469)

#### CinC Challenge 2020 — Classification of 12-Lead ECGs

- **Records:** 66,361 from 6 international databases (CPSC 2018, CPSC-Extra, INCART, PTB, PTB-XL, Georgia).
- **Leads:** 12-lead.
- **Sampling rates:** 257 Hz (INCART), 500 Hz (CPSC, PTB-XL, Georgia), 1,000 Hz (PTB).
- **Labels:** 27 scored diagnoses using SNOMED-CT codes (multi-label). Full label set: 133 unique conditions across all databases.
- **Evaluation metric:** Custom weighted accuracy awarding partial credit for clinically similar misdiagnoses.

> Perez Alday EA, Gu A, Shah AJ, et al. "Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020." *Physiological Measurement* 41(12), 124003 (2021). DOI: [10.1088/1361-6579/abc960](https://doi.org/10.1088/1361-6579/abc960)

#### CinC Challenge 2021 — Reduced-Lead ECG Classification

- **Records:** 88,253 from 8 databases (adds Chapman-Shaoxing and Ningbo to 2020 sources).
- **Lead configurations:** 12-, 6-, 4-, 3-, and 2-lead.
- **Labels:** 30 scored conditions out of 133 total (multi-label).
- **Key finding:** Performance on reduced-lead sets (2–6 leads) was comparable to 12-lead, suggesting feasibility for wearable deployment.

> Reyna MA, Sadr N, Perez Alday EA, et al. "Will Two Do? Varying Dimensions in Electrocardiography: The PhysioNet/CinC Challenge 2021." *Computers in Cardiology* (2021). DOI: [10.23919/CinC53138.2021.9662687](https://doi.org/10.23919/CinC53138.2021.9662687)

---

### 2.3 MIMIC-IV-ECG

- **Records:** ~800,000 diagnostic 12-lead ECGs.
- **Patients:** ~160,000 unique patients (Beth Israel Deaconess Medical Center, 2008–2019).
- **Sampling rate:** 500 Hz; 10 seconds.
- **Labels:** Machine-generated waveform measurements (RR interval, QRS onset/end, axis, intervals) + >600,000 free-text cardiologist reports. Linkable to MIMIC-IV clinical data (ICD-9/10 diagnoses, lab values, medications, notes) via `subject_id`.
- **Clinical value:** Enables construction of multi-modal models combining ECG + EHR data, and downstream outcome prediction (mortality, rehospitalisation, incident disease).
- **Access:** Credentialed access via PhysioNet (requires CITI training + data use agreement).

> Gow B, Pollard T, Nathanson LA, et al. "MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset" (v1.0). PhysioNet (2023). DOI: [10.13026/4nqg-sb35](https://physionet.org/content/mimic-iv-ecg/1.0/)

---

### 2.4 CODE-15%

- **Source:** Telehealth Network of Minas Gerais (TNMG), Brazil, 2010–2016. Stratified 15% sample of the full CODE dataset (>2 million ECGs).
- **Records:** 345,779 exams from 233,770 unique patients.
- **Sampling rate:** 400 Hz; 7 or 10 seconds (zero-padded).
- **Labels:** 6 binary abnormality flags: 1st-degree AV block, right bundle branch block, left bundle branch block, sinus bradycardia, atrial fibrillation, sinus tachycardia; plus normal ECG label.
- **Clinical significance:** Used to train the state-of-the-art Ribeiro et al. (2020) model that outperformed cardiology residents.

> Ribeiro AH, Ribeiro MH, Paixão GMM, et al. "Automatic diagnosis of the 12-lead ECG using a deep neural network." *Nature Communications* 11, 1760 (2020). DOI: [10.1038/s41467-020-15432-4](https://doi.org/10.1038/s41467-020-15432-4)

> Ribeiro AH, et al. "CODE-15%: a large scale annotated dataset of 12-lead ECGs." Zenodo (2021). DOI: [10.5281/zenodo.4916206](https://zenodo.org/records/4916206)

---

### 2.5 Chapman-Shaoxing Database

- **Records:** 10,646 (one record per patient).
- **Sampling rate:** 500 Hz; 10 seconds.
- **Labels:** 11 rhythm types + 67 additional diagnostic labels (multi-label). Covers AF, AFL, sinus bradycardia/tachycardia, and various conduction abnormalities.
- **Usage:** Included in both CinC 2021 and CinC 2020 training data.

> Zheng J, Zhang J, Danioko S, et al. "A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients." *Scientific Data* 7, 48 (2020). DOI: [10.1038/s41597-020-0386-x](https://doi.org/10.1038/s41597-020-0386-x)

---

### 2.6 CPSC 2018 (China Physiological Signal Challenge)

- **Records:** 6,877 (variable duration, 6–60 seconds).
- **Sampling rate:** 500 Hz; 12-lead.
- **Labels:** 9 classes — AF (1,221), 1st-degree AV block (722), LBBB (235), Normal sinus rhythm (918), PAC (615), PVC (700), RBBB (1,857), ST depression (869), ST elevation (220).
- **Challenge winner:** CNN-BiGRU-Attention model (Chen et al., 2020) achieved median F1 of 0.84 on the hidden test set.

> Chen TM, Huang CH, Shih ESC, Hwang MJ. "Detection and Classification of Cardiac Arrhythmias by a Challenge-Best Deep Learning Neural Network Model." *iScience* 23(3), 100886 (2020). DOI: [10.1016/j.isci.2020.100886](https://doi.org/10.1016/j.isci.2020.100886)

---

### 2.7 Ningbo Database

- **Records:** ~34,905; 12-lead; 500 Hz.
- **Labels:** Multi-label with standardised diagnostic statements per AHA/ACC/HRS recommendations.

> Zheng J, Chu H, Struppa D, et al. "A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements." *Scientific Data* 9, 272 (2022). DOI: [10.1038/s41597-022-01403-5](https://doi.org/10.1038/s41597-022-01403-5)

---

## 3. Clinical / Non-ECG Cohorts for CVD Risk Prediction

These datasets are used for traditional risk score modeling and for multi-modal ECG + clinical feature approaches.

### 3.1 UK Biobank

- **Participants:** ~500,000 (ages 40–69 at enrollment, 2006–2010), United Kingdom.
- **ECG data:** ~90,000 participants have 12-lead resting ECG data (10 seconds, 500 Hz); ~3,000 have repeat measurements.
- **Other modalities:** Cardiac MRI, genotyping (500K SNP array), blood biomarkers, lifestyle questionnaires, linked hospital records (ICD-coded outcomes).
- **CVD applications:** Deep learning models for ECG heart age prediction, AF risk estimation, left ventricular hypertrophy prediction, coronary artery calcification scoring, MACE prediction.
- **Access:** Application-based at [ukbiobank.ac.uk](https://www.ukbiobank.ac.uk/).

### 3.2 NHANES

- **Description:** Cross-sectional survey of the U.S. civilian population (CDC/NCHS). Includes demographics, lab tests (lipids, glucose, HbA1c), blood pressure, anthropometrics, medical history.
- **CVD prediction use:** 30–40+ variable models for CVD risk stratification.
- **Availability:** Fully public at [cdc.gov/nchs/nhanes](https://www.cdc.gov/nchs/nhanes/).

### 3.3 Framingham Heart Study

- **Description:** Longitudinal cohort started 1948, Framingham MA. The source of the Framingham Risk Score (FRS).
- **Commonly used variables (simplified version):** Age, sex, cigarettes/day, total cholesterol, systolic/diastolic BP, BMI, heart rate, glucose, diabetes, BP medication, prevalent stroke/hypertension (~4,240 patients, 15 variables).
- **ML results:** Logistic regression AUC ~0.72; neural networks accuracy ~0.77.
- **Availability:** Restricted via BioLINCC / dbGaP; simplified version widely available on Kaggle.

### 3.4 MIMIC-III / MIMIC-IV (Clinical EHR)

- **Description:** De-identified EHR from BIDMC ICU admissions. MIMIC-IV covers 2008–2019 with structured (ICD diagnoses, labs, vitals, medications) and unstructured (clinical notes) data.
- **CVD applications:** Heart failure prediction, mortality prediction, 90-day CVD rehospitalisation, multimodal ECG + EHR models.
- **Availability:** Credentialed access via [PhysioNet](https://physionet.org/content/mimiciv/).

---

## 4. Commonly Used Non-ECG Clinical Variables

The following variables appear most frequently across CVD prediction studies using EHR/clinical data:

| Category | Variables |
|----------|-----------|
| Demographics | Age, sex/gender, race/ethnicity |
| Vitals | Systolic/diastolic BP, heart rate, BMI |
| Lipid panel | Total cholesterol, LDL, HDL, triglycerides |
| Metabolic | Fasting glucose, HbA1c, diabetes status |
| Lifestyle | Smoking (current/former/never), cigarettes/day, alcohol, physical activity |
| Medical history | Prior MI, stroke, hypertension, diabetes, family CVD history |
| Medications | Antihypertensives, statins, anticoagulants |
| Lab values | Creatinine, BNP/NT-proBNP (HF), troponin, CRP |
| Imaging-derived | Ejection fraction, left atrial size, coronary artery calcium score |
