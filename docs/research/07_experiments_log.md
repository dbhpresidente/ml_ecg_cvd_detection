# Experiments Log: Training Iterations and Findings

This document tracks all training runs, architecture investigations, and key decisions made during model development for PTB-XL superclass classification. All results are on the **validation set** unless noted as test set.

**Task:** PTB-XL multilabel classification, 5 superclasses (NORM / MI / STTC / CD / HYP).  
**Splits:** train=17,084 | val=2,146 | test=2,158 (patient-level, PTB-XL folds).  
**Primary metric:** Macro-averaged AUC (val set, best checkpoint).

---

## 1. XResNet1D Experiments (Nov 2026)

All xresnet runs use `OneCycleLR` (unless noted), AMP fp16, batch=32, GTX 1070 8 GB VRAM.

### 1.1 Run History

| Version | Arch | Params | Best AUC | Best Ep | Stop Ep | Key Change |
|---------|------|--------|----------|---------|---------|------------|
| v1 | xresnet1d101 | 8.3M | 0.9133 | 19 | 34 | Baseline: lr=1e-3, cosine, no augment |
| v2 | xresnet1d101 | 8.3M | 0.9138 | 14 | 27 | lr=1e-4, wd=1e-3, augment |
| v3 | xresnet1d50  | 4.6M | 0.9114 |  7 | 32 | Smaller arch |
| v4 | xresnet1d50  | 4.6M | 0.9108 |  7 | 32 | pos_weight (HYP=7.06) |
| v5 | xresnet1d50  | 4.6M | 0.9120 | 12 | 37 | ReduceLROnPlateau (factor=0.3) |
| v6 | xresnet1d50  | 4.6M | **0.9174** | 18 | 33 | OneCycleLR, lr=1e-3, wd=1e-4 |
| v6.1 | xresnet1d50 | 4.6M | 0.9170 | 18 | 43 | patience 15→25 |
| v7 | xresnet1d101 | 8.3M | 0.9150 | 27 | 42 | OneCycleLR same recipe |
| v7.1 | xresnet1d101 | 8.3M | 0.9161 | 27 | 42 | dropout=0.7, wd=3e-4 |

### 1.2 Key Findings

**v1–v5 (AUC 0.911–0.914):** Overfitting in ep7–19. Attempts: reduce capacity (v3), pos_weight (v4), ReduceLROnPlateau (v5). Plateau was partially correct (broke the ep7 ceiling) but `factor=0.3` too aggressive.

**v6 — decisive experiment (AUC 0.9174):** Investigation of Strodthoff benchmark code confirmed it uses OneCycleLR (fastai default), `lr=1e-3` (10× higher than v2–v5), `wd=1e-4` (10× lower). OneCycleLR: phase 1 (30% = ep1–15) lr rises 4e-5→1e-3; phase 2 (70% = ep16–50) lr falls 1e-3→1e-7. Best at ep18 when lr starts descending from maximum.

**v6.1 — ceiling confirmed:** patience=25 adds nothing — best still ep18, early stop ep43. **Conclusion: xresnet1d50 ceiling ≈ 0.917 with this dataset and recipe.**

**v7/v7.1 — larger model does not help:** xresnet1d101 (8.3M vs 4.6M) overfit severely. v7.1 (dropout=0.7, wd=3e-4) partially controls overfitting but AUC 0.9161 < v6 (0.9174). **Conclusion: dataset size limits capacity utilisation. The xresnet ceiling ≈ 0.917. Gap vs benchmark (0.937) requires a different architecture.**

### 1.3 Preprocessing Bug Retrospective

All xresnet runs v1–v7.1 were affected by the same preprocessing bug discovered during S4D analysis: `scripts/train.py` did not forward `bandpass`/`notch`/`normalization` from YAML config to `PTBXLDataset`, always applying bandpass 0.5–40 Hz (default). The Strodthoff benchmark (xresnet1d101, **0.937 AUC**) explicitly uses zero filtering.

For xresnet, this means every run trained on a signal with QRS high-frequency morphology removed. Conv 1D kernels learning directly from time series likely benefit more from this than S4D (which uses global FFT kernels).

**All v1–v7.1 AUC results carry this systematic limitation.**

### 1.4 Next Experiments (pending)

Three experiments prepared, not yet launched. Sequential dependency: v8 must confirm improvement before v9/v10 are warranted.

| Version | Arch | Key Change vs Baseline | Expected AUC | Config |
|---------|------|------------------------|--------------|--------|
| **v8** | xresnet1d50 | No bandpass + aug fix (vs v6) | ≥0.920 | `xresnet_v8.yaml` |
| **v9** | xresnet1d101 | No bandpass + aug fix (vs v7.1) | ≥0.925 | `xresnet_v9.yaml` |
| **v10** | xresnet1d50 | No bandpass + label smoothing ε=0.1 (vs v8) | ≥0.922 | `xresnet_v10.yaml` |

**v8** is the critical ablation: isolates the preprocessing fix on the proven v6 recipe.  
**v9** tests whether xresnet1d101 recovers with raw signal (Strodthoff's exact architecture).  
**v10** tests label smoothing for HYP class calibration (HYP=2,649 samples, hardest class).

**Code changes made (not yet validated on xresnet):**
- `src/training/trainer.py`: `TrainerConfig.label_smoothing: float = 0.0` added
- `src/training/trainer.py`: `_build_criterion()` applies `y*(1-ε)+ε/2` smoothing when `label_smoothing > 0`
- `src/training/trainer.py`: `_augment_signal()` fixed (σ=0.01 absolute, per-sample scaling)

---

## 2. S4D (Diagonal State Space Model) Experiments (Nov 2026)

S4D-Lin variant (Gu et al., NeurIPS 2022). FFT convolution kernel, O(L log L) complexity.  
Reference implementation: Mehari & Strodthoff, *Cardiovascular Engineering and Technology* (2024); repo: `github.com/NeuroSyd/S4D-ECG`.

### 2.1 Run History

| Version | d_model | d_state | n_layers | lr | Scheduler | Epochs | Best AUC | Best Ep | Notes |
|---------|---------|---------|----------|----|-----------|----|----------|---------|-------|
| v1 | 256 | 32 | 6 | 3e-4 | OneCycleLR | 50 | 0.8890 | 13 | Baseline, C init σ=0.5 |
| v2 | 256 | 32 | 6 | 1e-4 | OneCycleLR | 50 | 0.8899 | 14 | Resume from v1 ep13; dropout=0.3, wd=3e-4 |
| v3 | 128 | 8 | 4 | 1e-3 | Cosine+warmup | 200 | 0.8933 | 11 | NeuroSyd config; killed ep42 (NaN — A_real drift) |
| v4 | 128 | 8 | 4 | 5e-4 | Cosine+warmup | 200 | 0.8959 | 14 | A_real fix (-softplus); lr=5e-4 more conservative |
| v5 | 128 | 8 | 4 | 1e-3 | Cosine+warmup | 200 | 0.8959 | ~14 | patience=195 (effectively disabled); killed ep62 — AUC drifting down |
| v6 | 128 | 8 | 4 | 1e-3 | Cosine+warmup | 200 | 0.8917 | 11 | Strodthoff protocol: bandpass=False, notch=False; killed ep23 |
| v7 | 128 | 8 | 4 | 1e-3 | Cosine+warmup | 200 | 0.8897 | 10 | +wd=0 on SSM params; killed ep11 (GPU conflict with v8) |
| v8 | 128 | 8 | 4 | 1e-3 | Cosine+warmup | 200 | **0.8880** | **9** | Full fix stack (no bandpass + wd fix + aug fix); killed ep110 — no recovery |

### 2.2 Key Findings — v1/v2

**v1 diagnosis:** Model peaked at ep13 (AUC 0.889) then overfit aggressively: loss 0.19→0.09 through ep28 while AUC fell to 0.862. First run (d_state=64) caused OOM (21 GB RAM); fixed with `gradient_checkpointing=True` + `d_state=32`.

**v2 (resume from v1 best.pt):** Added resume support to `Trainer` (saves `last.pt` with full optimizer+scheduler state each epoch). More regularisation (dropout 0.1→0.3, wd 1e-4→3e-4, lr 3e-4→1e-4) delayed early stopping (ep28→ep34) but did not break the AUC ceiling (0.8899 vs 0.8890). **Conclusion: ceiling is not a regularisation problem.**

### 2.3 Findings — v3/v4/v5 (NeuroSyd config applied)

**v3:** Applied full NeuroSyd reference config (d_model=128, d_state=8, n_layers=4, cosine+warmup5, lr=1e-3, 200ep). Best ep11 AUC=0.8933 — marginally better but still the same ceiling. Crashed ep42 with NaN: `A_real` drifted positive during training → `exp(A_real * dt * l) → inf` for L=5000 (Vandermonde explosion).

**v4 (A_real stability fix):** Applied `-F.softplus(A_real)` in `S4DKernel.forward()`, guaranteeing eigenvalues stay in left half-plane. Lowered lr=5e-4 for extra stability. Completed 200 epochs without NaN. Best ep14 AUC=0.8959. Early stop ep53. Same ceiling.

**v5 (scheduler hypothesis):** Hypothesis was that cosine over 200 epochs decays too slowly (at ep53, lr had fallen only 8%: 1e-3→9.2e-4). Set `patience=195` (effectively disabled) to let the full schedule run. Killed at ep62: AUC drifted DOWN through ep62 (0.895→0.882), no recovery. **Conclusion: the ceiling is not a scheduler problem.**

### 2.4 Root Cause Analysis — Gap vs Reference (0.889 vs 0.921)

Comparison of our implementation with NeuroSyd/S4D-ECG (Mehari 2024) revealed systematic mis-configuration:

| Parameter | v1/v2 | NeuroSyd paper | Direction of error |
|-----------|-------|----------------|-------------------|
| `d_model` | 256 | **128** | too large |
| `d_state` | 32 | **8** | too large → Vandermonde OOM, slow, overfit |
| `n_layers` | 6 | **4** | too many |
| `lr` | 3e-4 | **1e-3** | too low |
| `epochs` | 50 | **200** | too few |
| `scheduler` | OneCycleLR | **cosine annealing** | mismatched |
| C init σ | 0.5 | **1.0** | attenuated kernel |

The reference model is **deliberately small** (d_state=8, 160K params vs our 1.1M). This avoids the Vandermonde memory explosion and trains faster, enabling 200 epochs with cosine annealing. Our v1/v2 used a 7× larger model which overfit within 13 epochs.

### 2.5 Preprocessing Investigation — Root Cause Found

After v3/v4/v5 all plateau at ~0.895, hypothesis shifted to preprocessing. The Strodthoff benchmark paper (PTB-XL reference, 0.936 AUC) explicitly states:

> *"The deep learning models in this paper were trained on the original time series data **without any further preprocessing** such as removing baseline wander and/or filtering"*

Our pipeline was applying a **Butterworth bandpass 0.5–40 Hz** to all runs. This removes high-frequency components of the QRS complex that carry morphological information (QRS spans up to ~150 Hz). By cutting off at 40 Hz, the model never sees the sharp slopes and fine structure of the QRS peaks — which are likely the most discriminative features for MI, STTC, and CD classification.

**Bug discovered:** `scripts/train.py` initialized `PTBXLDataset` without passing `bandpass`/`notch`/`normalization` from the YAML config. These parameters always fell back to class defaults (`bandpass=True, notch=True`), making YAML overrides silently ignored. This affected **all v1–v5 S4D runs** and **all xresnet runs**.

**Fix applied (commit `07deac6`):**
- `scripts/train.py`: forward `cfg["data"].get("bandpass", True)`, `notch`, `normalization` to `PTBXLDataset`
- `configs/s4d_v6.yaml`: `bandpass: false, notch: false, normalization: zscore`

**v6/v7/v8 — final results (all three fixes):**

| Fix | Run | Best AUC | Best Ep | Conclusion |
|-----|-----|----------|---------|------------|
| Preprocessing only (no bandpass) | v6 | 0.8917 | 11 | Marginally worse than v4 (0.8959) |
| + SSM weight decay fix | v7 | 0.8897 | 10 | No improvement in first 11 eps (ablation incomplete) |
| + Augmentation fix (σ=0.01, per-sample) | v8 | 0.8880 | 9 | Same ceiling; ran to ep110, no recovery |

**v8 full run analysis (ep9–110, patience=150):**  
Val AUC peaked at ep9 (0.8880), then declined as train loss continued falling (0.26→0.04). By ep43 AUC was 0.8600, stabilised in band 0.856–0.864 through ep90. At ep100 (lr=5.2e-4, 50% reduction) no inflexion occurred. Killed at ep110.

**Structural overfitting conclusion:** None of the fixes (preprocessing, weight decay exclusion, augmentation) broke the ceiling. The pattern — peak at ep9-14, sustained decline, no recovery — is consistent across v1-v8. The model memorises the 17K training samples faster than the cosine schedule can counteract it. The best S4D result remains **0.8880** (v8, ep9).

### 2.6 Code Changes Made

- `src/models/s4d.py`:
  - Corrected C initialisation scale from `σ=0.5` to `σ=1.0` (matches paper)
  - `A_r = -F.softplus(self.A_real.float())` — guarantees eigenvalues in left half-plane (SSM stability)
- `src/training/trainer.py`:
  - Added `resume_from: Path | None` to `TrainerConfig`
  - `fit()` now saves `last.pt` every epoch (model + optimizer + scheduler + patience state)
  - Resume from `last.pt`: full state restoration
  - Resume from `best.pt` (partial): rebuilds OneCycleLR for remaining epochs
- `scripts/train.py`:
  - Added `--resume <path>` CLI argument
  - **Fixed preprocessing bug**: forward `bandpass`, `notch`, `normalization` from config to `PTBXLDataset`

---

## 3. Architecture Investigation (Nov 2026)

### 3.1 Literature Search Results (3 parallel agents)

Three concurrent searches were run:
1. SOTA architectures achieving AUC ≥ 0.93 on PTB-XL superclasses
2. Mamba and lightweight Transformer implementations for L=5000
3. 2026 publications

**Key findings:**

- **No Mamba model individually surpasses xresnet1d101 (0.937) on PTB-XL** (ECGMamba 2024: 0.917; S2M2ECG: 0.925; MSCA-Mamba: 0.900).
- **Standard Transformers are impractical** at L=5000 (self-attention = 25M ops/batch); sliding-window variants (MSW-Transformer) reach 0.923, comparable to xresnet.
- **S4D-ECG (Mehari & Strodthoff 2024)** is the highest reported single-model AUC on PTB-XL: **0.921 macro AUC** with only 160K parameters. This is achievable on GTX 1070 (2–3 GB VRAM).
- **2026 SOTA (PTB-XL leaderboard):** ER-JEPA and Mert et al. stacking ensemble both reach **0.936**, matching Strodthoff 2021. No single-model breakthrough observed in 2026.
- **Foundation models** (ECGFounder: 0.930, ECG-FM: ~0.925) require LoRA fine-tuning to fit in 8 GB VRAM.

### 3.2 Prioritised Roadmap (updated Nov 2026)

S4D supervised training is exhausted at ~0.888. Next steps to reach 0.920+:

| Priority | Approach | Expected AUC | VRAM | Status |
|----------|----------|-------------|------|--------|
| 1 | **Ensemble xresnet1d50_v6 + S4D_v8** | ~0.925+ | 5-6 GB | pending |
| 2 | S4D with strong regularisation (dropout=0.3–0.5, label smoothing=0.1) | ~0.900-0.910 | 2-3 GB | pending |
| 3 | BEAT-Net (arXiv:2601.07316, Jan 2026) | 0.931 | 2-3 GB | awaiting code |
| 4 | ECGFounder LoRA fine-tune | ~0.930 | 5-6 GB | exploratory |

**Gap analysis:** reference S4D-ECG (0.9244) vs our S4D (0.8880) = 0.036 gap. The 0.9244 likely requires a training detail not publicly disclosed (possibly much higher dropout, label smoothing, or a different scheduler recipe). Our supervised xresnet (0.9174) is the practical ceiling for now.

---

## 4. Infrastructure Notes

- **Ritmo por epoch:** xresnet1d50 ≈1.5 min; xresnet1d101 ≈3.5 min; S4D (d_model=256) ≈17 min; S4D v3 (d_model=128, d_state=8) estimated ≈3-5 min
- **Carga inicial PTB-XL:** ≈16 min (num_workers=0 on Windows, all signals loaded into RAM)
- **Resume support:** `--resume checkpoints/<run>/last.pt` for full state; `--resume checkpoints/<run>/best.pt` for weights-only (rebuilds scheduler for remaining epochs)
- **Log convention:** `tail -20 <run>.log` in repo root

---

## References

- Mehari T, Strodthoff N. "Self-supervised representation learning from 12-lead ECG data." *Cardiovascular Engineering and Technology* 15, 183–199 (2024). DOI: [10.1007/s13239-024-00716-3](https://doi.org/10.1007/s13239-024-00716-3)
- Gu A, Goel K, Ré C. "On the Parameterization and Initialization of Diagonal State Space Models." *NeurIPS* (2022). arXiv: [2206.11893](https://arxiv.org/abs/2206.11893)
- NeuroSyd/S4D-ECG repository: [github.com/NeuroSyd/S4D-ECG](https://github.com/NeuroSyd/S4D-ECG)
- Mert YE, Akdoğan E, Üvet H. "Multi-Label 12-Lead ECG Classification on the PTB-XL Dataset." arXiv:2609.12803 (2026). [UNVERIFIED — verify before citing]
