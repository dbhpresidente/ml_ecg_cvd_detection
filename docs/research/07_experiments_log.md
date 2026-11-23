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

---

## 2. S4D (Diagonal State Space Model) Experiments (Nov 2026)

S4D-Lin variant (Gu et al., NeurIPS 2022). FFT convolution kernel, O(L log L) complexity.  
Reference implementation: Mehari & Strodthoff, *Cardiovascular Engineering and Technology* (2024); repo: `github.com/NeuroSyd/S4D-ECG`.

### 2.1 Run History

| Version | d_model | d_state | n_layers | lr | Scheduler | Epochs | Best AUC | Best Ep | Notes |
|---------|---------|---------|----------|----|-----------|----|----------|---------|-------|
| v1 | 256 | 32 | 6 | 3e-4 | OneCycleLR | 50 | 0.8890 | 13 | Baseline, C init σ=0.5 |
| v2 | 256 | 32 | 6 | 1e-4 | OneCycleLR | 50 | 0.8899 | 14 | Resume from v1 ep13; dropout=0.3, wd=3e-4 |
| v3 | 128 | 8 | 4 | 1e-3 | Cosine+warmup | 200 | — | — | Config from NeuroSyd paper; in progress |

### 2.2 Key Findings — v1/v2

**v1 diagnosis:** Model peaked at ep13 (AUC 0.889) then overfit aggressively: loss 0.19→0.09 through ep28 while AUC fell to 0.862. First run (d_state=64) caused OOM (21 GB RAM); fixed with `gradient_checkpointing=True` + `d_state=32`.

**v2 (resume from v1 best.pt):** Added resume support to `Trainer` (saves `last.pt` with full optimizer+scheduler state each epoch). More regularisation (dropout 0.1→0.3, wd 1e-4→3e-4, lr 3e-4→1e-4) delayed early stopping (ep28→ep34) but did not break the AUC ceiling (0.8899 vs 0.8890). **Conclusion: ceiling is not a regularisation problem.**

### 2.3 Root Cause Analysis — Gap vs Reference (0.889 vs 0.921)

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

### 2.4 Code Changes Made

- `src/models/s4d.py`: corrected C initialisation scale from `σ=0.5` to `σ=1.0`
- `src/training/trainer.py`:
  - Added `resume_from: Path | None` to `TrainerConfig`
  - `fit()` now saves `last.pt` every epoch (model + optimizer + scheduler + patience state)
  - Resume from `last.pt`: full state restoration
  - Resume from `best.pt` (partial): rebuilds OneCycleLR for remaining epochs
- `scripts/train.py`: added `--resume <path>` CLI argument

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

### 3.2 Prioritised Roadmap

| Priority | Approach | Expected AUC | VRAM | Timeline |
|----------|----------|-------------|------|----------|
| 1 | S4D v3 (corrected config) | ~0.921 | 2-3 GB | in progress |
| 2 | Ensemble S4D v3 + xresnet1d50 v6 | ~0.930+ | 5-6 GB | after v3 converges |
| 3 | BEAT-Net (arXiv:2601.07316, Jan 2026) | 0.931 | 2-3 GB | awaiting code release |
| 4 | ECGFounder LoRA fine-tune | ~0.930 | 5-6 GB | exploratory |

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
