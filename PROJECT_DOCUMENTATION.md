# BioAge-KOA-Prediction — Complete Project Documentation

**Last updated:** May 2026  
**Project:** Predicting Symptomatic Knee Osteoarthritis from CHARLS Data with Cross-Cultural Replication  
**Data:** CHARLS (Mendeley CC BY 4.0), HRS (RAND), ELSA (UK Data Service)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Analysis Pipeline — All Steps](#2-analysis-pipeline--all-steps)
3. [Key Results Summary](#3-key-results-summary)
4. [AUC Gap Decomposition](#4-auc-gap-decomposition)
5. [Reduced Aging Score (RAS) Experiment](#5-reduced-aging-score-ras-experiment)
6. [Manuscript Status & Revisions Needed](#6-manuscript-status--revisions-needed)
7. [File Inventory](#7-file-inventory)

---

## 1. Project Overview

**Objective:** Predict symptomatic knee osteoarthritis (KOA) from sociodemographic and clinical variables using machine learning, with internal validation on CHARLS (China) and preliminary cross-cultural replication on HRS (USA) and ELSA (UK).

**Primary Model:** Random Forest, `n_estimators=300`, `random_state=42`, no class-weight adjustment, no hyperparameter tuning. Feature set: raw17 (17 sociodemographic + clinical variables including Biological Age from CHARLS Sheet1).

**KOA Definition:** `Arthritis == 'yes' AND position_knees == 'yes'` (symptomatic KOA per Fu et al. 2025)

**Cohort:** CHARLS Dataset A — 12,329 rows after excluding missing `position_knees`. KOA prevalence = 13.3%.

---

## 2. Analysis Pipeline — All Steps

### Step 01: Data Preparation (`step_01_data_prep/`)
- **Script:** `scripts/step_01_data_prep.py`
- **What it does:** Loads CHARLS Sheet1 (15,545 rows → 12,329 after KOA verification). Defines KOA target, removes leakage columns (Arthritis, position_knees), creates early 80/20 stratified split.
- **Output:** `dataset_A_internal_train.csv` (n=9,863), `dataset_A_unseen_test.csv` (n=2,466), `dataset_B_*.csv` (n=7,635, BA≥55 subset)
- **Key decisions:** Stratified split with `random_state=42` before any model development. Train-fitted imputation.

### Step 02: Baseline Models (`step_02_baseline/`)
- **Script:** `scripts/step_02_baseline.py`
- **What it does:** Sweeps model × feature-set combinations. Tests RF, XGBoost, LightGBM against raw17, raw17+biomarkers, etc.
- **Key result:** RF + raw17 is best: internal AUC=0.847, unseen AUC=0.907. XGBoost and LightGBM underperform significantly on raw17.
- **Output:** `baseline_results.csv`

### Step 03: KDM Biological Age (`step_03_kdm_ba/`)
- **Script:** `scripts/step_03_kdm_ba.py`
- **What it does:** Computes Klemera-Doubal Method Biological Age from 8 log-transformed biomarkers. Also computes PhenoAge_Adapted (non-canonical weighted composite) and BIR (biological age residual).
- **Key finding:** KDM-BA correlates r=0.91 with CHARLS Sheet1 Biological Age, but has weaker KOA association (r=0.018 vs Sheet1 BA r=0.038).
- **Output:** `dataset_A_with_kdm_ba.csv`, `dataset_B_with_kdm_ba.csv`

### Step 04: BA Impact Analysis (`step_04_ba_impact/`)
- **Script:** `scripts/step_04_ba_impact.py`
- **What it does:** Comprehensive SHAP analysis, LASSO feature selection, feature impact comparison, paper replication experiment.
- **Key findings:**
  - XGBoost SHAP: Biological Age=0.44, Gender=0.42, BMI=0.40 (top 3)
  - Adding KDM-BA or biomarkers **decreases** AUC (0.894→0.863)
  - raw17 alone beats all augmented feature sets
  - LASSO selected 21/28 features
- **Output:** SHAP plots, interaction heatmaps, waterfall plots, `aging_clock_summary.csv`

### Step 07: Final Model (`step_07_final_model/`)
- **Script:** `scripts/step_05_07_final.py`
- **What it does:** Phase 2 final model confirmation and comparison with Phase 1.
- **Key finding:** KDM-BA does NOT improve performance because Sheet1's Biological Age is already a strong feature. Adding biomarkers introduces noise for tree-based models.
- **Output:** `tuning_results.csv`

### Step 08: XGBoost Fix (`step_08_xgboost_fix/`)
- **Script:** `scripts/step_08_xgboost_fix.py`
- **What it does:** Tests one-hot encoding, PCA on biomarkers, scale_pos_weight, probability calibration.
- **Key finding:** Integer encoding still best. RF dominates all configurations. XGBoost + scale_pos_weight doesn't close the gap.
- **Output:** `step08_results.csv`

### Step 09: Advanced / Final Push (`step_09_advanced/`)
- **Script:** `scripts/step_09_final_push.py`
- **What it does:** Optuna-tuned LGBM, stacking ensembles (RF+XGB+LGBM).
- **Key finding:** Stacking AUC=0.893, doesn't beat single RF (0.897). Optuna LGBM AUC=0.865.
- **Output:** `step09_results.csv`

### Step 10: Paper Replication (`step_10_paper_replication/`)
- **Script:** `scripts/step_10_paper_replication.py`
- **What it does:** Attempts exact replication of Fu et al. (2025) methodology: LASSO-selected 11 features, one-hot encoding, Z-score, XGBoost, 70/30 split.
- **Key finding:** Best AUC=0.891 (RF, not XGBoost), gap=0.016 vs paper's 0.908. RF outperforms XGBoost in both our and paper's feature sets.

### Step 11: Clinical Validation (`step_11_clinical_validation/`)
- **Script:** `scripts/step_11_clinical_validation.py`
- **What it does:** Full clinical validation: OOF + holdout evaluation with calibration (Brier, ECE, Cox slope/intercept), isotonic calibration, Decision Curve Analysis, publication-ready text generation.
- **Key results (raw17, raw):**
  - Internal OOF: AUC=0.847, PR-AUC=0.688, Brier=0.070, ECE=0.045, Slope=0.989
  - Unseen holdout: AUC=0.907, PR-AUC=0.791, Brier=0.053, ECE=0.056, Slope=1.216
  - DCA: 56.0 (internal) / 63.8 (holdout) avoided interventions/100 patients vs treat-all
- **Output:** `clinical_metrics_summary.csv`, calibration tables, DCA curves, publication paragraphs (EN/TR)

### Step 13: Reliability Diagnostics (`step_13_reliability_diagnostics/`)
- **Script:** `scripts/step_13_reliability_diagnostics.py`
- **What it does:** Investigates the isotonic calibration slope anomaly (slope=1.6-1.7 for isotonic variants).
- **Key finding:** The paradox is due to logit compression from isotonic regression's piecewise-constant output. ECE/Brier and decile reliability diagrams are the appropriate metrics for isotonic-calibrated variants — not Cox slope.

### Step EXT-VAL-HRS (`step_ext_val_hrs/`)
- **Script:** `scripts/step_ext_val_hrs.py`
- **What it does:** Applies CHARLS-trained RF (raw16) to HRS Wave 13 (US, n=20,605).
- **KOA proxy:** Current arthritis + difficulty stooping/kneeling (prevalence 33.9%)
- **Key result:** ROC-AUC=0.604, PR-AUC=0.399, Brier=0.258, ECE=0.194, Slope=0.184
- **Limitations:** 12 of 17 features available. Biological Age, Residence, Dyslipidemia, wave excluded.

### Step EXT-VAL-ELSA (`step_ext_val_elsa/`)
- **Script:** `scripts/step_ext_val_elsa.py`
- **What it does:** Applies CHARLS-trained RF (raw16) to ELSA Wave 8 (UK, n=8,416).
- **KOA proxy:** Ever-diagnosed arthritis + difficulty stooping/kneeling (prevalence 24.9%)
- **Key result:** Raw: AUC=0.604, PR-AUC=0.297, Slope=0.271. Isotonic: AUC=0.606, Slope=0.938, ECE=0.005.
- **Key finding:** Isotonic recalibration restores calibration but NOT discrimination.

### Step EXT-VAL-ELSA-W9 (`step_ext_val_elsa_wave9/`)
- **Script:** `scripts/step_ext_val_elsa.py` (WAVE=9)
- **What it does:** Temporal sensitivity analysis with Wave 9 (2018-2019, n=8,683).
- **Key result:** Raw AUC=0.596, Isotonic AUC=0.615. Consistent with Wave 8.

---

### 🆕 Step CHARLS-RAW16 (`step_charls_raw16/`)
- **Script:** `scripts/step_charls_raw16.py`
- **Created:** May 2026
- **What it does:** Evaluates CHARLS model restricted to the exact same raw16 features used in ELSA external validation. This isolates the feature-subset penalty within CHARLS (no cross-cultural attenuation).
- **Key results (raw):**
  - Internal OOF: AUC=0.804, PR-AUC=0.482, Brier=0.086, ECE=0.062, Slope=0.366
  - Unseen holdout: AUC=0.869, PR-AUC=0.585, Brier=0.066, ECE=0.064, Slope=0.526
- **Δ from raw17:** Internal: -0.043, Holdout: -0.038
- **Output:** `charls_raw16_metrics.csv`, calibration tables, DCA curves

### 🆕 Step COMPARE-RAW16 (`step_compare_raw16/`)
- **Script:** `scripts/step_compare_raw16.py`
- **Created:** May 2026
- **What it does:** Unified cross-cohort comparison table and plots across all 3 cohorts (CHARLS raw17/raw16, ELSA, HRS).
- **Key output:** `cross_cohort_metrics.csv`, `cross_cohort_comparison.png`, `cross_cohort_reliability.png`

### 🆕 Step UNIFIED-AGING (`step_unified_aging/`)
- **Script:** `scripts/step_unified_aging.py`
- **Created:** May 2026
- **What it does:** Attempts to compute consistent Biological Age across all three cohorts. Documents where full KDM-BA is possible and where reduced proxies must be used.
- **Key finding:** Full KDM-BA possible only in CHARLS. ELSA nurse visit has 4/8 biomarkers. HRS VBS has ~4/8 but missing creatinine, BUN, platelets.
- **Output:** `unified_aging_summary.csv`, `charls_aging_distributions.png`

### 🆕 Step ELSA-RAS (`step_elsa_ras/`)
- **Script:** `scripts/step_elsa_ras.py`
- **Created:** May 2026
- **What it does:** Computes Reduced Aging Score (RAS) from CRP + HbA1c + SBP + TC for ELSA, retrains CHARLS model with RAS, measures AUC recovery.
- **Key result:** raw16 AUC=0.605, raw16+RAS AUC=0.611 → **Δ=+0.006**
- **Conclusion:** Adding the best possible aging proxy recovers essentially nothing. Cross-cultural factors dominate the AUC gap.

---

## 3. Key Results Summary

### Primary Model (CHARLS raw17, Random Forest)

| Setting | Variant | ROC-AUC | PR-AUC | Brier | ECE | Slope |
|---------|---------|---------|--------|-------|-----|-------|
| Internal OOF | raw | 0.847 | 0.688 | 0.070 | 0.045 | 0.989 |
| Internal OOF | isotonic | 0.839 | 0.676 | 0.069 | 0.042 | 1.607 |
| Unseen holdout | raw | 0.907 | 0.791 | 0.053 | 0.056 | 1.216 |
| Unseen holdout | isotonic | 0.900 | 0.779 | 0.051 | 0.060 | 1.742 |

### Cross-Cohort Comparison (raw, uncalibrated)

| Cohort | n | KOA% | Features | ROC-AUC | PR-AUC | Brier | ECE | Slope |
|--------|---|------|----------|---------|--------|-------|-----|-------|
| CHARLS raw17 Internal | 9,863 | 13.3 | 17 | 0.847 | 0.688 | 0.070 | 0.045 | 0.989 |
| CHARLS raw17 Holdout | 2,466 | 13.3 | 17 | 0.907 | 0.791 | 0.053 | 0.056 | 1.216 |
| CHARLS raw16 Internal | 9,863 | 13.3 | 15 | 0.804 | 0.482 | 0.086 | 0.062 | 0.366 |
| CHARLS raw16 Holdout | 2,466 | 13.3 | 15 | 0.869 | 0.585 | 0.066 | 0.064 | 0.526 |
| ELSA W8 raw16 | 8,416 | 24.9 | 14 | 0.604 | 0.297 | 0.201 | 0.097 | 0.271 |
| ELSA W8 raw16+RAS | 8,416 | 24.9 | 15 | 0.611 | 0.316 | 0.188 | — | — |
| HRS W13 raw16 | 20,605 | 33.9 | 12 | 0.604 | 0.399 | 0.258 | 0.194 | 0.184 |

---

## 4. AUC Gap Decomposition

This is the central new finding of the May 2026 analysis:

```
CHARLS raw17 Holdout → CHARLS raw16 Holdout:  ΔAUC = -0.038  (13% of total gap)
CHARLS raw16 Holdout → ELSA/HRS raw16:         ΔAUC = -0.265  (87% of total gap)
───────────────────────────────────────────────────────────────────────────
Total gap (CHARLS raw17 → ELSA/HRS):           ΔAUC = -0.303  (100%)
```

**Interpretation:**

- **Feature-subset penalty** (dropping BA + wave/Time + adding wave_id): only **13%** of the gap
- **Cross-cultural attenuation** (outcome proxy mismatch, population differences, diagnostic ascertainment): **87%** of the gap

The manuscript's claim that "Biological Age accounts for the majority of the ROC-AUC loss" is incorrect. The dominant driver is cross-cultural shift, not missing BA.

---

## 5. Reduced Aging Score (RAS) Experiment

**Method:** Train a linear model on CHARLS to predict KDM-BA from the 4 biomarkers available in both CHARLS and ELSA:

```
RAS = β₀ + β₁·log(CRP) + β₂·log(HbA1c) + β₃·log(TC) + β₄·SBP
```

**CHARLS training:**
- R² = 0.417 (RAS captures 42% of KDM-BA variance)
- RAS vs KOA correlation: r = 0.014 (KDM-BA vs KOA: r = 0.018)
- Sheet1 BA vs KOA correlation: r = 0.038 (still the strongest BA-KOA signal)

**ELSA application (n=8,416, 2,499 with complete biomarkers):**

| Model | ROC-AUC | Δ from raw16 |
|-------|---------|-------------|
| raw16 only | 0.6050 | — |
| raw16 + RAS | 0.6111 | +0.0061 |

**Conclusion:** Even the best possible aging proxy recoverable from publicly available ELSA data adds negligible discriminative power. The AUC gap is not a BA problem — it's a cross-cultural transportability problem.

---

## 6. Manuscript Status & Revisions Needed

**Current:** `manuscript_v3_with_elsa.tex` (878 lines, TRIPOD-compliant)

**Required revisions based on new findings:**

1. **Abstract:** Remove "the exclusion of Biological Age (the dominant raw17 predictor)" as the primary explanation for HRS attenuation. Replace with decomposition showing cross-cultural factors as dominant.
2. **Results:** Add AUC gap decomposition table and ELSA RAS experiment results.
3. **Discussion:** Rewrite the HRS/ELSA replication interpretation. The key sentence to change is the claim that "BA accounts for the majority of the ROC-AUC gap." Replace with: "Feature-subset penalty accounts for ~13% of the total AUC loss; cross-cultural factors (outcome proxy heterogeneity, population differences) account for ~87%. A Reduced Aging Score constructed from biomarkers available in ELSA recovers <0.01 AUC, confirming that BA's marginal contribution is small."
4. **New table:** Add complete cross-cohort comparison table showing all 3 cohorts at raw17, raw16, and raw16+RAS levels.
5. **Limitations:** Add explicit discussion of outcome proxy mismatch as the primary limitation.

**Template manuscript:** `manuscript_v4_corrected.tex` (created alongside this document)

---

## 7. File Inventory

### Scripts (`scripts/`)
| Script | Purpose |
|--------|---------|
| `step_01_data_prep.py` | Data loading, KOA definition, early split |
| `step_02_baseline.py` | Baseline model sweep |
| `step_03_kdm_ba.py` | KDM-BA + PhenoAge_Adapted computation |
| `step_04_ba_impact.py` | SHAP analysis, LASSO, feature impact |
| `step_05_07_final.py` | Final model confirmation |
| `step_08_xgboost_fix.py` | Encoding experiments |
| `step_09_final_push.py` | Optuna + stacking ensembles |
| `step_10_paper_replication.py` | Fu et al. replication attempt |
| `step_11_clinical_validation.py` | Calibration, DCA, full evaluation |
| `step_13_reliability_diagnostics.py` | Isotonic slope anomaly investigation |
| `step_ext_val_hrs.py` | HRS external validation |
| `step_ext_val_elsa.py` | ELSA external validation (W8/W9) |
| **`step_charls_raw16.py`** | 🆕 CHARLS raw16 internal baseline |
| **`step_compare_raw16.py`** | 🆕 Unified cross-cohort comparison |
| **`step_unified_aging.py`** | 🆕 BA computation across cohorts |
| **`step_elsa_ras.py`** | 🆕 RAS computation + ELSA re-evaluation |

### Output Folders
| Folder | Contents |
|--------|----------|
| `step_01_data_prep/` | Dataset splits, metadata |
| `step_02_baseline/` | Baseline results |
| `step_03_kdm_ba/` | Datasets with KDM-BA columns |
| `step_04_ba_impact/` | SHAP plots, feature comparisons |
| `step_07_final_model/` | Final model report |
| `step_08_xgboost_fix/` | Encoding results |
| `step_09_advanced/` | Optuna/stacking results |
| `step_10_paper_replication/` | Paper replication results |
| `step_11_clinical_validation/` | Clinical metrics, DCA, paragraphs |
| `step_13_reliability_diagnostics/` | Calibration diagnostics |
| `step_ext_val_hrs/` | HRS validation outputs |
| `step_ext_val_elsa/` | ELSA W8 validation outputs |
| `step_ext_val_elsa_wave9/` | ELSA W9 sensitivity outputs |
| **`step_charls_raw16/`** | 🆕 CHARLS raw16 metrics + plots |
| **`step_compare_raw16/`** | 🆕 Cross-cohort comparison + plots |
| **`step_unified_aging/`** | 🆕 Aging proxy summary + plots |
| **`step_elsa_ras/`** | 🆕 RAS experiment results |

### Manuscripts
| File | Status |
|------|--------|
| `manuscript_v3_with_elsa.tex` | Current version — needs revision |
| `manuscript_v4_corrected.tex` | 🆕 Template with corrected narrative |
| `supplement_methods_engineering.tex` | Methods supplement |
| `supplement_tripod_checklist.tex` | TRIPOD checklist |
| `draft_external_validation.tex` | Draft external validation section |
| `first-review-raport.tex` | First review response |
| `submission_strategy.md` | Journal targeting plan |

---

*Document generated: May 2026. All new steps (CHARLS raw16, comparison, unified aging, ELSA RAS) were created and executed on May 29, 2026.*
