# BioAge-KOA-Prediction

**Predicting Symptomatic Knee Osteoarthritis Risk from CHARLS Data**  
Internal Validation with Early Unseen Holdout, Calibration, and Decision Analysis

---

## Study Overview

This project implements a rigorous internal-validation machine-learning pipeline for predicting symptomatic knee osteoarthritis (KOA) in a CHARLS-derived cohort (n = 12,329; KOA prevalence 13.3%). Reporting follows the TRIPOD guidance and the PROBAST framework.

**Primary model:** Random Forest (n_estimators=300) trained on the raw17 feature set (17 sociodemographic and clinical variables).

**Key design choices:**
- Early 80/20 stratified unseen holdout split, created before any model development
- Train-fitted missing-data imputation (no leakage)
- 5-fold stratified OOF validation on the internal partition
- Calibration, Decision Curve Analysis (DCA), and SHAP interpretability

**Primary results (Random Forest + raw17):**

| Setting | Variant | ROC-AUC | PR-AUC | Brier | ECE |
|---------|---------|---------|--------|-------|-----|
| Internal OOF | Raw | 0.847 | 0.688 | 0.070 | 0.045 |
| Internal OOF | Isotonic | 0.839 | 0.676 | 0.069 | 0.042 |
| Unseen holdout | Raw | 0.907 | 0.791 | 0.053 | 0.056 |
| Unseen holdout | Isotonic | 0.900 | 0.779 | 0.051 | 0.060 |

---

## Project Structure

```
BioAge-KOA-Prediction/
│
├── scripts/                          # Active manuscript-facing pipeline
│   ├── step_01_data_prep.py          # Data loading, KOA definition, early split
│   ├── step_02_baseline.py           # Baseline model comparison
│   ├── step_03_kdm_ba.py             # KDM biological age computation
│   ├── step_04_ba_impact.py          # SHAP analysis and feature impact
│   ├── step_11_clinical_validation.py # Calibration + DCA (primary evaluation)
│   └── step_13_reliability_diagnostics.py # Reliability diagrams (post-hoc)
│
├── step_01_data_prep/                # [generated] Split CSVs and metadata
├── step_02_baseline/                 # [generated] Baseline results
├── step_03_kdm_ba/                   # [generated] KDM-enriched dataset
├── step_04_ba_impact/                # [generated] SHAP outputs
├── step_11_clinical_validation/      # [generated] Metrics, calibration, DCA
├── step_13_reliability_diagnostics/  # Reliability diagrams (pre-generated)
│   ├── reliability_diagram_raw17_internal_oof.png
│   ├── reliability_diagram_raw17_unseen_holdout.png
│   └── calibration_slope_diagnostic_note.txt
│
├── manuscript_main.tex               # Main manuscript (LaTeX)
├── supplement_methods_engineering.tex # Engineering workflow supplement
├── supplement_tripod_checklist.tex   # TRIPOD checklist supplement
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Data Requirements

- **Source:** "Raw Data of Biological Age" from CHARLS, published openly by Fanyu Fu on Mendeley Data
- **Download:** [https://data.mendeley.com/datasets/3rv7mf5pv9/1](https://data.mendeley.com/datasets/3rv7mf5pv9/1)
- **License:** CC BY 4.0 — free to use and reproduce with attribution
- **Place the file at:** `BioAge-KOA-Prediction/Raw Data .xlsx`

The dataset used in this study:
- 15,545 baseline rows, 28 columns
- After exclusion (position_knees non-missing): **Dataset A** (n = 12,329, KOA prevalence 13.32%)
- Biological Age ≥ 55 filter: **Dataset B** (n = 7,635) for sensitivity analyses

**Citation for dataset:** Fu F. Raw Data of Biological Age. Mendeley Data, V1. 2025. doi:10.17632/3rv7mf5pv9.1

---

## Installation

```powershell
# 1. Create and activate a virtual environment (optional but recommended)
python -m venv env
.\env\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Execution Order

Run scripts from the **`BioAge-KOA-Prediction/`** directory:

```powershell
# Step 1: Data preparation and early split
python scripts/step_01_data_prep.py

# Step 2: Baseline model comparison
python scripts/step_02_baseline.py

# Step 3: KDM biological age computation
python scripts/step_03_kdm_ba.py

# Step 4: SHAP feature impact analysis
python scripts/step_04_ba_impact.py

# Step 11: Clinical validation (calibration + DCA) - PRIMARY RESULTS
python scripts/step_11_clinical_validation.py

# Step 13: Reliability diagnostics (post-hoc, reads Step 11 outputs)
python scripts/step_13_reliability_diagnostics.py
```

**Total estimated runtime:** 15–30 minutes (depends on CPU; Step 04 SHAP is the slowest).

---

## Key Output Files

| File | Description |
|------|-------------|
| `step_01_data_prep/dataset_A_internal_train.csv` | Internal training partition (n ≈ 9,863) |
| `step_01_data_prep/dataset_A_unseen_test.csv` | Unseen holdout partition (n ≈ 2,466) |
| `step_11_clinical_validation/clinical_metrics_summary.csv` | Full metric table (all configs × variants × eval sets) |
| `step_11_clinical_validation/calibration_summary.csv` | Calibration metrics (Brier, ECE, MCE, slope, intercept) |
| `step_11_clinical_validation/dca_clinical_decision_summary.csv` | DCA 10%–30% threshold band summary |
| `step_13_reliability_diagnostics/reliability_diagram_raw17_internal_oof.png` | Reliability diagram (OOF) |
| `step_13_reliability_diagnostics/reliability_diagram_raw17_unseen_holdout.png` | Reliability diagram (holdout) |

---

## Feature Set (raw17)

| Variable | Type | Description |
|----------|------|-------------|
| wave | integer | Survey wave |
| Time | integer | Survey year |
| Gender | binary | Sex (1=male, 2=female) |
| Age_New | ordinal | Age bucket (1=45–59, 2=≥60) |
| Marital | binary | Marital status (1=married, 2=other) |
| Education | ordinal | Educational attainment (1=low, 2=medium, 3=high) |
| Residence | binary | Urban vs rural (1=urban, 2=rural) |
| Hypertension | binary | Physician-diagnosed (0/1) |
| Dyslipidemia | binary | Physician-diagnosed (0/1) |
| Diabetes | binary | Physician-diagnosed (0/1) |
| Cancer | binary | Physician-diagnosed (0/1) |
| CVD | binary | Physician-diagnosed cardiovascular disease (0/1) |
| Smoke | binary | Ever smoker (0/1) |
| Drink | binary | Ever drinker (0/1) |
| BMI | continuous | Measured body-mass index (kg/m²) |
| BMI_New | ordinal | BMI category (1=underweight/normal, 2=overweight, 3=obese) |
| Biological Age | continuous | CHARLS-supplied biological-age field |

---

## Reproducibility Notes

1. **Random seed:** All splits and models use `random_state=42`.
2. **Split:** A single 80/20 stratified split is created in Step 01 and reused by all downstream steps.
3. **No hyperparameter tuning:** The primary Random Forest uses `n_estimators=300` and scikit-learn defaults. No grid search was applied to primary model claims.
4. **Imputation:** `SimpleImputer` (median for numeric, most-frequent for categorical) is fitted on training data only within each CV fold.
5. **Leakage controls:** `Arthritis` and `position_knees` (the two source columns defining the KOA outcome) are excluded from the feature set. Rows with missing `position_knees` are dropped.

---

## Limitations and Next Steps

This study provides **internal validation** evidence only. Key limitations:

- Evidence is limited to one overall cohort (CHARLS); results may not generalize to other populations or healthcare systems.
- The unseen holdout, while more rigorous than single-split reporting, is drawn from the same cohort — it is not a true external validation.
- Comorbidity and lifestyle variables are self-reported.
- Bootstrap confidence intervals for discrimination and calibration metrics are planned for the next submission revision.

**Next milestone:** External validation on an independent cohort (e.g., ELSA, HRS, KLoSA) is required before clinical deployment claims can be made.

---

## Citation

If you use this code or the methodology described here, please cite the companion manuscript:

> BioAge-KOA-Prediction Team. Predicting Symptomatic Knee Osteoarthritis Risk from CHARLS Data: Internal Validation with Early Unseen Holdout, Calibration, and Decision Analysis. *[Journal]*, 2026.

---

## Manuscript and Supplements

| Document | Description |
|----------|-------------|
| `manuscript_main.tex` | Main paper body (LaTeX) |
| `supplement_methods_engineering.tex` | Engineering workflow details |
| `supplement_tripod_checklist.tex` | TRIPOD 22-item checklist mapping |
