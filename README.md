# KOA (Knee Osteoarthritis) Prediction and Clinical Translation Pipeline

Last Updated: 2026-04-18

This repository contains an end-to-end KOA risk modeling workflow, from preprocessing experiments to clinical interpretability and publication-ready reporting.

## Scope

- Data preparation and baseline modeling
- Multi-stage feature engineering and robustness evaluation
- Biological aging signal integration (KDM, adapted PhenoAge)
- Real SHAP interpretability and SHAP interaction analysis
- Clinical validation with calibration and Decision Curve Analysis (DCA)
- Automated clinician-facing recommendation text and publication paragraphs

## End-to-End Process Map

| Phase | Main Script(s) | Purpose | Key Output Folder |
|---|---|---|---|
| Step 0 | scripts/step0_baseline_comparison.py, scripts/step0_imputation_comparison.py, scripts/step0_prepare_imputed_dataset.py | Preprocessing strategy comparisons | step0_baseline_comparison, phase1_preliminary |
| Step 1 | scripts/step1_comprehensive_comparison.py, scripts/main_pipeline.py | Baseline + FE comparison | step1_comprehensive_comparison |
| Step 2 | scripts/step2_robustness_evaluation.py | Multi-seed robustness validation | step2_robustness_evaluation |
| Step 3 (old track) | scripts/step3_position_knees_experiments.py, scripts/step3_robust_feature_engineering.py | Feature dominance and robust FE checks | step3_position_knees_experiments |
| Step 4 (old track) | scripts/step4_comprehensive_feature_selection.py, scripts/step4_feature_selection_comparison.py | Feature selection method comparison | step4_feature_selection_comparison |
| Step 5-9 | scripts/step5_domain_specific_features.py, scripts/step6_feature_quality_audit.py, scripts/step7_groupwise_ablation.py, scripts/step8_repeated_cv_validation.py, scripts/step9_composite_score_fe.py | Domain features, quality audit, ablations, repeated CV | phase1_preliminary |
| Step 10+ optimization | scripts/step10_default_model_comparison.py, scripts/step11_to_14_model_optimization.py | Model family comparison + tuning workflow | phase1_preliminary |
| Step 01 (new pipeline) | scripts/step_01_data_prep.py | Build modeling dataset(s) | step_01_data_prep |
| Step 02 (new pipeline) | scripts/step_02_baseline.py | New baseline modeling stage | step_02_baseline |
| Step 03 (new pipeline) | scripts/step_03_kdm_ba.py | KDM and adapted aging clock generation | step_03_kdm_ba |
| Step 04 (new pipeline) | scripts/step_04_ba_impact.py | Impact analysis, real SHAP, SHAP interactions | step_04_ba_impact |
| Step 05-10 (new pipeline) | scripts/step_05_07_final.py, scripts/step_08_xgboost_fix.py, scripts/step_09_advanced_pipeline.py, scripts/step_10_paper_replication.py | Model finalization, advanced runs, replication | step_05_replication to step_10_paper_replication |
| Step 11 (new pipeline) | scripts/step_11_clinical_validation.py | Calibration, DCA, clinical recommendation/report text | step_11_clinical_validation |

## Latest Integrated Updates (April 2026)

### 1) Step 03: Aging Clock Layer Extended

Script: scripts/step_03_kdm_ba.py

Added/updated outputs:
- BA_KDM_log
- BA_KDM_orig
- BIR_log
- BIR_orig
- PhenoAge_Adapted
- PhenoAge_Adapted_Accel
- Quartiles including PhenoAge_Adapted_Qint

Main artifacts:
- step_03_kdm_ba/dataset_A_with_kdm_ba.csv
- step_03_kdm_ba/dataset_B_with_kdm_ba.csv
- step_03_kdm_ba/step03_report.txt

### 2) Step 04: Real SHAP + Clinical Interaction Filtering

Script: scripts/step_04_ba_impact.py

What is now enforced:
- Real SHAP with TreeExplainer (no feature-importance proxy)
- SHAP interaction values for XGBoost
- SHAP waterfall local explanations for representative high-risk/median-risk cases
- Clinical interaction filtering layer for medically meaningful pairs

Clinical interaction filter targets:
- Biological Age
- BMI/BMI_New
- Hypertension
- Gender
- hs-CRP (crp_mg.L / crp_original, when available)

SHAP analyses currently generated:
- raw17
- raw17 + PhenoAge_Adapted
- raw17 + hs-CRP (optional, if crp_mg.L exists)

Main artifacts:
- step_04_ba_impact/shap_importance_xgb.csv
- step_04_ba_impact/shap_importance_xgb_raw17_plus_pheno.csv
- step_04_ba_impact/shap_importance_xgb_raw17_plus_hscrp.csv
- step_04_ba_impact/shap_interactions_xgb_raw17.csv
- step_04_ba_impact/shap_interactions_xgb_raw17_plus_pheno.csv
- step_04_ba_impact/shap_interactions_xgb_raw17_plus_hscrp.csv
- step_04_ba_impact/shap_interactions_clinical_raw17.csv
- step_04_ba_impact/shap_interactions_clinical_raw17_plus_pheno.csv
- step_04_ba_impact/shap_interactions_clinical_raw17_plus_hscrp.csv
- step_04_ba_impact/shap_summary_xgb_raw17.png
- step_04_ba_impact/shap_summary_xgb_raw17_plus_pheno.png
- step_04_ba_impact/shap_summary_xgb_raw17_plus_hscrp.png
- step_04_ba_impact/shap_waterfall_xgb_raw17_high_risk_positive.png
- step_04_ba_impact/shap_waterfall_xgb_raw17_high_risk_negative.png
- step_04_ba_impact/shap_waterfall_xgb_raw17_median_risk_case.png
- step_04_ba_impact/shap_waterfall_cases_raw17.csv
- step_04_ba_impact/shap_interaction_heatmap_xgb_raw17.png
- step_04_ba_impact/shap_interaction_heatmap_xgb_raw17_plus_pheno.png
- step_04_ba_impact/shap_interaction_heatmap_xgb_raw17_plus_hscrp.png
- step_04_ba_impact/step04_report.txt

### 3) Step 11: Clinical Translation + Publication Text Automation

Script: scripts/step_11_clinical_validation.py

What is now included:
- OOF probability evaluation
- Raw vs isotonic comparison
- Calibration tables/curves
- DCA net-benefit curves
- 10%-30% threshold-band clinical decision summary
- Automated recommendation text (clinician-facing)
- Automated publication-ready Methods/Results paragraphs (EN/TR)

Main artifacts:
- step_11_clinical_validation/clinical_metrics_summary.csv
- step_11_clinical_validation/step11_clinical_validation_report.txt
- step_11_clinical_validation/dca_clinical_decision_summary.csv
- step_11_clinical_validation/dca_clinical_recommendation_report.txt
- step_11_clinical_validation/step11_publication_paragraphs.txt
- step_11_clinical_validation/calibration_curve_raw17.png
- step_11_clinical_validation/dca_curve_raw17.png

## Current Key Results Snapshot

Based on latest runs in this workspace:

- Best discrimination among evaluated Step 11 configs:
  - raw17 (raw)
  - ROC-AUC: 0.8916
  - PR-AUC: 0.7689
  - Brier: 0.0581
  - ECE: 0.0462

- DCA clinical utility in 10%-30% threshold band:
  - Best strategy: raw17 (raw)
  - Estimated avoided unnecessary referrals/imaging: 61.9 per 100 patients vs treat-all
  - Relative reduction: 71.4%

- Clinical SHAP interactions (raw17 + hs-CRP example):
  - BMI x Biological Age: INT=0.1622
  - Biological Age x crp_mg.L: INT=0.1255
  - BMI x crp_mg.L: INT=0.0888
  - Gender x Biological Age: INT=0.0619
  - Hypertension x Biological Age: INT=0.0357

## Evidence-Backed Findings (English)

1. Biomechanical stress is a dominant model mechanism.
  Evidence: [SHAP interaction heatmap (raw17 + hs-CRP)](step_04_ba_impact/shap_interaction_heatmap_xgb_raw17_plus_hscrp.png), [Clinical interaction table](step_04_ba_impact/shap_interactions_clinical_raw17_plus_hscrp.csv)

2. Inflammaging signal is explicitly captured as a high-impact interaction.
  Evidence: [SHAP summary (raw17 + hs-CRP)](step_04_ba_impact/shap_summary_xgb_raw17_plus_hscrp.png), [Clinical interaction table](step_04_ba_impact/shap_interactions_clinical_raw17_plus_hscrp.csv)

3. Local explanations are clinically interpretable at patient level.
  Evidence: [Waterfall - high-risk positive case](step_04_ba_impact/shap_waterfall_xgb_raw17_high_risk_positive.png), [Waterfall - high-risk negative case](step_04_ba_impact/shap_waterfall_xgb_raw17_high_risk_negative.png), [Waterfall case index](step_04_ba_impact/shap_waterfall_cases_raw17.csv)

4. The raw17 model provides meaningful net clinical benefit in the 10%-30% threshold band.
  Evidence: [DCA curve (raw17)](step_11_clinical_validation/dca_curve_raw17.png), [DCA decision summary](step_11_clinical_validation/dca_clinical_decision_summary.csv), [Clinical recommendation report](step_11_clinical_validation/dca_clinical_recommendation_report.txt)

5. Probability reliability is documented with explicit calibration diagnostics.
  Evidence: [Calibration curve (raw17)](step_11_clinical_validation/calibration_curve_raw17.png), [Calibration table (raw)](step_11_clinical_validation/calibration_table_raw17_raw.csv), [Calibration table (isotonic)](step_11_clinical_validation/calibration_table_raw17_isotonic.csv)

## Core Figure Set (for Manuscript)

Use these three figure types as the primary visual package:

1. SHAP explanation (waterfall + interaction)
  - step_04_ba_impact/shap_waterfall_xgb_raw17_high_risk_positive.png
  - step_04_ba_impact/shap_waterfall_xgb_raw17_high_risk_negative.png
  - step_04_ba_impact/shap_interaction_heatmap_xgb_raw17_plus_hscrp.png
  - step_04_ba_impact/shap_interactions_clinical_raw17_plus_hscrp.csv

2. Decision Curve Analysis (DCA)
  - step_11_clinical_validation/dca_curve_raw17.png
  - step_11_clinical_validation/dca_clinical_decision_summary.csv
  - step_11_clinical_validation/dca_clinical_recommendation_report.txt

3. Calibration curve
  - step_11_clinical_validation/calibration_curve_raw17.png
  - step_11_clinical_validation/calibration_table_raw17_raw.csv
  - step_11_clinical_validation/calibration_table_raw17_isotonic.csv

## Recommended Execution Order

Use this order for a clean full rerun of the new pipeline:

```powershell
# 1) Activate environment
cd c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data
.\env\Scripts\Activate.ps1

# 2) Build datasets
python scripts/step_01_data_prep.py

# 3) Baseline
python scripts/step_02_baseline.py

# 4) KDM + adapted aging clocks
python scripts/step_03_kdm_ba.py

# 5) SHAP + impact + replication
python scripts/step_04_ba_impact.py

# 6) Clinical validation + DCA + publication text
python scripts/step_11_clinical_validation.py
```

Optional advanced/final stages:

```powershell
python scripts/step_05_07_final.py
python scripts/step_08_xgboost_fix.py
python scripts/step_09_advanced_pipeline.py
python scripts/step_10_paper_replication.py
```

## Environment and Dependencies

Install/update dependencies:

```powershell
cd c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Important packages include:
- scikit-learn
- xgboost
- lightgbm
- catboost
- shap (real SHAP analysis)
- pandas, numpy, matplotlib

## Output Folders at a Glance

- step_03_kdm_ba: aging clocks and datasets with BA extensions
- step_04_ba_impact: SHAP, SHAP interactions, clinical interaction subsets, replication/impact outputs
- step_11_clinical_validation: calibration, DCA, clinical recommendation reports, publication paragraphs

## Notes for Manuscript Preparation

If manuscript drafting starts immediately, use these two files first:
- step_11_clinical_validation/step11_publication_paragraphs.txt
- step_11_clinical_validation/dca_clinical_recommendation_report.txt

For mechanistic interpretability claims, cite:
- step_04_ba_impact/shap_interactions_clinical_raw17.csv
- step_04_ba_impact/shap_interactions_clinical_raw17_plus_hscrp.csv
- step_04_ba_impact/shap_interactions_clinical_raw17_plus_pheno.csv

## Legacy Notes

Older exploratory stage summaries are still preserved in repository outputs and scripts.
This README is now the consolidated master process document for current and future runs.
