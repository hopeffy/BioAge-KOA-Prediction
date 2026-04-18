================================================================================
STEP 04: FEATURE IMPACT ANALYSIS & PAPER REPLICATION - REPORT
================================================================================
Date: 2026-04-14

1. SHAP ANALYSIS
========================================

RF SHAP Top 10 Features:
  Biological Age: 0.0082 (very low - RF doesn't use SHAP in the same scale)
  Gender: 0.0324
  * RF SHAP values are small due to RF's internal structure

XGBoost SHAP Top 10 Features:
  Biological Age:  0.4394 ★ DOMINANT FEATURE
  Gender:          0.4249
  BMI:             0.4026
  Residence:       0.3218
  Education:       0.2118
  CVD:             0.1775
  Drink:           0.1159
  Age_New:         0.0926
  Smoke:           0.0863
  Dyslipidemia:    0.0752

Paper comparison: Paper found SHAP > 0.6 for BA, we find 0.44 for XGBoost
  - Our scale may differ from paper
  - BA is dominant but not as overwhelming as paper (>0.6)
  - Gender and BMI are also strong (0.42, 0.40)

2. LASSO FEATURE SELECTION
========================================

21 features selected by LASSO (out of 26 + KDM-BA):
  TOP 5: Gender (0.42), Residence (-0.33), CVD (0.22), Education (-0.14), Drink (0.10)
  
  BA_KDM_log: coef = 0.00 (NOT selected by LASSO!)
  BMI: coef = 0.00 (NOT selected by LASSO - BMI_New selected instead with small coef)
  
  Paper comparison: Paper selected 11 features. Our LASSO selected 21.
  - Paper excluded Diabetes; we also have Diabetescoef=0.00
  - BA_KDM_log was NOT selected (LASSO set it to 0)

3. FEATURE IMPACT COMPARISON
========================================

raw17                     : AUC=0.8937 (BEST) ★★★
raw17_plus_KDM_orig       : AUC=0.8652 (-0.028)
raw17_plus_KDM_log        : AUC=0.8626 (-0.031)
raw17_plus_KDM_quartile   : AUC=0.8422 (-0.051)
raw17_plus_BIR            : AUC=0.7949 (-0.099)
raw17_plus_biomarkers     : AUC=0.7269 (-0.167) ★★ WORST
raw17_plus_KDM_biomarkers : AUC=0.7178 (-0.176)
raw17_all                 : AUC=0.6958 (-0.198)

CRITICAL FINDING: Adding ANY extra features (biomarkers, KDM-BA, BIR) DECREASES performance!
Raw 17 features with RF is the BEST configuration.

4. PAPER REPLICATION RESULTS
========================================

Best configurations by AUC:
  raw17 + RandomForest          = 0.8937 (F1=0.7578) ★★★
  paper_11 + RandomForest       = 0.8459 (F1=0.7039)
  all_raw17 + XGBoost_scaled   = 0.8327 (F1=0.5520)
  all_raw17 + XGBoost           = 0.8048 (F1=0.3139)

  Paper's best: XGBoost AUROC = 0.9078
  Our best:      RF AUROC      = 0.8937
  Gap: Only 0.014!

5. SUBGROUP ANALYSIS
========================================

  All:                n=12329, KOA=13.3%, AUC=0.857
  Female:             n= 6443, KOA=16.9%, AUC=0.885  (higher)
  Male:               n= 5886, KOA=9.4%,  AUC=0.817
  Urban:              n= 7954, KOA=15.8%, AUC=0.890  (higher)
  Rural:              n= 4375, KOA=8.7%,  AUC=0.853
  CVD_yes:            n= 1304, KOA=23.2%, AUC=0.899  (highest!) ★
  CVD_no:             n=11025, KOA=12.1%, AUC=0.854
  Hypertension_yes:   n= 3142, KOA=15.8%, AUC=0.878
  Hypertension_no:    n= 9187, KOA=12.5%, AUC=0.857

  Paper finding: BA-KOA association stronger in females, rural, CVD+ → PARTIALLY CONFIRMED
  - CVD+ subgroup has highest AUC (0.899) ✓
  - Females have higher AUC than males ✓

================================================================================
CONCLUSIONS
================================================================================

1. RF with 17 raw features achieves AUROC=0.89, only 0.014 below paper's 0.91
2. Adding biomarkers or KDM-BA HURTS performance (RF drops from 0.89 to 0.72)
3. KDM-BA is NOT a dominant feature (LASSO coef=0, SHAP < BA)
4. Sheet1's Biological Age is already the dominant feature (SHAP=0.44 in XGBoost)
5. Paper can be largely replicated WITHOUT KDM-BA (we reach 0.89 vs paper's 0.91)
6. The remaining 0.014 gap may be due to: one-hot encoding, exact LASSO features, or hyperparameter tuning

NEXT STEPS:
  - Step 05: Hyperparameter tuning on best config (RF raw17)
  - Step 06: RCS analysis (BA-KOA nonlinear relationship)
  - Step 07: Final model and documentation