# Scripts

Python scripts for KOA prediction pipeline.

---

## 📜 Script Overview

### main_pipeline.py
**Steps**: 1 + 2  
**Purpose**: Raw baseline + Feature engineering + ANOVA F-test

**Includes**:
- Data loading from Raw Data.xlsx
- Data cleaning (Age ≥ 18, missing value handling)
- **Step 1**: Raw baseline with 17 features (80/20 split)
- **Step 2**: Feature engineering (37 new features) + ANOVA F-test (70/30 split)
- Visualizations: ANOVA feature analysis, rankings

**Runtime**: ~2-3 minutes

**Outputs**:
- `step1_raw_baseline/baseline_results.txt`
- `step2_feature_engineering/anova_feature_scores.csv`
- `step2_feature_engineering/*.png` (2 visualizations)

**Usage**:
```powershell
.\env\Scripts\Activate.ps1
cd scripts
python main_pipeline.py
```

---

### step3_position_knees_experiments.py
**Step**: 3  
**Purpose**: Test Position_Knees_Encoded dominance with 4 experiments

**Includes**:
- **Exp 1**: Without Position_Knees_Encoded (43 features)
- **Exp 2**: p < 0.05 features only (28 features)
- **Exp 3**: 0.01 < p < 0.05 features only (3 features)
- **Exp 4**: Hierarchical Clustering representatives (5 features) 🏆

**Runtime**: ~3-5 minutes

**Outputs**:
- `step3_position_knees_experiments/feature_experiments_results.txt`
- `step3_position_knees_experiments/experiments_summary.csv`
- `step3_position_knees_experiments/feature_scores_detailed.csv`
- `step3_position_knees_experiments/feature_clusters.csv`
- `step3_position_knees_experiments/*.png` (2 visualizations)

**Key Finding**: Hierarchical Clustering (5 features) achieves best F1 score (0.4263)

**Usage**:
```powershell
.\env\Scripts\Activate.ps1
cd scripts
python step3_position_knees_experiments.py
```

---

### step4_feature_selection_comparison.py
**Step**: 4  
**Purpose**: Compare 6 feature selection methods comprehensively

**Includes**:
1. **ANOVA F-test** (SelectKBest, top 15)
2. **Mutual Information** (mutual_info_classif, top 15)
3. **L1 LASSO** (LogisticRegression with L1, top 15)
4. **Tree-Based** (RF + XGBoost importance avg, top 15)
5. **RFECV** (Recursive Feature Elimination + CV, optimal 33)
6. **Hierarchical Clustering** (correlation-based, 15 clusters)

**Runtime**: ~5-8 minutes (RFECV is slow due to CV)

**Outputs**:
- `step4_feature_selection_comparison/feature_selection_comparison_results.txt`
- `step4_feature_selection_comparison/feature_selection_methods_comparison.csv`
- `step4_feature_selection_comparison/*.png` (2 visualizations)

**Key Findings**:
- **Best ROC AUC**: L1 LASSO (0.5927) 🥇
- **Best F1/Accuracy**: Hierarchical Clustering (0.4253 / 0.5931) 🥈
- **Consensus features**: BioAge_60_Plus, BMI_x_BioAge (6/6 methods)

**Usage**:
```powershell
.\env\Scripts\Activate.ps1
cd scripts
python step4_feature_selection_comparison.py
```

---

## 🔄 Execution Order

Run scripts in this order for full pipeline:

```powershell
# 1. Activate environment
.\env\Scripts\Activate.ps1
cd scripts

# 2. Run main pipeline (Steps 1-2)
python main_pipeline.py

# 3. Run Position_Knees experiments (Step 3)
python step3_position_knees_experiments.py

# 4. Run feature selection comparison (Step 4)
python step4_feature_selection_comparison.py
```

**Total runtime**: ~10-15 minutes

---

## 📦 Dependencies

All scripts require:
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn
- scipy
- openpyxl

Install via:
```powershell
pip install -r requirements.txt
```

---

## 🎯 Best Results Summary

| Metric | Best Method | Score | Features |
|--------|-------------|-------|----------|
| ROC AUC | L1 LASSO | 0.5927 | 15 |
| F1 Score | Hierarchical Clustering | 0.4263 | 15 |
| Accuracy | Hierarchical Clustering | 0.5931 | 15 |

**Recommended for Production**: Hierarchical Clustering (most balanced)

---

## 🔍 Troubleshooting

### ImportError: No module named 'xxx'
```powershell
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

### FileNotFoundError: Raw Data.xlsx not found
- Ensure `Raw Data .xlsx` is in the project root
- Check file path in script (line ~30-50)

### MemoryError / Performance Issues
- Reduce dataset size for testing
- Comment out visualization sections
- Run one experiment at a time

---

**Last Updated**: 2026-03-10  
**Total Scripts**: 3  
**Next**: Model comparison (Step 5) - 7 models planned
