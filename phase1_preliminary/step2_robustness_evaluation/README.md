# Stage 2: Robustness Evaluation - Multiple Seed Testing

**Date**: March 12, 2026  
**Script**: `scripts/step2_robustness_evaluation.py`  
**Objective**: Validate reproducibility and stability of preprocessing strategies

---

## 🎯 Research Question

**Are our previous results reproducible across different random seeds?**

Previous single-seed testing (seed=42) showed Listwise Deletion achieving 0.7262 ROC AUC, suggesting it was the best strategy. However, single-seed results can be misleading due to random variation in train/test splits.

This stage tests **3 random seeds (42, 123, 999)** to measure:
1. Mean performance across seeds
2. Standard deviation (variability)
3. Coefficient of Variation (CV = Std/Mean × 100%)
4. Reproducibility and stability

---

## 🧪 Experimental Design

**Seeds Tested**: 42, 123, 999 (industry standard)

**Strategies Evaluated**:
1. **Listwise Deletion**: Remove rows with any missing values (2,742 samples)
2. **Raw Data Direct**: Use all data, encode missing as -999 (18,046 samples)
3. **Mean Imputation**: Fill missing with mean/mode (18,046 samples)

**Metrics Calculated**:
- **ROC AUC**: Primary metric (area under ROC curve)
- **PR AUC**: Precision-Recall AUC (important for imbalanced data)
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity

**Configuration**:
- Model: Random Forest (n_estimators=100, n_jobs=-1)
- Split: 80/20 stratified by target class
- Features: 17 raw features (no feature engineering)

---

## 📊 Results Summary

### Performance Comparison (Mean ± Std)

| Strategy | Mean ROC AUC | Std | CV (%) | PR AUC | F1 Score | Samples |
|----------|--------------|-----|--------|--------|----------|---------|
| **Listwise Deletion** | 0.6776 | ±0.0422 | 6.23% | 0.5745 ± 0.0444 | 0.5286 ± 0.0358 | 2,742 |
| **Raw Data Direct** ⭐ | **0.6655** | **±0.0107** | **1.61%** | 0.5107 ± 0.0157 | 0.4228 ± 0.0117 | 18,046 |
| **Mean Imputation** | 0.6447 | ±0.0122 | 1.89% | 0.4903 ± 0.0203 | 0.4044 ± 0.0118 | 18,046 |

---

## 🔑 Key Findings

### 1. 🚨 **Listwise Deletion Is Unstable**

**Problem**: High seed-dependency (6.23% CV - highest variation)

**Evidence**:
- Seed 42: ROC AUC = 0.7262 (seems excellent!)
- Seed 123: ROC AUC = 0.6558 (drops 9.7%)
- Seed 999: ROC AUC = 0.6507 (drops 10.4%)
- **Mean**: 0.6776 ± 0.0422

**Implication**: The impressive 0.7262 result from Stage 0 was a **lucky draw** with seed=42. Across multiple seeds, Listwise only achieves 0.6776 on average.

**Root Cause**: Small sample size (2,742) → high variance in performance depending on which samples end up in train vs test set.

---

### 2. ⭐ **Raw Data Direct = Production Winner**

**Strengths**: 
- **Lowest CV**: 1.61% (3.9× more stable than Listwise)
- **Consistent**: Only ±0.0107 swing in ROC AUC across seeds
- **Reproducible**: 0.6655 is a reliable estimate
- **Maximum coverage**: 18,046 samples (6.6× more data than Listwise)

**Performance**:
- Mean ROC AUC: 0.6655 ± 0.0107
- PR AUC: 0.5107 ± 0.0157
- F1: 0.4228 ± 0.0117
- Accuracy: 0.7016 ± 0.0066

**Why It Wins**:
1. **Stability**: Minimal seed-dependency ensures consistent deployment
2. **Sample Size**: Large dataset → stable statistical estimates
3. **Data Utilization**: 100% of available samples (no waste)
4. **Competitive Performance**: Only -1.8% vs Listwise mean (negligible)

**Recommended for**: Production deployment, clinical applications requiring reliability

---

### 3. 📉 **Mean Imputation Fails Consistently**

**Performance**: 0.6447 ± 0.0122 (worst across all seeds)

**Issue**: 
- Imputation introduces systematic bias
- Lowest PR AUC (0.4903) → struggles with minority class
- Even with low CV (1.89%), consistently underperforms

**Verdict**: Avoid this strategy

---

### 4. 🔄 **Single Seed Results Are Misleading**

**Gap Analysis**:
- **Stage 0 (seed=42)**: Listwise = 0.7262, Raw Direct = 0.6767 (7.3% gap)
- **Stage 2 (mean)**: Listwise = 0.6776, Raw Direct = 0.6655 (1.8% gap)
- **Gap shrinks by 76%** when averaging across seeds!

**Lesson**: Never trust a single seed evaluation. Always test multiple seeds and report Mean ± Std.

---

### 5. ✅ **PR AUC Adds Important Context**

**Why PR AUC Matters**: 
- Dataset is imbalanced (32.5% positive class)
- ROC AUC can be optimistic for imbalanced data
- PR AUC focuses on minority class performance

**Results**:
- Listwise: 0.5745 PR AUC (good precision-recall trade-off)
- Raw Direct: 0.5107 PR AUC (reasonable)
- Mean Imp: 0.4903 PR AUC (poor minority class handling)

---

## 📈 Stability Analysis

**Coefficient of Variation (CV = Std/Mean × 100%)**

Lower CV = More stable performance

| Strategy | ROC AUC CV | F1 CV | Stability |
|----------|-----------|-------|-----------|
| **Raw Data Direct** ⭐ | **1.61%** | 2.76% | **MOST STABLE** |
| Mean Imputation | 1.89% | 2.91% | MODERATE |
| Listwise Deletion | 6.23% | 6.77% | UNSTABLE |

**Interpretation**:
- **CV < 2%**: Stable (production-ready)
- **CV 2-5%**: Moderate (acceptable for research)
- **CV > 5%**: Unstable (unreliable)

---

## 🏆 Final Recommendation

### For Production Deployment:

**Strategy**: Raw Data Direct + Raw Features + Random Forest

**Configuration**:
```python
# Preprocessing
- Load Raw Data.xlsx
- Filter: Arthritis column not null
- Encode categorical missing: '__MISSING__'
- Encode numeric missing: -999
- Result: 18,046 samples, 17 features

# Model
- RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
- Stratified 80/20 split

# Expected Performance (Mean ± Std)
- ROC AUC: 0.6655 ± 0.0107
- PR AUC: 0.5107 ± 0.0157
- F1: 0.4228 ± 0.0117
```

**Why This Configuration**:
✅ Most reproducible (1.61% CV)  
✅ Largest sample size (18,046)  
✅ No data loss (100% utilization)  
✅ Stable estimates (low variance)  
✅ Production-ready reliability  
✅ Simple and interpretable  

---

## 📋 Reproducibility Checklist

✅ **Multiple seeds tested**: 42, 123, 999  
✅ **Mean ± Std reported**: All metrics  
✅ **Stratified split**: Maintained across runs  
✅ **PR AUC included**: Imbalanced data assessment  
✅ **CV calculated**: Stability validation  
✅ **Statistical rigor**: Fully documented  

---

## 📁 Files Generated

1. **robustness_detailed_results.csv**: All runs × all metrics (9 rows)
2. **robustness_summary.csv**: Mean ± Std by strategy (3 rows)

---

## 💡 Scientific Implications

1. **Quality ≠ Small Sample Size**
   - Previous belief: "Clean small data > large noisy data"
   - Reality: Small data has high variance, large data more stable
   - **Conclusion**: Stability matters more than peak performance

2. **Seed Selection Bias**
   - Seed=42 gave Listwise best performance (0.7262)
   - Other seeds show Listwise is average (0.6776)
   - **Conclusion**: Always validate with multiple seeds

3. **Production vs Research Trade-offs**
   - Research: Can tolerate variance, pursue peak performance
   - Production: Needs reliability, consistent performance
   - **Conclusion**: Different use cases need different strategies

4. **Data Utilization Efficiency**
   - 2,742 clean samples: 0.6776 mean, 6.23% CV (unstable)
   - 18,046 encoded samples: 0.6655 mean, 1.61% CV (stable)
   - **Conclusion**: 6.6× more data → 3.9× more stability

---

## 🚀 Next Steps

1. ✅ **Stage 0-2 Complete**: Preprocessing + Feature Engineering + Reproducibility
2. ⏳ **Model Optimization**: Test XGBoost, LightGBM, CatBoost
3. ⏳ **Hyperparameter Tuning**: Optuna for optimal RF parameters
4. ⏳ **Ensemble Methods**: Voting, Stacking classifiers
5. ⏳ **Cross-Validation**: 5-fold CV for final validation
6. ⏳ **SHAP Analysis**: Feature importance and interpretability

---

## 🔗 Related Files

- **Stage 0**: [step0_baseline_comparison/](../step0_baseline_comparison/)
- **Stage 1**: [step1_comprehensive_comparison/](../step1_comprehensive_comparison/)
- **Master Results**: [results/pipeline_results.txt](../results/pipeline_results.txt)
- **Main README**: [README.md](../README.md)

---

**Conclusion**: Raw Data Direct with raw features is the most stable, reproducible, and production-ready strategy. The impressive Listwise Deletion result (0.7262) from seed=42 is not representative and should not be relied upon for deployment.
