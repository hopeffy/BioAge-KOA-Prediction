# Biological Age and Symptomatic KOA: Reproduction and Extension Analysis

## Executive Summary
This study reproduces and extends a published claim that Biological Age (BA) is a strong predictor of symptomatic knee osteoarthritis (KOA). While BA-related measures show statistical association with KOA in prior literature, our experiments highlight a key distinction between statistical association and incremental predictive utility.

Key findings:

- Best raw-feature model (RandomForest): ROC-AUC about 0.89, F1 about 0.75
- Adding BA clocks (KDM, PhenoAge):
- ROC-AUC decreases
- F1 drops substantially (to about 0.29 in some settings)
- BA clocks alone: near-chance discrimination (AUROC about 0.52)
- BA strongly correlates with chronological age (r > 0.90)
- BA shows negligible correlation with KOA outcome (r about 0.02)

Overall, BA appears statistically relevant but not predictively useful in this pipeline.

## 1. Statistical Association vs Predictive Utility

| Aspect | Statistical Association | Predictive Modeling |
|---|---|---|
| Core question | Is BA related to KOA? | Does BA improve prediction? |
| Typical metrics | p-values, odds ratios | ROC-AUC, F1, calibration |
| Sensitivity | High (sample-size dependent) | Depends on generalization gain |
| Failure mode | Significant but trivial effects | No incremental performance gain |
| Interpretation | Epidemiological relevance | Clinical utility |

A variable can be statistically significant while contributing no measurable improvement in out-of-sample prediction.

## 2. Why BA May Be Significant but Not Improve ML Performance

Several mechanisms can explain this discrepancy:

- Redundancy with chronological age: BA clocks are highly age-correlated, so they may add little new information.
- Feature overlap with clinical covariates: existing predictors may already capture BA-related signal.
- Collinearity effects: redundant variables can destabilize decision boundaries and threshold tuning, which can hurt F1.
- Importance-utility mismatch: high feature importance (including SHAP ranking) does not guarantee better ablation performance.
- Different optimization targets: association tests evaluate relationship strength, not incremental predictive contribution.

## 3. Empirical Interpretation of Results

- Strong raw-feature performance suggests most KOA signal is already captured without BA.
- Adding BA clocks consistently reduced performance, indicating negative incremental utility in this dataset.
- BA-only models remained near chance (AUROC about 0.52), confirming weak standalone signal.
- High BA-age correlation (r > 0.90) plus near-zero BA-KOA correlation (r about 0.02) indicates strong age tracking but weak disease specificity.

## 4. Comparison with the Original Study

Original findings reported:

- XGBoost AUROC about 0.91
- BA identified as a top SHAP feature

This reproduction shows:

- Comparable performance can be achieved without BA features
- BA inclusion does not improve prediction and may degrade it

This does not imply the original study is incorrect. It suggests BA utility is context-dependent and sensitive to dataset composition, preprocessing choices, and evaluation design.

## 5. Possible Explanations for Discrepancy

- Dataset shift: differences in population, KOA definition, and measurement protocol
- Evaluation differences: repeated CV versus single split and other validation design choices
- Overfitting risk: high-capacity models may over-credit correlated proxies
- Information leakage risk: subtle preprocessing dependencies can inflate apparent importance
- Prevalence and class balance effects: threshold-dependent metrics (especially F1) are sensitive to distribution shifts

## 6. Key Insight

**Statistical association does not imply predictive utility.**

A feature can be biologically meaningful and statistically significant while still providing no incremental value in a predictive system.

## 7. Reproducibility and Robustness Findings

- Performance degradation after BA inclusion is consistent across tested configurations
- Both discrimination (ROC-AUC) and threshold performance (F1) deteriorate
- BA-only models remain near-random
- Raw-feature models are more stable and robust than BA-augmented variants

## 8. Implications for Clinical Machine Learning

- Feature selection should prioritize incremental predictive value, not statistical ranking alone
- Ablation studies are essential to confirm real feature contribution
- ROC-AUC alone is insufficient; calibration and decision utility must also be evaluated
- Highly correlated biological proxies may add instability without improving decision support

## Conclusion
Biological Age is epidemiologically associated with KOA, but in this reproduction and extension study it did not provide measurable predictive benefit and could degrade model performance.

For clinical ML systems, utility should be defined by out-of-sample improvement and decision relevance, not by statistical association or feature-importance ranking alone.
