# Experiment Log - Phase 2 (FINAL)

---

## Exp-P2-004 | 2026-04-14 | SHAP + Feature Impact + Paper Replication

**Amac**: RF'in 0.89'unu dogrulamak, feature impact'i anlamak, makale replikasyon

**Sonuclar**:

### SHAP Analizi (XGBoost)
1. Biological Age: SHAP=0.44 (dominant ama makalenin 0.6'sindan dusuk)
2. Gender: SHAP=0.42
3. BMI: SHAP=0.40
4. Residence: SHAP=0.32
5. Education: SHAP=0.21

**Paper karsilastirma**: Makale SHAP>0.6 buldu, biz SHAP=0.44 bulduk

### LASSO Feature Selection
- 21/27 feature secildi (C=0.1)
- Top 5: Gender(0.42), Residence(-0.33), CVD(0.22), Education(-0.14), Drink(0.10)
- **BA_KDM_log LASSO coef = 0** (secilmedi!)
- **BMI LASSO coef = 0** (BMI_New secildi)

### Feature Impact Comparison
| Feature Set | AUROC | Fark |
|-------------|-------|------|
| raw17 (BEST) | 0.8937 | - |
| raw17+KDM_orig | 0.8652 | -0.029 |
| raw17+KDM_log | 0.8626 | -0.031 |
| raw17+KDM_quartile | 0.8422 | -0.052 |
| raw17+BIR | 0.7949 | -0.099 |
| raw17+biomarkers | 0.7269 | -0.167 |
| raw17+all | 0.6958 | -0.198 |

**Sonuc**: Ekstra feature eklemek her zaman ZARARLI

### Paper Replication (Top 5)
| Config | AUROC | F1 |
|--------|-------|-----|
| raw17 + RF | 0.8937 | 0.76 |
| raw17 + RF_tuned | 0.8924 | - |
| paper_11 + RF | 0.8459 | 0.70 |
| raw17 + XGB_scaled | 0.8327 | 0.55 |
| raw17 + XGB | 0.8048 | 0.31 |

### Subgroup Analysis
| Grup | n | KOA% | AUROC |
|------|---|------|-------|
| CVD+ | 1304 | 23.2% | **0.90** |
| Female | 6443 | 16.9% | 0.89 |
| Urban | 7954 | 15.8% | 0.89 |
| All | 12329 | 13.3% | 0.86 |
| Male | 5886 | 9.4% | 0.82 |

---

## Exp-P2-005 | 2026-04-14 | Hyperparameter Tuning + Final Model

**Amac**: En iyi modeli optimize etmek ve final raporu yazmak

**Sonuclar**:
| Model | AUROC (CV) |
|-------|-----------|
| RF_tuned | **0.8924** |
| Ensemble (RF+XGB+LGBM) | 0.8740 |
| XGB_tuned | 0.8497 |
| LGBM_tuned | 0.8370 |
| CB_tuned | 0.8239 |

**RF Best Params**: n_estimators=500, max_depth=None, min_samples_leaf=1

### RCS-like Analysis (BA-KOA)
- BA OR per year: 1.012
- BA AUROC (univariate): 0.5356 (dusuk!)
- Q4 vs Q1 OR: 1.414 (makale: 1.4519 benzer)
- Threshold: ~55-58 yasinda KOA artis egilimi
- Paper bulgusu (BA ~66.7 threshold) kismen dogrulandi

---

## FINAL MODEL

**Model**: RandomForest (n_estimators=500, raw17 features)
**AUROC**: 0.8924 (5-fold CV)
**F1**: ~0.76
**Dataset**: Sheet1, 12,329 rows, 13.3% KOA

**Makale ile karsilastirma**:
| Metrik | Paper | Bizim | Fark |
|--------|-------|-------|------|
| AUROC | 0.9078 | 0.8924 | -0.015 |
| Feature | 11 (LASSO+KDM-BA) | 17 (raw) | - |
| Target | KOA 10.5% | KOA 13.3% | - |
| Sample | 9,505 | 12,329 | - |

**Kalan fark kaynaklari**: One-hot encoding, LASSO feature selection, XGBoost hiperparametreleri

---

*Last Updated: 2026-04-14*
*Phase 2 Experiments: 5 completed*
*Best Result: RF_tuned, AUROC = 0.8924*