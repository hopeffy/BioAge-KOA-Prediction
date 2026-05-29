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

---

## Exp-P2-006 | 2026-04-21 | Step 11: Klinik Validasyon (Kalibrasyon + DCA)

**Amac**: Kalibrasyon, DCA ve isotonic calibration ile modelin klinik guvenilirligini degerlendirmek

**Sonuclar (raw17, raw)**:
| Set | ROC-AUC | PR-AUC | Brier | ECE | Slope | DCA (avoided/100) |
|-----|---------|--------|-------|-----|-------|-------------------|
| Internal OOF | 0.847 | 0.688 | 0.070 | 0.045 | 0.989 | 56.0 |
| Unseen holdout | 0.907 | 0.791 | 0.053 | 0.056 | 1.216 | 63.8 |

**Isotonic calibration**:
- Internal OOF: ECE 0.045→0.042 (iyilesme)
- Holdout: ECE 0.056→0.060 (benzer)
- Cox slope isotonic'te paradoksal olarak 1.6-1.7'ye yukseldi (logit compression artifact)

**DCA**: 10%-30% esik bandinda tum modeller treat-all ve treat-none'dan ustun

---

## Exp-P2-007 | 2026-04-24 | Step 13: Reliability Diagnostics

**Amac**: Isotonic calibration slope anomalisini (1.6-1.7) arastirmak

**Bulgu**: Cox logistic recalibration slope, isotonic regression'un piecewise-constant ciktisi nedeniyle logit compression'a ugruyor. Bu bilinen bir pitfall (Niculescu-Mizil & Caruana 2005, Van Calster 2019). Dogru metrikler: ECE/Brier + decile reliability diagram.

**Oneri**: Isotonic varyantlar icin ECE ve reliability diagram kullan, Cox slope'u tek basina yorumlama.

---

## Exp-P2-008 | 2026-04-24 | HRS External Validation (raw16)

**Amac**: CHARLS'te egitilen RF modelini HRS Wave 13'e (ABD, n=20,605) uygulamak

**KOA proxy**: Mevcut artrit + diz cokme zorlugu (prevalans %33.9, CHARLS'ten cok daha yuksek)

**Sonuclar (raw16, 12/17 feature)**:
- ROC-AUC: 0.604 (CHARLS holdout 0.907'den -0.303)
- PR-AUC: 0.399
- Calibration slope: 0.184 (kabul edilemez)
- DCA: Negatif (treat-all'dan kotu)

**Eksik feature'lar**: Biological Age, Residence, Dyslipidemia, wave identifier

---

## Exp-P2-009 | 2026-05-02 | ELSA External Validation (raw16) + Isotonic

**Amac**: CHARLS modelini ELSA Wave 8'e (UK, n=8,416) uygulamak

**KOA proxy**: Hayat boyu artrit tanisi + diz cokme zorlugu (prevalans %24.9)

**Sonuclar (raw16, 14/17 feature)**:
| Variant | ROC-AUC | PR-AUC | Brier | ECE | Slope |
|---------|---------|--------|-------|-----|-------|
| Raw | 0.604 | 0.297 | 0.201 | 0.097 | 0.271 |
| Isotonic (split-calibrate-evaluate) | 0.606 | 0.305 | 0.182 | 0.005 | 0.938 |

**Kritik bulgu**: Isotonic calibration, slope'u 0.27→0.94 ve ECE'yi 0.097→0.005 duzeltti AMA ROC-AUC'yi iyilestirmedi (0.604→0.606). Kalibrasyon ayri, ayrim gucu ayri.

**HRS ile kiyas**: AUC neredeyse ayni (0.604 vs 0.604) — iki farkli kita, iki farkli KOA tanimi, farkli feature sayilari ama ayni transportability tavan.

---

## Exp-P2-010 | 2026-05-02 | ELSA Wave 9 Sensitivity

**Amac**: ELSA Wave 9 (2018-2019) ile temporal robustness kontrolu

**Not**: Wave 9'da BMI degiskeni yok — CHARLS'ten full impute edildi.

**Sonuclar (n=8,683, KOA=23.3%)**:
- Raw: AUC=0.596, Slope=0.187
- Isotonic: AUC=0.615, Slope=1.225, ECE=0.009

Wave 8 ile tutarli sonuclar. BMI imputasyonu raw discrimination'i hafif dusurmus.

---

## 🆕 Exp-P3-001 | 2026-05-29 | CHARLS raw16 Baseline

**Amac**: CHARLS modelini ELSA'daki ayni raw16 feature set'i ile degerlendirip, AUC kaybinin ne kadarinin feature-subset (BA eksikligi) vs cross-cultural oldugunu ayristirmak.

**Metod**: CHARLS raw17'den Biological Age, wave, Time cikarildi; wave_id eklendi. Ayni holdout split, ayni RF(n=300, rs=42).

**Sonuclar**:
| Set | ROC-AUC | PR-AUC | Brier | ECE | Slope |
|-----|---------|--------|-------|-----|-------|
| Internal OOF (raw) | 0.804 | 0.482 | 0.086 | 0.062 | 0.366 |
| Unseen holdout (raw) | 0.869 | 0.585 | 0.066 | 0.064 | 0.526 |

**CHARLS raw17 → raw16 drop**:
- Internal: 0.847 → 0.804 (Δ=-0.043)
- Holdout: 0.907 → 0.869 (Δ=-0.038)

**Kritik bulgu**: Feature-subset penalty sadece ~0.038. CHARLS raw16 holdout'tan ELSA'ya dusus ~0.265. Yani toplam AUC kaybinin **%87'si cross-cultural, %13'u feature-subset**.

---

## 🆕 Exp-P3-002 | 2026-05-29 | AUC Gap Decomposition (Cross-Cohort Comparison)

**Amac**: Tum kohortlari tek tabloda birlestirip AUC kaybini bilesenlerine ayirmak.

**Complete cross-cohort table**:
| Cohort | n | KOA% | Features | ROC-AUC | Δ from raw17 holdout |
|--------|---|------|----------|---------|---------------------|
| CHARLS raw17 Internal | 9,863 | 13.3% | 17 | 0.847 | — |
| CHARLS raw17 Holdout | 2,466 | 13.3% | 17 | 0.907 | baseline |
| CHARLS raw16 Holdout | 2,466 | 13.3% | 15 | 0.869 | -0.038 (13%) |
| ELSA W8 raw16 | 8,416 | 24.9% | 14 | 0.604 | -0.303 |
| HRS W13 raw16 | 20,605 | 33.9% | 12 | 0.604 | -0.303 |

**Decomposition**:
- Feature-subset component (raw17→raw16): **-0.038 (13%)**
- Cross-cultural component (CHARLS→ELSA/HRS): **-0.265 (87%)**

**Makale revizyonu gerekli**: Mevcut manuscript "BA eksikligi AUC kaybinin cogunu acikliyor" diyor — bu YANLIS. Cross-cultural faktorler (outcome proxy mismatch, populasyon farki) dominant.

---

## 🆕 Exp-P3-003 | 2026-05-29 | Reduced Aging Score (RAS) Experiment

**Amac**: CHARLS ve ELSA'da ortak bulunan 4 biyobelirtecten (CRP, HbA1c, SBP, TC) Reduced Aging Score hesaplayip, ELSA'ya eklemek ve AUC kazanimini olcmek.

**RAS formulu**: RAS = β₀ + β₁·log(CRP) + β₂·log(HbA1c) + β₃·log(TC) + β₄·SBP

**CHARLS training**:
- R² = 0.417 (RAS, KDM-BA varyansinin %42'sini yakaliyor)
- RAS vs KOA korelasyonu: r = 0.014 (KDM-BA vs KOA: r = 0.018)
- Sheet1 BA vs KOA: r = 0.038 (hala en guclu)

**ELSA uygulamasi** (n=8,416, 2,499 complete biomarker):
| Model | ROC-AUC | Δ |
|-------|---------|---|
| raw16 only | 0.6050 | — |
| raw16 + RAS | 0.6111 | +0.0061 |

**Sonuc**: RAS, elde edilebilir en iyi BA proxy'si olmasina ragmen sadece 0.006 AUC kazandiriyor. Bu, toplam cross-cultural gap'in (~0.265) sadece %2'si. BA eksikligi hikayesi istatistiksel olarak desteklenmiyor.

**Implication**: Makalenin "BA eksikligi ana faktor" anlatisi tamamen degismeli. Asil sorun outcome tanimi uyumsuzlugu ve populasyon farkliliklari.

---

## 🆕 Exp-P3-004 | 2026-05-29 | Unified Aging Computation

**Amac**: CHARLS, ELSA, HRS icin ayni yontemle Biological Age hesaplamaya calismak.

**Bulgular**:
- **CHARLS**: Full KDM-BA hesaplanabilir (8 biomarker mevcut)
- **ELSA**: Nurse visit dosyasinda 4/8 biomarker var (CRP, HbA1c, TC, SBP). Creatinine, BUN, platelets EKSIK.
- **HRS**: VBS'de ~4/8 biomarker var. Creatinine, BUN, platelets EKSIK. Cystatin C var ama KDM'de kullanilmiyor.

**Sonuc**: Full KDM-BA sadece CHARLS'te mumkun. ELSA ve HRS icin RAS gibi reduced proxy'ler kullanilabilir ama bunlarin KOA ile iliskisi cok zayif.

---

## GENEL DEGERLENDIRME (May 2026)

### Bugune Kadar Uretilen Tum Kanitlar

1. **Internal validation basarili**: CHARLS raw17, holdout'ta AUC=0.907, PR-AUC=0.791, pozitif DCA
2. **External validation zayif**: ELSA ve HRS'de AUC ~0.60
3. **Feature-subset penalty kucuk**: CHARLS icinde raw17→raw16 sadece -0.038 AUC
4. **Cross-cultural dominant**: Toplam kaybin %87'si kulturlerarasi faktorler
5. **RAS ise yaramiyor**: En iyi BA proxy'si sadece +0.006 AUC kazandiriyor
6. **BA-KOA iliskisi zayif**: KDM-BA vs KOA r=0.018; Sheet1 BA vs KOA r=0.038

### Makale Icin Onerilen Revizyonlar

- Abstract: "BA eksikligi" yerine "cross-cultural faktorler dominant" vurgusu
- Results: AUC gap decomposition tablosu ekle
- Results: RAS deney sonucu ekle (Δ=+0.006)
- Discussion: BA'nin rolu hakkinda duzeltilmis yorum
- Limitations: Outcome proxy uyumsuzlugu en buyuk limitasyon olarak vurgulanmali

### Yeni Step Klasorleri
- `step_charls_raw16/` — CHARLS raw16 baseline
- `step_compare_raw16/` — Cross-cohort karsilastirma
- `step_unified_aging/` — Unified BA computation
- `step_elsa_ras/` — RAS experiment

### Yeni Dosyalar
- `scripts/step_charls_raw16.py`
- `scripts/step_compare_raw16.py`
- `scripts/step_unified_aging.py`
- `scripts/step_elsa_ras.py`
- `PROJECT_DOCUMENTATION.md` — Tum projenin ingilizce dokumantasyonu
- `manuscript_v4_corrected.tex` — Duzeltilmis anlatili manuscript sablonu

---

*Last Updated: 2026-05-29*
*Phase 2 Experiments: 10 completed*
*Phase 3 Experiments (Cross-Cultural): 4 completed*
*Best Internal Result: RF raw17, AUROC = 0.907 (holdout)*
*Best External Result: ELSA raw16+RAS, AUROC = 0.611*
*Key Finding: AUC gap %87 cross-cultural, %13 feature-subset*