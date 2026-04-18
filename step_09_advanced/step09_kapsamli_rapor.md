================================================================================
STEP 09: ADVANCED PIPELINE - CLOSING THE GAP & THE EDGE
================================================================================
Date: 2026-04-14

================================================================================
1. YAPILAN İYİLEŞTİRMELER
================================================================================

Aşama 1: Baseline Düzeltmeleri
---------------------------------
- PR-AUC ve F1-Macro metrikleri eklendi (sadece AUROC değil)
- Optimal threshold: Youden's J ve F1-maximizing threshold arandı
- Orijinal ölçek biyobelirteçler (reverse log-transform) test edildi
- Klinik interaksiyon değişkenleri test edildi:
  * delta_BA = KDM-BA - Chronological Age (yaşlanma ivmesi)
  * CVD_x_KDM_BA = CVD * KDM-BA (risk çarpanı)
  * CVD_x_delta_BA = CVD * delta_BA

Aşama 2: Makalenin Üstüne Çıkmak
----------------------------------
- Optuna ile hiperparametre optimizasyonu:
  * XGBoost: 80 trial, 3-fold CV (PR-AUC + AUROC combined)
  * LightGBM: 50 trial, 3-fold CV
- Stacking Ensemble:
  * RF + XGBoost_opt -> LR meta-learner
  * RF + XGBoost_opt + LightGBM_opt -> LR meta-learner
  * RF_balanced + XGBoost_opt -> LR meta-learner

================================================================================
2. SONUÇLAR
================================================================================

Feature Set'ler:
  FS1_raw17 (17): wave, Time, Gender, Age_New, Marital, Education, Residence,
                   Hypertension, Dyslipidemia, Diabetes, Cancer, CVD, Smoke,
                   Drink, BMI, BMI_New, Biological Age
  FS2_raw17_ohe (19): One-hot encoded version
  FS3_raw17_interactions (21): raw17 + BA_KDM_orig, delta_BA, CVD_x_KDM_BA, CVD_x_delta_BA
  FS4_raw17_ohe_interactions (23): OHE + interactions

BASELINE RESULTS (Phase 1):
-----------------------------
| Feature Set          | Model      | AUC@0.5 | PR-AUC  | F1-Macro@0.5 | F1-Macro@f1 |
|----------------------|------------|---------|---------|---------------|--------------|
| raw17                | RF         | 0.8967  | 0.7783  | 0.8644        | 0.8695       |
| raw17                | XGB_spw    | 0.7654  | 0.3796  | 0.6173        | 0.6507       |
| raw17                | XGB        | 0.7428  | 0.3593  | 0.5080        | 0.6332       |
| raw17                | LGBM_spw   | 0.7984  | 0.4453  | 0.6534        | 0.6842       |
| raw17_ohe            | RF         | 0.8954  | 0.7760  | 0.8634        | 0.8691       |
| raw17_ohe            | XGB_spw    | 0.7651  | 0.3849  | 0.6167        | 0.6533       |
| raw17_ohe            | LGBM_spw   | 0.8012  | 0.4469  | 0.6593        | 0.6892       |
| raw17_ohe_interact   | RF         | 0.8181  | 0.5101  | 0.5803        | 0.7093       |
| raw17_ohe_interact   | XGB_spw    | 0.7071  | 0.2932  | 0.5828        | 0.5866       |
| raw17_ohe_interact   | LGBM_spw   | 0.7295  | 0.3181  | 0.6105        | 0.6185       |

OPTUNA RESULTS (Phase 2.1):
-----------------------------
| XGBoost Optuna (best from 80 trials):
  - Best combined score: 0.7354 (PR-AUC + AUROC)
  - AUROC: 0.8563 | PR-AUC: 0.7272 | F1-Macro@f1: 0.8630

| LightGBM Optuna (best from 50 trials):
  - Best combined score: 0.7520 (PR-AUC + AUROC)
  - AUROC: 0.8652 | PR-AUC: 0.7639 | F1-Macro@f1: 0.8696

STACKING RESULTS (Phase 2.3):
-------------------------------
| Model                        | AUC     | PR-AUC  | F1-Macro@0.5 | F1-Macro@f1 | Std     |
|------------------------------|---------|---------|---------------|-------------|---------|
| Stack_RF+XGB+LGBM_LR        | 0.8925  | 0.7876  | 0.8579        | 0.8718      | 0.0116  |
| Stack_RF+XGB_LR              | 0.8914  | 0.7829  | 0.8536        | 0.8707      | 0.0107  |
| Stack_RFbal+XGB_LR           | 0.8903  | 0.7812  | 0.8542        | 0.8712      | 0.0111  |
| RF (baseline)                | 0.8967  | 0.7783  | 0.8644        | 0.8695      | 0.0078  |

================================================================================
3. KARŞILAŞTIRMA
================================================================================

| Model                    | AUROC  | PR-AUC | F1-Macro | Kaynak           |
|--------------------------|--------|--------|----------|------------------|
| Paper (Fu et al. 2025)   | 0.9078 |   -    |    -     | XGBoost, LASSO  |
| RF baseline (Step 02/08) | 0.8967 | 0.7783 | 0.8695   | raw17, th=f1     |
| Stack (RF+XGB+LGBM)     | 0.8925 | 0.7876 | 0.8718   | th=f1            |
| XGBoost Optuna           | 0.8563 | 0.7272 | 0.8630   | th=f1            |
| LightGBM Optuna           | 0.8652 | 0.7639 | 0.8696   | th=f1            |

Gap from paper: 0.9078 - 0.8967 = 0.011 (AUROC)
PR-AUC improvement with stacking: 0.7876 vs 0.7783 (+0.0093)
F1-Macro improvement with stacking: 0.8718 vs 0.8695 (+0.0023)

================================================================================
4. KRİTİK BULGULAR
================================================================================

1. RF HALİ EN İYİ MODEL (AUROC açısından)
   - AUROC=0.8967 ile en yüksek skor
   - Ancak stacking PR-AUC ve F1-Macro'da hafif avantaj sağlıyor

2. XGBoost VE LightGBM BU VERİDE RF'DEN GERİ
   - Sınıf dengesizliği (%13.3) RF'in avantajı
   - scale_pos_weight parametresi performansı düşürüyor (0.89 → 0.76)
   - Optuna ile iyileştirme var ama RF'yi geçemiyor

3. KLİNİK İNTERAKSİYON DEĞİŞKENLERİ ZARAR VERİYOR
   - delta_BA, CVD*KDM-BA eklendiğinde AUROC 0.90'den 0.82'e düşüyor
   - Bu değişkenler modele gürültü ekliyor

4. BİYOBELİRTEÇLER (ORİJİNAL ÖLÇEK) ZARAR VERİYOR
   - Orijinal ölçek biomarker'lar eklendiğinde RF performansı düşüyor
   - Log-transform ölçeğinde de aynı etki (önceki adımlarda görüldü)

5. THRESHOLD OPTİMİZASYONU F1'İ ARTIRIYOR, AUROC'YU DEĞİŞTİRMİYOR
   - Youden's J: recall'ı artırır ama precision düşer
   - F1-optimal: F1-Macro'yu maksimize eder (0.86 → 0.87)
   - AUROC threshold'tan bağımsızdır

6. MAKALE ILE KALAN FARK (0.011)
   Muhtemel nedenler:
   - Farklı veri filtreleme (BA ≥ 55 alt kümesi)
   - LASSO ile 11 özellik seçimi
   - Farklı kohort/popülasyon
   - Sheet vs Sheet1 veri farklılıkları

================================================================================
5. OPTUNA EN İYİ PARAMETRELER
================================================================================

XGBoost (80 trials):
  n_estimators: 504, max_depth: 7, learning_rate: 0.242
  subsample: 0.78, colsample_bytree: 0.65
  scale_pos_weight: 5.86, reg_alpha: 0.035
  min_child_weight: 1

LightGBM (50 trials):
  n_estimators: 575, max_depth: 12, num_leaves: 49
  learning_rate: 0.244, subsample: 0.62, colsample_bytree: 0.97
  scale_pos_weight: 4.66, reg_alpha: 0.037
  min_child_samples: 10, class_weight: None