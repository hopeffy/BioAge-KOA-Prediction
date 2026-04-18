# FAZ 2 FINAL RAPORU: KOA Tahmin Modeli

**Tarih**: 2026-04-14  
**Proje**: CHARLS verisi ile Diz Osteoartriti (KOA) Risk Tahmini  
**Makale**: Fu et al. (2025) PLOS ONE - "Biological age threshold is associated with symptomatic knee osteoarthritis risk"

---

## 1. Proje Özeti

Bu çalışma, CHARLS (China Health and Retirement Longitudinal Study) verisini kullanarak semptomatik diz osteoartriti (KOA) riskini tahmin eden bir binary classification modeli geliştirmeyi amaçlamaktadır. Çalışma iki faza ayrılmıştır:

| | Faz 1 (Arsiv) | Faz 2 (Guncel) |
|---|---|---|
| Veri Kaynagi | Sheet (18 kolon, 38,800 satir) | Sheet1 (28 kolon, 15,545 satir) |
| Hedef Degisken | Arthritis (genel, %32.5) | KOA (symptomatic, %13.3) |
| Biyobelirtec | YOK | 8 adet (log-transformed) |
| En Iyi AUROC | 0.7058 | **0.8924** |
| En Iyi Model | Soft Voting Ensemble | RandomForest (500 tree) |

---

## 2. Kritik Kesif: Sheet1

Excel dosyasindaki ikinci sheet (Sheet1) daha once hic kullanilmamisti. Bu sheet icerisinde:

- **KOA hedef degiskeni** (makalenin "Symptomatic KOA" tanimina esit)
- **8 serum biyobelirteci** (KDM-BA hesaplamasi icin gereken: kolesterol, trigliserid, HbA1c, BUN, kreatinin, sistolik KB, hs-CRP, trombosit)
- **Daha zengin feature seti** (28 kolon vs Sheet'in 18 kolonu)
- **Daha temiz veri** (biyobelirteclerde %0 eksik, sadece position_knees %20.7)

---

## 3. Veri Hazirlama (Step 01)

### KOA Tanimi ve Dogrulama
- **KOA = Arthritis=yes AND position_knees=yes** (%100 eslesme dogrulandi)
- **position_knees ve Arthritis leakage olarak cikarildi** (KOA bunlardan turetildi)
- **position_knees=NaN olan 3,216 satir cikarildi**

### Veri Setleri
| Veri Seti | Satir | KOA Orani | Aciklama |
|-----------|-------|-----------|----------|
| Dataset A | 12,329 | 13.3% | position_knees dolu |
| Dataset B | 7,635 | 14.2% | BA >= 55 filtreli |

### Encoding Mapping
| Kolon | Encoding |
|-------|----------|
| Gender | 1=Male, 2=Female |
| Age_New | 1=younger (45-59), 2=older (60+) |
| BMI_New | 1=underweight/normal, 2=overweight, 3=obese |
| Marital | 1=married, 2=other |
| Education | 1=low, 2=medium, 3=high |
| Residence | 1=urban, 2=rural |
| Binary disease | 0=no, 1=yes |

---

## 4. Baseline Model Sonuclari (Step 02)

| Veri Seti | Feature Set | Model | AUROC | F1 |
|-----------|-------------|-------|-------|-----|
| A (12,329) | raw17 | **RandomForest** | **0.8937** | **0.7578** |
| B (7,635) | raw17 | **RandomForest** | **0.8939** | **0.7440** |
| A | raw17+biomarkers | RandomForest | 0.7269 | 0.0283 |
| A | raw17 | XGBoost | 0.8048 | 0.3139 |
| A | raw17 | CatBoost | 0.7113 | 0.0598 |
| A | raw17 | LogisticRegression | 0.6579 | 0.0072 |

**Kritik bulgu**: Biyobelirtec eklemek RF'yi 0.89'dan 0.73'e dusuruyor!

### RF Feature Importance (raw17)
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | Biological Age | 36.9% |
| 2 | BMI | 36.7% |
| 3 | Education | 3.2% |
| 4 | Drink | 2.6% |
| 5 | Hypertension | 2.5% |

---

## 5. KDM-Biyolojik Yas Hesaplamasi (Step 03)

### Yontem
Klemera-Doubal Method (KDM) ile 8 biyobelirtecten biyolojik yas hesaplandi.

### Biyobelirtec-Yas Korelasyonlari
| Biyobelirtec | r (log) | R^2 |
|-------------|---------|-----|
| sbp.mean | 0.4214 | 0.1776 |
| creatinine | 0.3471 | 0.1205 |
| BUN | 0.3089 | 0.0954 |
| crp | 0.2080 | 0.0433 |
| HbA1c | 0.1420 | 0.0202 |
| plt | -0.1423 | 0.0202 |
| TC | 0.0683 | 0.0047 |
| TG | -0.0392 | 0.0015 |

### KDM-BA Sonuclari
| Metrik | Biological Age | KDM-BA (log) | KDM-BA (orig) |
|--------|---------------|--------------|----------------|
| KOA korelasyonu | 0.0383 | 0.0182 | 0.0208 |
| AUROC (univariate) | 0.5356 | 0.5158 | 0.5187 |
| OR (per 1 year) | 1.0115 | 1.0050 | 1.0057 |

**Kritik bulgu**: KDM-BA, Biological Age'den DAHA DUSUK KOA korelasyonuna sahip. Makalenin SHAP>0.6 bulgusu burada gecerli degil.

---

## 6. SHAP ve Feature Impact Anailzi (Step 04)

### XGBoost SHAP Feature Importance
| Rank | Feature | SHAP |
|------|---------|------|
| 1 | **Biological Age** | **0.4394** |
| 2 | Gender | 0.4249 |
| 3 | BMI | 0.4026 |
| 4 | Residence | 0.3218 |
| 5 | Education | 0.2118 |
| 6 | CVD | 0.1775 |
| 7 | Drink | 0.1159 |
| 8 | Age_New | 0.0926 |
| 9 | Smoke | 0.0863 |
| 10 | Dyslipidemia | 0.0752 |

Paper karsilastirma: Paper SHAP>0.6 (BA), Bizim SHAP=0.44 (BA)

### Feature Impact Comparison
| Feature Set | AUROC | Fark (vs raw17) |
|-------------|-------|------------------|
| **raw17** | **0.8937** | - |
| raw17+KDM_orig | 0.8652 | -0.029 |
| raw17+KDM_log | 0.8626 | -0.031 |
| raw17+KDM_quartile | 0.8422 | -0.052 |
| raw17+BIR | 0.7949 | -0.099 |
| raw17+biomarkers | 0.7269 | **-0.167** |
| raw17+KDM_biomarkers | 0.7178 | -0.176 |
| raw17+all | 0.6958 | -0.198 |

**Sonuc**: Her tur ekstra feature eklemek performansi dusuruyor. Raw 17 feature en iyi.

### LASSO Feature Selection
21/27 feature secildi (C=0.1). Not: BA_KDM_log LASSO coef = 0 (secilmadi!)

---

## 7. Makale Replikasyonu (Step 04)

| Config | Model | AUROC | F1 |
|--------|-------|-------|-----|
| raw17 | RandomForest | **0.8937** | **0.7578** |
| paper_11 | RandomForest | 0.8459 | 0.7039 |
| raw17 | XGBoost+scaled | 0.8327 | 0.5520 |
| raw17 | XGBoost | 0.8048 | 0.3139 |
| raw17 | LightGBM | 0.7757 | 0.1342 |
| raw17 | CatBoost | 0.7113 | 0.0598 |

---

## 8. Hiperparametre Tuning (Step 05)

| Model | AUROC (CV) | Best Params |
|-------|-----------|-------------|
| **RF_tuned** | **0.8924** | n_estimators=500, max_depth=None, min_samples_leaf=1 |
| Ensemble | 0.8740 | RF+XGB+LGBM soft voting |
| XGB_tuned | 0.8497 | n=300, depth=9, lr=0.1, scale_pos_weight=6.5 |
| LGBM_tuned | 0.8370 | n=200, depth=15, lr=0.1, class_weight=balanced |
| CB_tuned | 0.8239 | n=300, depth=8, lr=0.1, class_weights=[1,6.5] |

---

## 9. RCS ve Alt Grup Analizleri (Step 06)

### BA-KOA Nonlinear Relationship
- BA OR per year: **1.012** (makale: 1.012)
- BA AUROC (univariate): 0.536
- Quartil OR (Q4 vs Q1): **1.414** (makale: 1.4519)
- Threshold: ~55-58 yasinda KOA artis egilimi

### Subgroup Analysis
| Grup | n | KOA% | AUROC |
|------|---|------|-------|
| **CVD+** | 1,304 | 23.2% | **0.899** |
| Female | 6,443 | 16.9% | 0.885 |
| Urban | 7,954 | 15.8% | 0.890 |
| All | 12,329 | 13.3% | 0.857 |
| Hypertension+ | 3,142 | 15.8% | 0.878 |
| Male | 5,886 | 9.4% | 0.817 |
| Rural | 4,375 | 8.7% | 0.853 |

Paper bulgusu: "BA-KOA iliskisi kadinlarda, kirsal bolgede, CVD+'da daha guclu" → **KISMEN DOGRULANDI**

---

## 10. Makale Karsilastirma Tablosu (Guncel)

| Metrik | Makale (Fu 2025) | Faz 2 (Bizim) | Faz 1 (Eski) | Fark (Makale-Faz2) |
|--------|-------------------|---------------|---------------|---------------------|
| **En Iyi AUROC** | 0.9078 | **0.8924** | 0.7058 | **-0.015** |
| En Iyi Model | XGBoost | RandomForest | Soft Voting | - |
| Feature Sayisi | 11 (LASSO) | 17 (raw) | 17 (raw) | -6 |
| Veri Buyuklugu | 9,505 | 12,329 | 18,046 | - |
| Sinif Orani | 10.5% | 13.3% | 32.5% | - |
| BA Hesaplama | KDM (SHAP>0.6) | Biological Age (SHAP=0.44) | Ham | - |
| Hedef Tanimi | Symptomatic KOA | Symptomatic KOA | Arthritis (genel) | Ayni |
| Encoding | One-hot | Integer | Label | Farkli |
| Standartizasyon | Z-score | Z-score | Z-score | Ayni |

---

## 11. Kritik Bulgular ve Cikarimlar

### 1. Hedef Degiskeni Degisimi En Buyuk Etki
- Arthritis (genel, %32.5) → KOA (symptomatic, %13.3)
- AUROC: 0.71 → 0.89 (**+0.18 artis**)
- Daha spesifik hedef = daha learnable problem

### 2. KDM-BA Bekleneni Vermedi
- Sheet1'deki `Biological Age` zaten guclu bir predictor (SHAP=0.44)
- KDM-BA hesaplandi ama KOA korelasyonu cok dusuk (r=0.02)
- LASSO KDM-BA'yi secmedi (coef=0)
- Sebep: Sheet1'deki biyobelirtecler yasla zayif korelasyonlu (en yuksek r=0.42)

### 3. Biyobelirtec Eklemek Zararli
- RF raw17: 0.89
- RF raw17+biyobelirtec: 0.73 (**-0.16**)
- Sebep: RF'in yuksek boyutlu veride overfitting yapmasi, biyobelirteclerin noise eklemesi

### 4. Random Forest En Iyi Model
- RF raw17: AUROC=0.89, F1=0.76
- XGBoost: AUROC=0.80, F1=0.31 (class imbalance'dan etkilendi)
- RF class imbalance'i daha iyi handle ediyor

### 5. Kalan 0.015'lik Farkin Olasi Kaynaklari
1. One-hot encoding vs integer encoding
2. LASSO ile secilmis 11 feature vs 17 raw feature
3. XGBoost hiperparametre optimizasyonu
4. Veri seti farki (9,505 vs 12,329; farkli dislama kriterleri)
5. Makalenin KDM-BA'si farkli veri kaynagindan hesaplanmis olabilir

---

## 12. Dosya Yapisi

```
bioinformatics-data/
├── phase1_preliminary/          (Faz 1 arsiv, 22 step klasoru)
├── step_01_data_prep/           (Dataset A/B, log ters donusum)
├── step_02_baseline/            (Baseline sonuclar)
├── step_03_kdm_ba/              (KDM-BA hesaplama sonuclar)
├── step_04_ba_impact/            (SHAP, feature impact, makale replikasyon)
├── step_07_final_model/          (Hiperparametre tuning, final rapor)
├── scripts/                      (Python scriptleri)
├── memory/                       (Proje hafizasi)
├── results/                      (Faz 1 sonuclar)
├── Raw Data .xlsx                (Ana veri dosyasi)
└── makale_karsilastirma_raporu.md
```

---


Geliştirme olarak yapılacaklar:

## 1. Kilitlenen XGBoost'u "Uyandırmak" (Threshold & Kalibrasyon)
Raporunuzdaki en büyük anormallik şurada: Tabular (tablo) verilerde XGBoost'un Random Forest'ın bu kadar gerisinde kalması (RF: 0.89 vs XGB: 0.80) nadir görülen bir durumdur. Özellikle XGBoost'un F1 skorunun 0.31'de kalması, modelin olasılık (probability) hesaplamalarında değil, sınıflandırma eşiğinde (decision threshold) çuvalladığını gösteriyor. Sınıf dengesizliği (%13.3 KOA) XGBoost'u şaşırtmış.

Hamle: Modeller varsayılan olarak tahmini >0.5 ise "Hasta (1)", <0.5 ise "Sağlıklı (0)" der. XGBoost için bu eşik değerini 0.5'te bırakmayın. ROC eğrisi üzerinden Youden İndeksi (Sensitivity + Specificity - 1) hesaplayarak en optimal kesim noktasını (örneğin 0.18 veya 0.22) bulun.

Kalibrasyon: Modelinizin çıktısını CalibratedClassifierCV (Isotonic Regression) sargısına alarak olasılık kalibrasyonu yapın. Bu, XGBoost'un performansını anında yukarı çekecektir.

## 2. Encoding Hatasını Düzeltmek (Integer vs. One-Hot)
Makale kategorik değişkenler için One-Hot Encoding kullanmış, siz Integer Encoding kullanmışsınız.

Neden Önemli? Integer encoding algoritmaya matematiksel bir hiyerarşi dayatır. Örneğin; Residence değişkeninde "Urban=1, Rural=2" yaptığınızda, model "Rural, Urban'ın iki katı büyüklüğündedir" gibi yanlış bir matematiksel ilişki kurabilir. Bu durum ağaç tabanlı modellerde (split noktalarında) verimliliği düşürür, LASSO gibi regresyon temelli feature selection yöntemlerini ise tamamen bozar.

Hamle: Kategorik değişkenlerinizi (Gender, Marital, Education, Residence) pd.get_dummies veya OneHotEncoder ile dönüştürüp modeli tekrar eğitin.

## 3. Biyobelirteç Gürültüsünü Filtrelemek (PCA Hamlesi)
Raporunuzdaki kilit bulgulardan biri: Biyobelirteçleri eklemek RF modelinin skorunu 0.89'dan 0.73'e düşürüyor. Bu, modellerinizin yüksek boyutluluk (curse of dimensionality) ve gürültü nedeniyle aşırı öğrenmeye (overfit) düştüğünü kanıtlar. Biyobelirteçleri doğrudan modele vermek yerine onları sıkıştırmalısınız.

Hamle: 8 biyobelirteç sütununa PCA (Temel Bileşen Analizi) uygulayın. 8 sütunu, varyansın %80'ini açıklayan 1 veya 2 ana bileşene (PC1, PC2) indirgeyin. Bu yeni sütunları (Örn: Metabolic_Component, Inflammatory_Component olarak isimlendirebilirsiniz) modele verin. Böylece gürültüyü silip sadece sinyali modele aktarmış olursunuz.

## 4. Makalenin SHAP Farlılaştırması
Makalede KDM-BA'nın SHAP değerinin >0.6 çıkması, sizin bulduğunuz Biyolojik Yaşın SHAP değerinden (0.44) çok daha yüksek. Bu durum, makale yazarlarının KDM-BA hesaplarken parametreleri (Klemera-Doubal formülündeki katsayıları) sadece CHARLS veri setine göre değil, belki de sağlıklı bir alt popülasyona göre normalize etmiş olabileceğinden kaynaklanıyor olabilir.

Hamle: LASSO'nun biyolojik yaşı hiç seçmemesi, biyolojik yaşın diğer değişkenler (örneğin Kronolojik Yaş veya Tansiyon) ile yüksek korelasyona (Multicollinearity) sahip olduğunu ve LASSO'nun katsayıyı sıfırladığını gösteriyor olabilir. Feature selection yaparken Biyolojik Yaşı modele "zorunlu (forced)" olarak dahil edip diğerlerini LASSO ile seçmeyi deneyin.

---

## 13. Step 08: XGBoost Fix & Encoding Improvements

### Yapılanlar
- One-hot encoding (Gender, Age_New, Marital, Education, Residence, BMI_New)
- XGBoost scale_pos_weight + Youden threshold optimizasyonu
- Probability calibration (Isotonic Regression)
- Forced BA LASSO feature selection
- PCA on biomarkers

### Sonuçlar
| Feature Set | Model | Threshold | AUROC | F1 |
|-------------|-------|-----------|-------|-----|
| raw17 (int) | RF | default | 0.8967 | 0.760 |
| raw17 (int) | RF | youden(0.25) | 0.8967 | 0.670 |
| raw17 (OHE) | RF | default | 0.8954 | 0.758 |
| raw17 (OHE) + KDM | RF | default | 0.8700 | 0.447 |
| raw17 (OHE) + PCA | RF | default | 0.7485 | 0.028 |
| raw17 (int) | XGBoost_spw | default | 0.7654 | 0.396 |
| raw17 (OHE) | XGBoost_spw | youden(0.48) | 0.8012 | 0.450 |

**Sonuc**: One-hot encoding RF'i hafif düşürdü (0.8967→0.8954), KDM-BA ve PCA eklemek bariz düşürdü.

---

## 14. Step 09: Advanced Pipeline (PR-AUC, Optuna, Stacking)

### Yapılanlar
- PR-AUC ve F1-Macro değerlendirme metrikleri eklendi
- Youden's J ve F1-optimal threshold araması
- Optuna ile hiperparametre optimizasyonu (XGBoost 80 trial, LightGBM 50 trial)
- Klinik interaksiyon değişkenleri: delta_BA, CVD×KDM_BA, CVD×delta_BA
- Stacking ensemble: RF + XGBoost → LR, RF + XGBoost + LightGBM → LR

### Sonuçlar
| Model | AUROC | PR-AUC | F1-Macro@0.5 | F1-Macro@f1 |
|-------|-------|--------|---------------|--------------|
| **RF (raw17)** | **0.8967** | **0.7783** | 0.8644 | 0.8695 |
| XGBoost_optuna | 0.8563 | 0.7272 | 0.8487 | 0.8630 |
| LightGBM_optuna | 0.8652 | 0.7639 | 0.8482 | 0.8696 |
| Stack RF+XGB_LR | 0.8914 | 0.7829 | 0.8536 | 0.8707 |
| Stack RF+XGB+LGBM_LR | 0.8925 | 0.7876 | 0.8579 | 0.8718 |
| Stack RFbal+XGB_LR | 0.8903 | 0.7812 | 0.8542 | 0.8712 |

### Klinik İnteraksiyon Değişkenleri Etkisi
- delta_BA (= KDM_BA - Biological Age), CVD×KDM_BA, CVD×delta_BA eklendiğinde AUROC **0.90'den 0.82'e düşüyor**
- Bu değişkenler modele gürültü ekliyor

### Optuna En İyi Parametreler
**XGBoost**: n=504, depth=7, lr=0.242, subsample=0.78, colsample=0.65, spw=5.86, reg_alpha=0.035
**LightGBM**: n=575, depth=12, leaves=49, lr=0.244, subsample=0.62, colsample=0.97, spw=4.66

---

## 15. Step 10: Makale Replikasyonu (Exact Feature Space)

### Yaklaşım
Makalenin (Fu et al. 2025) SHAP plot'undaki 13 feature'ı kullanarak birebir replikasyon:
- Paper 11 (LASSO-selected): BA, Gender, Education, Residence, Hypertension, Dyslipidemia, CVD, Smoke, Drink, BMI_New, Cancer
- Paper 13 (full SHAP): +Marital, Diabetes
- Paper 13 cont BMI: BMI_New yerine continuous BMI

### Metodoloji
- One-hot encoding (kategorik değişkenler)
- Z-score standardizasyonu (sürekli değişkenler)
- 70/30 stratified train/test split (makalenin yöntemi)
- 5-fold CV × 3 seed (robust değerlendirme)
- Optuna XGBoost optimizasyonu (100 trial)

### Sonuçlar (5-Fold CV × 3 Seed)

| Feature Set | Model | AUROC | PR-AUC | F1-Macro@f1 |
|-------------|-------|-------|--------|-------------|
| **paper_13_cont_BMI** | **RF** | **0.8914** | **0.7628** | **0.8692** |
| paper_13_cont_BMI | XGB_optuna | 0.8603 | 0.7314 | 0.8598 |
| paper_13 (BMI_New) | RF | 0.8544 | 0.6030 | 0.8394 |
| paper_13 | XGB_default | 0.7677 | 0.3987 | 0.6572 |
| paper_11 | RF | 0.8497 | 0.5837 | 0.8361 |
| paper_11 | XGB_spw | 0.7279 | 0.3142 | 0.6053 |
| paper_11 | XGB_default | 0.7613 | 0.3865 | 0.6530 |

### Sonuçlar (70/30 Split)

| Feature Set | Model | AUROC | PR-AUC | F1-Macro@f1 |
|-------------|-------|-------|--------|-------------|
| paper_13_cont_BMI | RF | 0.8622 | 0.6982 | 0.8314 |
| paper_13_cont_BMI | XGB_optuna | 0.8292 | 0.6526 | 0.8181 |
| paper_13_cont_BMI | XGB_spw | 0.8097 | 0.5585 | 0.7432 |
| paper_11 | RF | 0.8183 | 0.5119 | 0.7976 |
| paper_11 | XGB_spw | 0.7772 | 0.4048 | 0.6771 |

### Optuna XGBoost En İyi Parametreler (Paper 13 features)
- n_estimators=438, max_depth=7, lr=0.141, subsample=0.82, colsample=0.89
- scale_pos_weight=3.66, reg_alpha=0.011

### Kritik Bulgular

1. **Continuous BMI > BMI_New (kategorik)**: 0.8914 vs 0.8544
   - BMI'yi kategorik (1/2/3) olarak kodlamak bilgi kaybına neden oluyor
   - Continuous BMI modelin ince gradyanları yakalamasına izin veriyor

2. **Paper 11 + Marital + Diabetes = 0.04 AUROC artışı** (0.8497 → 0.8914)
   - Marital ve Diabetes önemli feature'lar

3. **RF > XGBoost (tüm feature setlerinde)**
   - Makale XGBoost ile 0.9078'e ulaşıyor ama bizim verimizde RF tutarlı şekilde daha iyi
   - XGBoost sınıf dengesizliğinde (%13.3) zorlanıyor

4. **Makale farkı (0.9078 - 0.8914 = 0.0164)** hâlâ kapanamadı

### Makale ile Kalan Farkın Nedenleri

1. **Veri seti farkı**: Makale 9,505 satır (BA≥55 filtresi + listwise deletion), biz 12,329
2. **Sınıf oranı**: Makale %10.5, biz %13.3
3. **XGBoost performans farkı**: Bizim verimizde RF > XGBoost; makalede XGBoost > RF (0.9078 vs 0.7393)
4. **KDM-BA hesaplama**: Makalenin KDM-BA'sı SHAP>0.6 ile baskın feature; bizim verimizde Biological Age SHAP=0.44

---

## 16. Tüm Adımların Karşılaştırma Özeti

| Step | Yaklaşım | En İyi AUROC | PR-AUC | F1-Macro | Model |
|------|----------|-------------|--------|----------|-------|
| 02 | Baseline (raw17) | 0.8937 | - | 0.7578 | RF |
| 03 | KDM-BA eklendi | 0.8652 | - | - | RF |
| 05-07 | Hiperparametre tuning | 0.8924 | - | - | RF_tuned |
| 08 | XGBoost fix + OHE | 0.8967 | - | 0.7596 | RF |
| 09 | Optuna + Stacking | 0.8925 | 0.7876 | 0.8718 | Stack_RF+XGB+LGBM |
| 10 | Paper replikasyon | **0.8914** | 0.7628 | 0.8692 | RF (paper_13_contBMI) |
| **Makale** | **LASSO 11 + XGBoost** | **0.9078** | - | - | **XGBoost** |

### Toplam İyileşme
- Faz 1'den (Arthritis): 0.7058 → 0.8967 = **+0.19**
- Makaleye kalan fark: 0.9078 - 0.8967 = **0.011**
- Paper replikasyon özelinde kalan fark: 0.9078 - 0.8914 = **0.016**

---

## 17. Güncellenmiş Dosya Yapisi

```
bioinformatics-data/
├── phase1_preliminary/          (Faz 1 arsiv)
├── step_01_data_prep/           (Dataset A/B, log ters donusum)
├── step_02_baseline/             (Baseline sonuclar)
├── step_03_kdm_ba/              (KDM-BA hesaplama sonuclar)
├── step_04_ba_impact/           (SHAP, feature impact)
├── step_07_final_model/         (Hiperparametre tuning, final rapor)
├── step_08_xgboost_fix/         (OHE, threshold, calibration)
├── step_09_advanced/             (PR-AUC, Optuna, Stacking)
├── step_09_advanced/step09_results.csv
├── step_09_advanced/step09_report.txt
├── step_09_advanced/step09_kapsamli_rapor.md
├── step_10_paper_replication/    (Makale replikasyonu)
├── step_10_paper_replication/step10_results.csv
├── step_10_paper_replication/step10_report.txt
├── scripts/
│   ├── step_09_advanced_pipeline.py
│   ├── step_09_final_push.py
│   └── step_10_paper_replication.py
├── Raw Data .xlsx
├── faz2_final_raporu.md
└── makale_karsilastirma_raporu.md
```

*Rapor Tarihi: 2026-04-14 (Step 08-10 güncellemesi)*
*Makale: PLOS ONE 20(12): e0335250 (Aralık 2025)*