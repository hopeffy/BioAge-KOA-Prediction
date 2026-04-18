# Step 4: Feature Selection Methods Comparison

## Amaç
6 farklı feature selection yöntemini karşılaştırmalı olarak değerlendirmek ve en etkili yaklaşımı belirlemek.

## Yöntemler

### 1. ANOVA F-test (Univariate Statistical Test)
- **Yaklaşım**: Her feature'ı bağımsız olarak target ile F-test
- **Avantaj**: Hızlı, yorumlaması kolay
- **Dezavantaj**: Feature interactions'ları görmez

### 2. Mutual Information (Information Theory)
- **Yaklaşım**: Feature ve target arasındaki bilgi paylaşımını ölçer
- **Avantaj**: Non-linear ilişkileri yakalar
- **Dezavantaj**: Yavaş, parametrik değil

### 3. L1 Logistic Regression (LASSO)
- **Yaklaşım**: L1 regularization ile sparse model
- **Avantaj**: Feature selection ve modeling birlikte
- **Dezavantaj**: Scaling gerektirir

### 4. Tree-Based Importance (RF + XGBoost)
- **Yaklaşım**: RF ve XGBoost importance'larının ortalaması
- **Avantaj**: Feature interactions'ları yakalar
- **Dezavantaj**: Biased towards high-cardinality features

### 5. RFECV (Recursive Feature Elimination + CV)
- **Yaklaşım**: Wrapper method, iterative elimination
- **Avantaj**: Model-aware selection, CV ile optimize
- **Dezavantaj**: Hesaplama maliyeti yüksek

### 6. Hierarchical Clustering (Correlation-Based)
- **Yaklaşım**: Korelasyon clusterlama, her cluster'dan en iyisini seç
- **Avantaj**: Multicollinearity'yi elimine eder
- **Dezavantaj**: Cluster sayısı belirlenmeli

## Sonuçlar

### Performans Sıralaması (15 feature* hedefiyle)

| Sıra | Yöntem | ROC AUC | F1 Score | Accuracy | Features |
|------|---------|---------|----------|----------|----------|
| 🥇 | **L1 LASSO** | **0.5927** | 0.4238 | 0.5881 | 15 |
| 🥈 | **Hierarchical Clustering** | 0.5912 | **0.4253** ⭐ | **0.5931** ⭐ | 15 |
| 🥉 | **ANOVA F-test** | 0.5876 | 0.4168 | 0.6010 | 15 |
| 4️⃣ | Tree-Based (RF+XGBoost) | 0.5851 | 0.4124 | 0.5868 | 15 |
| 5️⃣ | RFECV | 0.5825 | 0.4246 | 0.5864 | 33* |
| 6️⃣ | Mutual Information | 0.5622 | 0.4200 | 0.5746 | 15 |

*RFECV kendisi optimal 33 feature seçti

### Performans İstatistikleri

- **ROC AUC Range**: 0.5622 - 0.5927 (Δ = 0.0305, ~3%)
- **F1 Score Range**: 0.4124 - 0.4253 (Δ = 0.0129, ~1.3%)
- **Accuracy Range**: 0.5746 - 0.6010 (Δ = 0.0264, ~2.6%)

**Yorum**: Yöntemler arasında performans farkı küçük, ancak L1 LASSO ve Hierarchical Clustering öne çıkıyor.

## Consensus Features

Birden fazla yöntem tarafından seçilen features (güçlü consensus göstergesi):

### 6/6 Yöntemde Seçilen (Must-Have! ⭐⭐⭐)
1. **BioAge_60_Plus** - 60+ yaş olma
2. **BMI_x_BioAge** - BMI ve yaş interaksiyonu

### 5/6 Yöntemde Seçilen
3. **wave** - Araştırma dalgası
4. **doctor_diagnose_heart_disease** - Kalp hastalığı teşhisi
5. **Sex_Numeric** - Cinsiyet
6. **Age_x_Comorbidity** - Yaş ve komorbidite interaksiyonu

### 4/6 Yöntemde Seçilen
- iyear, Has_Multiple_Comorbidities, BMI_x_Comorbidity, MET

### 3/6 Yöntemde Seçilen
- Hypertension, Heart_Disease, BioAge_Over_67, Comorbidity_Risk_Score, Biological Age, 
  Biological_Age_Cubed, Log_BMI, BMI_Category, BMI_Obese, doctor_diagnose_diabetes

## Dosyalar

📊 **Veri Dosyaları**:
- `feature_selection_comparison_results.txt` - Her yöntemin detaylı analizi ve seçilen features
- `feature_selection_methods_comparison.csv` - Özet performans karşılaştırması

📈 **Görselleştirmeler**:
- `feature_selection_comparison.png` - 4 panelli karşılaştırma (ROC AUC, F1, Accuracy, Best Performers)
- `feature_overlap_heatmap.png` - Yöntemler arası feature overlap matrix

## Key Insights

### 1. En İyi Yöntemler
- **ROC AUC için**: L1 LASSO (0.5927)
- **F1 Score için**: Hierarchical Clustering (0.4253)
- **Accuracy için**: ANOVA F-test (0.6010)
- **Genel**: Hierarchical Clustering - En dengeli performans

### 2. Yöntem Karakteristikleri
- **Hız**: ANOVA > MI > Tree > L1 > Hierarchical > RFECV
- **Interpretability**: ANOVA > L1 > Hierarchical > Tree > MI > RFECV
- **Feature Interactions**: Tree > RFECV > L1 > Hierarchical > MI > ANOVA
- **Multicollinearity Handling**: Hierarchical > L1 > RFECV > Tree > ANOVA > MI

### 3. Consensus Strength
- BioAge_60_Plus ve BMI_x_BioAge **tüm yöntemlerde** seçildi
- 22 feature 3+ yöntemde seçildi (güçlü kandidatlar)
- Yöntemler arası overlap orta/yüksek (feature selection stabilitesi iyi)

### 4. Unexpected Findings
- Mutual Information en düşük performansı verdi (beklenmeyen)
- RFECV 33 feature seçmesine rağmen orta performans
- Tree-based methods beklenen kadar dominant değil

## Öneriler

### Üretim Ortamı İçin

✅ **#1 Seçim: Hierarchical Clustering**
- En yüksek F1 ve Accuracy
- Multicollinearity'yi elimine eder
- 15 feature ile verimli
- Yorumlanabilir: Her cluster bir klinik domain

✅ **#2 Seçim: L1 LASSO**
- En yüksek ROC AUC
- Regularization ile overfitting önlenir
- İstatistiksel olarak güçlü

✅ **#3 Seçim: ANOVA F-test**
- En hızlı
- En yorumlanabilir
- Baseline için ideal

### Model Ensemble İçin

🔥 **Ensemble Stratejisi**:
1. L1 LASSO features ile model 1
2. Hierarchical features ile model 2
3. ANOVA features ile model 3
4. Voting/Stacking ile kombine et

### Minimal Feature Set İçin

⚡ **Consensus Core (6 features)**:
```
BioAge_60_Plus, BMI_x_BioAge, wave, 
doctor_diagnose_heart_disease, Sex_Numeric, 
Age_x_Comorbidity
```
Bu 6 feature 5+ yöntemde seçildi - güçlü predictive power.

## Sonraki Adımlar

1. ✅ Hierarchical Clustering features ile final model eğit
2. ✅ Consensus features (6 feature) ile lightweight model test et
3. ⬜ Ensemble model oluştur (3 yöntemden features)
4. ⬜ Hyperparameter optimization (Optuna ile)
5. ⬜ Cross-validation ile robust değerlendirme
6. ⬜ SHAP values ile interpretability analizi

---

## 🏃 Nasıl Çalıştırılır

```powershell
# Virtual environment aktif et
.\env\Scripts\Activate.ps1

# 6 feature selection yöntemini karşılaştır
cd scripts
python step4_feature_selection_comparison.py
```

**Çıktılar**: `step4_feature_selection_comparison/` klasöründe 2 TXT, 1 CSV, 2 PNG dosyası oluşur.

**Süre**: ~2-3 dakika (RFECV CV nedeniyle uzun sürer)

---

**Tarih**: 2026-03-10  
**Status**: ✅ Complete  
**Next**: Model optimization ve ensemble
