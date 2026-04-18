# Step 2: Feature Engineering + ANOVA F-test

## Amaç
Raw data üzerine feature engineering yaparak 37 yeni özellik türetmek ve ANOVA F-test ile en önemli özellikleri belirlemek.

## Yapılanlar

### Feature Engineering
- **Aging Features** (10): Age^2, Age^3, log(Age), age thresholds (60+, 70+, >67)
- **BMI Features** (7): BMI^2, BMI categories, obesity flags, log(BMI)
- **Comorbidity Features** (3): Count, risk score, multiple comorbidity flag
- **Interaction Features** (6): BMI×Age, Age×Comorbidity, BMI×Comorbidity
- **Metabolic Risk** (2): Metabolic risk score, high risk flag
- **Gender-Specific** (2): Sex×BMI, Sex×Age
- **Socioeconomic** (3+): Education, residence, marital status encoding

**Toplam**: 18 → 55 features

### ANOVA F-test Analysis
- **Top 30** features seçildi
- **29/38** feature istatistiksel olarak anlamlı (p < 0.05)
- **En güçlü feature**: Position_Knees_Encoded (F=349.73)

## Sonuçlar

### Performans
- **Raw Baseline (80/20)**: ROC AUC = 0.6543
- **ANOVA Selected (70/30)**: ROC AUC = 0.6384 (-2.43%)
- **Bulgu**: Feature engineering beklendiği gibi performans artışı sağlamadı

### Top 5 Features (ANOVA F-score)
1. Position_Knees_Encoded (F=349.73)
2. Residence_Level (F=148.79)
3. Heart_Disease (F=133.77)
4. Sex_Numeric (F=118.34)
5. BioAge_60_Plus (F=104.41)

### Multicollinearity
- **7 yüksek korelasyon çifti** tespit edildi (r > 0.8)
- Örnek: Comorbidity_Count ↔ Comorbidity_Risk_Score (r=1.000)

## Dosyalar

📊 **Veri Dosyaları**:
- `anova_feature_scores.csv` - Tüm features için F-score ve p-value

📈 **Görselleştirmeler**:
- `anova_feature_analysis.png` - 6 panelli analiz (top 20, p-value dist, volcano plot, categories, correlation, cumulative)
- `anova_all_features_ranked.png` - Tüm 38 feature'ın ranking'i

## Çıkarsamalar

✅ **Başarılar**:
- Kapsamlı feature engineering pipeline oluşturuldu
- İstatistiksel analiz ile önemli features belirlendi
- Multicollinearity tespiti yapıldı

⚠️ **Sorunlar**:
- Feature engineering performansı düşürdü
- Position_Knees_Encoded aşırı dominant (F=349 vs next F=149)
- Farklı train/test split'leri karşılaştırmayı zorlaştırdı

---

## 🏃 Nasıl Çalıştırılır

```powershell
# Virtual environment aktif et
.\env\Scripts\Activate.ps1

# Ana pipeline'ı çalıştır (Step 1 + Step 2 içerir)
cd scripts
python main_pipeline.py
```

**Not**: `main_pipeline.py` hem Step 1 (raw baseline) hem de Step 2 (feature engineering + ANOVA) içerir.

➡️ **Sonraki Adım**: Position_Knees dominance'ını test etmek için Step 3'e geç
