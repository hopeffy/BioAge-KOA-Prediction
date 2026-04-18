# Step 1: Raw Baseline

## Amaç
Feature engineering'den önce raw features ile baseline performans ölçmek.

## Metodoloji

### Veri
- **Kaynak**: Raw Data .xlsx (38,800 rows, 18 columns)
- **Temizleme**: Age ≥ 18, missing values kaldırıldı
- **Final Dataset**: 18,046 samples

### Train/Test Split
- **Oran**: 80/20 stratified split
- **Reason**: Daha büyük training set ile daha güçlü baseline

### Model
- **Algorithm**: Logistic Regression
- **Hyperparameters**: Default (max_iter=1000)
- **Feature Scaling**: StandardScaler

### Raw Features (18)
```
wave, iyear, Biological Age, bmi_kg.m2, Sex, Residence_Type,
doctor_diagnose_hypertension, doctor_diagnose_diabetes, 
doctor_diagnose_heart_disease, doctor_diagnose_stroke, 
doctor_diagnose_cancer, Arthritis (target), MET, edu_level, 
r_smoking, r_drinking, position_knees, marital_status
```

## Sonuçlar

### Performans
- **ROC AUC**: 0.6543
- **F1 Score**: 0.4189
- **Accuracy**: 0.6964
- **Precision**: 0.5006
- **Recall**: 0.3634

### Feature Preprocessing
- **Encoding**: position_knees (ordinal → numeric)
- **Scaling**: StandardScaler (mean=0, std=1)
- **Stratification**: Target balanced in train/test

## Dosyalar

📊 **Veri Dosyaları**:
- `baseline_results.txt` - Raw baseline performans sonuçları

## Key Insights

✅ **Başarılar**:
- ROC AUC 0.6543 - Makul başlangıç performansı
- Class imbalance (67.5% / 32.5%) ile başarılı train

⚠️ **Gözlemler**:
- F1 Score düşük (0.4189) - imbalanced dataset etkisi
- Recall düşük (0.3634) - pozitif sınıf tespit zorluğu

## Benchmark

Bu baseline, sonraki tüm feature engineering ve model optimization çalışmalarının karşılaştırma noktasıdır.

**Target**: ROC AUC > 0.65 ve F1 Score > 0.42

---

## 🏃 Nasıl Çalıştırılır

```powershell
# Virtual environment aktif et
.\env\Scripts\Activate.ps1

# Ana pipeline'ı çalıştır (Step 1 + Step 2 içerir)
cd scripts
python main_pipeline.py
```

**Not**: `main_pipeline.py` hem Step 1 (raw baseline) hem de Step 2 (feature engineering) içerir.

➡️ **Sonraki Adım**: Feature engineering ile performans artışı için Step 2'ye geç
