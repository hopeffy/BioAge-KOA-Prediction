# Step 3: Position_Knees Experiments

## Amaç
Position_Knees_Encoded'ın aşırı yüksek F-score'u (349.73) nedeniyle alternatif feature seçim stratejilerini test etmek.

## 4 Deney

### Deney 1: Position_Knees_Encoded Olmadan
- **Strateji**: Position_Knees_Encoded'ı tamamen çıkar
- **Features**: 43
- **Sonuç**: ROC AUC = 0.5832, F1 = 0.4214
- **Bulgu**: Sadece %8 performans kaybı - o kadar da kritik değilmiş!

### Deney 2: p < 0.05 (Statistically Significant)
- **Strateji**: Sadece p < 0.05 olan features
- **Features**: 28 (35% azalma)
- **Sonuç**: ROC AUC = 0.5822, F1 = 0.4197
- **Bulgu**: Minimal performans kaybı, iyi feature reduction

### Deney 3: 0.01 < p < 0.05 (Moderate Significance)
- **Strateji**: Sadece orta anlamlılıktaki features
- **Features**: 3 (BMI_Squared, BMI, bmi_kg.m2)
- **Sonuç**: ROC AUC = 0.5266, F1 = 0.3920
- **Bulgu**: Zayıf performans - güçlü features ile kombine edilmeli

### Deney 4: Hierarchical Clustering (Winner! 🏆)
- **Strateji**: Ward linkage ile clustering, her cluster'dan 1 representative
- **Clusters**: 5
- **Features**: 5 (BioAge_60_Plus, Heart_Disease, BMI_Category, Cancer, Sex_Numeric)
- **Sonuç**: ROC AUC = 0.5793, F1 = **0.4263** ⭐ (En yüksek F1!)
- **Bulgu**: **88% feature reduction**, minimal performans kaybı, multicollinearity eliminasyonu

## Karşılaştırma Tablosu

| Deney | ROC AUC | F1 Score | Accuracy | Features | Feature Reduction |
|-------|---------|----------|----------|----------|-------------------|
| Raw Baseline | 0.6543 | 0.4189 | 0.6964 | 17 | - |
| Exp1: No Position_Knees | 0.5832 | 0.4214 | 0.5861 | 43 | 0% |
| Exp2: p < 0.05 | 0.5822 | 0.4197 | 0.5863 | 28 | 35% |
| Exp3: 0.01 < p < 0.05 | 0.5266 | 0.3920 | 0.5393 | 3 | 93% |
| **Exp4: Hierarchical** | **0.5793** | **0.4263** ⭐ | **0.5815** | **5** | **88%** |

## Top Features by ANOVA (without Position_Knees)

1. Heart_Disease (F=133.77)
2. Sex_Numeric (F=118.34)
3. BioAge_60_Plus (F=104.41)
4. Age_x_Comorbidity (F=92.92)
5. BMI_x_Comorbidity (F=79.47)

## Dosyalar

📊 **Veri Dosyaları**:
- `feature_experiments_results.txt` - Tüm deneylerin detaylı sonuçları
- `experiments_summary.csv` - Özet karşılaştırma
- `feature_scores_detailed.csv` - 43 feature'ın ANOVA skorları
- `feature_clusters.csv` - Hierarchical clustering atamaları (5 cluster)

📈 **Görselleştirmeler**:
- `hierarchical_clustering_dendrogram.png` - 2 panelli dendrogram (renkli threshold)
- `representative_features_heatmap.png` - 5 representative feature korelasyon matrisi

## Key Insights

1. **Position_Knees Not Essential**: F-score yüksek olmasına rağmen, olmadan da %58 AUC
2. **Feature Efficiency**: 5 iyi seçilmiş feature, 43 feature kadar etkili
3. **Best Strategy**: Hierarchical clustering - en yüksek F1, en az feature
4. **Multicollinearity Solution**: Clustering ile redundancy eliminate edildi

## Öneriler

✅ **Üretim için**: Hierarchical clustering ile 5 feature kullan
✅ **Alternatif**: p < 0.05 filter ile 28 feature (daha konservatif)
⚠️ **Dikkat**: Raw baseline hala en iyi performansı veriyor - farklı split'ler araştırılmalı

---

## 🏃 Nasıl Çalıştırılır

```powershell
# Virtual environment aktif et
.\env\Scripts\Activate.ps1

# Position_Knees deneylerini çalıştır
cd scripts
python step3_position_knees_experiments.py
```

**Çıktılar**: `step3_position_knees_experiments/` klasöründe 4 TXT, 2 CSV, 2 PNG dosyası oluşur.

➡️ **Sonraki Adım**: Farklı feature selection yöntemlerini karşılaştırmak için Step 4'e geç
