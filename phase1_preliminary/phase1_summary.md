# FAZ 1 ÖZETİ: Preliminary Work (Arthritis Hedefi, Sheet Verisi)

**Tarih**: 2026-03-10 ile 2026-04-14 arası
**Durum**: Tamamlandı → phase1_preliminary/ altına arşivlendi
**Sonuç**: AUROC tavanı **0.7058** (Soft Voting Ensemble)

---

## Kritik Bulgular

### Yanlış Veri Kaynağı
- **Kullanılan**: Sheet (18 kolon, 38,800 satır) — biyobelirteç YOK, Arthritis hedefi
- **Kullanılmamış**: Sheet1 (28 kolon, 15,545 satır) — 8 biyobelirteç, KOA hedefi mevcut
- Sheet1'de KDM-BA hesaplamak için gereken tüm 8 serum biyobelirteci var ve %100 dolu

### Yanlış Hedef Değişken
- **Kullanılan**: Arthritis (genel, %32.5 pozitif) — heterojen, az spesifik
- **Olması gereken**: KOA (Symptomatic Knee OA, %10.6 pozitif) — makalenin hedefi, çok daha spesifik
- KOA = Arthritis=yes AND position_knees=yes (Sheet1'de zaten tanımlı)

### Feature Engineering Zararlı
- 9 farklı FE yaklaşımı denendi (37 yeni feature)
- Hiçbiri ham feature'lardan anlamlı iyileşme sağlamadı
- **Temel neden**: "Genel" feature transformation'lar (kare, log, etkileşim) işe yaramıyor
- Makalenin KDM-BA'sı gibi **domain-specific hesaplama** gerekiyor

### Model Performans Tavanı
| Model | AUROC | Not |
|---|---|---|
| Logistic Regression (baseline) | 0.6199 | |
| Random Forest (default) | 0.6671 | |
| CatBoost (default) | 0.6985 | |
| LightGBM (tuned) | 0.7055 | |
| Soft Voting Ensemble | **0.7058** | En iyi |
| Makale (XGBoost) | **0.9078** | +0.20 fark |

### Feature Engineering Sonuçları
- Domain-specific features (5 yaklaşıım): En iyi +0.22% (istatistiksel anlamsız)
- Composite score FE: Improvements < 1 std dev (not validated)
- Position_knees: F=349.73 (leakage şüphesi) → çıkarılınca -8% AUROC

### Sınıf Dengesi
- Arthritis: %32.5 pozitif (moderat dengesiz)
- SMOTE: -2.93% AUROC düşüşü
- Balanced Weight: neredeyse aynı
- Threshold optimization: F1 iyileştirmesi (0.283 threshold → F1: 0.54)

---

## Deney Listesi (Exp-001 ile Exp-014)

| Exp | Adım | AUROC | F1 | Sonuç |
|---|---|---|---|---|
| 001 | Raw Baseline (LR) | 0.6543 | 0.4189 | Referans baseline |
| 002 | ANOVA FE + top-30 | 0.6384 | 0.4178 | FE zararlı |
| 003 | Position_Knees çıkarıldı | 0.5832 | 0.4214 | -8% AUROC |
| 004-005 | Feature selection denemeleri | 0.53-0.59 | 0.39-0.42 | FE tutarsız |
| 006 | Hierarchical Clust. (5 feat) | 0.5793 | **0.4263** | En iyi F1 |
| 007-012 | 6 feature selection yöntemi | 0.56-0.59 | 0.41-0.43 | LASSO en yüksek AUC |
| 10 | 8 model karşılaştırması | 0.57-0.70 | - | GB en iyi |
| 11 | Hyperparameter tuning | 0.7055 | - | LightGBM |
| 12 | Class balancing | 0.68-0.71 | 0.42-0.54 | SMOTE zararlı |
| 13 | Ensemble/Stacking | 0.70 | - | Soft Voting en iyi |
| 14 | Final staircase | **0.7058** | 0.4484 | Tavanına ulaşıldı |

---

## Faz 2'ye Geçiş Nedenleri

1. **Sheet1 keşfedildi** — KOA hedefi + 8 biyobelirteç + daha zengin veri
2. **Arthritis hedefi yetersiz** — heterojen ve az spesifik
3. **KDM-BA hesaplanabilir** — 8 biyobelirteç mevcut, makalenin SHAP>0.6 feature'ı
4. **Feature engineering'in doğru versiyonu** — generic transformation'lar değil, KDM gibi domain-specific hesaplama gerekli
5. **Performans açığı** — 0.70 vs 0.91, KDM-BA ve KOA hedefi ile kapatılabilir

---

*Arşivlenme tarihi: 2026-04-14*