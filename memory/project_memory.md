# KOA (Knee Osteoarthritis) Prediction - Phase 2 FINAL

**Guncelleme Tarihi**: 2026-04-14
**Durum**: Tamamlandi
**En Iyi Model**: RF (500 tree, raw17 features, AUROC=0.89)

---

## Proje Ozeti

### Faz 1 (Arsiv): Sheet verisi, Arthritis hedefi
- En iyi AUROC: 0.7058 (Soft Voting Ensemble)
- 14 adim denendi: FE, FS, model comparison, tuning
- Bulgu: Feature engineering zararli
- Arsiv: phase1_preliminary/

### Faz 2 (Guncel): Sheet1 verisi, KOA hedefi
- En iyi AUROC: **0.89** (Random Forest, raw17 features)
- 7 adim tamamlandi
- Makale ile fark: sadece **-0.014** (0.89 vs 0.91)
- Bulgu: Hedef degiskeni degisimi en buyuk etki (+0.18)

---

## Final Model

| Parametre | Deger |
|-----------|-------|
| Model | RandomForest (n_estimators=500) |
| Features | 17 raw (Sheet1) |
| AUROC | 0.8924 (CV) |
| F1 | 0.76 |
| Hedef | KOA (Symptomatic KOA) |
| Veri | Sheet1, 12,329 satir, %13.3 KOA |
| Validasyon | 5-fold CV x 3 seed |

### Feature Importance (XGBoost SHAP)
1. Biological Age (0.44)
2. Gender (0.42)
3. BMI (0.40)
4. Residence (0.32)
5. Education (0.21)

---

## Kritik Bulgular

1. **Hedef degiskeni degisimi en buyuk etki**: Arthritis→KOA tek basina +0.18 AUROC
2. **KDM-BA bekleneni vermedi**: Sheet1'in Biological Age'i zaten guclu (SHAP=0.44)
3. **Biyobelirtec eklemek RF'yi dusuruyor**: 0.89 → 0.73 (-0.16)
4. **RF ham feature'larla en iyi model**: Tuning sonrasi AUROC=0.89
5. **Alt grup analizleri makaleyi dogruluyor**: CVD+ (0.90), Kadin (0.89), Sehir (0.89)

---

## Veri Setleri

| Ad | Satir | KOA Orani | Aciklama |
|----|-------|-----------|----------|
| Dataset A | 12,329 | 13.3% | position_knees dolu, leakage cikarildi |
| Dataset B | 7,635 | 14.2% | BA>=55 filtreli |

---

*Son guncelleme: 2026-04-14*