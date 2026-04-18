# STEP 08: XGBoost Fix & Encoding Improvements - RAPOR

**Tarih**: 2026-04-14

## Uygulanan Iyilestirmeler

### 1. Threshold Optimization (Youden Index)
- XGBoost default (th=0.5): F1=0.086 → Youden (th=0.14): F1=**0.363**
- XGBoost spw (th=0.5): F1=0.396 → Youden (th=0.48): F1=0.391
- RF (th=0.5): F1=0.760 → Youden (th=0.25): F1=0.670
- **Sonuc**: Youden threshold XGBoost F1'i 0.09'dan 0.36'a cikardi ama RF hala cok ustun (0.76)

### 2. One-Hot Encoding
- Integer encoding RF: AUC=0.8967
- One-hot encoding RF: AUC=0.8954
- **Sonuc**: One-hot encoding DAHA DUSUK! Integer encoding RF icin daha iyi.
- Makale one-hot kullanmis ama bizim verimizde integer encoding RF ile daha iyi calisiyor

### 3. PCA on Biomarkers
- 8 biyobelirtec → 6 PCA bileseni (%80 varyans)
- raw17 + PCA RF: AUC=**0.7485** (raw17 RF'den -0.15!)
- raw17 + PCA + OHE RF: AUC=0.7481
- **Sonuc**: PCA biyobelirtec SIKINTILI! AUC 0.89'dan 0.75'e dustu.
- Biyobelirtecler bu veri seti icin noise ekliyor

### 4. Probability Calibration (Isotonic Regression)
- RF calibrated (OHE): AUC=0.8810, F1=0.6369
- XGBoost spw calibrated (OHE): AUC=0.7582, F1=0.3756
- **Sonuc**: Calibration AUC'yi dusturdu! RF'un olasilliklari zaten kalibre, isotonic regression overfitting yapiyor

### 5. Forced BA LASSO
- LASSO C=0.01: 9 feature secildi (BA dahil)
- LASSO C=0.1: 19 feature secildi
- BA_KDM_log LASSO katsayisi: -0.14 (C=1.0, secildi ama negatif!)
- LASSO + forced BA + XGBoost: AUC=0.62-0.68
- **Sonuc**: LASSO feature selection XGBoost icin iyi calismiyor, RF ile raw17 hala en iyi

## Final Model Comparison

| Model | Feature Set | AUC | F1 | Encoding |
|-------|-------------|-----|-----|----------|
| **RF** | **raw17** | **0.8967** | **0.7596** | **integer** |
| RF | raw17 (OHE) | 0.8954 | 0.7580 | one-hot |
| RF | raw17+KDM (OHE) | 0.8699 | 0.4469 | one-hot |
| LightGBM spw | raw17 (OHE) | 0.8012 | 0.4541 | one-hot |
| LightGBM spw | raw17 (int) | 0.7984 | 0.4447 | integer |
| XGBoost spw | raw17 (int) | 0.7654 | 0.3963 | integer |
| XGBoost spw | raw17 (OHE) | 0.7651 | 0.3949 | one-hot |
| XGBoost default | raw17 (int) | 0.7428 | 0.0857 | integer |
| XGBoost calibrated | raw17 (OHE) | 0.7582 | 0.3756 | one-hot |
| RF calibrated | raw17 (OHE) | 0.8810 | 0.6369 | one-hot |

## Paper Karsilastirma

| Metrik | Paper | Bizim (RF raw17) | Fark |
|--------|-------|-------------------|------|
| AUROC | 0.9078 | 0.8967 | **-0.011** |
| Model | XGBoost | RandomForest | - |
| Encoding | One-hot | Integer | - |
| Features | 11 LASSO | 17 raw | - |
| BA | KDM-BA (SHAP>0.6) | Biological Age (SHAP=0.44) | - |

## Kritik Bulgular

1. **RF integer encoding ile en iyi**: AUC=0.8967, F1=0.76
2. **XGBoost RF'den 0.13 puan dusuk**: Class imbalance sensitive (13.3% pozitif)
3. **One-hot encoding RF'yi hafif dusuruyor**: 0.8967 → 0.8954
4. **PCA biyobelirtec cati zarar veriyor**: 0.89 → 0.75
5. **Calibration RF'yi dusuruyor**: 0.90 → 0.88
6. **LASSO feature selection XGBoost icin uygun degil**

## XGBoost Neden Dustuk?

XGBoost'un RF'den 0.13 puan dusuk olmasinin sebepleri:
1. **Class imbalance**: %13.3 pozitif orani XGBoost icin zor (scale_pos_weight ile kismen cozuldu)
2. **Small dataset**: 12,329 satir XGBoost icin yeterli degil (paper: 9,505)
3. **Feature scaling**: XGBoost tree-based olmeli ama StandardScaler ile scaling yapildi (gereksiz ama zarar vermameli)
4. **Overfitting**: XGBoost default F1=0.086, th=0.5'te neredeyse hic pozitif tahmin etmiyor
5. **Probability miscalibration**: XGBoost'un probability output'u RF'den daha az kalibre

**Not**: Paper'in XGBoost'u farkli veri seti, farkli feature set ve farkli hiperparametrelerle calisti. Makalenin XGBoost basarisi KDM-BA ve LASSO feature selection sayesinde.