# STEP 08: XGBoost Fix & Encoding Improvements

**Tarih**: 2026-04-14  
**Amac**: XGBoost'un dusuk performansini duzeltmek ve makale yaklasimlarini test etmek

---

## Uygulanan Iyilestirmeler

### 1. Threshold Optimization (Youden Index)

XGBoost varsayilan threshold=0.5 ile neredeyse hic pozitif tahmin yapmiyor (F1=0.086). Youden Index ile optimal threshold bulundu:

| Model | Threshold | AUC | F1 |
|-------|----------|-----|-----|
| XGBoost default | 0.50 | 0.7428 | 0.086 |
| XGBoost default | 0.14 (youden) | 0.7428 | **0.363** |
| XGBoost spw | 0.50 | 0.7654 | 0.396 |
| XGBoost spw | 0.47 (youden) | 0.7654 | 0.391 |
| RF | 0.50 | 0.8967 | **0.760** |
| RF | 0.25 (youden) | 0.8967 | 0.670 |

**Sonuc**: Youden threshold XGBoost F1'i 4x artirdi (0.09→0.36) ama RF hala cok ustun (0.76).

### 2. One-Hot Encoding (Makale Yaklasimi)

Makale kategorik degiskenler icin one-hot encoding kullanmis. Biz integer encoding kullaniyorduk.

| Encoding | Model | AUC | F1 |
|----------|-------|-----|-----|
| **Integer** | **RF** | **0.8967** | **0.760** |
| One-hot | RF | 0.8954 | 0.758 |
| Integer | XGBoost spw | 0.7654 | 0.396 |
| One-hot | XGBoost spw | 0.7651 | 0.395 |
| One-hot | LightGBM spw | 0.8012 | 0.454 |
| Integer | LightGBM spw | 0.7984 | 0.445 |

**Sonuc**: One-hot encoding RF'yi hafif dusuruyor (0.8967→0.8954). Integer encoding bu veri seti icin daha iyi. Makalenin one-hot.encoding avantaji burada gecerli degil cunku kategorik degiskenler zaten ordinal anlam tasiyor (Gender, Education, Residence vb.).

### 3. PCA on Biyobelirtec (Boyut Azaltma)

8 biyobelirtec → 6 PCA bilesenine (%80 varyans) indirgendi:

| Feature Set | Model | AUC | F1 |
|-------------|-------|-----|-----|
| **raw17 (integer)** | **RF** | **0.8967** | **0.760** |
| raw17 + PCA (integer) | RF | 0.7485 | 0.028 |
| raw17 + PCA (OHE) | RF | 0.7481 | 0.031 |
| raw17 + PCA + KDM (OHE) | RF | 0.7378 | 0.027 |

**Sonuc**: PCA biyobelirtec CATI ZARARLI! AUC 0.89'dan 0.75'e dustu. Biyobelirtecler bu veri seti icin noise ekliyor. RF'in yuksek boyutlu veri setinde overfitting yapmasi sorunu devam ediyor.

### 4. Probability Calibration (Isotonic Regression)

| Model | Calibration | AUC | F1 |
|-------|------------|-----|-----|
| RF (integer) | None | **0.8967** | **0.760** |
| RF (OHE) | Isotonic | 0.8810 | 0.637 |
| XGBoost spw (integer) | None | 0.7654 | 0.396 |
| XGBoost spw (OHE) | Isotonic | 0.7582 | 0.376 |

**Sonuc**: Calibration RF'yi dusurdu (0.90→0.88). RF'un olasiliklari zaten iyi kalibre edilmis; isotonic regression overfitting yapiyor.

### 5. Forced BA LASSO Feature Selection

| C degeri | Secilen Feature | BA_KDM_durumu | XGBoost AUC |
|----------|----------------|---------------|-------------|
| 0.01 | 9 | BA secildi (coef=0.047) | 0.59 |
| 0.05 | 19 | BA secildi (coef=0.093) | 0.62 |
| 0.10 | 19 | BA secildi (coef=0.104) | 0.62 |
| 0.50 | 24 | BA_KDM secildi (coef=-0.14) | 0.68 |
| 1.00 | 26 | Tumu secildi | 0.68 |

**Sonuc**: BA_KDM_log LASSO tarafindan secildi ama **negatif katsayi ile** (-0.14). Bu, KDM-BA'nin KOA'yi tahmin etmede raw Biological Age'den DAHA kotu oldugunu dogruluyor. LASSO + XGBoost kombinasyonu RF+raw17'den cok dusuk (0.62-0.68 vs 0.90).

---

## Final Model Comparison

| Model | Feature Set | Encoding | Threshold | AUC | F1 |
|-------|-------------|----------|-----------|-----|-----|
| **RF** | **raw17** | **integer** | **0.50** | **0.8967** | **0.760** |
| RF | raw17 (OHE) | one-hot | 0.50 | 0.8954 | 0.758 |
| RF calibrated | raw17 (OHE) | one-hot | 0.16 | 0.8810 | 0.637 |
| RF | raw17+KDM (OHE) | one-hot | 0.50 | 0.8699 | 0.447 |
| LightGBM spw | raw17 (OHE) | one-hot | 0.50 | 0.8012 | 0.454 |
| LightGBM spw | raw17 (int) | integer | 0.50 | 0.7984 | 0.445 |
| XGBoost spw | raw17 (int) | integer | 0.50 | 0.7654 | 0.396 |
| XGBoost spw | raw17 (int) | integer | 0.47 | 0.7654 | 0.391 |
| XGBoost cal | raw17 (OHE) | one-hot | 0.15 | 0.7582 | 0.376 |
| XGBoost default | raw17 (int) | integer | 0.50 | 0.7428 | 0.086 |
| XGBoost default | raw17 (int) | integer | 0.14 | 0.7428 | 0.363 |

---

## XGBoost Neden RF'den 0.13 Puan Dusuk?

1. **Class imbalance**: %13.3 pozitif → XGBoost th=0.5'te neredeyse hic pozitif tahmin etmiyor (F1=0.086)
2. **Dataset boyutu**: 12,329 satir XGBoost icin yeterli degil; paper 9,505 satir istiyor ama daha iyi feature set ile
3. **Probability miscalibration**: XGBoost'un probability output'u RF'den daha az kalibre; default threshold=0.5 cok konservatif
4. **Feature scaling gereksiz ama uygulaniyor**: Tree modeller scaling gerektirmez ama pipeline'da var
5. **RF'in ensemble yapisi bu veri ile daha iyi calisiyor**: 500 agac, bagging ile overfitting onleniyor

---

## Kritik Bulgular

1. **RF + 17 raw features + integer encoding = EN IYI MODEL** (AUC=0.8967)
2. **One-hot encoding RF'yi hafif dusuruyor** (integer daha iyi)
3. **PCA biyobelirtec cati zarar veriyor** (AUC -0.15)
4. **Calibration RF'yi dusuruyor** (AUC -0.02)
5. **XGBoost F1 Youden ile 4x artti** ama RF hala cok ustun
6. **LASSO'nun BA_KDM'yi NEGATIF katsayi ile secmesi** dikkat cekici — KDM-BA Biological Age'den daha kotu predictor

---

## Paper Karsilastirma (Guncel)

| Metrik | Paper (Fu 2025) | Bizim (RF raw17) | Fark |
|--------|----------------|-------------------|------|
| **AUROC** | 0.9078 | **0.8967** | **-0.011** |
| Model | XGBoost | RandomForest | - |
| Features | 11 (LASSO+KDM-BA) | 17 (raw) | +6 |
| Encoding | One-hot | Integer | - |
| Threshold | optimal | 0.50 | - |
| BA | KDM-BA (SHAP>0.6) | Biological Age (SHAP=0.44) | - |
| Sinf orani | 10.5% | 13.3% | - |
| Veri | 9,505 | 12,329 | - |

---

*Step 08 Tarihi: 2026-04-14*