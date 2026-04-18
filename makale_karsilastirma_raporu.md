# Makale (Fu et al., 2025 - PLOS ONE) vs Proje: Detaylı Karşılaştırma

**Makale**: Fu F, Dong L, Lian J, et al. "Biological age threshold is associated with symptomatic knee osteoarthritis risk in chinese adults: Insights from machine learning analysis of a national cohort." *PLOS One*. 2025;20(12):e0335250.

**DOI**: https://doi.org/10.1371/journal.pone.0335250

---

## 1. Veri Kaynağı & Örneklem

| | **Makale** | **Projemiz** |
|---|---|---|
| Kaynak | CHARLS 2011+2015 | CHARLS (aynı) |
| Toplam kayıt | 38,800 | 38,800 |
| Analiz örneklemi | **9,505** (listwise deletion + yaş≥45) | **18,046** (eksik verileri -999 ile kodlama) |
| Sınıf oranı | 1,000 KOA(+) / 8,505 KOA(-) = **10.5%** pozitif | 5,865(+) / 12,181(-) = **32.5%** pozitif |

**Kritik fark**: Makale katı dışlama kriterleri uygulayarak 9,505'e inmiş; biz ise tüm veriyi koruyarak 18,046'de kaldık. Sınıf dengesi çok farklı (%10.5 vs %32.5).

---

## 2. Sonuç (Outcome) Tanımı

| | **Makale** | **Projemiz** |
|---|---|---|
| Tanım | **Symptomatic KOA**: Hekim tanılı osteoarthritis + eşzamanlı diz ağrısı | **Arthritis** (genel hekim tanılı artrit) |
| Spesifilik | Çok spesifik (sadece KOA + diz ağrısı) | Daha geniş (her türlü artrit) |

Bu fark, hedef değişkenimizin daha heterojen olması anlamına geliyor; bu da sınıflandırmayı zorlaştırır.

---

## 3. EN ÖNEMLİ FARK: Biyolojik Yaş (BA)

| | **Makale** | **Projemiz** |
|---|---|---|
| BA hesaplama | **KDM yöntemi ile 8 serum biyobelirteçinden hesaplandı** (kolesterol, trigliserid, HbA1c, üre nitrojen, kreatinin, sistolik KB, hs-CRP, trombosit) | Ham "Biological Age" değişkeni (CHARLS'den) |
| BA önemi | **SHAP > 0.6** (ağırlıklı en önemli feature) | BioAge_60_Plus feature olarak test edildi ama çok güçlü değil |
| Etki | Her 1 yıl BA artışı = %1.23 daha fazla KOA riski | Benzer etki bulunamadı |

**Bu, performans farkının ana nedeni.** Makale, BA'yı kan biyobelirteçlerinden matematiksel olarak türetiyor; bu çok daha bilgilendirici bir predictor oluşturuyor. Bizim ham "Biological Age" değişkenimizin aynı hesaplama yöntemiyle elde edilip edilmediği kritik bir soru.

### RCS (Restricted Cubic Spline) Analizi Bulguları

Makale, BA ile KOA riski arasında **doğrusal olmayan** bir ilişki buldu:
- BA ~66.7 yılın altında: risk nispeten sabit
- BA ~66.7 yılın üzerinde: risk hızla artıyor (non-linearity p = 0.013)
- En yüksek BA kartilinde (Q4): Q1'e göre %45.19 daha fazla KOA riski (OR = 1.4519)

Bu bulgu, makalenin BA'yı neden bu kadar güçlü bir feature haline getirdiğini açıklıyor.

---

## 4. Eksik Veri İşleme

| | **Makale** | **Projemiz** |
|---|---|---|
| Strateji | BA ve KOA bilgisi eksik olanları dışladı + MICE (kovaryatlar için) | Eksik verileri -999 ile kodlama (Raw Data Direct) |
| Veri kaybı | ~62% (38,800 → 9,505), sonra 1,000 vaka seçimi | %0 (18,046) |

Makale, eksik veri sorununu iki aşamada çözüyor:
1. BA hesaplaması için tam veri gerektirdiğinden, eksik olan 15,394 kişiyi dışlıyor
2. Kalan kovaryatlar için %2'den az eksik veriyi MICE ile tamamlıyor

---

## 5. Feature Seçimi

| | **Makale** | **Projemiz** |
|---|---|---|
| Yöntem | LASSO regression → 11 feature | 7 yöntem denenmiş, ham 17 feature en iyi |
| Seçilenler | BA, gender, education, residence, hypertension, dyslipidemia, CVD, smoking, drinking, BMI category, cancer | Ham 17 feature (domain FE dahil 33, selection dahil 5-15 arası) |
| Sonuç | BA tek başına baskın feature | Feature engineering tutarlı şekilde zararlı |

### Makalenin LASSO ile Seçtiği 11 Feature

1. Biological Age (BA) — **EN ÖNEMLİ**
2. Gender
3. Education
4. Residence
5. Hypertension
6. Dyslipidemia
7. Cardiovascular Disease (CVD)
8. Smoking
9. Drinking
10. BMI Category
11. Cancer

**Dikkat**: Diabetes dışlandı (LASSO ile katsayısı sıfır).

---

## 6. Model Performansları (HEAD-TO-HEAD)

| Model | **Makale AUROC** | **Bizim ROC AUC** | Fark |
|---|---|---|---|
| XGBoost | **0.9078** | ~0.6985 (CatBoost yakın) | **+0.21** |
| LightGBM | **0.8973** | 0.7055 | **+0.19** |
| CatBoost | **0.8616** | 0.6985 | **+0.16** |
| Random Forest | **0.7393** | 0.6671 | **+0.07** |
| SVM | 0.7159 | - | - |
| Decision Tree | 0.6965 | - | - |
| **En İyimiz** | - | **0.7058** (Soft Voting) | - |
| **En İyi 5-fold CV** | **0.9106** | 0.7058 | - |

---

## 7. Performans Farkının Ana Nedenleri

### Neden 1: Biyolojik Yaş Hesaplaması (EN BÜYÜK FARK)

Makale BA'yı 8 kan biyobelirtečinden **Klemera-Doubal Method (KDM)** ile hesaplıyor. Bu, yaşla ilişkili hastalık riskini tek bir güçlü feature'da yoğunlaştırıyor. SHAP analizi BA'yı açık ara en önemli feature olarak gösteriyor (>0.6).

Bizim ham "Biological Age" değişkenimiz muhtemelen aynı bilgiyi içermiyor. Domain-specific feature engineering'imiz (BioAge_60_Plus, BMI_x_BioAge) bu hesaplamayı yapmadığı için etkisiz kaldı.

### Neden 2: Sonuç Tanımı Farkı

Makale "symptomatic KOA" (diz ağrısı + hekim tanısı) kullanıyor; bizim "Arthritis" değişkenimiz daha genel ve heterojen. Daha spesifik bir outcome tanımı, modelin ayrıştırma gücünü artırır.

### Neden 3: Örneklem Seçimi

Makale eksik verisi olanları dışlayarak daha temiz ama daha küçük bir veri setiyle çalışıyor (9,505). Biz 18,046'de kalarak daha fazla veri koruyoruz ama daha gürültülü veriyle çalışıyoruz.

### Neden 4: Sınıf Dengesi

- Makale: %10.5 pozitif (aşırı dengesiz)
- Bizim: %32.5 pozitif (moderat dengesiz)

Farklı sınıf dengesi AUROC'u etkiler. Makalenin %10.5'lik pozitif oranı, modelin negatif sınıfı öğrenmesini kolaylaştırır ama pozitif sınıf recall'ını zorlaştırır.

### Neden 5: Model Parametreleri ve Validasyon

| | **Makale** | **Projemiz** |
|---|---|---|
| Split | 70/30 | 80/20 |
| Validasyon | 5-fold CV | 5-fold CV × 3 seed |
| Feature scaling | Z-score standardizasyonu | Farklı stratejiler |
| Encoding | One-hot (kategorik) | Label encoding |
| Eksik veri (kovaryatlar) | MICE | -999 kodlama |
| Düzenlileştirme | L1/L2 + early stopping | RandomSearchCV |

---

## 8. Makalenin Diğer Önemli Bulguları

### Lojistik Regresyon Bulguları

- **Sürekli BA**: Her 1 yıl artış için OR = 1.0123 (p = 0.0010)
- **Kartil karşılaştırması**:
  - Q3 vs Q1: OR = 1.4655 (p = 0.0002)
  - Q4 vs Q1: OR = 1.4519 (p = 0.0001)
- Tüm modellerde anlamlı trend (p < 0.001)

### Alt Grup Analizleri

- BA-KOA ilişkisi tüm alt gruplarda tutarlı
- Kadınlarda, kırsal bölgede yaşayanlarda, CVD'li olanlarda biraz daha güçlü
- Hiçbir interaksiyon istatistiksel olarak anlamlı değil (p > 0.05)

### SHAP Analizi (Feature Önem Sırası)

1. **Biological Age** (SHAP > 0.6) — baskın
2. Residence (kırsal = yüksek risk)
3. Gender (kadın = yüksek risk)
4. Education
5. Marital status
6. CVD
7. BMI category
8. Hypertension

---

## 9. Öneriler

### Kısa Vadeli (Hemen Uygulanabilir)

1. **KDM-BA hesapla**: CHARLS'de 8 serum biyobelirteci varsa (kolesterol, trigliserid, HbA1c, üre nitrojen, kreatinin, sistolik kan basıncı, hs-CRP, trombosit sayısı), KDM yöntemiyle BA'yı türet. Bu makalenin SHAP > 0.6 veren dominant feature'ıdır.

2. **Sonuç tanımını daralt**: "Arthritis" yerine "symptomatic KOA" (diz ağrısı + hekim tanısı) kullan. Daha spesifik outcome = daha yüksek discriminability.

3. **MICE ile eksik veri tamamlama**: Eksik kovaryatlar için MICE uygula, -999 yerine.

### Orta Vadeli

4. **Listwise deletion ile küçük veri seti oluştur**: Makalenin yaklaşımını tekrarla (9,505 örneklem) ve sonuçları karşılaştır.

5. **Outcome tanımını ikiye ayır**: 
   - "Symptomatic KOA" (diz ağrısı + hekim tanısı) → makalenin tanımına yakın
   - "General Arthritis" → mevcut yaklaşımımız

6. **LASSO feature selection**: Makalenin yaptığı gibi sadece LASSO ile feature seç, sonra XGBoost ile modelle.

### Uzun Vadeli

7. **Makalenin tam metodolojisini tekrarla**: Aynı dışlama kriterleri, aynı feature'lar, aynı model parametreleri ile çalış ve sonuçları birebir karşılaştır.

8. **RCS analizi**: BA ile KOA riski arasındaki doğrusal olmayan ilişkiyi test et (threshold efekti).

---

## 10. Özet

| Metrik | **Makale** | **Projemiz** | Fark | Ana Neden |
|---|---|---|---|---|
| En İyi AUROC | 0.9078 | 0.7058 | +0.20 | KDM-BA hesaplaması + outcome tanımı |
| En İyi Model | XGBoost | Soft Voting (RF+XGB+LGBM) | - | Model seçimi benzer |
| Veri Büyüklüğü | 9,505 | 18,046 | -9,505 | Dışlama kriterleri |
| Feature Sayısı | 11 (LASSO) | 17 (ham) | -6 | Feature selection yaklaşımı |
| Sınıf Oranı | 10.5% pozitif | 32.5% pozitif | -22% | Outcome tanımı |
| BA Hesaplama | KDM (8 biyobelirteç) | Ham değişken | - | **Kritik fark** |

**Temel Çıkarım**: "Feature engineering zararlı" bulgumuz, aslında **doğru feature engineering yapmadığımız** için geçerli. KDM-BA gibi domain-specific hesaplama, generic transformation'lardan (kare, log, etkileşim) çok daha güçlü. Makalenin yaklaşımını taklit ederek AUROC'umuzu ~0.90'a yaklaştırabiliriz.

---

## 11. Faz 2 Güncelleme (2026-04-14)

### Sheet1 Keşfi ve Kritik Değişiklik

**Sheet1** (28 kolon, 15,545 satır) daha önce hiç kullanılmamış bir veri seti olarak keşfedildi. Sheet ile karşılaştırma:

| | Sheet (Faz 1) | Sheet1 (Faz 2) |
|---|---|---|
| Satır | 38,800 | 15,545 |
| Kolon | 18 | 28 |
| Hedef | Arthritis (%32.5) | KOA (%10.6) |
| Biyobelirteç | YOK | 8 adet (log-transformed) |
| Biological Age | Ham değişken | Mevcut |

### KOA Hedef Değişkeni

- **KOA = Arthritis=yes AND position_knees=yes** (makalenin "Symptomatic KOA" tanımına eşit)
- %10.6 pozitif oran makalenin %10.5'ine çok yakın
- position_knees ve Arthritis leakage olarak çıkarıldı

### Faz 2 Sonuçları

| Konfigürasyon | AUROC | F1 | Not |
|---|---|---|---|
| **RF raw17 (Dataset A)** | **0.8937** | 0.7578 | Faz 1'den +0.19 artış! |
| **RF raw17 (Dataset B)** | **0.8939** | 0.7440 | Makalenin 0.9078'ine çok yakın |
| XGB raw17 (Dataset A) | 0.8048 | 0.3139 | |
| XGB raw17 (Dataset B) | 0.8275 | 0.4355 | |
| CatBoost raw17 | 0.71 | 0.06 | |
| LR raw17 | 0.66 | 0.01 | |
| Faz 1 en iyi (Arthritis) | 0.7058 | 0.45 | Karşılaştırma |

### KDM-BA Hesaplama Sonuçları (Beklenmedik)

KDM-BA hesaplaması yapıldı ama beklenen etkiyi göstermedi:

| Metrik | Biological Age | KDM-BA (log) | KDM-BA (orig) |
|---|---|---|---|
| KOA korelasyonu | 0.0383 | 0.0182 | 0.0208 |
| AUROC (univariate) | 0.5356 | 0.5158 | 0.5187 |
| OR (per 1 year) | 1.0115 | 1.0050 | 1.0057 |

**KDM-BA, Biological Age'den DAHA ZAYIF bir KOA predictor!** Bunun olası sebepleri:
1. Sheet1'deki biyobelirteçler yaşla zayıf korelasyona sahip (en güçlü sbp.mean r=0.42)
2. Sheet1'deki Biological Age zaten güçlü bir feature (%37 importance RF'de)
3. Biyobelirteçler log-transformed olarak saklanmış, KDM için orijinal ölçek gerekli olabilir

### Güncelleştirilmiş Özet

| Metrik | **Makale** | **Faz 1** | **Faz 2** | Fark (Faz 2 vs Makale) |
|---|---|---|---|---|
| En İyi AUROC | 0.9078 | 0.7058 | **0.8939** | **-0.014** |
| En İyi Model | XGBoost | Soft Voting | RF | |
| Feature Sayısı | 11 | 17 | 17 | |
| Sınıf Oranı | 10.5% | 32.5% | 13.3% | |
| BA Hesaplama | KDM | Ham | Ham (KDM zayıf) | |

**Temel Çıkarım (Güncelleme)**: KOA hedef değişkeni değişimi tek başına AUROC'yu 0.71'den 0.89'a taşıdı. KDM-BA'nın beklenen dominant etkiyi göstermemesi, Sheet1'deki Biological Age'in zaten güçlü bir feature olmasından kaynaklanıyor olabilir. RF'in yüksek performansı daha detaylı SHAP analizi ile doğrulanmalı.

---

*Rapor Tarihi: 2026-04-14 (Faz 2 güncellemesi)*
*Makale: PLOS ONE 20(12): e0335250 (Aralık 2025)*