# Session Summaries - Phase 2

---

## Session 5 | 2026-04-14 | Sheet1 Discovery & Phase 2 Setup

### Bugun Yapilanlar

1. **Sheet1 kesfi**: Excel dosyasinda 2. sheet (28 kolon, 15,545 satir) daha once hic kullanilmamis
   - 8 biyobelirtec log-transformed olarak mevcut
   - KOA hedef degiskeni zaten mevcut (makalenin tanimina esit)
   - Sheet1'de 0 eksik veri (position_knees haric)

2. **Faz 1 arsivleme**: Step 0-14 klasorleri phase1_preliminary/ altina tasindi

3. **Step 01**: Sheet1 veri hazirlama
   - Log ters donusumu (exp()) yapildi
   - KOA tanimi dogrulandi (100% match)
   - Leakage kontrolu (Arthritis + position_knees cikarildi)
   - Dataset A: 12,329 satir, Dataset B: 7,635 satir

4. **Step 02**: Baseline modeller
   - RF raw17: **AUROC = 0.89** (Faz 1'den +0.19 artis!)
   - Biyobelirtec eklemek RF'yi dusurdu (0.89 → 0.74)
   - Feature importance: BA(37%) + BMI(37%) = dominant

5. **Step 03**: KDM-BA hesaplama
   - KDM-BA hesaplandi ama KOA korelasyonu cok dusuk (r=0.02)
   - Biological Age KOA korelasyonu: r=0.04 (daha yuksek ama yine dusuk)
   - Bu beklenmedik bir sonuc — makaledeki SHAP>0.6 etkisi gorulmedi

### Elde Edilen Sonuclar

| Konfig | AUROC | F1 |
|--------|-------|-----|
| RF raw17 (Dataset A) | 0.8937 | 0.7578 |
| RF raw17 (Dataset B) | 0.8939 | 0.7440 |
| XGB raw17 (Dataset A) | 0.8048 | 0.3139 |
| XGB raw17 (Dataset B) | 0.8275 | 0.4355 |

### Kritik Bulgular

1. **Hedef degiskeni degisimi en buyuk etki**: Arthritis → KOA tek basina +0.19 AUROC artisi
2. **RF'in yuksek performansi**: Makalenin 0.9078'ine 0.89 ile cok yakin
3. **KDM-BA bekleneni vermedi**: Sheet1'deki Biological Age zaten guclu, KDM ekleme fayda saglamadi
4. **Biyobelirtec eklemek RF'yi dusurdu**: Noise ekledi, information/art oracle degil

### Acik Sorular

1. RF'in 0.89'u overfitting mi? Detayli SHAP analizi gerekli
2. Sheet1'deki Biological Age nasil hesaplanmis? (CHARLS'in kendi yontemi?)
3. Makale ile 0.01'lik fark nerden? (Encoding, LASSO, parametre farklari)
4. XGB neden yuksek AUROC vermiyor (0.80 vs makalenin 0.91)?

### Sonraki Adimlar

- Step 04: SHAP analizi ile RF'in yuksek performansini dogrulama
- Step 05: Makale replikasyonu (LASSO, one-hot, Z-score)
- Step 06: RCS ve alt grup analizleri
- Step 07: Final model optimizasyonu

---

**Last Updated**: 2026-04-14
**Phase 2 Experiments**: 3 completed
**Best Result**: RF raw17, AUROC = 0.8939