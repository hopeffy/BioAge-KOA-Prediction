# Dataset Versions - Phase 2

## Dataset v3 | 2026-04-14 | Sheet1 Data (Phase 2)

### Kaynak
- Dosya: Raw Data .xlsx, Sheet1
- 15,545 satir x 28 kolon
- 8 biyobelirtec (log-transformed) + KOA hedefi + encoded features

### Islemler
1. position_knees=NaN satirlar cikarildi → 12,329 satir
2. position_knees ve Arthritis leakage olarak cikarildi
3. Log-transformed biyobelirtecler orijinal olcege cevrildi (exp())
4. KDM-BA hesaplandi (log ve orig)
5. BIR (Biological Illness Risk) hesaplandi
6. BA kartilleri olusturuldu

### Dosyalar
- dataset_A_all.csv: 12,329 satir, 33 feature, %13.3 KOA
- dataset_B_ba55.csv: 7,635 satir, 33 feature, %14.2 KOA
- dataset_A_with_kdm_ba.csv: 12,329 satir, 33+5 feature (KDM-BA, BIR, kartiller)
- dataset_B_with_kdm_ba.csv: 7,635 satir, 33+5 feature

### Feature List (33 raw features, leakage kolonu cikarildi)
1. wave
2. Time
3. Gender (1=Male, 2=Female)
4. Age_New (1=younger, 2=older)
5. Marital (1=married, 2=other)
6. Education (1=low, 2=medium, 3=high)
7. Residence (1=urban, 2=rural)
8. Hypertension (0/1)
9. Dyslipidemia (0/1)
10. Diabetes (0/1)
11. Cancer (0/1)
12. CVD (0/1)
13. Smoke (0/1)
14. Drink (0/1)
15. BMI (continuous)
16. BMI_New (1=normal, 2=overweight, 3=obese)
17. Biological Age (continuous)
18-25. 8 biyobelirtec (log-transformed)
26-33. 8 biyobelirtec (original scale)

### Ek Ozellikler (KDM-BA dosyalarinda)
34. BA_KDM_log
35. BA_KDM_orig
36. BIR_log
37. BIR_orig
38. Biological Age_Qint
39. BA_KDM_log_Qint
40. BA_KDM_orig_Qint

### Target
- KOA: 0=negatif (86.7%), 1=pozitif (13.3%)
- KOA = Arthritis=yes AND position_knees=yes

### Leakage Control
- position_knees: CIKARILDI (KOA'nin taniminda kullanildi)
- Arthritis: CIKARILDI (KOA'nin taniminda kullanildi)

---

## Dataset v2 (Phase 1 - Arsiv)
- Ham: 18,046 satir, 18 kolon
- Engineered: 18,046 satir, 43 feature
- Hedef: Arthritis (%32.5 pozitif)
- Faz 1 tamamlanoi, phase1_preliminary/ altinda arsivlendi

## Dataset v1 (Phase 1 - Arsiv)
- Raw: 38,800 satir, 18 kolon (Sheet)
- Faz 1 tamamlanoi, phase1_preliminary/ altinda arsivlendi