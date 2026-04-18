# Feature Notes

Feature engineering detayları, feature set versiyonları, şüpheli features, ve feature importance notları.

---

## Feature Set Versions

### Feature Set v1: Raw Features (18 features)

**Used in**: Exp-001 (Raw Baseline)

```
1. wave                               # Study wave (temporal)
2. iyear                              # Interview year
3. Biological Age                     # Age in years
4. bmi_kg.m2                          # Body Mass Index
5. Sex                                # Gender
6. Residence_Type                     # Urban/Rural
7. doctor_diagnose_hypertension       # Hypertension diagnosis
8. doctor_diagnose_diabetes           # Diabetes diagnosis
9. doctor_diagnose_heart_disease      # Heart disease diagnosis
10. doctor_diagnose_stroke            # Stroke diagnosis
11. doctor_diagnose_cancer            # Cancer diagnosis
12. Arthritis                         # TARGET VARIABLE
13. MET                               # Metabolic Equivalent Task
14. edu_level                         # Education level
15. r_smoking                         # Smoking status
16. r_drinking                        # Drinking status
17. position_knees                    # Knee position/pain (ordinal)
18. marital_status                    # Marital status
```

**Performance**: ROC AUC = 0.6543, F1 = 0.4189

---

### Feature Set v2: Engineered Features (55 → 43 active)

**Used in**: Exp-002 to Exp-012

#### Aging Features (10)
```python
# Polynomial transforms
Age_Squared = Biological_Age ** 2
Age_Cubed = Biological_Age ** 3
Biological_Age_Cubed = Biological_Age ** 3  # Duplicate?

# Log transform
Log_Age = log(Biological_Age)

# Age thresholds
BioAge_60_Plus = (Biological_Age >= 60).astype(int)  # 🏆 Consensus feature!
BioAge_70_Plus = (Biological_Age >= 70).astype(int)
BioAge_Over_67 = (Biological_Age > 67).astype(int)

# Age categories
Age_Category = pd.cut(Biological_Age, bins=[0,40,60,80,120])

# Interactions (listed separately below)
Age_x_Comorbidity
Age_x_BMI  # Same as BMI_x_BioAge?
```

#### BMI Features (7)
```python
# Polynomial
BMI_Squared = bmi_kg.m2 ** 2

# Log transform
Log_BMI = log(bmi_kg.m2)

# Categories
BMI_Category = pd.cut(bmi_kg.m2, bins=[0,18.5,25,30,100], 
                       labels=['underweight','normal','overweight','obese'])
BMI_Obese = (bmi_kg.m2 >= 30).astype(int)

# Interactions
BMI_x_BioAge = bmi_kg.m2 * Biological_Age  # 🏆 Consensus feature!
BMI_x_Comorbidity
Sex_x_BMI
```

#### Comorbidity Features (3)
```python
# Count of comorbidities
Comorbidity_Count = (
    doctor_diagnose_hypertension + 
    doctor_diagnose_diabetes + 
    doctor_diagnose_heart_disease + 
    doctor_diagnose_stroke + 
    doctor_diagnose_cancer
)

# Weighted risk score
Comorbidity_Risk_Score = (
    1.0 * doctor_diagnose_hypertension +
    1.5 * doctor_diagnose_diabetes +
    2.0 * doctor_diagnose_heart_disease +
    2.5 * doctor_diagnose_stroke +
    2.0 * doctor_diagnose_cancer
)

# Multiple comorbidity flag
Has_Multiple_Comorbidities = (Comorbidity_Count >= 2).astype(int)
```

**Note**: Comorbidity_Count ve Comorbidity_Risk_Score perfect correlation (r=1.000) → redundant!

#### Disease-Specific Features (5)
```python
# Renamed/encoded versions (same as raw)
Hypertension = doctor_diagnose_hypertension
Diabetes = doctor_diagnose_diabetes
Heart_Disease = doctor_diagnose_heart_disease
Stroke = doctor_diagnose_stroke
Cancer = doctor_diagnose_cancer
```

#### Metabolic Risk Features (2)
```python
# Composite metabolic risk
Metabolic_Risk_Score = (
    BMI_Obese * 2 + 
    Hypertension + 
    Diabetes * 1.5
)

# High risk flag
High_Metabolic_Risk = (Metabolic_Risk_Score > threshold).astype(int)
```

#### Gender-Specific Features (2)
```python
Sex_Numeric = Sex.map({'Male': 0, 'Female': 1})
Sex_x_Age = Sex_Numeric * Biological_Age
```

#### Socioeconomic Features (3+)
```python
# Encoding details not fully documented
Residence_Level = encode(Residence_Type)
Education_Encoded = encode(edu_level)
Marital_Status_Encoded = encode(marital_status)
```

#### Other Features
```python
Position_Knees_Encoded = ordinal_encode(position_knees)  # ⚠️ F=349.73!
MET  # Same as raw
wave  # Same as raw
iyear  # Same as raw
r_smoking  # Same as raw
r_drinking  # Same as raw
```

**Total**: 55 features created, **43 used** in pipeline (12 redundant/unused)

---

## 🏆 Consensus Features (Selected by All 6 Methods)

These features were selected by ALL 6 feature selection methods (ANOVA, MI, L1 LASSO, Tree-Based, RFECV, Hierarchical):

1. **BioAge_60_Plus** - Age ≥ 60 indicator
   - Selected by: 6/6 methods ⭐⭐⭐
   - Reason: Strong age effect on arthritis risk
   - Type: Binary (0/1)

2. **BMI_x_BioAge** - BMI × Age interaction
   - Selected by: 6/6 methods ⭐⭐⭐
   - Reason: Joint effect of obesity and aging
   - Type: Continuous

**Core Features (Selected by ≥5 methods)**:

3. **wave** - Study wave (5/6)
4. **doctor_diagnose_heart_disease** - Heart disease (5/6)
5. **Sex_Numeric** - Gender (5/6)
6. **Age_x_Comorbidity** - Age × Comorbidity interaction (5/6)

**Strong Features (Selected by ≥4 methods)**:

7. iyear
8. Has_Multiple_Comorbidities
9. BMI_x_Comorbidity
10. MET

**Moderate Features (Selected by ≥3 methods)**:

11-22. (22 total features selected by 3+ methods)

See `step4_feature_selection_comparison/` for full overlap analysis.

---

## ⚠️ Şüpheli Features (Leakage / Problematic)

### 1. Position_Knees_Encoded

**Şüphe**: 
- ANOVA F-score = 349.73 (aşırı yüksek, 2nd highest = 148.79)
- Diz pozisyonu/ağrısı arthritis'in semptomu olabilir (leakage riski)

**Test Sonuçları** (Exp-003):
- Without Position_Knees: ROC AUC = 0.5832 (only -8% drop)
- Yorum: Kritik değil, leakage riski düşük

**Karar**:
- ✅ Production'da kullanılabilir
- ⚠️ Dikkatle izlenmeli
- Alternative: Position_Knees olmadan model de deploy edilebilir

---

### 2. Comorbidity_Count vs Comorbidity_Risk_Score

**Problem**:
- Perfect correlation (r = 1.000)
- Redundant features

**Çözüm**:
- Hierarchical clustering ile birini seç
- Comorbidity_Risk_Score daha anlamlı (weighted)

---

### 3. Age_Cubed vs Biological_Age_Cubed

**Problem**:
- Duplicate features (aynı hesaplama)

**Çözüm**:
- Birini kaldır

---

### 4. Feature Interactions

**Potential Redundancy**:
- Age_x_BMI vs BMI_x_BioAge (aynı mı?)
- Check: İkisi de aynı feature olabilir

**Multicollinearity**:
- 7 high-correlation pairs detected (r > 0.8)
- Feature selection ile handle edildi

---

## 🎯 Recommended Feature Sets

### Minimal Set (5 features) - Exp-006
**Best F1 Score** (0.4263)

```
1. BioAge_60_Plus
2. Heart_Disease
3. BMI_Category
4. Cancer
5. Sex_Numeric
```

**Avantajları**:
- 88% feature reduction
- Multicollinearity-free
- Yorumlanabilir (her feature bir clinical domain)
- Best F1 ve Accuracy

**Kullanım**: Lightweight deployment, edge devices

---

### Balanced Set (15 features) - Exp-012
**Best Overall Performance** (Hierarchical Clustering)

```
1. BioAge_60_Plus
2. BMI_x_BioAge
3. wave
4. doctor_diagnose_heart_disease
5. Sex_Numeric
6. Age_x_Comorbidity
7. Comorbidity_Risk_Score
8. Hypertension
9. BMI_Category
10. MET
11. Log_BMI
12. (5 more - see step4 results)
```

**Performans**:
- ROC AUC: 0.5912 (2nd best)
- F1: 0.4253 (best)
- Accuracy: 0.5931 (best)

**Kullanım**: Production deployment (recommended)

---

### Maximum Set (15 features) - Exp-009
**Best ROC AUC** (L1 LASSO)

```
Features selected by L1 regularization
(see step4_feature_selection_comparison/feature_selection_comparison_results.txt)
```

**Performans**:
- ROC AUC: 0.5927 (best)
- F1: 0.4238
- Accuracy: 0.5881

**Kullanım**: When AUC is primary metric

---

### Consensus Core (6 features) - Not Yet Tested
**High Confidence Set**

```
1. BioAge_60_Plus (6/6 methods)
2. BMI_x_BioAge (6/6 methods)
3. wave (5/6 methods)
4. doctor_diagnose_heart_disease (5/6 methods)
5. Sex_Numeric (5/6 methods)
6. Age_x_Comorbidity (5/6 methods)
```

**Next Experiment**: Exp-013 (planned)

---

## Feature Importance Rankings

### By ANOVA F-score (Top 10)

1. Position_Knees_Encoded: F=349.73 ⚠️
2. Residence_Level: F=148.79
3. Heart_Disease: F=133.77
4. Sex_Numeric: F=118.34
5. BioAge_60_Plus: F=104.41
6. Age_x_Comorbidity: F=92.92
7. Marital_Status_Encoded: F=86.14
8. BMI_x_Comorbidity: F=79.47
9. Hypertension: F=74.52
10. Comorbidity_Risk_Score: F=73.89

### By Mutual Information (Top 10)

See `step4_feature_selection_comparison/feature_selection_comparison_results.txt`

### By L1 Coefficients (Top 10)

See `step4_feature_selection_comparison/feature_selection_comparison_results.txt`

### By Tree Importance (RF + XGBoost Average, Top 10)

See `step4_feature_selection_comparison/feature_selection_comparison_results.txt`

---

## Feature Engineering İçin Gelecek Fikirler

### 1. Temporal Features
- Time since first wave
- Change in BMI over waves
- Disease progression indicators

### 2. Ratio Features
- BMI / Age ratio
- Comorbidity density (count / age)
- Physical activity vs age ratio

### 3. Clustering Features
- Patient clusters (K-means on demographics)
- Risk groups (hierarchical clustering)

### 4. Domain-Specific Features
- Frailty index
- Cardiovascular risk score (established formula)
- Obesity-age interaction categories

### 5. External Data
- Regional health statistics
- Socioeconomic indicators by residence
- Healthcare access metrics

---

## Multicollinearity Analysis

### Detected High-Correlation Pairs (r > 0.8)

1. Comorbidity_Count ↔ Comorbidity_Risk_Score (r=1.000) ⚠️
2. Age_Squared ↔ Age_Cubed (r=0.95+)
3. BMI_Squared ↔ BMI (r=0.90+)
4. ... (7 pairs total)

**Solution**: Hierarchical clustering eliminates these automatically

---

## Feature Pre-inference Availability Check

**Critical for Production**: Hangi features inference sırasında bulunmayacak?

✅ **Available at Inference**:
- Demographics: Age, Sex, Residence
- Anthropometrics: BMI, weight, height
- Lifestyle: Smoking, drinking, MET
- Medical history: Diagnosed diseases
- Socioeconomic: Education, marital status

❓ **Questionable**:
- position_knees: Arthritis semptomu olabilir
- wave, iyear: Temporal features (future waves?)

❌ **Not Available**:
- Future diagnoses
- Longitudinal changes

**Recommendation**: Review feature availability with clinical team

---

**Last Updated**: 2026-03-10  
**Current Feature Set**: v2 (43 active features)  
**Recommended Set**: Hierarchical Clustering (15 features)  
**Next Steps**: Test consensus core (6 features)
