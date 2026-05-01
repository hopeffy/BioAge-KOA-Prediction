# External Validation Cohort Assessment

**Project:** BioAge-KOA-Prediction  
**Purpose:** Identify suitable external cohorts for validating the Random Forest + raw17 symptomatic KOA prediction model  
**Date:** April 2026

---

## Model Feature Requirements (raw17)

To be a valid external validation cohort, the dataset must contain (or allow derivation of) the following variables:

| Variable | Type | Required | Notes |
|----------|------|----------|-------|
| wave / Time | integer | Optional | Survey wave identifiers |
| Gender | binary | Required | Male/Female |
| Age (bucket) | ordinal | Required | Age 45–59 vs ≥60 |
| Marital status | binary | Required | Married vs other |
| Education | ordinal | Required | Low/medium/high |
| Residence | binary | Required | Urban vs rural |
| Hypertension | binary | Required | Self-reported or diagnosed |
| Dyslipidemia | binary | Required | Self-reported or diagnosed |
| Diabetes | binary | Required | Self-reported or diagnosed |
| Cancer | binary | Required | Self-reported or diagnosed |
| CVD | binary | Required | Self-reported or diagnosed |
| Smoking | binary | Required | Ever vs never |
| Alcohol | binary | Required | Ever vs never |
| BMI | continuous | Required | Measured or self-reported |
| BMI category | ordinal | Derivable | Underweight/normal/overweight/obese |
| Biological Age | continuous | Required | Or equivalent aging biomarker |
| **KOA outcome** | binary | Required | Physician-diagnosed arthritis + knee pain |

**Minimum feasibility:** At least 12 of 17 predictors must be available or derivable.

---

## Candidate External Cohorts

### 1. ELSA — English Longitudinal Study of Ageing

| Attribute | Details |
|-----------|---------|
| **Country** | United Kingdom |
| **Sample size** | ~18,000 adults aged ≥50 |
| **Waves** | 9 waves (2002–2018+) |
| **KOA proxy** | Doctor-diagnosed arthritis + knee-specific pain module |
| **Key predictors available** | Gender, age, marital, education, BMI, smoking, alcohol, CVD, diabetes, hypertension |
| **Biological Age** | Can be computed from ELSA biomarker panel (CRP, HbA1c, total cholesterol, creatinine, albumin) using KDM or PhenoAge formulas |
| **Access** | UK Data Service (free, registration required): https://ukdataservice.ac.uk |
| **Feasibility** | HIGH – all core raw17 variables present; strong methodological similarity to CHARLS |
| **Main challenge** | British vs Chinese population differences (BMI cut-offs, comorbidity prevalence) |

---

### 2. HRS — Health and Retirement Study (USA)

| Attribute | Details |
|-----------|---------|
| **Country** | United States |
| **Sample size** | ~20,000 adults aged ≥51 |
| **Waves** | Biennial since 1992; most recent: 2020 |
| **KOA proxy** | Self-reported doctor-diagnosed arthritis + specific joint pain questions |
| **Key predictors available** | Gender, age, marital, education, BMI, smoking, alcohol, CVD, diabetes, hypertension, cancer |
| **Biological Age** | Computable via KDM/PhenoAge from HRS Venous Blood Study biomarker panel (available 2006+) |
| **Access** | RAND HRS public files (free): https://hrs.isr.umich.edu |
| **Feasibility** | HIGH – excellent biomarker coverage; KOA outcome derivable |
| **Main challenge** | US-specific healthcare utilization patterns; BMI distributions differ |

---

### 3. KLoSA — Korean Longitudinal Study of Aging

| Attribute | Details |
|-----------|---------|
| **Country** | South Korea |
| **Sample size** | ~10,000 adults aged ≥45 |
| **Waves** | 8 waves (2006–2020) |
| **KOA proxy** | Self-reported musculoskeletal conditions + pain module |
| **Key predictors available** | Gender, age, marital, education, BMI, smoking, alcohol, chronic disease history |
| **Biological Age** | Limited biomarker data (some waves); BA may need proxy computation |
| **Access** | Korea Employment Information Service (free, registration): https://survey.keis.or.kr |
| **Feasibility** | MEDIUM-HIGH – Asian population most comparable to CHARLS; some biomarker gaps |
| **Main challenge** | Biomarker availability varies by wave; KOA definition harmonisation needed |

---

### 4. WHO SAGE — Study on Global AGEing and Adult Health

| Attribute | Details |
|-----------|---------|
| **Country** | Multi-country (China, India, Ghana, Mexico, Russia, South Africa) |
| **Sample size** | ~50,000 adults aged ≥50 (pooled) |
| **Waves** | Wave 1 (2007–2010), Wave 2 (2014–2015) |
| **KOA proxy** | Self-reported arthritis + functional limitation |
| **Key predictors available** | Gender, age, marital, education, BMI, smoking, alcohol, chronic conditions |
| **Biological Age** | Not directly available; no biomarker panel in SAGE |
| **Access** | WHO data repository (free): https://www.who.int/teams/healthier-populations/sage |
| **Feasibility** | MEDIUM – China sub-sample (n ≈ 15,000) is most relevant; no Biological Age |
| **Main challenge** | No direct Biological Age equivalent; KOA definition less specific |

---

### 5. SHARE — Survey of Health, Ageing and Retirement in Europe

| Attribute | Details |
|-----------|---------|
| **Country** | 27 European countries |
| **Sample size** | ~140,000 adults aged ≥50 |
| **Waves** | 8 waves (2004–2020) |
| **KOA proxy** | Self-reported arthritis/rheumatism + pain modules |
| **Key predictors available** | Gender, age, marital, education, BMI, smoking, alcohol, CVD, diabetes, hypertension, cancer |
| **Biological Age** | Computable from SHARE Biomarker data (Wave 4, 6): grip strength, CRP, cholesterol |
| **Access** | SHARE-ERIC (free, registration): https://share-eric.eu |
| **Feasibility** | MEDIUM-HIGH – very large sample; European populations differ culturally |
| **Main challenge** | Multi-country heterogeneity; Biological Age computation approximated |

---

## Prioritization Recommendation

| Priority | Cohort | Rationale |
|----------|--------|-----------|
| **1st** | ELSA | Best biomarker coverage for Biological Age; accessible; similar study design to CHARLS |
| **2nd** | HRS | Large sample; strong biomarker panel; well-documented |
| **3rd** | KLoSA | Most culturally comparable (East Asian); CHARLS sister study |
| **4th** | WHO SAGE China | Same country, different wave; partially overlapping population |
| **5th** | SHARE | European context; useful for cross-cultural generalisability |

---

## Validation Analysis Protocol

When external validation data is secured, the following protocol should be applied:

1. **Feature harmonisation:** Map external cohort variables to raw17 equivalents using the harmonisation table (see `feature_harmonization_protocol.md`).
2. **Model application:** Apply the fitted Random Forest (trained on the full CHARLS Dataset A internal partition, n = 9,863) to the harmonised external dataset.
3. **Performance evaluation:**
   - Discrimination: ROC-AUC, PR-AUC
   - Calibration: Brier score, ECE, Cox calibration slope/intercept, decile reliability diagram
   - Clinical utility: DCA net benefit at 10%–30% threshold band
4. **Comparison to internal results:** Report performance alongside internal OOF and holdout metrics in a matched table.
5. **Model updating (if needed):** If calibration degrades substantially (slope < 0.8 or > 1.2, ECE > 0.08), apply logistic recalibration-in-the-large to adjust intercept.

---

## Data Access Timeline Estimates

| Cohort | Access Type | Estimated Time to Data |
|--------|-------------|------------------------|
| ELSA | Free registration | 2–4 weeks |
| HRS | Free, RAND public files | 1–2 weeks |
| KLoSA | Free registration | 3–6 weeks |
| WHO SAGE | Free download | 1–2 weeks |
| SHARE | Free registration + ethics | 4–8 weeks |
