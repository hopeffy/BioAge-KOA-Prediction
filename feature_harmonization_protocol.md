# Feature Harmonization Protocol for External Validation

**Project:** BioAge-KOA-Prediction  
**Purpose:** Map raw17 CHARLS variables to equivalent variables in external cohorts  
**Date:** April 2026

---

## Overview

The raw17 feature set contains 17 variables drawn from CHARLS survey questions. External validation requires mapping each variable to the closest equivalent in the target cohort, documenting any definitional differences, and applying consistent encoding rules.

**Decision rule:** A feature is considered "harmonisable" if the external cohort contains either (a) the identical or near-identical question, or (b) a variable from which the same binary/ordinal encoding can be derived.

---

## Variable-by-Variable Harmonization Map

### 1. Survey Wave Variables

| CHARLS Variable | Encoding | ELSA Equivalent | HRS Equivalent | KLoSA Equivalent |
|----------------|----------|----------------|----------------|-----------------|
| `wave` | integer (1, 2, 3...) | `wave` | `wave` | `wave` |
| `Time` | year integer | year of interview | year of interview | year of interview |
| **Action** | Retain as survey identifiers | Direct map | Direct map | Direct map |

**Note:** These are time indicators, not predictors in the clinical sense. Include if available, otherwise omit and re-assess feature importance without these two variables.

---

### 2. Demographics

#### Gender
| CHARLS | `Gender` (1=male, 2=female) |
|--------|----------------------------|
| ELSA | `ragender` or `gender` (1=male, 2=female) |
| HRS | `ragender` in RAND file (1=male, 2=female) |
| KLoSA | `gender` (1=male, 2=female) |
| WHO SAGE | `sex` (1=male, 2=female) |
| **Mapping:** | Direct — no transformation needed |

#### Age Bucket
| CHARLS | `Age_New` (1=45–59, 2=≥60) |
|--------|-----------------------------|
| ELSA | Derive from `age_at_interview`: if 45–59 → 1, if ≥60 → 2 |
| HRS | Derive from `ragey_b` (age at last birthday) |
| KLoSA | Derive from `birth_year` and interview year |
| WHO SAGE | Derive from `age` variable |
| **Mapping:** | Thresholded derivation — standard |

#### Marital Status
| CHARLS | `Marital` (1=married, 2=other) |
|--------|--------------------------------|
| ELSA | `ramliv` or `mstat`: partnered/married → 1, else → 2 |
| HRS | `rmastat`: married/partnered → 1, widowed/divorced/single → 2 |
| KLoSA | Marital status field: married → 1, other → 2 |
| **Mapping:** | Binary collapse of multi-category marital variable |

#### Education
| CHARLS | `Education` (1=low, 2=medium, 3=high) |
|--------|---------------------------------------|
| ELSA | `edqual` or ISCED: primary→1, secondary→2, tertiary→3 |
| HRS | `raedyrs` (years): <9→1, 9–12→2, >12→3 |
| KLoSA | Education years or level: apply same cut-offs as HRS |
| WHO SAGE | `q1009` (education level): apply ISCED mapping |
| **Mapping:** | Map to low/medium/high categories using ISCED or years-of-education thresholds |
| **Caveat:** | Chinese education levels (no schooling / elementary / middle / high / college) may not map perfectly to Western ISCED; document any definitional difference |

#### Residence
| CHARLS | `Residence` (1=urban, 2=rural) |
|--------|--------------------------------|
| ELSA | No direct equivalent (UK is largely urban); use region type if available |
| HRS | `raceregion` or geographic codes; rural/non-rural dichotomy |
| KLoSA | Urban vs rural residence (direct match) |
| WHO SAGE | Urban/rural sampling strata (direct match) |
| **Mapping:** | Direct where available; ELSA is a challenge (predominantly urban UK) — document as partial |

---

### 3. Comorbidities (Binary 0/1)

All comorbidities in raw17 are self-reported physician diagnoses in CHARLS. The mapping priority is:
1. Self-reported physician diagnosis (preferred, same measurement approach)
2. Biomarker-confirmed diagnosis (e.g., HbA1c ≥ 6.5% for diabetes) as secondary

#### Hypertension
| CHARLS | Self-reported physician-diagnosed hypertension (0/1) |
| ELSA | `hedawhy_3` or `bp_ever_diagnosed`: Yes → 1 |
| HRS | `diab` series or `hibpe`: diagnosed → 1 |
| KLoSA | Chronic disease module: hypertension (직접 yes/no) |
| **Mapping:** | Direct |

#### Dyslipidemia
| CHARLS | Self-reported physician-diagnosed dyslipidemia (0/1) |
| ELSA | `hedawhy_14` (high cholesterol ever diagnosed) |
| HRS | `cholst` (cholesterol medication or diagnosis) |
| KLoSA | Dyslipidemia/hyperlipidemia diagnosis field |
| **Mapping:** | Direct; note that dyslipidemia prevalence differs between countries |

#### Diabetes
| CHARLS | Self-reported physician-diagnosed diabetes (0/1) |
| ELSA | `hedawhy_5`: diabetes diagnosis |
| HRS | `diabe` in RAND file |
| KLoSA | Diabetes field in chronic disease module |
| **Mapping:** | Direct |

#### Cancer
| CHARLS | Self-reported physician-diagnosed cancer (0/1) |
| ELSA | `hedawhy_24` or cancer ever |
| HRS | `cancre` in RAND file |
| KLoSA | Cancer diagnosis field |
| **Mapping:** | Direct; note very low prevalence (~0.8% in CHARLS) — external prevalence may differ |

#### Cardiovascular Disease (CVD)
| CHARLS | Self-reported physician-diagnosed CVD (0/1) |
| ELSA | Heart disease + stroke composite; `hedawhy_6` + `hedawhy_9` |
| HRS | `hearte` or `stroke` (composite) in RAND file |
| KLoSA | CVD/heart disease field |
| **Mapping:** | Composite of heart disease and stroke diagnoses; document composite definition |

---

### 4. Lifestyle

#### Smoking
| CHARLS | `Smoke` — ever vs never smoker (0=never, 1=ever) |
|--------|--------------------------------------------------|
| ELSA | `smoker`: ever smoked → 1; never → 0 |
| HRS | `rsmoken` (ever smoked regularly) → binary |
| KLoSA | Smoking history: ever/never binary |
| **Mapping:** | Direct; "ever" definition is consistent across cohorts |

#### Alcohol
| CHARLS | `Drink` — ever vs never drinker (0=never, 1=ever) |
|--------|---------------------------------------------------|
| ELSA | `drinker`: ever drank → 1; abstainer → 0 |
| HRS | `rdrinkr` (drink at all) |
| KLoSA | Alcohol consumption history: ever/never |
| **Mapping:** | Direct; note cultural differences in alcohol reporting (non-Muslim bias in CHARLS is negligible given Chinese cohort) |

---

### 5. Physical Measures

#### BMI (continuous)
| CHARLS | Measured BMI (kg/m²) |
|--------|----------------------|
| ELSA | Nurse visit measured BMI (waves 2, 4, 6, 8) |
| HRS | Self-reported height and weight → derived BMI (some waves have measured) |
| KLoSA | Self-reported or measured BMI |
| **Mapping:** | Use measured where available; document if self-reported (generally leads to slight underestimation) |

#### BMI Category (ordinal)
| CHARLS | `BMI_New` (1=underweight/normal <25, 2=overweight 25–30, 3=obese ≥30) |
|--------|----------------------------------------------------------------------|
| External | Derive from continuous BMI using the same cut-offs: |
|  | <25 → 1, 25–29.99 → 2, ≥30 → 3 |
| **Caveat:** | WHO Asian BMI cut-offs differ (overweight: ≥23, obese: ≥27.5). CHARLS uses standard Western WHO cut-offs (≥25, ≥30). Apply the same standard Western cut-offs for cross-cohort consistency; document this choice. |

---

### 6. Biological Age (critical variable)

`Biological Age` is the most challenging variable to harmonise because CHARLS supplies it as a pre-computed field.

#### Computation Options

| Method | Inputs Required | Cohorts Feasible |
|--------|----------------|-----------------|
| CHARLS-supplied value | Pre-computed in CHARLS | CHARLS only |
| KDM-BA (Klemera-Doubal) | 7–10 biomarkers + chronological age | ELSA, HRS (biomarker waves) |
| PhenoAge (Levine 2018) | 9 biomarkers (CRP, albumin, creatinine, glucose, lymph %, MCV, RDW, alkaline phosphatase, WBC) | ELSA, HRS, partial SHARE |
| Simplified BA proxy | Chronological age + comorbidity count | All cohorts (lower validity) |

#### Recommended approach per cohort:

- **ELSA:** Use ELSA Nurse Visit biomarkers (CRP, total cholesterol, HDL, HbA1c, creatinine) to compute KDM-BA or PhenoAge using `step_03_kdm_ba.py` adapted for ELSA variable names.
- **HRS:** Use HRS Venous Blood Study (waves 2006, 2008, 2010, 2012, 2014, 2016) biomarkers to compute KDM-BA or PhenoAge.
- **KLoSA:** Biological age biomarkers not available in all waves; use chronological age + comorbidity sum as proxy if needed; document sensitivity analysis.
- **WHO SAGE:** No biomarker panel; use chronological age as proxy; clearly flag as a limitation.

#### Sensitivity analysis plan:
Run models both with and without `Biological Age` for each external cohort where the Biological Age field must be approximated. Report the difference in performance.

---

## KOA Outcome Harmonization

The CHARLS KOA outcome is defined as:
```
KOA = (Arthritis == "yes") AND (position_knees == "yes")
```

This corresponds to **physician-diagnosed arthritis with knee-localised joint pain**.

| Cohort | KOA Proxy Definition |
|--------|---------------------|
| ELSA | `hedawhy_12` (arthritis/rheumatism ever) AND knee pain in past month |
| HRS | `arthrhr` (arthritis diagnosed) AND knee pain item |
| KLoSA | Arthritis/degenerative joint disease AND knee location |
| WHO SAGE | Q6000 series (chronic conditions: arthritis) AND knee-specific pain |

**Documentation requirement:** Record the exact question text and response options from each external survey that map to the CHARLS outcome definition.

---

## Harmonization Quality Scoring

Apply the following scoring rubric for each external cohort before validation:

| Score | Meaning |
|-------|---------|
| 3 | Direct match — same or equivalent question, same encoding |
| 2 | Derivable — slight recoding required |
| 1 | Approximation — conceptually similar but definitionally different |
| 0 | Unavailable — must be excluded or proxied |

Target: Average harmonization score ≥ 2.0 across all 17 variables for cohort to be used in primary external validation. Cohorts scoring 1.5–2.0 can be used as secondary sensitivity analyses.

---

## Harmonization Tracking Template

Create a spreadsheet with columns:
`variable | charls_definition | external_variable | external_definition | harmonization_score | notes`

Save as: `external_validation/harmonization_map_{cohort_name}.xlsx`

---

## Next Steps After Harmonization

1. Apply `step_01_data_prep.py` exclusion logic to external cohort (drop rows where KOA outcome cannot be verified).
2. Encode variables using the same schema as raw17.
3. Load the trained Random Forest model from Step 11 (retrain on full CHARLS n = 9,863 internal partition).
4. Apply model to harmonised external data using the same preprocessing (train-fitted imputer on CHARLS data, applied to external data).
5. Evaluate using the full metric suite (ROC-AUC, PR-AUC, Brier, ECE, DCA).
6. Compare against internal OOF and holdout benchmarks.
