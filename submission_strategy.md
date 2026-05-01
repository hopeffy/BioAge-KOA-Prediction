# Journal Submission Strategy

**Project:** BioAge-KOA-Prediction  
**Current Status:** Internal validation study — publishable at mid-tier level  
**Date:** April 2026

---

## Stage 1: Immediate Submission (Current Manuscript)

### Manuscript Framing

The current manuscript should be submitted as:

> **"A TRIPOD-compliant internal validation study for machine-learning-based symptomatic knee osteoarthritis risk prediction in a CHARLS-derived cohort"**

Key framing elements:
- Explicitly an **internal validation** study (not a deployment or external validation claim)
- Methodological contribution: rigorous TRIPOD-aligned design with calibration + DCA
- Not competing with the Fu et al. (2025) paper — complementing it with better validation methodology
- Reproducible, transparent pipeline

---

## Target Journals — Stage 1 (Internal Validation)

### Tier 1 Targets (Impact Factor 3–6, likely accept)

| Journal | IF (approx.) | Scope Fit | Open Access | Submission Notes |
|---------|-------------|-----------|-------------|-----------------|
| **International Journal of Medical Informatics** | 3.8 | Direct — clinical ML, prediction models | Hybrid | Strong fit; accepts TRIPOD-structured prediction model papers |
| **BMC Medical Informatics and Decision Making** | 3.2 | Direct — prediction, decision support, EHR | Full OA | BMC journals are generally favorable to internal validation studies |
| **Journal of Biomedical Informatics** | 4.5 | Direct — ML for clinical informatics | Hybrid | Rigorous peer review; calibration/DCA work well-received |
| **Computers in Biology and Medicine** | 6.7 | Applied ML in medicine | Hybrid | Large aging/disease prediction literature |

### Tier 2 Targets (Impact Factor 4–8, slightly higher bar)

| Journal | IF (approx.) | Scope Fit | Notes |
|---------|-------------|-----------|-------|
| **Journal of Medical Internet Research (JMIR)** | 7.1 | Digital health, ML prediction | Requires public code+data; dataset is openly available at Mendeley (CC BY 4.0) — fully eligible |
| **Osteoarthritis and Cartilage** | 6.8 | Direct KOA focus | Focused orthopaedic audience; expect high clinical evidence bar |
| **Aging (Albany NY)** | 5.5 | Aging + biological age focus | Good fit for BA-centred narrative |
| **Journals of Gerontology: Medical Sciences** | 5.1 | Aging cohort studies | CHARLS-based studies well-received |

### Tier 3 Targets (Backup, IF 2–3)

| Journal | IF (approx.) | Notes |
|---------|-------------|-------|
| **PLOS ONE** | 3.4 | High acceptance rate; calibration/DCA not commonly required — advantage |
| **Scientific Reports** | 4.3 | Broad scope; ML methods accepted |
| **Frontiers in Medicine** | 3.1 | Open access; rapid review |

### Recommended First Submission

**Primary:** *Journal of Medical Internet Research (JMIR)* or *International Journal of Medical Informatics*  
**Rationale:** JMIR requires publicly available data and code, which is now fully satisfied — the dataset is openly available at [Mendeley Data (CC BY 4.0)](https://data.mendeley.com/datasets/3rv7mf5pv9/1) and the pipeline code is reproducible. This is a strong competitive advantage. JMIR's IF (~7) and audience (digital health, clinical ML) make it an ideal Stage 1 venue. International Journal of Medical Informatics is the backup if JMIR's scope is too digital-health-focused for reviewers.

---

## Pre-Submission Checklist

### Must-have before submission

- [ ] Bootstrap 95% confidence intervals for ROC-AUC, PR-AUC, Brier, and ECE (TRIPOD item 16)
- [ ] Title page with author affiliations and corresponding author
- [ ] Funding statement (TRIPOD item 22)
- [ ] Conflict of interest declaration
- [ ] Ethics statement (CHARLS ethics approval reference)
- [x] Data availability statement — dataset publicly available at Mendeley Data (CC BY 4.0): https://data.mendeley.com/datasets/3rv7mf5pv9/1
- [ ] Code availability statement (GitHub/Zenodo link)
- [ ] Cover letter highlighting methodological contributions

### Should-have

- [ ] Patient/public involvement statement (if applicable)
- [ ] Supplementary figures: SHAP beeswarm plots, DCA curves for all configs
- [ ] Supplementary table: full metric table (all feature sets × all variants × all eval sets)

### Manuscript completeness check

| Section | Status | Notes |
|---------|--------|-------|
| Title + abstract | Complete | External validation limit clearly stated |
| Introduction | Complete | TRIPOD + PROBAST framing explicit |
| Methods | Complete | All TRIPOD items addressed |
| Results — primary metrics | Complete | Table 3 |
| Results — calibration/DCA | Complete | Tables 4, 5; Figures pending |
| Results — secondary branches | Complete | Reported as sensitivity analyses |
| Discussion | **Updated** | External validation gap prominently addressed |
| Limitations | **Updated** | Bootstrap CI gap added; longitudinal caveat added |
| Conclusion | **Updated** | Candidate external cohorts named |
| TRIPOD checklist | Complete | supplement_tripod_checklist.tex |
| Methods engineering | Complete | supplement_methods_engineering.tex |
| References | Complete | 15 references |

---

## Stage 2: External Validation Submission (6–18 months)

Once external validation on at least one independent cohort is completed:

### Higher-Impact Target Journals

| Journal | IF (approx.) | Why |
|---------|-------------|-----|
| **npj Digital Medicine** | 15.0 | Top clinical AI journal; multi-cohort prediction studies common |
| **JAMA Network Open** | 13.8 | High impact; rigorous TRIPOD compliance rewarded |
| **The Lancet Digital Health** | 35.0 | Highest bar; needs multi-cohort + clinical impact story |
| **BMJ** | 30.0 | Multi-cohort prediction model studies common |
| **Age and Ageing** | 7.8 | Aging + clinical prediction focus; CHARLS studies cited |

### Evidence Required for High-Impact Submission

1. External validation on ≥1 independent cohort (different country or time period)
2. Model transportability assessment (calibration slope in external cohort)
3. Net benefit in the external setting
4. Clear multi-cohort Table 1 comparing development and validation populations
5. Sensitivity analysis: model performance with and without Biological Age
6. Discussion of implementation pathway (who would use this, at what point of care)

---

## Cover Letter Template (Stage 1)

```
Dear Editors of [Journal Name],

We are pleased to submit our manuscript entitled "Predicting Symptomatic Knee 
Osteoarthritis Risk from CHARLS Data: Internal Validation with Early Unseen 
Holdout, Calibration, and Decision Analysis" for consideration as an original 
research article.

Knee osteoarthritis (KOA) is a leading cause of disability in aging populations, 
yet most machine-learning prediction models focus exclusively on discrimination 
(ROC-AUC) without adequate calibration evaluation or decision-relevance testing. 
Our study addresses this gap by presenting a TRIPOD-compliant internal validation 
pipeline that goes beyond discrimination to include calibration (Brier score, 
ECE, Cox slope/intercept, reliability diagrams), Decision Curve Analysis (DCA), 
and SHAP-based interpretability.

Key methodological contributions of our work include:
(i) an early 80/20 stratified holdout split created before any model development, 
(ii) train-fitted imputation to prevent data leakage, 
(iii) a pre-specified primary model-feature combination (Random Forest + raw17 
    sociodemographic and comorbidity variables), and 
(iv) multi-dimensional validation beyond ROC-AUC, following TRIPOD guidance.

We explicitly frame this as an internal validation study; external validation 
on independent cohorts is the acknowledged next step. We believe this manuscript 
makes a methodological contribution to the growing body of TRIPOD-compliant 
clinical prediction model studies and will be of interest to readers of [Journal].

The manuscript has not been submitted elsewhere, and all authors have approved 
the submission. There are no conflicts of interest to declare.

We look forward to your consideration.

Sincerely,
[Author Names]
```

---

## Timeline

| Milestone | Target Date |
|-----------|-------------|
| Bootstrap CIs added to manuscript | 2–3 weeks |
| GitHub repository prepared (code + README) | 2–3 weeks |
| First journal submission | 4–6 weeks |
| Expected first decision | 8–16 weeks post-submission |
| External cohort data access (ELSA/HRS) | 4–8 weeks |
| External validation analysis | 3–6 months |
| Revised/extended manuscript for high-impact journal | 6–9 months |
