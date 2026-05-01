Methods supplement — pipeline artifacts and reproduction notes

Generated: 2026-04-24

Overview
- This brief supplement documents the key pipeline artifacts, primary reproduce commands, and minimal notes about preprocessing choices used in the manuscript run.

Key artifacts (paths relative to repo root):
- `step_01_data_prep/dataset_A_internal_train.csv` (n = 9863)
- `step_01_data_prep/dataset_A_unseen_test.csv` (n = 2466)
- `step_02_baseline/baseline_metrics_summary.csv` (CSV with internal + unseen per-model metrics)
- `step_11_clinical_validation/clinical_metrics_summary.csv` (detailed metrics shown in manuscript)
- `step_11_clinical_validation/calibration_summary.csv` (calibration slopes/intercepts and ECE/MCE)
- `step_11_clinical_validation/dca_clinical_decision_summary.csv` (DCA threshold-band aggregates)
- `step_11_clinical_validation/step11_publication_paragraphs.txt` (ready-to-insert result sentences)

Primary preprocessing rules
- Missing values: train-fitted `SimpleImputer` (numeric: `median`, categorical: `most_frequent`).
- Categorical mapping: fit on training fold and map unseen categories to `-1` prior to scaling.
- Scaling: `StandardScaler` fit on training data only.
- Calibration: optional isotonic calibration via `CalibratedClassifierCV(method='isotonic')`.

Reproduce commands (from repo root)
```powershell
# create early split (step 01)
.\env\Scripts\python.exe scripts/step_01_data_prep.py

# baseline comparison (step 02)
.\env\Scripts\python.exe scripts/step_02_baseline.py

# clinical validation and calibration/DCA (step 11)
.\env\Scripts\python.exe scripts/step_11_clinical_validation.py
```

Notes and caveats
- Scripts assume a Python venv at `env` with dependencies from `requirements.txt` installed.
- The early 80/20 split is persisted; when engineered features required by a branch are absent from persisted split, the scripts apply a deterministic fallback split (same random seed) and note the `split_source` in outputs.
- All reported metrics are internal (OOF) or unseen holdout within the same CHARLS-derived cohort; external validation is outside the scope of this supplement.

Contact
- For reproduction questions, inspect `scripts/step_01_data_prep.py`, `scripts/step_02_baseline.py`, and `scripts/step_11_clinical_validation.py` for implementation details and parameter knobs.