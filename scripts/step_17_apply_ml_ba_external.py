"""
STEP 17: Apply ML-BA Predictors to External Cohorts
====================================================
Apply the two ML-BA predictors trained in step_16 to ELSA and HRS, then
evaluate KOA prediction under four feature scenarios:

  1. raw16 only
  2. raw16 + ML_BA
  3. raw16 + ML_KDM_BA
  4. raw16 + ML_BA + ML_KDM_BA (ensemble BA feature)

This directly implements the feedback: predicted BA applied to all cohorts,
single-feature BA outputs, and ensemble BA prediction.

Inputs
------
  step_16_ml_ba_predictor/ml_ba_combo_ensemble_model.pkl
  step_ext_val_elsa/elsa_analysis_cohort.csv
  step_ext_val_hrs/hrs_analysis_cohort.csv
  step_01_data_prep/dataset_A_internal_train.csv

Outputs
-------
  step_17_ml_ba_external/
    elsa_with_ml_ba.csv
    hrs_with_ml_ba.csv
    step17_metrics.csv
    step17_report.txt
"""

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ML_BA_ARTIFACT = os.path.join(BASE_DIR, "step_16_ml_ba_predictor", "ml_ba_combo_ensemble_model.pkl")
ELSA_COHORT = os.path.join(BASE_DIR, "step_ext_val_elsa", "elsa_analysis_cohort.csv")
HRS_COHORT = os.path.join(BASE_DIR, "step_ext_val_hrs", "hrs_analysis_cohort.csv")
CHARLS_TRAIN = os.path.join(BASE_DIR, "step_01_data_prep", "dataset_A_internal_train.csv")
OUT_DIR = os.path.join(BASE_DIR, "step_17_ml_ba_external")

N_TREES = 300
RANDOM_STATE = 42

# raw16 feature set (must match step_16)
RAW16 = [
    "Gender", "Age_New", "Marital", "Education", "Residence",
    "Hypertension", "Dyslipidemia", "Diabetes", "Cancer", "CVD",
    "Smoke", "Drink", "BMI", "BMI_New",
]

# HRS uses _proxy suffixes for Residence/Dyslipidemia
HRS_RAW16 = [
    "Gender", "Age_New", "Marital", "Education", "Residence_proxy",
    "Hypertension", "Dyslipidemia_proxy", "Diabetes", "Cancer", "CVD",
    "Smoke", "Drink", "BMI", "BMI_New",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensemble_predict(artifact, df, feature_cols):
    """Apply one ML-BA ensemble to a DataFrame."""
    X = df[feature_cols].values.astype(float)
    X_imp = artifact["imputer"].transform(X)
    X_sc = artifact["scaler"].transform(X_imp)
    preds = np.column_stack([m.predict(X_sc) for m in artifact["models"].values()])
    return preds.mean(axis=1)


def apply_ml_ba_to_cohort(cohort_df, combo_artifact, raw16_cols, cohort_name):
    """Add ML_BA and ML_KDM_BA columns to an external cohort."""
    df = cohort_df.copy()

    # Determine which features the combo artifact was trained on
    artifact_feature_cols = combo_artifact["feature_cols"]

    for col_name, artifact in combo_artifact["artifacts"].items():
        # Align features: combo artifact may have slightly different feature_cols
        feat_cols = artifact["feature_cols"]
        missing = [c for c in feat_cols if c not in df.columns]
        if missing:
            print(f"  WARNING for {cohort_name}/{col_name}: missing {missing}; filling with NaN")
            for c in missing:
                df[c] = np.nan
        df[col_name] = ensemble_predict(artifact, df, feat_cols)

    return df


def train_and_evaluate(charls_df, ext_df, feature_cols, scenario_name, cohort_name):
    """Train RF on CHARLS and evaluate on external cohort."""
    X_charls = charls_df[feature_cols].values.astype(float)
    y_charls = charls_df["KOA"].values
    X_ext = ext_df[feature_cols].values.astype(float)
    y_ext = ext_df["KOA"].values

    # Drop rows with missing target
    valid_charls = ~np.isnan(y_charls)
    valid_ext = ~np.isnan(y_ext)
    if not valid_charls.all():
        X_charls = X_charls[valid_charls]
        y_charls = y_charls[valid_charls]
    if not valid_ext.all():
        X_ext = X_ext[valid_ext]
        y_ext = y_ext[valid_ext]

    # Preprocess: impute + scale fit on CHARLS
    imp = SimpleImputer(strategy="median")
    X_charls_imp = imp.fit_transform(X_charls)
    X_ext_imp = imp.transform(X_ext)
    scaler = StandardScaler()
    X_charls_sc = scaler.fit_transform(X_charls_imp)
    X_ext_sc = scaler.transform(X_ext_imp)

    rf = RandomForestClassifier(
        n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_charls_sc, y_charls)
    proba = rf.predict_proba(X_ext_sc)[:, 1]

    metrics = {
        "cohort": cohort_name,
        "scenario": scenario_name,
        "n": len(y_ext),
        "prevalence": float(y_ext.mean()),
        "roc_auc": roc_auc_score(y_ext, proba),
        "pr_auc": average_precision_score(y_ext, proba),
        "brier": brier_score_loss(y_ext, proba),
    }

    # Best F1 threshold
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1 = -1.0
    best_thr = 0.5
    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        f1 = f1_score(y_ext, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    best_preds = (proba >= best_thr).astype(int)
    metrics.update({
        "best_f1_threshold": float(best_thr),
        "f1": best_f1,
        "precision": precision_score(y_ext, best_preds, zero_division=0),
        "recall": recall_score(y_ext, best_preds, zero_division=0),
    })

    return metrics, proba


def load_or_create_cohort(path, cohort_name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{cohort_name} cohort file not found at {path}. "
            f"Run the corresponding external validation script first."
        )
    return pd.read_csv(path)


def main():
    print("=" * 80)
    print("STEP 17: APPLY ML-BA PREDICTORS TO EXTERNAL COHORTS")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    ensure_dir(OUT_DIR)

    # -----------------------------------------------------------------------
    # Load artifacts and data
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING ARTIFACTS AND COHORTS")
    print("=" * 60)

    if not os.path.exists(ML_BA_ARTIFACT):
        raise FileNotFoundError(f"ML-BA combo artifact not found: {ML_BA_ARTIFACT}")
    combo_artifact = joblib.load(ML_BA_ARTIFACT)
    print(f"  Loaded combo artifact: {ML_BA_ARTIFACT}")
    print(f"  Targets: {combo_artifact['targets']}")
    print(f"  Feature cols: {combo_artifact['feature_cols']}")

    elsa_df = load_or_create_cohort(ELSA_COHORT, "ELSA")
    hrs_df = load_or_create_cohort(HRS_COHORT, "HRS")
    print(f"  ELSA cohort: {elsa_df.shape}")
    print(f"  HRS cohort:  {hrs_df.shape}")

    charls_df = pd.read_csv(CHARLS_TRAIN)
    print(f"  CHARLS train: {charls_df.shape}")

    # -----------------------------------------------------------------------
    # Apply ML-BA to external cohorts AND to CHARLS train
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("APPLYING ML-BA PREDICTORS")
    print("=" * 60)

    elsa_with_ba = apply_ml_ba_to_cohort(elsa_df, combo_artifact, RAW16, "ELSA")
    hrs_with_ba = apply_ml_ba_to_cohort(hrs_df, combo_artifact, HRS_RAW16, "HRS")
    charls_with_ba = apply_ml_ba_to_cohort(charls_df, combo_artifact, RAW16, "CHARLS")

    print(f"\n  ELSA ML_BA     : mean={elsa_with_ba['ML_BA'].mean():.2f}, std={elsa_with_ba['ML_BA'].std():.2f}")
    print(f"  ELSA ML_KDM_BA : mean={elsa_with_ba['ML_KDM_BA'].mean():.2f}, std={elsa_with_ba['ML_KDM_BA'].std():.2f}")
    print(f"  HRS ML_BA      : mean={hrs_with_ba['ML_BA'].mean():.2f}, std={hrs_with_ba['ML_BA'].std():.2f}")
    print(f"  HRS ML_KDM_BA  : mean={hrs_with_ba['ML_KDM_BA'].mean():.2f}, std={hrs_with_ba['ML_KDM_BA'].std():.2f}")
    print(f"  CHARLS ML_BA   : mean={charls_with_ba['ML_BA'].mean():.2f}, std={charls_with_ba['ML_BA'].std():.2f}")
    print(f"  CHARLS ML_KDM_BA: mean={charls_with_ba['ML_KDM_BA'].mean():.2f}, std={charls_with_ba['ML_KDM_BA'].std():.2f}")

    # Save augmented cohorts
    elsa_with_ba.to_csv(os.path.join(OUT_DIR, "elsa_with_ml_ba.csv"), index=False)
    hrs_with_ba.to_csv(os.path.join(OUT_DIR, "hrs_with_ml_ba.csv"), index=False)
    charls_with_ba.to_csv(os.path.join(OUT_DIR, "charls_train_with_ml_ba.csv"), index=False)

    # Use CHARLS-with-BA for training in all scenarios
    charls_train_df = charls_with_ba

    # -----------------------------------------------------------------------
    # Evaluate KOA prediction scenarios
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KOA PREDICTION SCENARIOS")
    print("=" * 60)

    metrics_rows = []

    for cohort_name, cohort_df, raw16_cols in [
        ("ELSA", elsa_with_ba, RAW16),
        ("HRS", hrs_with_ba, HRS_RAW16),
    ]:
        print(f"\n--- {cohort_name} ---")

        # Determine available columns
        available_raw16 = [c for c in raw16_cols if c in cohort_df.columns and c in charls_train_df.columns]

        scenarios = {
            "raw16": available_raw16,
            "raw16+ML_BA": available_raw16 + ["ML_BA"],
            "raw16+ML_KDM_BA": available_raw16 + ["ML_KDM_BA"],
            "raw16+ML_BA+ML_KDM_BA": available_raw16 + ["ML_BA", "ML_KDM_BA"],
        }

        for scenario_name, feat_cols in scenarios.items():
            charls_cols = [c for c in feat_cols if c in charls_train_df.columns]
            ext_cols = [c for c in feat_cols if c in cohort_df.columns]
            common_cols = [c for c in charls_cols if c in ext_cols]

            if len(common_cols) < 5:
                print(f"  SKIPPING {scenario_name}: only {len(common_cols)} common columns")
                continue

            metrics, proba = train_and_evaluate(
                charls_train_df, cohort_df, common_cols, scenario_name, cohort_name
            )
            metrics_rows.append(metrics)
            print(
                f"  {scenario_name:<24s} AUC={metrics['roc_auc']:.4f}  "
                f"PR-AUC={metrics['pr_auc']:.4f}  Brier={metrics['brier']:.4f}  "
                f"F1={metrics['f1']:.3f}  n={metrics['n']}"
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["cohort", "scenario"])
    metrics_path = os.path.join(OUT_DIR, "step17_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Saved metrics: {metrics_path}")

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    report_lines = [
        "=" * 80,
        "STEP 17: ML-BA EXTERNAL VALIDATION REPORT",
        "=" * 80,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "PURPOSE",
        "-" * 40,
        "Apply ML-predicted Biological Age variants to ELSA/HRS and compare",
        "KOA prediction scenarios: no BA, ML_BA, ML_KDM_BA, and both.",
        "",
        "ML-BA PREDICTOR QUALITY (from step_16)",
        "-" * 40,
        "  ML_BA target     : Biological Age (Sheet1 BA)",
        "  ML_KDM_BA target : BA_KDM_log (KDM formula)",
        "  Expected R2      : ~0.64 for ML_BA, ~0.48 for ML_KDM_BA",
        "",
        "PREDICTED BA DISTRIBUTIONS IN EXTERNAL COHORTS",
        "-" * 40,
        f"  ELSA ML_BA     : mean={elsa_with_ba['ML_BA'].mean():.2f}, std={elsa_with_ba['ML_BA'].std():.2f}",
        f"  ELSA ML_KDM_BA : mean={elsa_with_ba['ML_KDM_BA'].mean():.2f}, std={elsa_with_ba['ML_KDM_BA'].std():.2f}",
        f"  HRS ML_BA      : mean={hrs_with_ba['ML_BA'].mean():.2f}, std={hrs_with_ba['ML_BA'].std():.2f}",
        f"  HRS ML_KDM_BA  : mean={hrs_with_ba['ML_KDM_BA'].mean():.2f}, std={hrs_with_ba['ML_KDM_BA'].std():.2f}",
        "",
        "KOA PREDICTION RESULTS",
        "-" * 40,
    ]

    for _, row in metrics_df.iterrows():
        report_lines.append(
            f"  [{row['cohort']}] {row['scenario']:<24s} "
            f"AUC={row['roc_auc']:.4f}  PR-AUC={row['pr_auc']:.4f}  "
            f"Brier={row['brier']:.4f}  F1={row['f1']:.3f}  n={int(row['n'])}"
        )

    report_lines.extend([
        "",
        "INTERPRETATION",
        "-" * 40,
        "If ML_BA or ML_KDM_BA improve AUROC over raw16 alone, the BA variable",
        "carries transportable KOA-relevant signal beyond Age_New. If not, the",
        "cross-cohort AUC gap is driven mainly by outcome/culture differences.",
        "",
        "OUTPUT FILES",
        "-" * 40,
        f"  {OUT_DIR}/",
        "    elsa_with_ml_ba.csv",
        "    hrs_with_ml_ba.csv",
        "    step17_metrics.csv",
        "    step17_report.txt",
    ])

    report_path = os.path.join(OUT_DIR, "step17_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Saved report: {report_path}")

    print("\n" + "=" * 80)
    print("STEP 17 COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
