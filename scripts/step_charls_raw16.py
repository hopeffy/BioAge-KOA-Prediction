"""
STEP CHARLS-RAW16: CHARLS Internal Validation with raw16 Feature Set
======================================================================
Evaluates the CHARLS-trained Random Forest model restricted to the raw16
feature set — the exact same 14 features used in ELSA Wave 8 external
validation. This provides the direct CHARLS-internal counterpart for the
cross-cohort comparison.

The goal: quantify how much performance is lost when we drop
Biological Age, wave, and Time from raw17 → raw16 WITHIN CHARLS,
before any cross-cultural attenuation.

Design
------
- Feature set: raw16 = raw17 minus {Biological Age, wave, Time} + wave_id
- Same columns as ELSA raw16: Gender, Age_New, Marital, Education,
  Residence, Hypertension, Dyslipidemia, Diabetes, Cancer, CVD, Smoke,
  Drink, BMI, BMI_New, wave_id
- Model: Random Forest (n=300, rs=42, no tuning/class_weight)
- Evaluation: 5-fold stratified OOF on internal partition +
  single-shot evaluation on early unseen holdout
- Both raw and isotonic-calibrated probabilities

Output folder
-------------
  step_charls_raw16/
    charls_raw16_metrics.csv
    charls_raw16_calibration_summary.csv
    charls_raw16_dca_summary.csv
    calibration_table_charls_raw16_*.csv
    dca_charls_raw16_*.csv
    calibration_curve_charls_raw16_*.png
    dca_curve_charls_raw16_*.png
    charls_raw16_report.txt
"""

import os
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EARLY_SPLIT_DIR = os.path.join(BASE_DIR, "step_01_data_prep")
EARLY_SPLIT_TRAIN = os.path.join(EARLY_SPLIT_DIR, "dataset_A_internal_train.csv")
EARLY_SPLIT_UNSEEN = os.path.join(EARLY_SPLIT_DIR, "dataset_A_unseen_test.csv")
OUT_DIR = os.path.join(BASE_DIR, "step_charls_raw16")

N_FOLDS = 5
RANDOM_STATE = 42
N_TREES = 300
WAVE_ID = 1  # constant wave identifier (same convention as ELSA/HRS)

# ELSA raw16 feature set (14 variables, no Biological Age)
RAW16 = [
    "wave_id",
    "Gender",
    "Age_New",
    "Marital",
    "Education",
    "Residence",
    "Hypertension",
    "Dyslipidemia",
    "Diabetes",
    "Cancer",
    "CVD",
    "Smoke",
    "Drink",
    "BMI",
    "BMI_New",
]

RAW17 = [
    "wave",
    "Time",
    "Gender",
    "Age_New",
    "Marital",
    "Education",
    "Residence",
    "Hypertension",
    "Dyslipidemia",
    "Diabetes",
    "Cancer",
    "CVD",
    "Smoke",
    "Drink",
    "BMI",
    "BMI_New",
    "Biological Age",
]


# ---------------------------------------------------------------------------
# Metrics & utilities (mirrors step_11)
# ---------------------------------------------------------------------------
def compute_binary_metrics(y_true, proba, threshold=0.5):
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
        "f1": f1_score(y_true, preds),
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds),
    }


def best_f1_threshold_metrics(y_true, proba):
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_thr, best_f1, best_stats = 0.5, -1.0, None
    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
            best_stats = {
                "f1": f1,
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
            }
    return best_thr, best_stats


def calibration_slope_intercept(y_true, proba, eps=1e-6):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    p = np.clip(proba, eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)
    try:
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(logit_p, y_true)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


def expected_calibration_error(y_true, proba, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba, bins[1:-1], right=True)
    ece, mce, n = 0.0, 0.0, len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        obs = y_true[mask].mean()
        pred = proba[mask].mean()
        gap = abs(obs - pred)
        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)
    return ece, mce


def train_fitted_impute_and_scale(X_train_df, X_test_df):
    X_tr = X_train_df.copy()
    X_te = X_test_df.copy()
    numeric_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_tr.columns if c not in numeric_cols]

    if numeric_cols:
        num_imp = SimpleImputer(strategy="median")
        X_tr[numeric_cols] = num_imp.fit_transform(X_tr[numeric_cols])
        X_te[numeric_cols] = num_imp.transform(X_te[numeric_cols])

    if categorical_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        X_tr[categorical_cols] = cat_imp.fit_transform(X_tr[categorical_cols])
        X_te[categorical_cols] = cat_imp.transform(X_te[categorical_cols])
        for col in categorical_cols:
            train_vals = pd.Series(X_tr[col]).astype(str)
            mapping = {v: i for i, v in enumerate(train_vals.unique())}
            X_tr[col] = train_vals.map(mapping).astype(float)
            X_te[col] = pd.Series(X_te[col]).astype(str).map(mapping).fillna(-1.0).astype(float)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.values)
    X_te_s = scaler.transform(X_te.values)
    return X_tr_s, X_te_s


def build_model(calibrated=False):
    base = RandomForestClassifier(n_estimators=N_TREES, random_state=RANDOM_STATE, n_jobs=-1)
    if calibrated:
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    return base


def oof_probabilities(X_df, y, calibrated=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in skf.split(X_df, y):
        X_tr = X_df.iloc[train_idx].copy()
        X_te = X_df.iloc[test_idx].copy()
        y_tr = y[train_idx]
        X_tr_s, X_te_s = train_fitted_impute_and_scale(X_tr, X_te)
        model = build_model(calibrated=calibrated)
        model.fit(X_tr_s, y_tr)
        oof[test_idx] = model.predict_proba(X_te_s)[:, 1]
    return oof


def fit_and_predict_unseen(X_train_df, y_train, X_unseen_df, calibrated=False):
    X_tr_s, X_un_s = train_fitted_impute_and_scale(X_train_df, X_unseen_df)
    model = build_model(calibrated=calibrated)
    model.fit(X_tr_s, y_train)
    return model.predict_proba(X_un_s)[:, 1]


def calibration_table(y_true, proba, n_bins=10):
    d = pd.DataFrame({"y": y_true, "p": proba})
    d["bin"] = pd.qcut(d["p"], q=n_bins, duplicates="drop")
    tbl = (
        d.groupby("bin", observed=False)
        .agg(n=("y", "size"), mean_pred=("p", "mean"), observed_rate=("y", "mean"))
        .reset_index()
    )
    tbl["abs_gap"] = (tbl["observed_rate"] - tbl["mean_pred"]).abs()
    return tbl


def decision_curve(y_true, proba, thresholds):
    n = len(y_true)
    prev = y_true.mean()
    rows = []
    for t in thresholds:
        pred_pos = proba >= t
        tp = ((pred_pos == 1) & (y_true == 1)).sum()
        fp = ((pred_pos == 1) & (y_true == 0)).sum()
        odds = t / (1.0 - t)
        nb_model = (tp / n) - (fp / n) * odds
        nb_all = prev - (1.0 - prev) * odds
        rows.append({
            "threshold": t,
            "net_benefit_model": nb_model,
            "net_benefit_all": nb_all,
            "net_benefit_none": 0.0,
        })
    return pd.DataFrame(rows)


def summarize_dca_band(dca_df, prevalence, threshold_low=0.10, threshold_high=0.30):
    band = dca_df[(dca_df["threshold"] >= threshold_low) & (dca_df["threshold"] <= threshold_high)].copy()
    if band.empty:
        return None
    band["gain_vs_all"] = band["net_benefit_model"] - band["net_benefit_all"]
    band["avoided_per_100"] = band["gain_vs_all"] * ((1.0 - band["threshold"]) / band["threshold"]) * 100.0
    baseline = (1.0 - prevalence) * 100.0
    mean_gain = float(band["gain_vs_all"].mean())
    mean_avoided = float(band["avoided_per_100"].mean())
    rel_reduction = (mean_avoided / baseline * 100.0) if baseline > 0 else 0.0
    return {
        "mean_gain_vs_treat_all": mean_gain,
        "mean_avoided_per_100": mean_avoided,
        "relative_reduction_pct": rel_reduction,
        "baseline_unnecessary_per_100": baseline,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_calibration(y_true, proba_raw, proba_cal, title, out_path):
    plt.figure(figsize=(7, 6))
    frac_raw, mean_raw = calibration_curve(y_true, proba_raw, n_bins=10, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_true, proba_cal, n_bins=10, strategy="quantile")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(mean_raw, frac_raw, marker="o", linewidth=2, label="CHARLS raw16 (raw)")
    plt.plot(mean_cal, frac_cal, marker="o", linewidth=2, label="CHARLS raw16 (isotonic)")
    plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dca(dca_raw, dca_cal, title, out_path):
    plt.figure(figsize=(8, 6))
    plt.plot(dca_raw["threshold"], dca_raw["net_benefit_model"], linewidth=2, label="CHARLS raw16 (raw)")
    plt.plot(dca_cal["threshold"], dca_cal["net_benefit_model"], linewidth=2, label="CHARLS raw16 (isotonic)")
    plt.plot(dca_raw["threshold"], dca_raw["net_benefit_all"], "--", linewidth=1.5, label="Treat all")
    plt.axhline(0, color="gray", linestyle=":", linewidth=1.5, label="Treat none")
    plt.title(title)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("STEP CHARLS-RAW16: CHARLS Internal Validation with raw16 Features")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. Load CHARLS early-split data ----
    train_df = pd.read_csv(EARLY_SPLIT_TRAIN)
    unseen_df = pd.read_csv(EARLY_SPLIT_UNSEEN)
    print(f"\nCHARLS internal train: {len(train_df):,} rows, KOA={train_df['KOA'].mean()*100:.1f}%")
    print(f"CHARLS unseen holdout: {len(unseen_df):,} rows, KOA={unseen_df['KOA'].mean()*100:.1f}%")

    # ---- 2. Build raw16 features from CHARLS columns ----
    # CHARLS has: wave, Time, Gender, Age_New, Marital, Education, Residence,
    #   Hypertension, Dyslipidemia, Diabetes, Cancer, CVD, Smoke, Drink, BMI,
    #   BMI_New, Biological Age, (and biomarkers)
    # raw16 drops: wave, Time, Biological Age
    # raw16 adds: wave_id (constant)

    # Identify which raw16 columns are in CHARLS (all should be)
    charls_raw16_cols = []
    for c in RAW16:
        if c == "wave_id":
            charls_raw16_cols.append(c)
        elif c in train_df.columns:
            charls_raw16_cols.append(c)
        else:
            print(f"  WARNING: {c} not found in CHARLS data!")

    print(f"\nraw16 feature set ({len(charls_raw16_cols)} columns): {charls_raw16_cols}")

    # Build feature DataFrames
    X_internal = train_df[[c for c in charls_raw16_cols if c != "wave_id"]].copy()
    X_internal["wave_id"] = WAVE_ID
    X_unseen = unseen_df[[c for c in charls_raw16_cols if c != "wave_id"]].copy()
    X_unseen["wave_id"] = WAVE_ID

    # Ensure column order matches RAW16
    X_internal = X_internal[charls_raw16_cols]
    X_unseen = X_unseen[charls_raw16_cols]

    y_internal = train_df["KOA"].values
    y_unseen = unseen_df["KOA"].values

    print(f"  Internal features: {X_internal.shape}")
    print(f"  Unseen features:   {X_unseen.shape}")
    print(f"  Missing values (internal): {X_internal.isna().sum().sum()}")
    print(f"  Missing values (unseen):   {X_unseen.isna().sum().sum()}")

    # ---- 3. OOF probabilities on internal partition ----
    print("\n" + "-" * 80)
    print("OOF on internal partition (5-fold CV)")
    print("-" * 80)

    proba_raw_internal = oof_probabilities(X_internal, y_internal, calibrated=False)
    proba_cal_internal = oof_probabilities(X_internal, y_internal, calibrated=True)

    # ---- 4. Unseen holdout prediction ----
    print("\n" + "-" * 80)
    print("Unseen holdout evaluation")
    print("-" * 80)

    proba_raw_unseen = fit_and_predict_unseen(X_internal, y_internal, X_unseen, calibrated=False)
    proba_cal_unseen = fit_and_predict_unseen(X_internal, y_internal, X_unseen, calibrated=True)

    # ---- 5. Compute metrics ----
    print("\n" + "=" * 80)
    print("METRICS")
    print("=" * 80)

    metric_jobs = [
        ("raw", "internal_oof", y_internal, proba_raw_internal),
        ("isotonic", "internal_oof", y_internal, proba_cal_internal),
        ("raw", "unseen_holdout", y_unseen, proba_raw_unseen),
        ("isotonic", "unseen_holdout", y_unseen, proba_cal_unseen),
    ]

    metrics_rows = []
    thresholds = np.arange(0.05, 0.61, 0.01)

    for variant, eval_set, y_eval, proba_eval in metric_jobs:
        m = compute_binary_metrics(y_eval, proba_eval)
        ece, mce = expected_calibration_error(y_eval, proba_eval)
        best_thr, best_stats = best_f1_threshold_metrics(y_eval, proba_eval)
        cal_slope, cal_intercept = calibration_slope_intercept(y_eval, proba_eval)

        row = {
            "config": "raw16",
            "variant": variant,
            "evaluation_set": eval_set,
            "n_samples": len(y_eval),
            "prevalence": y_eval.mean(),
            **m,
            "ece": ece,
            "mce": mce,
            "best_f1_threshold": best_thr,
            "f1_best_threshold": best_stats["f1"],
            "precision_best_threshold": best_stats["precision"],
            "recall_best_threshold": best_stats["recall"],
            "calibration_slope": cal_slope,
            "calibration_intercept": cal_intercept,
        }
        metrics_rows.append(row)

        slope_flag = "OK" if 0.80 <= cal_slope <= 1.20 else "OUTSIDE"
        print(f"\n  [{eval_set}] {variant:8s}: "
              f"ROC-AUC={m['roc_auc']:.4f}, PR-AUC={m['pr_auc']:.4f}, "
              f"Brier={m['brier']:.4f}, ECE={ece:.4f}, "
              f"Slope={cal_slope:.3f} [{slope_flag}], Intercept={cal_intercept:.3f}, "
              f"F1@0.5={m['f1']:.4f}")

    metrics_df = pd.DataFrame(metrics_rows)

    # ---- 6. Calibration tables, DCA, and plots ----
    for eval_set, y_eval, proba_raw_e, proba_cal_e in [
        ("internal_oof", y_internal, proba_raw_internal, proba_cal_internal),
        ("unseen_holdout", y_unseen, proba_raw_unseen, proba_cal_unseen),
    ]:
        # Calibration tables
        ct_raw = calibration_table(y_eval, proba_raw_e)
        ct_cal = calibration_table(y_eval, proba_cal_e)
        ct_raw.to_csv(os.path.join(OUT_DIR, f"calibration_table_charls_raw16_raw_{eval_set}.csv"), index=False)
        ct_cal.to_csv(os.path.join(OUT_DIR, f"calibration_table_charls_raw16_isotonic_{eval_set}.csv"), index=False)

        # DCA
        dca_raw = decision_curve(y_eval, proba_raw_e, thresholds)
        dca_cal = decision_curve(y_eval, proba_cal_e, thresholds)
        dca_raw.to_csv(os.path.join(OUT_DIR, f"dca_charls_raw16_raw_{eval_set}.csv"), index=False)
        dca_cal.to_csv(os.path.join(OUT_DIR, f"dca_charls_raw16_isotonic_{eval_set}.csv"), index=False)

        # Plots
        plot_calibration(
            y_eval, proba_raw_e, proba_cal_e,
            title=f"Calibration: CHARLS raw16 ({eval_set})",
            out_path=os.path.join(OUT_DIR, f"calibration_curve_charls_raw16_{eval_set}.png"),
        )
        plot_dca(
            dca_raw, dca_cal,
            title=f"DCA: CHARLS raw16 ({eval_set})",
            out_path=os.path.join(OUT_DIR, f"dca_curve_charls_raw16_{eval_set}.png"),
        )

    # ---- 7. DCA clinical summary ----
    dca_summary_rows = []
    for _, row in metrics_df.iterrows():
        eval_set = row["evaluation_set"]
        variant = row["variant"]
        y_eval = y_internal if eval_set == "internal_oof" else y_unseen
        proba_e = proba_raw_internal if (eval_set == "internal_oof" and variant == "raw") else \
                   proba_cal_internal if (eval_set == "internal_oof") else \
                   proba_raw_unseen if variant == "raw" else proba_cal_unseen
        band = summarize_dca_band(decision_curve(y_eval, proba_e, thresholds), y_eval.mean())
        if band:
            dca_summary_rows.append({
                "config": "raw16",
                "variant": variant,
                "evaluation_set": eval_set,
                **band,
            })

    dca_summary_df = pd.DataFrame(dca_summary_rows)
    if not dca_summary_df.empty:
        dca_summary_df.to_csv(os.path.join(OUT_DIR, "charls_raw16_dca_summary.csv"), index=False)

        for eval_set in ["internal_oof", "unseen_holdout"]:
            subset = dca_summary_df[dca_summary_df["evaluation_set"] == eval_set]
            if not subset.empty:
                for _, r in subset.iterrows():
                    print(f"\n  DCA {eval_set} [{r['variant']}]: "
                          f"mean gain vs all={r['mean_gain_vs_treat_all']:.4f}, "
                          f"avoided={r['mean_avoided_per_100']:.1f}/100, "
                          f"rel reduction={r['relative_reduction_pct']:.1f}%")

    # ---- 8. Save metrics ----
    metrics_df.to_csv(os.path.join(OUT_DIR, "charls_raw16_metrics.csv"), index=False)
    cal_summary = metrics_df[[
        "config", "variant", "evaluation_set", "n_samples", "prevalence",
        "brier", "ece", "mce", "calibration_slope", "calibration_intercept",
    ]]
    cal_summary.to_csv(os.path.join(OUT_DIR, "charls_raw16_calibration_summary.csv"), index=False)

    # ---- 9. Comparison with raw17 (from step_11) ----
    step11_metrics_path = os.path.join(BASE_DIR, "step_11_clinical_validation", "clinical_metrics_summary.csv")
    raw17_comparison = ""
    if os.path.exists(step11_metrics_path):
        step11 = pd.read_csv(step11_metrics_path)
        raw17_raw = step11[(step11["config"] == "raw17") & (step11["variant"] == "raw")]
        raw16_raw = metrics_df[(metrics_df["config"] == "raw16") & (metrics_df["variant"] == "raw")]

        comparison_lines = ["", "COMPARISON raw17 vs raw16 (CHARLS only)", "-" * 50]
        for eval_set in ["internal_oof", "unseen_holdout"]:
            r17 = raw17_raw[raw17_raw["evaluation_set"] == eval_set]
            r16 = raw16_raw[raw16_raw["evaluation_set"] == eval_set]
            if not r17.empty and not r16.empty:
                r17 = r17.iloc[0]; r16 = r16.iloc[0]
                auc_drop = r17["roc_auc"] - r16["roc_auc"]
                pr_drop = r17["pr_auc"] - r16["pr_auc"]
                comparison_lines.append(
                    f"  [{eval_set}] raw17 AUC={r17['roc_auc']:.4f} → raw16 AUC={r16['roc_auc']:.4f} "
                    f"(Δ={auc_drop:.4f}); PR-AUC: {r17['pr_auc']:.4f} → {r16['pr_auc']:.4f} (Δ={pr_drop:.4f})"
                )
        raw17_comparison = "\n".join(comparison_lines)
        print(raw17_comparison)

    # ---- 10. Report ----
    report = [
        "=" * 80,
        "STEP CHARLS-RAW16 REPORT",
        "=" * 80,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Feature set: raw16 ({len(charls_raw16_cols)} features — raw17 minus Biological Age/wave/Time)",
        f"Model: Random Forest (n={N_TREES}, seed={RANDOM_STATE})",
        f"Primary CHARLS cohort: Dataset A (n=12,329, KOA=13.3%)",
        "",
        "PURPOSE",
        "-" * 40,
        "Establish the CHARLS-internal raw16 baseline for direct comparison with",
        "ELSA and HRS raw16 external validation results. This isolates the feature-",
        "subset penalty within CHARLS (no cross-cultural attenuation).",
        "",
        "RAW16 FEATURE SET",
        "-" * 40,
    ]
    for i, c in enumerate(charls_raw16_cols):
        report.append(f"  {i+1:2d}. {c}")
    report.append("")
    report.append("METRICS")
    report.append("-" * 40)
    for _, row in metrics_df.iterrows():
        slope = row["calibration_slope"]
        flag = "OK" if 0.80 <= slope <= 1.20 else "OUTSIDE"
        report.append(
            f"  [{row['evaluation_set']}] {row['variant']:8s}: "
            f"ROC-AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}, "
            f"Brier={row['brier']:.4f}, ECE={row['ece']:.4f}, "
            f"Slope={slope:.3f} [{flag}], Intercept={row['calibration_intercept']:.3f}"
        )
    if raw17_comparison:
        report.append(raw17_comparison)

    report.extend([
        "",
        "OUTPUT FILES",
        "-" * 40,
        f"  {OUT_DIR}/",
        "    charls_raw16_metrics.csv",
        "    charls_raw16_calibration_summary.csv",
        "    charls_raw16_dca_summary.csv",
        "    calibration_table_charls_raw16_raw_*.csv",
        "    calibration_table_charls_raw16_isotonic_*.csv",
        "    dca_charls_raw16_raw_*.csv",
        "    dca_charls_raw16_isotonic_*.csv",
        "    calibration_curve_charls_raw16_*.png",
        "    dca_curve_charls_raw16_*.png",
        "    charls_raw16_report.txt  (this file)",
    ])

    report_path = os.path.join(OUT_DIR, "charls_raw16_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nReport saved: {report_path}")

    print("\n" + "=" * 80)
    print("STEP CHARLS-RAW16 COMPLETED")
    print("=" * 80)
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
