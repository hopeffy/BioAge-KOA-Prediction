"""
STEP 11: Clinical Validation Layer (Calibration + DCA)
=======================================================
Evaluate clinical reliability for KOA risk models using:
1. Out-of-fold probability calibration analysis
2. Brier score and calibration error metrics (ECE/MCE)
3. Isotonic calibration comparison
4. Decision curve analysis (net benefit)

This script complements discrimination metrics (AUROC/PR-AUC)
with calibration and clinical utility metrics.
"""

import os
from datetime import datetime
import warnings

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DATA = os.path.join(BASE_DIR, "step_03_kdm_ba", "dataset_A_with_kdm_ba.csv")
OUT_DIR = os.path.join(BASE_DIR, "step_11_clinical_validation")
EARLY_SPLIT_DIR = os.path.join(BASE_DIR, "step_01_data_prep")
EARLY_SPLIT_TRAIN = os.path.join(EARLY_SPLIT_DIR, "dataset_A_internal_train.csv")
EARLY_SPLIT_UNSEEN = os.path.join(EARLY_SPLIT_DIR, "dataset_A_unseen_test.csv")
EARLY_SPLIT_TEST_SIZE = 0.20

N_FOLDS = 5
RANDOM_STATE = 42
N_TREES = 300


def safe_slug(value):
    return (
        value.lower()
        .replace(" ", "_")
        .replace("+", "plus")
        .replace("-", "_")
        .replace("/", "_")
    )


def build_feature_sets(df):
    raw17 = [
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

    feature_sets = {"raw17": raw17}

    adapted_aging_candidates = [
        "BA_KDM_orig",
        "PhenoAge_Adapted",
        "PhenoAge_Adapted_Accel",
        "BIR_orig",
    ]
    available_candidates = [c for c in adapted_aging_candidates if c in df.columns]

    if len(available_candidates) > 0:
        feature_sets["raw17_plus_adapted_aging"] = raw17 + available_candidates

    if "PhenoAge_Adapted" in df.columns:
        feature_sets["raw17_plus_pheno"] = raw17 + ["PhenoAge_Adapted"]

    return feature_sets


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
    best_threshold = 0.5
    best_f1 = -1.0
    best_stats = None

    for thr in thresholds:
        preds = (proba >= thr).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(thr)
            best_stats = {
                "f1": f1,
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
            }

    return best_threshold, best_stats


def calibration_slope_intercept(y_true, proba, eps=1e-6):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan

    p = np.clip(proba, eps, 1.0 - eps)
    logit_p = np.log(p / (1.0 - p)).reshape(-1, 1)

    try:
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(logit_p, y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:
        slope = np.nan
        intercept = np.nan

    return slope, intercept


def train_fitted_impute_and_scale(X_train_df, X_test_df):
    X_train_proc = X_train_df.copy()
    X_test_proc = X_test_df.copy()

    numeric_cols = X_train_proc.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train_proc.columns if c not in numeric_cols]

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        X_train_proc[numeric_cols] = num_imputer.fit_transform(X_train_proc[numeric_cols])
        X_test_proc[numeric_cols] = num_imputer.transform(X_test_proc[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_proc[categorical_cols] = cat_imputer.fit_transform(X_train_proc[categorical_cols])
        X_test_proc[categorical_cols] = cat_imputer.transform(X_test_proc[categorical_cols])

        for col in categorical_cols:
            train_values = pd.Series(X_train_proc[col]).astype(str)
            mapping = {v: i for i, v in enumerate(train_values.unique())}
            X_train_proc[col] = train_values.map(mapping).astype(float)
            X_test_proc[col] = pd.Series(X_test_proc[col]).astype(str).map(mapping).fillna(-1.0).astype(float)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proc.values)
    X_test_scaled = scaler.transform(X_test_proc.values)
    return X_train_scaled, X_test_scaled


def build_model(calibrated=False):
    base_model = RandomForestClassifier(
        n_estimators=N_TREES,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    if calibrated:
        return CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    return base_model


def expected_calibration_error(y_true, proba, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(proba, bins[1:-1], right=True)

    ece = 0.0
    mce = 0.0
    n = len(y_true)

    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if mask.sum() == 0:
            continue

        obs = y_true[mask].mean()
        pred = proba[mask].mean()
        gap = abs(obs - pred)

        ece += (mask.sum() / n) * gap
        mce = max(mce, gap)

    return ece, mce


def oof_probabilities(X_df, y, calibrated=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in skf.split(X_df, y):
        X_train_df = X_df.iloc[train_idx].copy()
        X_test_df = X_df.iloc[test_idx].copy()
        y_train = y[train_idx]

        X_train_s, X_test_s = train_fitted_impute_and_scale(X_train_df, X_test_df)
        model = build_model(calibrated=calibrated)

        model.fit(X_train_s, y_train)
        oof[test_idx] = model.predict_proba(X_test_s)[:, 1]

    return oof


def fit_and_predict_unseen(X_train_df, y_train, X_unseen_df, calibrated=False):
    X_train_s, X_unseen_s = train_fitted_impute_and_scale(X_train_df, X_unseen_df)
    model = build_model(calibrated=calibrated)
    model.fit(X_train_s, y_train)
    return model.predict_proba(X_unseen_s)[:, 1]


def load_early_split_data():
    if os.path.exists(EARLY_SPLIT_TRAIN) and os.path.exists(EARLY_SPLIT_UNSEEN):
        train_df = pd.read_csv(EARLY_SPLIT_TRAIN)
        unseen_df = pd.read_csv(EARLY_SPLIT_UNSEEN)
        if "KOA" in train_df.columns and "KOA" in unseen_df.columns:
            return train_df, unseen_df
    return None, None


def deterministic_fallback_split(df):
    train_df, unseen_df = train_test_split(
        df,
        test_size=EARLY_SPLIT_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["KOA"],
    )
    return train_df.reset_index(drop=True), unseen_df.reset_index(drop=True)


def calibration_table(y_true, proba, n_bins=10):
    df = pd.DataFrame({"y": y_true, "p": proba})
    df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")

    table = (
        df.groupby("bin", observed=False)
        .agg(
            n=("y", "size"),
            mean_pred=("p", "mean"),
            observed_rate=("y", "mean"),
        )
        .reset_index()
    )
    table["abs_gap"] = (table["observed_rate"] - table["mean_pred"]).abs()
    return table


def decision_curve(y_true, proba, thresholds):
    n = len(y_true)
    prevalence = y_true.mean()

    rows = []
    for t in thresholds:
        pred_pos = proba >= t

        tp = ((pred_pos == 1) & (y_true == 1)).sum()
        fp = ((pred_pos == 1) & (y_true == 0)).sum()

        odds = t / (1.0 - t)
        net_benefit_model = (tp / n) - (fp / n) * odds
        net_benefit_all = prevalence - (1.0 - prevalence) * odds

        rows.append(
            {
                "threshold": t,
                "net_benefit_model": net_benefit_model,
                "net_benefit_all": net_benefit_all,
                "net_benefit_none": 0.0,
            }
        )

    return pd.DataFrame(rows)


def summarize_dca_clinical_band(
    dca_df,
    config_name,
    variant,
    prevalence,
    evaluation_set,
    split_source,
    threshold_low=0.10,
    threshold_high=0.30,
):
    band = dca_df[(dca_df["threshold"] >= threshold_low) & (dca_df["threshold"] <= threshold_high)].copy()
    if band.empty:
        return None

    band["gain_vs_treat_all"] = band["net_benefit_model"] - band["net_benefit_all"]
    band["gain_vs_treat_none"] = band["net_benefit_model"] - band["net_benefit_none"]

    # Vickers DCA net reduction interpretation: avoided unnecessary interventions per 100
    # compared to treat-all at each threshold.
    band["avoided_unnecessary_per_100"] = (
        band["gain_vs_treat_all"] * ((1.0 - band["threshold"]) / band["threshold"]) * 100.0
    )

    baseline_unnecessary_per_100 = (1.0 - prevalence) * 100.0
    best_idx = band["avoided_unnecessary_per_100"].idxmax()
    best_row = band.loc[best_idx]

    mean_avoided = float(band["avoided_unnecessary_per_100"].mean())
    max_avoided = float(best_row["avoided_unnecessary_per_100"])
    best_threshold = float(best_row["threshold"])
    mean_gain_vs_all = float(band["gain_vs_treat_all"].mean())
    mean_gain_vs_none = float(band["gain_vs_treat_none"].mean())
    positive_share_vs_all = float((band["gain_vs_treat_all"] > 0).mean())
    relative_reduction_pct = (
        (mean_avoided / baseline_unnecessary_per_100) * 100.0 if baseline_unnecessary_per_100 > 0 else 0.0
    )

    return {
        "config": config_name,
        "variant": variant,
        "evaluation_set": evaluation_set,
        "split_source": split_source,
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
        "mean_gain_vs_treat_all": mean_gain_vs_all,
        "mean_gain_vs_treat_none": mean_gain_vs_none,
        "positive_share_vs_treat_all": positive_share_vs_all,
        "mean_avoided_unnecessary_per_100": mean_avoided,
        "max_avoided_unnecessary_per_100": max_avoided,
        "best_threshold_for_avoidance": best_threshold,
        "baseline_unnecessary_per_100_treat_all": baseline_unnecessary_per_100,
        "relative_reduction_vs_treat_all_pct": relative_reduction_pct,
    }


def build_publication_paragraphs(metrics_df, decision_df, prevalence, threshold_low=0.10, threshold_high=0.30):
    if "evaluation_set" in metrics_df.columns:
        internal_df = metrics_df[metrics_df["evaluation_set"] == "internal_oof"].copy()
        unseen_df = metrics_df[metrics_df["evaluation_set"] == "unseen_holdout"].copy()
    else:
        internal_df = metrics_df.copy()
        unseen_df = pd.DataFrame()

    best_auc_row = internal_df.sort_values("roc_auc", ascending=False).iloc[0]

    unseen_match = None
    if len(unseen_df) > 0:
        same_config = unseen_df[
            (unseen_df["config"] == best_auc_row["config"]) &
            (unseen_df["variant"] == best_auc_row["variant"])
        ]
        if len(same_config) > 0:
            unseen_match = same_config.iloc[0]
        else:
            unseen_match = unseen_df.sort_values("roc_auc", ascending=False).iloc[0]

    methods_en = (
        "Methods: We evaluated KOA risk models using stratified 5-fold out-of-fold prediction on the internal training partition, "
        "with a Random Forest classifier as the base learner and optional isotonic probability calibration. "
        "Missing values were handled with train-fitted SimpleImputer strategies (median for numeric, most-frequent for categorical) "
        "within each modeling split to avoid leakage. We additionally performed an early 80/20 stratified holdout split as an unseen "
        "test set. Model performance was quantified using discrimination (ROC-AUC, PR-AUC), calibration (Brier score, expected "
        "calibration error, calibration slope/intercept), and decision curve analysis (DCA). Clinical utility estimates were interpreted "
        "as internal-validation decision-support signals, not external prospective validation."
    )

    methods_tr = (
        "Yöntem: KOA risk modelleri internal eğitim bölümünde stratified 5-fold out-of-fold yaklaşımı ile değerlendirildi; "
        "temel öğrenici olarak Random Forest kullanıldı ve isteğe bağlı isotonic kalibrasyon uygulandı. Eksik veriler, sızıntıyı "
        "önlemek için her modelleme bölünmesinde yalnızca eğitimden fit edilen SimpleImputer stratejileriyle (sayısal: median, "
        "kategorik: en sık) ele alındı. Ayrıca erken 80/20 stratified holdout ile unseen test set oluşturuldu. Performans ayrım gücü "
        "(ROC-AUC, PR-AUC), kalibrasyon (Brier, beklenen kalibrasyon hatası, kalibrasyon slope/intercept) ve karar eğrisi analizi (DCA) "
        "ile değerlendirildi. Klinik fayda bulguları external/prospektif doğrulama değil, internal validation karar-destek sinyali olarak yorumlandı."
    )

    if "evaluation_set" in decision_df.columns:
        decision_internal = decision_df[decision_df["evaluation_set"] == "internal_oof"].copy()
    else:
        decision_internal = decision_df.copy()

    if len(decision_internal) > 0:
        best_dca_row = decision_internal.sort_values("mean_gain_vs_treat_all", ascending=False).iloc[0]
        dca_text_en = (
            f"In DCA, {best_dca_row['config']} ({best_dca_row['variant']}) showed the highest internal net benefit in the "
            f"{int(threshold_low*100)}%-{int(threshold_high*100)}% threshold range, with an estimated "
            f"{best_dca_row['mean_avoided_unnecessary_per_100']:.1f} avoided unnecessary referrals/imaging per 100 patients versus treat-all "
            f"({best_dca_row['relative_reduction_vs_treat_all_pct']:.1f}% relative reduction)."
        )
        dca_text_tr = (
            f"DCA analizinde {best_dca_row['config']} ({best_dca_row['variant']}) modeli internal değerlendirmede "
            f"%{int(threshold_low*100)}-%{int(threshold_high*100)} eşik aralığında en yüksek net faydayı gösterdi; treat-all stratejisine göre "
            f"100 hasta başına tahmini {best_dca_row['mean_avoided_unnecessary_per_100']:.1f} gereksiz sevk/görüntüleme azalımı "
            f"({best_dca_row['relative_reduction_vs_treat_all_pct']:.1f}% relatif azalma) sağladı."
        )
    else:
        dca_text_en = "DCA summary statistics were not available for publication text generation."
        dca_text_tr = "Yayın metni için DCA özet istatistikleri üretilemedi."

    if unseen_match is not None:
        unseen_text_en = (
            f"For the matched unseen holdout evaluation, {unseen_match['config']} ({unseen_match['variant']}) achieved "
            f"ROC-AUC={unseen_match['roc_auc']:.3f}, PR-AUC={unseen_match['pr_auc']:.3f}, Brier={unseen_match['brier']:.3f}, "
            f"and ECE={unseen_match['ece']:.3f}."
        )
        unseen_text_tr = (
            f"Eşleşen unseen holdout değerlendirmesinde {unseen_match['config']} ({unseen_match['variant']}) modeli "
            f"ROC-AUC={unseen_match['roc_auc']:.3f}, PR-AUC={unseen_match['pr_auc']:.3f}, Brier={unseen_match['brier']:.3f}, "
            f"ECE={unseen_match['ece']:.3f} değerlerine ulaştı."
        )
    else:
        unseen_text_en = "Unseen holdout metrics were not available."
        unseen_text_tr = "Unseen holdout metrikleri mevcut değildi."

    results_en = (
        f"Results: Internal out-of-fold discrimination was highest for {best_auc_row['config']} ({best_auc_row['variant']}), "
        f"with ROC-AUC={best_auc_row['roc_auc']:.3f}, PR-AUC={best_auc_row['pr_auc']:.3f}, Brier={best_auc_row['brier']:.3f}, "
        f"ECE={best_auc_row['ece']:.3f}, calibration slope={best_auc_row['calibration_slope']:.3f}, and calibration intercept={best_auc_row['calibration_intercept']:.3f}. "
        f"{unseen_text_en} {dca_text_en}"
    )

    results_tr = (
        f"Bulgular: Internal out-of-fold ayrım gücü en yüksek {best_auc_row['config']} ({best_auc_row['variant']}) modelinde gözlendi "
        f"(ROC-AUC={best_auc_row['roc_auc']:.3f}, PR-AUC={best_auc_row['pr_auc']:.3f}, Brier={best_auc_row['brier']:.3f}, "
        f"ECE={best_auc_row['ece']:.3f}, kalibrasyon slope={best_auc_row['calibration_slope']:.3f}, kalibrasyon intercept={best_auc_row['calibration_intercept']:.3f}). "
        f"{unseen_text_tr} {dca_text_tr}"
    )

    return {
        "methods_en": methods_en,
        "results_en": results_en,
        "methods_tr": methods_tr,
        "results_tr": results_tr,
    }


def plot_calibration(y_true, proba_raw, proba_cal, title, out_path):
    plt.figure(figsize=(7, 6))

    frac_raw, mean_raw = calibration_curve(y_true, proba_raw, n_bins=10, strategy="quantile")
    frac_cal, mean_cal = calibration_curve(y_true, proba_cal, n_bins=10, strategy="quantile")

    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(mean_raw, frac_raw, marker="o", linewidth=2, label="Model (raw)")
    plt.plot(mean_cal, frac_cal, marker="o", linewidth=2, label="Model (isotonic)")

    plt.title(title)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed event rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_dca(dca_raw, dca_cal, title, out_path):
    plt.figure(figsize=(8, 6))

    plt.plot(
        dca_raw["threshold"],
        dca_raw["net_benefit_model"],
        linewidth=2,
        label="Model (raw)",
    )
    plt.plot(
        dca_cal["threshold"],
        dca_cal["net_benefit_model"],
        linewidth=2,
        label="Model (isotonic)",
    )
    plt.plot(
        dca_raw["threshold"],
        dca_raw["net_benefit_all"],
        "--",
        linewidth=1.5,
        label="Treat all",
    )
    plt.plot(
        dca_raw["threshold"],
        dca_raw["net_benefit_none"],
        ":",
        linewidth=1.5,
        label="Treat none",
    )

    plt.title(title)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 80)
    print("STEP 11: CLINICAL VALIDATION (CALIBRATION + DCA)")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(OUT_DIR, exist_ok=True)

    df_full = pd.read_csv(INPUT_DATA)
    y_full = df_full["KOA"].values

    print(f"\nLoaded dataset: {df_full.shape}")
    print(f"KOA prevalence (full): {y_full.mean() * 100:.2f}%")

    early_train_df, early_unseen_df = load_early_split_data()
    if early_train_df is not None:
        print(
            "Loaded early split from Step 01: "
            f"internal={len(early_train_df)}, unseen={len(early_unseen_df)}"
        )
    else:
        print("Step 01 early split files not found; using deterministic fallback split in Step 11.")

    feature_sets = build_feature_sets(df_full)
    print(f"Feature configurations: {list(feature_sets.keys())}")

    thresholds = np.arange(0.05, 0.61, 0.01)
    metrics_rows = []
    dca_decision_rows = []

    for config_name, cols in feature_sets.items():
        available_cols = [c for c in cols if c in df_full.columns]

        if early_train_df is not None and all(c in early_train_df.columns for c in available_cols):
            internal_df = early_train_df[available_cols + ["KOA"]].copy().reset_index(drop=True)
            unseen_df = early_unseen_df[available_cols + ["KOA"]].copy().reset_index(drop=True)
            split_source = "step01_early_split"
        else:
            fallback_df = df_full[available_cols + ["KOA"]].copy()
            internal_df, unseen_df = deterministic_fallback_split(fallback_df)
            split_source = "step11_deterministic_fallback"

        print("\n" + "-" * 80)
        print(f"Running config: {config_name} ({len(available_cols)} features)")
        print(f"Split source: {split_source}")
        print("-" * 80)

        X_internal = internal_df[available_cols].copy()
        y_internal = internal_df["KOA"].values
        X_unseen = unseen_df[available_cols].copy()
        y_unseen = unseen_df["KOA"].values

        proba_raw_internal = oof_probabilities(X_internal, y_internal, calibrated=False)
        proba_cal_internal = oof_probabilities(X_internal, y_internal, calibrated=True)

        proba_raw_unseen = fit_and_predict_unseen(
            X_internal,
            y_internal,
            X_unseen,
            calibrated=False,
        )
        proba_cal_unseen = fit_and_predict_unseen(
            X_internal,
            y_internal,
            X_unseen,
            calibrated=True,
        )

        metric_jobs = [
            ("raw", "internal_oof", y_internal, proba_raw_internal),
            ("isotonic", "internal_oof", y_internal, proba_cal_internal),
            ("raw", "unseen_holdout", y_unseen, proba_raw_unseen),
            ("isotonic", "unseen_holdout", y_unseen, proba_cal_unseen),
        ]

        for variant, evaluation_set, y_eval, proba_eval in metric_jobs:
            metric_values = compute_binary_metrics(y_eval, proba_eval)
            ece_val, mce_val = expected_calibration_error(y_eval, proba_eval, n_bins=10)
            best_thr, best_stats = best_f1_threshold_metrics(y_eval, proba_eval)
            cal_slope, cal_intercept = calibration_slope_intercept(y_eval, proba_eval)

            metric_values["ece"] = ece_val
            metric_values["mce"] = mce_val
            metric_values["best_f1_threshold"] = best_thr
            metric_values["f1_best_threshold"] = best_stats["f1"]
            metric_values["precision_best_threshold"] = best_stats["precision"]
            metric_values["recall_best_threshold"] = best_stats["recall"]
            metric_values["calibration_slope"] = cal_slope
            metric_values["calibration_intercept"] = cal_intercept

            metrics_rows.append(
                {
                    "config": config_name,
                    "variant": variant,
                    "evaluation_set": evaluation_set,
                    "split_source": split_source,
                    "n_samples": len(y_eval),
                    "prevalence": y_eval.mean(),
                    **metric_values,
                }
            )

        internal_raw = metrics_rows[-4]
        internal_cal = metrics_rows[-3]
        unseen_raw = metrics_rows[-2]
        unseen_cal = metrics_rows[-1]

        print(
            f"Internal raw      -> AUC={internal_raw['roc_auc']:.4f}, "
            f"Brier={internal_raw['brier']:.4f}, ECE={internal_raw['ece']:.4f}, "
            f"Slope={internal_raw['calibration_slope']:.3f}, Intercept={internal_raw['calibration_intercept']:.3f}"
        )
        print(
            f"Internal isotonic -> AUC={internal_cal['roc_auc']:.4f}, "
            f"Brier={internal_cal['brier']:.4f}, ECE={internal_cal['ece']:.4f}, "
            f"Slope={internal_cal['calibration_slope']:.3f}, Intercept={internal_cal['calibration_intercept']:.3f}"
        )
        print(
            f"Unseen raw        -> AUC={unseen_raw['roc_auc']:.4f}, "
            f"Brier={unseen_raw['brier']:.4f}, ECE={unseen_raw['ece']:.4f}, "
            f"Slope={unseen_raw['calibration_slope']:.3f}, Intercept={unseen_raw['calibration_intercept']:.3f}"
        )
        print(
            f"Unseen isotonic   -> AUC={unseen_cal['roc_auc']:.4f}, "
            f"Brier={unseen_cal['brier']:.4f}, ECE={unseen_cal['ece']:.4f}, "
            f"Slope={unseen_cal['calibration_slope']:.3f}, Intercept={unseen_cal['calibration_intercept']:.3f}"
        )

        slug = safe_slug(config_name)

        for evaluation_set, y_eval, proba_raw_eval, proba_cal_eval in [
            ("internal_oof", y_internal, proba_raw_internal, proba_cal_internal),
            ("unseen_holdout", y_unseen, proba_raw_unseen, proba_cal_unseen),
        ]:
            table_raw = calibration_table(y_eval, proba_raw_eval)
            table_cal = calibration_table(y_eval, proba_cal_eval)
            table_raw.to_csv(
                os.path.join(OUT_DIR, f"calibration_table_{slug}_raw_{evaluation_set}.csv"),
                index=False,
            )
            table_cal.to_csv(
                os.path.join(OUT_DIR, f"calibration_table_{slug}_isotonic_{evaluation_set}.csv"),
                index=False,
            )

            dca_raw = decision_curve(y_eval, proba_raw_eval, thresholds)
            dca_cal = decision_curve(y_eval, proba_cal_eval, thresholds)

            dca_raw["config"] = config_name
            dca_raw["variant"] = "raw"
            dca_raw["evaluation_set"] = evaluation_set
            dca_raw["split_source"] = split_source

            dca_cal["config"] = config_name
            dca_cal["variant"] = "isotonic"
            dca_cal["evaluation_set"] = evaluation_set
            dca_cal["split_source"] = split_source

            dca_raw.to_csv(
                os.path.join(OUT_DIR, f"dca_{slug}_raw_{evaluation_set}.csv"),
                index=False,
            )
            dca_cal.to_csv(
                os.path.join(OUT_DIR, f"dca_{slug}_isotonic_{evaluation_set}.csv"),
                index=False,
            )

            raw_decision = summarize_dca_clinical_band(
                dca_raw,
                config_name=config_name,
                variant="raw",
                prevalence=y_eval.mean(),
                evaluation_set=evaluation_set,
                split_source=split_source,
                threshold_low=0.10,
                threshold_high=0.30,
            )
            if raw_decision is not None:
                dca_decision_rows.append(raw_decision)

            cal_decision = summarize_dca_clinical_band(
                dca_cal,
                config_name=config_name,
                variant="isotonic",
                prevalence=y_eval.mean(),
                evaluation_set=evaluation_set,
                split_source=split_source,
                threshold_low=0.10,
                threshold_high=0.30,
            )
            if cal_decision is not None:
                dca_decision_rows.append(cal_decision)

            plot_calibration(
                y_eval,
                proba_raw_eval,
                proba_cal_eval,
                title=f"Calibration Curve: {config_name} ({evaluation_set})",
                out_path=os.path.join(OUT_DIR, f"calibration_curve_{slug}_{evaluation_set}.png"),
            )

            plot_dca(
                dca_raw,
                dca_cal,
                title=f"Decision Curve Analysis: {config_name} ({evaluation_set})",
                out_path=os.path.join(OUT_DIR, f"dca_curve_{slug}_{evaluation_set}.png"),
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["evaluation_set", "config", "variant"])
    metrics_path = os.path.join(OUT_DIR, "clinical_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)

    calibration_summary = metrics_df[
        [
            "config",
            "variant",
            "evaluation_set",
            "split_source",
            "n_samples",
            "prevalence",
            "brier",
            "ece",
            "mce",
            "calibration_slope",
            "calibration_intercept",
        ]
    ].copy()
    calibration_summary_path = os.path.join(OUT_DIR, "calibration_summary.csv")
    calibration_summary.to_csv(calibration_summary_path, index=False)

    decision_df = pd.DataFrame(dca_decision_rows)
    decision_df = decision_df.sort_values(
        ["evaluation_set", "mean_gain_vs_treat_all"],
        ascending=[True, False],
    )
    decision_path = os.path.join(OUT_DIR, "dca_clinical_decision_summary.csv")
    decision_df.to_csv(decision_path, index=False)

    publication_paragraphs = build_publication_paragraphs(
        metrics_df,
        decision_df,
        prevalence=y_full.mean(),
        threshold_low=0.10,
        threshold_high=0.30,
    )

    report_lines = [
        "=" * 80,
        "STEP 11: CLINICAL VALIDATION REPORT",
        "=" * 80,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input dataset: {INPUT_DATA}",
        f"Samples (full): {len(df_full)}, KOA prevalence: {y_full.mean() * 100:.2f}%",
        "Validation protocol:",
        f"- Internal: OOF on internal partition ({(1-EARLY_SPLIT_TEST_SIZE)*100:.0f}%)",
        f"- Unseen: holdout partition ({EARLY_SPLIT_TEST_SIZE*100:.0f}%)",
        "- Missing data handling: train-fitted SimpleImputer (median/mode)",
        "- Clinical utility language: internal validation only",
        "",
        "Metrics by configuration:",
    ]

    for eval_set in ["internal_oof", "unseen_holdout"]:
        subset_eval = metrics_df[metrics_df["evaluation_set"] == eval_set]
        if len(subset_eval) == 0:
            continue
        report_lines.append(f"\n[{eval_set}]")
        for config_name in subset_eval["config"].unique():
            subset_cfg = subset_eval[subset_eval["config"] == config_name]
            report_lines.append(f"- {config_name}")
            for _, row in subset_cfg.iterrows():
                report_lines.append(
                    f"  {row['variant']:<8s} "
                    f"AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}, "
                    f"Brier={row['brier']:.4f}, ECE={row['ece']:.4f}, "
                    f"Slope={row['calibration_slope']:.3f}, Intercept={row['calibration_intercept']:.3f}, "
                    f"F1@0.50={row['f1']:.4f}, best_thr={row['best_f1_threshold']:.2f}, "
                    f"F1@best={row['f1_best_threshold']:.4f}"
                )

    report_lines.extend(
        [
            "",
            "Interpretation notes:",
            "- Isotonic calibration improves probability reliability when Brier/ECE and slope/intercept move toward ideal.",
            "- AUROC may stay similar or decline slightly after calibration.",
            "- DCA curve above treat-all and treat-none indicates internal clinical decision-support signal.",
            "- Clinical utility claims here are internal-validation only, not external prospective validation.",
        ]
    )

    decision_internal = decision_df[decision_df["evaluation_set"] == "internal_oof"].copy()
    if len(decision_internal) > 0:
        best = decision_internal.sort_values("mean_gain_vs_treat_all", ascending=False).iloc[0]
        report_lines.extend(
            [
                "",
                "DCA clinical recommendation (10%-30% threshold band):",
                (
                    f"- Best strategy: {best['config']} ({best['variant']}). "
                    f"Estimated mean avoided unnecessary interventions: "
                    f"{best['mean_avoided_unnecessary_per_100']:.1f} per 100 patients "
                    f"vs treat-all."
                ),
                (
                    f"- Relative reduction vs treat-all unnecessary interventions: "
                    f"{best['relative_reduction_vs_treat_all_pct']:.1f}%"
                ),
                (
                    f"- Peak avoidance: {best['max_avoided_unnecessary_per_100']:.1f} per 100 "
                    f"at threshold={best['best_threshold_for_avoidance']:.2f}."
                ),
            ]
        )

    report_path = os.path.join(OUT_DIR, "step11_clinical_validation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    decision_lines = [
        "=" * 80,
        "DCA CLINICAL DECISION RECOMMENDATION REPORT",
        "=" * 80,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "Clinical threshold band: 10% to 30% KOA risk",
        f"Population KOA prevalence (full): {y_full.mean() * 100:.2f}%",
        "Context: internal validation decision-support interpretation only",
        "",
        "Interpretation key:",
        "- avoided_unnecessary_per_100: expected avoided unnecessary referrals/imaging per 100 patients vs treat-all.",
        "- relative_reduction_vs_treat_all_pct: percentage reduction against treat-all unnecessary interventions.",
        "",
    ]

    if len(decision_internal) == 0:
        decision_lines.append("No DCA clinical decision summaries were generated.")
    else:
        best = decision_internal.sort_values("mean_gain_vs_treat_all", ascending=False).iloc[0]
        decision_lines.append("Top recommendation:")
        decision_lines.append(
            (
                f"- Use {best['config']} ({best['variant']}) for patients in 10%-30% risk range. "
                f"Estimated mean reduction: {best['mean_avoided_unnecessary_per_100']:.1f} "
                f"unnecessary knee MRI/advanced imaging referrals per 100 patients "
                f"({best['relative_reduction_vs_treat_all_pct']:.1f}% relative reduction vs treat-all)."
            )
        )
        decision_lines.append(
            (
                f"- Strongest point estimate is at threshold={best['best_threshold_for_avoidance']:.2f}, "
                f"with up to {best['max_avoided_unnecessary_per_100']:.1f} avoided per 100."
            )
        )
        decision_lines.append("")
        decision_lines.append("All evaluated internal strategies:")
        for _, row in decision_internal.sort_values("mean_gain_vs_treat_all", ascending=False).iterrows():
            decision_lines.append(
                (
                    f"- {row['config']} ({row['variant']}): "
                    f"mean gain vs treat-all={row['mean_gain_vs_treat_all']:.4f}, "
                    f"mean avoided={row['mean_avoided_unnecessary_per_100']:.1f}/100, "
                    f"relative reduction={row['relative_reduction_vs_treat_all_pct']:.1f}%, "
                    f"positive-threshold share={row['positive_share_vs_treat_all']*100:.1f}%"
                )
            )

    decision_report_path = os.path.join(OUT_DIR, "dca_clinical_recommendation_report.txt")
    with open(decision_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(decision_lines))

    publication_report_path = os.path.join(OUT_DIR, "step11_publication_paragraphs.txt")
    publication_lines = [
        "=" * 80,
        "PUBLICATION-READY PARAGRAPHS (METHODS / RESULTS)",
        "=" * 80,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "English - Methods",
        publication_paragraphs["methods_en"],
        "",
        "English - Results",
        publication_paragraphs["results_en"],
        "",
        "Turkish - Methods",
        publication_paragraphs["methods_tr"],
        "",
        "Turkish - Results",
        publication_paragraphs["results_tr"],
    ]
    with open(publication_report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(publication_lines))

    print("\n" + "=" * 80)
    print("STEP 11 CLINICAL VALIDATION COMPLETED")
    print("=" * 80)
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved calibration summary: {calibration_summary_path}")
    print(f"Saved report:  {report_path}")
    print(f"Saved DCA decision summary: {decision_path}")
    print(f"Saved DCA recommendation report: {decision_report_path}")
    print(f"Saved publication paragraphs: {publication_report_path}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
