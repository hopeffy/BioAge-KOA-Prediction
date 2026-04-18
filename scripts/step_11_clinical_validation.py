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

BASE_DIR = r"C:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data"
INPUT_DATA = os.path.join(BASE_DIR, "step_03_kdm_ba", "dataset_A_with_kdm_ba.csv")
OUT_DIR = os.path.join(BASE_DIR, "step_11_clinical_validation")

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


def oof_probabilities(X, y, calibrated=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        base_model = RandomForestClassifier(
            n_estimators=N_TREES,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        if calibrated:
            model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
        else:
            model = base_model

        model.fit(X_train_s, y_train)
        oof[test_idx] = model.predict_proba(X_test_s)[:, 1]

    return oof


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
    best_auc_row = metrics_df.sort_values("roc_auc", ascending=False).iloc[0]

    methods_en = (
        "Methods: We evaluated KOA risk models using stratified 5-fold out-of-fold prediction, "
        "with a Random Forest classifier as the base learner and optional isotonic probability calibration. "
        "Model performance was quantified using discrimination (ROC-AUC, PR-AUC), calibration "
        "(Brier score, expected calibration error), and decision curve analysis (DCA). "
        f"Clinical utility was assessed across threshold probabilities from {int(threshold_low*100)}% to "
        f"{int(threshold_high*100)}%, against treat-all and treat-none strategies."
    )

    methods_tr = (
        "Yöntem: KOA risk modelleri stratified 5-fold out-of-fold tahmin yaklaşımı ile değerlendirildi; "
        "temel öğrenici olarak Random Forest kullanıldı ve isteğe bağlı isotonic kalibrasyon uygulandı. "
        "Model performansı ayrım gücü (ROC-AUC, PR-AUC), kalibrasyon (Brier skoru, beklenen kalibrasyon hatası) "
        "ve karar eğrisi analizi (DCA) ile ölçüldü. "
        f"Klinik fayda, %{int(threshold_low*100)}-%{int(threshold_high*100)} risk eşik aralığında "
        "treat-all ve treat-none stratejilerine karşı değerlendirildi."
    )

    if len(decision_df) > 0:
        best_dca_row = decision_df.sort_values("mean_gain_vs_treat_all", ascending=False).iloc[0]
        results_en = (
            f"Results: The highest discrimination was observed for {best_auc_row['config']} "
            f"({best_auc_row['variant']}), with ROC-AUC={best_auc_row['roc_auc']:.3f}, "
            f"PR-AUC={best_auc_row['pr_auc']:.3f}, Brier={best_auc_row['brier']:.3f}, "
            f"and ECE={best_auc_row['ece']:.3f}. In DCA, {best_dca_row['config']} "
            f"({best_dca_row['variant']}) yielded the greatest clinical net benefit in the "
            f"{int(threshold_low*100)}%-{int(threshold_high*100)}% threshold range, corresponding to an estimated "
            f"{best_dca_row['mean_avoided_unnecessary_per_100']:.1f} avoided unnecessary referrals/imaging per 100 patients "
            f"versus treat-all ({best_dca_row['relative_reduction_vs_treat_all_pct']:.1f}% relative reduction)."
        )

        results_tr = (
            f"Bulgular: En yüksek ayrım gücü {best_auc_row['config']} ({best_auc_row['variant']}) modelinde gözlendi "
            f"(ROC-AUC={best_auc_row['roc_auc']:.3f}, PR-AUC={best_auc_row['pr_auc']:.3f}, "
            f"Brier={best_auc_row['brier']:.3f}, ECE={best_auc_row['ece']:.3f}). DCA analizinde "
            f"{best_dca_row['config']} ({best_dca_row['variant']}) modeli %{int(threshold_low*100)}-%{int(threshold_high*100)} "
            f"eşik aralığında en yüksek klinik net faydayı sağladı ve treat-all stratejisine kıyasla "
            f"100 hasta başına tahmini {best_dca_row['mean_avoided_unnecessary_per_100']:.1f} gereksiz sevk/görüntüleme azalımı "
            f"({best_dca_row['relative_reduction_vs_treat_all_pct']:.1f}% relatif azalma) sundu."
        )
    else:
        results_en = (
            f"Results: The highest discrimination was observed for {best_auc_row['config']} "
            f"({best_auc_row['variant']}), with ROC-AUC={best_auc_row['roc_auc']:.3f}, "
            f"PR-AUC={best_auc_row['pr_auc']:.3f}, Brier={best_auc_row['brier']:.3f}, and ECE={best_auc_row['ece']:.3f}. "
            "DCA summary statistics were not available for publication text generation."
        )
        results_tr = (
            f"Bulgular: En yüksek ayrım gücü {best_auc_row['config']} ({best_auc_row['variant']}) modelinde gözlendi "
            f"(ROC-AUC={best_auc_row['roc_auc']:.3f}, PR-AUC={best_auc_row['pr_auc']:.3f}, "
            f"Brier={best_auc_row['brier']:.3f}, ECE={best_auc_row['ece']:.3f}). "
            "Yayın metni için DCA özet istatistikleri üretilemedi."
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

    df = pd.read_csv(INPUT_DATA)
    y = df["KOA"].values

    print(f"\nLoaded dataset: {df.shape}")
    print(f"KOA prevalence: {y.mean() * 100:.2f}%")

    feature_sets = build_feature_sets(df)
    print(f"Feature configurations: {list(feature_sets.keys())}")

    thresholds = np.arange(0.05, 0.61, 0.01)
    metrics_rows = []
    dca_decision_rows = []

    for config_name, cols in feature_sets.items():
        available_cols = [c for c in cols if c in df.columns]
        X = df[available_cols].values

        print("\n" + "-" * 80)
        print(f"Running config: {config_name} ({len(available_cols)} features)")
        print("-" * 80)

        proba_raw = oof_probabilities(X, y, calibrated=False)
        proba_cal = oof_probabilities(X, y, calibrated=True)

        raw_metrics = compute_binary_metrics(y, proba_raw)
        cal_metrics = compute_binary_metrics(y, proba_cal)

        raw_ece, raw_mce = expected_calibration_error(y, proba_raw, n_bins=10)
        cal_ece, cal_mce = expected_calibration_error(y, proba_cal, n_bins=10)

        raw_best_thr, raw_best = best_f1_threshold_metrics(y, proba_raw)
        cal_best_thr, cal_best = best_f1_threshold_metrics(y, proba_cal)

        raw_metrics["ece"] = raw_ece
        raw_metrics["mce"] = raw_mce
        raw_metrics["best_f1_threshold"] = raw_best_thr
        raw_metrics["f1_best_threshold"] = raw_best["f1"]
        raw_metrics["precision_best_threshold"] = raw_best["precision"]
        raw_metrics["recall_best_threshold"] = raw_best["recall"]

        cal_metrics["ece"] = cal_ece
        cal_metrics["mce"] = cal_mce
        cal_metrics["best_f1_threshold"] = cal_best_thr
        cal_metrics["f1_best_threshold"] = cal_best["f1"]
        cal_metrics["precision_best_threshold"] = cal_best["precision"]
        cal_metrics["recall_best_threshold"] = cal_best["recall"]

        metrics_rows.append({"config": config_name, "variant": "raw", **raw_metrics})
        metrics_rows.append({"config": config_name, "variant": "isotonic", **cal_metrics})

        print(
            f"Raw      -> AUC={raw_metrics['roc_auc']:.4f}, "
            f"Brier={raw_metrics['brier']:.4f}, ECE={raw_metrics['ece']:.4f}"
        )
        print(
            f"            Best F1 threshold={raw_metrics['best_f1_threshold']:.2f}, "
            f"F1={raw_metrics['f1_best_threshold']:.4f}"
        )
        print(
            f"Isotonic -> AUC={cal_metrics['roc_auc']:.4f}, "
            f"Brier={cal_metrics['brier']:.4f}, ECE={cal_metrics['ece']:.4f}"
        )
        print(
            f"            Best F1 threshold={cal_metrics['best_f1_threshold']:.2f}, "
            f"F1={cal_metrics['f1_best_threshold']:.4f}"
        )

        slug = safe_slug(config_name)

        table_raw = calibration_table(y, proba_raw)
        table_cal = calibration_table(y, proba_cal)
        table_raw.to_csv(os.path.join(OUT_DIR, f"calibration_table_{slug}_raw.csv"), index=False)
        table_cal.to_csv(os.path.join(OUT_DIR, f"calibration_table_{slug}_isotonic.csv"), index=False)

        dca_raw = decision_curve(y, proba_raw, thresholds)
        dca_cal = decision_curve(y, proba_cal, thresholds)

        dca_raw["config"] = config_name
        dca_raw["variant"] = "raw"
        dca_cal["config"] = config_name
        dca_cal["variant"] = "isotonic"

        dca_raw.to_csv(os.path.join(OUT_DIR, f"dca_{slug}_raw.csv"), index=False)
        dca_cal.to_csv(os.path.join(OUT_DIR, f"dca_{slug}_isotonic.csv"), index=False)

        raw_decision = summarize_dca_clinical_band(
            dca_raw,
            config_name=config_name,
            variant="raw",
            prevalence=y.mean(),
            threshold_low=0.10,
            threshold_high=0.30,
        )
        if raw_decision is not None:
            dca_decision_rows.append(raw_decision)

        cal_decision = summarize_dca_clinical_band(
            dca_cal,
            config_name=config_name,
            variant="isotonic",
            prevalence=y.mean(),
            threshold_low=0.10,
            threshold_high=0.30,
        )
        if cal_decision is not None:
            dca_decision_rows.append(cal_decision)

        plot_calibration(
            y,
            proba_raw,
            proba_cal,
            title=f"Calibration Curve: {config_name}",
            out_path=os.path.join(OUT_DIR, f"calibration_curve_{slug}.png"),
        )

        plot_dca(
            dca_raw,
            dca_cal,
            title=f"Decision Curve Analysis: {config_name}",
            out_path=os.path.join(OUT_DIR, f"dca_curve_{slug}.png"),
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df = metrics_df.sort_values(["config", "variant"])
    metrics_path = os.path.join(OUT_DIR, "clinical_metrics_summary.csv")
    metrics_df.to_csv(metrics_path, index=False)

    decision_df = pd.DataFrame(dca_decision_rows)
    decision_df = decision_df.sort_values("mean_gain_vs_treat_all", ascending=False)
    decision_path = os.path.join(OUT_DIR, "dca_clinical_decision_summary.csv")
    decision_df.to_csv(decision_path, index=False)

    publication_paragraphs = build_publication_paragraphs(
        metrics_df,
        decision_df,
        prevalence=y.mean(),
        threshold_low=0.10,
        threshold_high=0.30,
    )

    report_lines = [
        "=" * 80,
        "STEP 11: CLINICAL VALIDATION REPORT",
        "=" * 80,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Input dataset: {INPUT_DATA}",
        f"Samples: {len(df)}, KOA prevalence: {y.mean() * 100:.2f}%",
        "",
        "Metrics by configuration:",
    ]

    for config_name in metrics_df["config"].unique():
        subset = metrics_df[metrics_df["config"] == config_name]
        report_lines.append(f"\n- {config_name}")
        for _, row in subset.iterrows():
            report_lines.append(
                f"  {row['variant']:<8s} "
                f"AUC={row['roc_auc']:.4f}, PR-AUC={row['pr_auc']:.4f}, "
                f"Brier={row['brier']:.4f}, ECE={row['ece']:.4f}, "
                f"F1@0.50={row['f1']:.4f}, "
                f"best_thr={row['best_f1_threshold']:.2f}, "
                f"F1@best={row['f1_best_threshold']:.4f}"
            )

    report_lines.extend(
        [
            "",
            "Interpretation notes:",
            "- Isotonic calibration improves probability reliability when Brier/ECE decreases.",
            "- AUROC may stay similar or decline slightly after calibration.",
            "- DCA curve above treat-all and treat-none indicates clinical net benefit.",
        ]
    )

    if len(decision_df) > 0:
        best = decision_df.iloc[0]
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
        f"Population KOA prevalence: {y.mean() * 100:.2f}%",
        "",
        "Interpretation key:",
        "- avoided_unnecessary_per_100: expected avoided unnecessary referrals/imaging per 100 patients vs treat-all.",
        "- relative_reduction_vs_treat_all_pct: percentage reduction against treat-all unnecessary interventions.",
        "",
    ]

    if len(decision_df) == 0:
        decision_lines.append("No DCA clinical decision summaries were generated.")
    else:
        best = decision_df.iloc[0]
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
        decision_lines.append("All evaluated strategies:")
        for _, row in decision_df.iterrows():
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
    print(f"Saved report:  {report_path}")
    print(f"Saved DCA decision summary: {decision_path}")
    print(f"Saved DCA recommendation report: {decision_report_path}")
    print(f"Saved publication paragraphs: {publication_report_path}")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
