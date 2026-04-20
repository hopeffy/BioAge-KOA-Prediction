"""
STEP 12: Minimal Dot Plots for Key Evidence Tables
===================================================
Creates simple, publication-friendly dot plots for 5 core result tables:
1) Feature-set AUROC comparison
2) AUROC vs F1 model comparison
3) Aging clock standalone performance
4) Clinical metric comparison
5) DCA clinical decision summary
"""

import os
import shutil
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "visualizations", "key_table_dots")
OUT_CORE_DIR = os.path.join(BASE_DIR, "visualizations", "presentation_core")

PATHS = {
    "feature_impact": os.path.join(BASE_DIR, "step_04_ba_impact", "feature_impact_comparison.csv"),
    "paper_replication": os.path.join(BASE_DIR, "step_04_ba_impact", "paper_replication_results.csv"),
    "aging_clock": os.path.join(BASE_DIR, "step_04_ba_impact", "aging_clock_summary.csv"),
    "clinical_metrics": os.path.join(BASE_DIR, "step_11_clinical_validation", "clinical_metrics_summary.csv"),
    "dca_decision": os.path.join(BASE_DIR, "step_11_clinical_validation", "dca_clinical_decision_summary.csv"),
    "calibration_curve": os.path.join(BASE_DIR, "step_11_clinical_validation", "calibration_curve_raw17.png"),
}


def _assert_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing input file: {path}")


def plot_feature_impact():
    df = pd.read_csv(PATHS["feature_impact"]).copy()
    df = df.sort_values("roc_auc_mean", ascending=True)

    base_auc = float(df.loc[df["feature_set"] == "raw17", "roc_auc_mean"].iloc[0]) if (df["feature_set"] == "raw17").any() else None
    colors = ["#d62728" if x == "raw17" else "#1f77b4" for x in df["feature_set"]]

    plt.figure(figsize=(10, 7))
    plt.scatter(df["roc_auc_mean"], df["feature_set"], c=colors, s=70)
    if base_auc is not None:
        plt.axvline(base_auc, linestyle="--", linewidth=1.5, color="#d62728", alpha=0.65, label=f"raw17 AUROC={base_auc:.3f}")
        plt.legend(loc="lower right")

    plt.xlabel("ROC-AUC (mean)")
    plt.ylabel("Feature set")
    plt.title("Feature Set Comparison: AUROC Dot Plot")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_feature_impact_auroc_dot.png"), dpi=180)
    plt.close()


def plot_paper_replication():
    df = pd.read_csv(PATHS["paper_replication"]).copy()
    # Keep highest-performing configurations to avoid label clutter.
    df = df.sort_values("roc_auc_mean", ascending=False).head(20)

    plt.figure(figsize=(9, 7))
    for model, sub in df.groupby("model"):
        plt.scatter(sub["roc_auc_mean"], sub["f1_mean"], s=75, label=model, alpha=0.85)

    top = df.nlargest(5, "roc_auc_mean")
    for _, row in top.iterrows():
        label = f"{row['model']} | {row['feature_set']}"
        plt.annotate(label, (row["roc_auc_mean"], row["f1_mean"]), xytext=(5, 5), textcoords="offset points", fontsize=8)

    plt.xlabel("ROC-AUC (mean)")
    plt.ylabel("F1 (mean)")
    plt.title("Model Comparison: AUROC vs F1 (Top 20)")
    plt.grid(alpha=0.3)
    plt.legend(title="Model", loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "02_paper_replication_auc_f1_scatter.png"), dpi=180)
    plt.close()


def plot_aging_clock_summary():
    df = pd.read_csv(PATHS["aging_clock"]).copy()
    df = df.sort_values("auc_univariate", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)

    axes[0].scatter(df["auc_univariate"], df["aging_clock"], s=70, color="#1f77b4")
    axes[0].axvline(0.5, linestyle="--", linewidth=1.2, color="gray", alpha=0.8)
    axes[0].set_title("Univariate AUROC")
    axes[0].set_xlabel("AUROC")
    axes[0].grid(axis="x", alpha=0.3)

    axes[1].scatter(df["corr_with_koa"], df["aging_clock"], s=70, color="#ff7f0e")
    axes[1].axvline(0.0, linestyle="--", linewidth=1.2, color="gray", alpha=0.8)
    axes[1].set_title("Correlation with KOA")
    axes[1].set_xlabel("corr_with_koa")
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("Aging Clocks: Standalone Predictive Signal", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_aging_clock_summary_dots.png"), dpi=180)
    plt.close()


def plot_clinical_metrics():
    df = pd.read_csv(PATHS["clinical_metrics"]).copy()
    df = df[df["variant"] == "raw"].copy()
    df = df.sort_values("roc_auc", ascending=True)

    metrics = [
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("f1", "F1@0.50"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 7), sharey=True)
    for ax, (col, title) in zip(axes, metrics):
        ax.scatter(df[col], df["config"], s=70, color="#2ca02c")
        ax.set_title(title)
        ax.set_xlabel(col)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Clinical Metrics (raw variants)", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_clinical_metrics_dot_panels.png"), dpi=180)
    plt.close()


def plot_dca_summary():
    df = pd.read_csv(PATHS["dca_decision"]).copy()
    df["label"] = df["config"] + " (" + df["variant"] + ")"
    df = df.sort_values("mean_avoided_unnecessary_per_100", ascending=True)

    best_label = df.iloc[-1]["label"]
    colors = ["#d62728" if x == best_label else "#9467bd" for x in df["label"]]

    plt.figure(figsize=(11, 7))
    plt.scatter(df["mean_avoided_unnecessary_per_100"], df["label"], s=75, c=colors)
    plt.xlabel("Mean avoided unnecessary interventions per 100")
    plt.ylabel("Configuration")
    plt.title("DCA Clinical Decision Summary")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "05_dca_clinical_decision_dot.png"), dpi=180)
    plt.close()


def write_summary_report():
    report_path = os.path.join(OUT_DIR, "key_table_dotplots_report.txt")
    lines = [
        "=" * 70,
        "KEY TABLE DOT PLOTS - GENERATED FILES",
        "=" * 70,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Outputs:",
        "1) 01_feature_impact_auroc_dot.png",
        "2) 02_paper_replication_auc_f1_scatter.png",
        "3) 03_aging_clock_summary_dots.png",
        "4) 04_clinical_metrics_dot_panels.png",
        "5) 05_dca_clinical_decision_dot.png",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def build_core_story_panel():
    """Build a 2x2 presentation panel with the recommended storyline."""
    df_feature = pd.read_csv(PATHS["feature_impact"]).copy()
    df_feature = df_feature.sort_values("roc_auc_mean", ascending=True)

    df_rep = pd.read_csv(PATHS["paper_replication"]).copy()
    df_rep = df_rep.sort_values("roc_auc_mean", ascending=False).head(20)

    df_dca = pd.read_csv(PATHS["dca_decision"]).copy()
    df_dca["label"] = df_dca["config"] + " (" + df_dca["variant"] + ")"
    df_dca = df_dca.sort_values("mean_avoided_unnecessary_per_100", ascending=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Feature impact AUROC dot plot
    colors_feature = ["#d62728" if x == "raw17" else "#1f77b4" for x in df_feature["feature_set"]]
    axes[0, 0].scatter(df_feature["roc_auc_mean"], df_feature["feature_set"], c=colors_feature, s=52)
    if (df_feature["feature_set"] == "raw17").any():
        raw_auc = float(df_feature.loc[df_feature["feature_set"] == "raw17", "roc_auc_mean"].iloc[0])
        axes[0, 0].axvline(raw_auc, linestyle="--", linewidth=1.3, color="#d62728", alpha=0.7)
    axes[0, 0].set_title("A) Feature Set AUROC")
    axes[0, 0].set_xlabel("ROC-AUC (mean)")
    axes[0, 0].set_ylabel("Feature set")
    axes[0, 0].grid(axis="x", alpha=0.25)

    # Panel B: AUROC vs F1 scatter
    for model, sub in df_rep.groupby("model"):
        axes[0, 1].scatter(sub["roc_auc_mean"], sub["f1_mean"], s=58, alpha=0.85, label=model)
    axes[0, 1].set_title("B) AUROC vs F1 (Top 20)")
    axes[0, 1].set_xlabel("ROC-AUC (mean)")
    axes[0, 1].set_ylabel("F1 (mean)")
    axes[0, 1].grid(alpha=0.25)
    axes[0, 1].legend(fontsize=8, loc="lower left")

    # Panel C: DCA clinical summary dot plot
    best_label = df_dca.iloc[-1]["label"]
    colors_dca = ["#d62728" if x == best_label else "#9467bd" for x in df_dca["label"]]
    axes[1, 0].scatter(df_dca["mean_avoided_unnecessary_per_100"], df_dca["label"], c=colors_dca, s=58)
    axes[1, 0].set_title("C) DCA Clinical Utility")
    axes[1, 0].set_xlabel("Avoided unnecessary interventions per 100")
    axes[1, 0].set_ylabel("Configuration")
    axes[1, 0].grid(axis="x", alpha=0.25)

    # Panel D: Calibration curve image (already produced in step 11)
    calibration_img = plt.imread(PATHS["calibration_curve"])
    axes[1, 1].imshow(calibration_img)
    axes[1, 1].set_title("D) Calibration (raw17)")
    axes[1, 1].axis("off")

    fig.suptitle("Core Presentation Storyline: Performance -> Balance -> Clinical Utility -> Reliability", y=0.995)
    plt.tight_layout()
    panel_path = os.path.join(OUT_CORE_DIR, "core_storyline_2x2_panel.png")
    plt.savefig(panel_path, dpi=180)
    plt.close()


def export_core_pack():
    """Export the recommended 3+1 core presentation files."""
    os.makedirs(OUT_CORE_DIR, exist_ok=True)

    core_files = [
        (os.path.join(OUT_DIR, "01_feature_impact_auroc_dot.png"), "01_feature_impact_auroc_dot.png"),
        (os.path.join(OUT_DIR, "02_paper_replication_auc_f1_scatter.png"), "02_paper_replication_auc_f1_scatter.png"),
        (os.path.join(OUT_DIR, "05_dca_clinical_decision_dot.png"), "03_dca_clinical_decision_dot.png"),
        (PATHS["calibration_curve"], "04_calibration_curve_raw17.png"),
    ]

    copied_names = []
    for src, name in core_files:
        _assert_exists(src)
        dst = os.path.join(OUT_CORE_DIR, name)
        shutil.copy2(src, dst)
        copied_names.append(name)

    build_core_story_panel()

    report_lines = [
        "=" * 70,
        "PRESENTATION CORE PACK",
        "=" * 70,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Recommended storyline order:",
        "1) 01_feature_impact_auroc_dot.png",
        "2) 02_paper_replication_auc_f1_scatter.png",
        "3) 03_dca_clinical_decision_dot.png",
        "4) 04_calibration_curve_raw17.png",
        "",
        "Combined panel:",
        "- core_storyline_2x2_panel.png",
    ]

    report_path = os.path.join(OUT_CORE_DIR, "presentation_core_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(OUT_CORE_DIR, exist_ok=True)

    for path in PATHS.values():
        _assert_exists(path)

    plot_feature_impact()
    plot_paper_replication()
    plot_aging_clock_summary()
    plot_clinical_metrics()
    plot_dca_summary()
    write_summary_report()
    export_core_pack()

    print("=" * 70)
    print("STEP 12 COMPLETE: Minimal dot plots generated")
    print(f"Output directory: {OUT_DIR}")
    print(f"Core presentation pack: {OUT_CORE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
