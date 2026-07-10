"""
STEP 18: Biological Age Replacement Experiment
================================================
Replace Sheet1's opaque "Biological Age" with our transparent KDM-BA / PhenoAge
inside the raw17 feature set. Same 17-feature count, same CV methodology as
step_04 (3 seeds x 5-fold StratifiedKFold, RF with StandardScaler).

Configurations:
  1. raw17           — baseline (Sheet1 BA)      [17 features]
  2. raw16_only      — no BA at all              [16 features]
  3. raw16+KDM_log   — KDM-BA (log) replaces BA  [17 features]
  4. raw16+KDM_orig  — KDM-BA (orig) replaces BA [17 features]
  5. raw16+PhenoAge  — PhenoAge_Adapted replaces [17 features]
  6. raw16+PhenoAge_Accel — PhenoAge_Accel replaces [17 features]

Also runs multi-model (XGBoost, LightGBM) for robustness.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STEP_DIR = os.path.join(BASE_DIR, 'step_18_ba_replace')

# Feature groups
RAW16 = ['wave', 'Time', 'Gender', 'Age_New', 'Marital', 'Education',
         'Residence', 'Hypertension', 'Dyslipidemia', 'Diabetes',
         'Cancer', 'CVD', 'Smoke', 'Drink', 'BMI', 'BMI_New']

RAW17 = RAW16 + ['Biological Age']


def run_cv(df, feature_cols, y, models=None, seeds=(42, 123, 999), n_splits=5):
    """Run 3-seed x 5-fold StratifiedKFold CV, return per-model metrics."""
    if models is None:
        models = {
            'RF': lambda: RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            'XGB': lambda: XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                         random_state=42, eval_metric='logloss',
                                         use_label_encoder=False),
            'LGBM': lambda: LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                          random_state=42, verbose=-1),
        }

    X = df[feature_cols].values
    all_results = {}

    for model_name, model_factory in models.items():
        auc_list, prauc_list, brier_list, f1_list = [], [], [], []

        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                model = model_factory()
                model.fit(X_train_s, y_train)
                proba = model.predict_proba(X_test_s)[:, 1]

                auc_list.append(roc_auc_score(y_test, proba))
                prauc_list.append(average_precision_score(y_test, proba))
                brier_list.append(brier_score_loss(y_test, proba))
                f1_list.append(f1_score(y_test, (proba >= 0.5).astype(int)))

        all_results[model_name] = {
            'auc_mean': np.mean(auc_list),
            'auc_std': np.std(auc_list),
            'pr_auc_mean': np.mean(prauc_list),
            'pr_auc_std': np.std(prauc_list),
            'brier_mean': np.mean(brier_list),
            'brier_std': np.std(brier_list),
            'f1_mean': np.mean(f1_list),
            'f1_std': np.std(f1_list),
        }

    return all_results


def main():
    print("=" * 80)
    print("STEP 18: BIOLOGICAL AGE REPLACEMENT EXPERIMENT")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    os.makedirs(STEP_DIR, exist_ok=True)

    # Load data with KDM-BA
    df = pd.read_csv(os.path.join(BASE_DIR, 'step_03_kdm_ba', 'dataset_A_with_kdm_ba.csv'))
    y = df['KOA'].values
    print(f"Dataset loaded: {df.shape}, KOA rate: {y.mean()*100:.1f}%\n")

    # Define configurations
    configs = {
        'raw17 (Sheet1 BA)':       RAW17,
        'raw16_only (no BA)':       RAW16,
        'raw16+KDM_log':            RAW16 + ['BA_KDM_log'],
        'raw16+KDM_orig':           RAW16 + ['BA_KDM_orig'],
        'raw16+PhenoAge_Adapted':   RAW16 + ['PhenoAge_Adapted'],
        'raw16+PhenoAge_Accel':     RAW16 + ['PhenoAge_Adapted_Accel'],
    }

    # Verify all columns exist
    for name, cols in configs.items():
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(f"WARNING: {name} missing columns: {missing}")

    # Run experiments
    all_results = []
    for name, cols in configs.items():
        print(f"\nRunning: {name} ({len(cols)} features)")
        print("-" * 60)
        results = run_cv(df, cols, y)
        for model_name, metrics in results.items():
            print(f"  {model_name:>6s}: AUROC={metrics['auc_mean']:.4f} +/- {metrics['auc_std']:.4f}  "
                  f"PR-AUC={metrics['pr_auc_mean']:.4f}  Brier={metrics['brier_mean']:.4f}  F1={metrics['f1_mean']:.4f}")
            all_results.append({
                'config': name,
                'n_features': len(cols),
                'model': model_name,
                'auc_mean': metrics['auc_mean'],
                'auc_std': metrics['auc_std'],
                'pr_auc_mean': metrics['pr_auc_mean'],
                'pr_auc_std': metrics['pr_auc_std'],
                'brier_mean': metrics['brier_mean'],
                'brier_std': metrics['brier_std'],
                'f1_mean': metrics['f1_mean'],
                'f1_std': metrics['f1_std'],
            })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(STEP_DIR, 'step18_results.csv'), index=False)

    # Summary table: RF only (matching step_04 methodology)
    rf_results = results_df[results_df['model'] == 'RF'].sort_values('auc_mean', ascending=False)
    rf_results.to_csv(os.path.join(STEP_DIR, 'step18_rf_summary.csv'), index=False)

    print("\n" + "=" * 80)
    print("SUMMARY (RF, sorted by AUROC)")
    print("=" * 80)
    print(f"{'Config':<32s} {'AUC':>8s} {'PR-AUC':>8s} {'Brier':>8s} {'F1':>8s}")
    print("-" * 80)
    for _, row in rf_results.iterrows():
        print(f"{row['config']:<32s} {row['auc_mean']:>8.4f} {row['pr_auc_mean']:>8.4f} "
              f"{row['brier_mean']:>8.4f} {row['f1_mean']:>8.4f}")

    # Delta analysis
    print("\n" + "=" * 80)
    print("DELTA ANALYSIS (vs raw17 baseline)")
    print("=" * 80)
    baseline_auc = rf_results[rf_results['config'] == 'raw17 (Sheet1 BA)']['auc_mean'].values[0]
    for _, row in rf_results.iterrows():
        delta = row['auc_mean'] - baseline_auc
        sign = '+' if delta >= 0 else ''
        print(f"  {row['config']:<32s} AUC={row['auc_mean']:.4f}  delta={sign}{delta:.4f}")

    # Generate report
    report = []
    report.append("=" * 80)
    report.append("STEP 18: BIOLOGICAL AGE REPLACEMENT EXPERIMENT - REPORT")
    report.append("=" * 80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("PURPOSE")
    report.append("-" * 40)
    report.append("Replace Sheet1's opaque 'Biological Age' with transparent KDM-BA")
    report.append("and PhenoAge_Adapted inside raw17. Same feature count, same CV.")
    report.append("Tests whether Sheet1 BA is irreplaceable or any biomarker-derived")
    report.append("BA achieves equivalent performance.")
    report.append("")
    report.append("METHODOLOGY")
    report.append("-" * 40)
    report.append("  Models: RF (200 trees), XGBoost (200), LightGBM (200)")
    report.append("  CV: 3 seeds (42,123,999) x 5-fold StratifiedKFold = 15 runs each")
    report.append("  Standardization: StandardScaler (fit on train only)")
    report.append("  Metrics: AUROC, PR-AUC, Brier, F1")
    report.append("")
    report.append("RESULTS (RF)")
    report.append("-" * 40)
    report.append(f"{'Config':<32s} {'AUC':>8s} {'PR-AUC':>8s} {'Brier':>8s} {'F1':>8s}")
    for _, row in rf_results.iterrows():
        report.append(f"{row['config']:<32s} {row['auc_mean']:>8.4f} {row['pr_auc_mean']:>8.4f} "
                       f"{row['brier_mean']:>8.4f} {row['f1_mean']:>8.4f}")
    report.append("")
    report.append("DELTA vs raw17 baseline (RF)")
    report.append("-" * 40)
    for _, row in rf_results.iterrows():
        delta = row['auc_mean'] - baseline_auc
        sign = '+' if delta >= 0 else ''
        report.append(f"  {row['config']:<32s} delta={sign}{delta:.4f}")
    report.append("")
    report.append("INTERPRETATION")
    report.append("-" * 40)
    best_replacement = rf_results[rf_results['config'] != 'raw17 (Sheet1 BA)'].iloc[0]
    worst = rf_results.iloc[-1]
    report.append(f"  Best replacement: {best_replacement['config']} (AUC={best_replacement['auc_mean']:.4f})")
    report.append(f"  Baseline (Sheet1 BA): AUC={baseline_auc:.4f}")
    report.append(f"  Gap: {baseline_auc - best_replacement['auc_mean']:.4f}")
    report.append("")
    if baseline_auc - best_replacement['auc_mean'] < 0.02:
        report.append("  CONCLUSION: KDM-BA or PhenoAge achieves equivalent performance.")
        report.append("  Sheet1 BA is NOT special -- any biomarker-derived BA works.")
    elif baseline_auc - best_replacement['auc_mean'] < 0.05:
        report.append("  CONCLUSION: Moderate gap. Sheet1 BA captures some additional signal,")
        report.append("  but transparent KDM-BA is a reasonable substitute.")
    else:
        report.append("  CONCLUSION: Large gap. Sheet1 BA encodes signal beyond standard KDM.")
        report.append("  Their BA formula may use additional variables or a different method.")
    report.append("")
    report.append("FILES SAVED")
    report.append("-" * 40)
    report.append(f"  step18_results.csv (all models)")
    report.append(f"  step18_rf_summary.csv (RF only, sorted)")
    report.append(f"  step18_report.txt (this report)")

    report_text = "\n".join(report)
    with open(os.path.join(STEP_DIR, 'step18_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nFiles saved in: {STEP_DIR}")
    print("  step18_results.csv")
    print("  step18_rf_summary.csv")
    print("  step18_report.txt")
    print("\n" + "=" * 80)
    print("STEP 18 COMPLETED")
    print("=" * 80)


if __name__ == '__main__':
    main()