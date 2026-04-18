"""
Step 10: Default Model Comparison (Professor's Step 6)
======================================================
Test a wide range of models (linear, boosting, bagging) with DEFAULT parameters.
Pick the best performing model.

Models tested:
  1. Logistic Regression (Linear)
  2. Decision Tree (Tree)
  3. K-Nearest Neighbors (Instance-based)
  4. Gradient Boosting (sklearn Boosting)
  5. Random Forest (Bagging)
  6. XGBoost (Boosting)
  7. LightGBM (Boosting)
  8. CatBoost (Boosting)

Evaluation: 5-fold Stratified CV x 3 seeds = 15 evaluations per model
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score,
                             average_precision_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import sklearn.base

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data"
RAW_DATA_PATH = os.path.join(BASE_DIR, "Raw Data .xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "step10_default_model_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [42, 123, 999]
N_FOLDS = 5

print("=" * 80)
print("STEP 10: DEFAULT MODEL COMPARISON")
print("Professor's Workflow Step 6: Test wide range of models with DEFAULT params")
print("=" * 80)

# ============================================================================
# DATA LOADING (Raw Data Direct - same as previous steps)
# ============================================================================
def load_data():
    """Load Raw Data Direct approach: 17 raw features, encode missing as -999"""
    df_raw = pd.read_excel(RAW_DATA_PATH)
    df = df_raw[df_raw['Arthritis'].notna()].copy()
    y = df['Arthritis']
    X = df.drop('Arthritis', axis=1)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('__MISSING__')
        X[col] = le.fit_transform(X[col].astype(str))
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        X[col] = X[col].fillna(-999)
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    return X, y

print("\n[*] Loading data...")
X, y = load_data()
print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f"   Class distribution: {np.bincount(y)}")
print(f"   Positive rate: {y.mean():.2%}")

# ============================================================================
# MODEL DEFINITIONS (ALL DEFAULT PARAMETERS)
# ============================================================================
def get_models():
    return {
        '1_LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        '2_DecisionTree': DecisionTreeClassifier(random_state=42),
        '3_KNN': KNeighborsClassifier(),
        '4_GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        '5_RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        '6_XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss',
                                    verbosity=0, n_jobs=-1),
        '7_LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1),
        '8_CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
    }

# ============================================================================
# EVALUATE EACH MODEL
# ============================================================================
print("\n" + "=" * 80)
print("[>] EVALUATING 8 MODELS (DEFAULT PARAMETERS)")
print("    Method: 5-fold Stratified CV x 3 seeds = 15 evaluations each")
print("=" * 80)

all_results = []
models = get_models()

for model_name, model_template in models.items():
    print(f"\n{'-' * 60}")
    print(f"  >> {model_name}")
    print(f"{'-' * 60}")

    model_scores = {'roc_auc': [], 'f1': [], 'accuracy': [], 'precision': [], 'recall': [], 'pr_auc': []}
    start_time = time.time()

    for seed in SEEDS:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if model_name in ['1_LogisticRegression', '3_KNN']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values

            model = sklearn.base.clone(model_template)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            model_scores['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
            model_scores['f1'].append(f1_score(y_test, y_pred))
            model_scores['accuracy'].append(accuracy_score(y_test, y_pred))
            model_scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            model_scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            model_scores['pr_auc'].append(average_precision_score(y_test, y_pred_proba))

            all_results.append({
                'Model': model_name, 'Seed': seed, 'Fold': fold_idx + 1,
                'ROC_AUC': model_scores['roc_auc'][-1],
                'PR_AUC': model_scores['pr_auc'][-1],
                'F1_Score': model_scores['f1'][-1],
                'Accuracy': model_scores['accuracy'][-1],
                'Precision': model_scores['precision'][-1],
                'Recall': model_scores['recall'][-1],
            })

    elapsed = time.time() - start_time
    roc_mean = np.mean(model_scores['roc_auc'])
    roc_std = np.std(model_scores['roc_auc'])
    f1_mean = np.mean(model_scores['f1'])
    pr_mean = np.mean(model_scores['pr_auc'])

    print(f"   ROC AUC: {roc_mean:.4f} +/- {roc_std:.4f}")
    print(f"   PR AUC:  {pr_mean:.4f}")
    print(f"   F1:      {f1_mean:.4f}")
    print(f"   Time:    {elapsed:.1f}s")

# ============================================================================
# ANALYSIS & SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("[*] COMPREHENSIVE RESULTS")
print("=" * 80)

results_df = pd.DataFrame(all_results)
detailed_path = os.path.join(OUTPUT_DIR, 'model_comparison_detailed.csv')
results_df.to_csv(detailed_path, index=False)

summary_data = []
for model in results_df['Model'].unique():
    subset = results_df[results_df['Model'] == model]
    summary_data.append({
        'Model': model,
        'ROC_AUC_mean': subset['ROC_AUC'].mean(),
        'ROC_AUC_std': subset['ROC_AUC'].std(),
        'PR_AUC_mean': subset['PR_AUC'].mean(),
        'PR_AUC_std': subset['PR_AUC'].std(),
        'F1_mean': subset['F1_Score'].mean(),
        'F1_std': subset['F1_Score'].std(),
        'Accuracy_mean': subset['Accuracy'].mean(),
        'Accuracy_std': subset['Accuracy'].std(),
        'Precision_mean': subset['Precision'].mean(),
        'Recall_mean': subset['Recall'].mean(),
    })

summary_df = pd.DataFrame(summary_data)
summary_df['CV_ROC'] = (summary_df['ROC_AUC_std'] / summary_df['ROC_AUC_mean']) * 100
summary_df = summary_df.sort_values('ROC_AUC_mean', ascending=False)

rf_baseline = summary_df[summary_df['Model'] == '5_RandomForest']['ROC_AUC_mean'].values[0]
summary_df['vs_RF_baseline'] = ((summary_df['ROC_AUC_mean'] - rf_baseline) / rf_baseline * 100).round(2)

print("\n" + "=" * 80)
print("[RANKING] MODEL RANKING BY ROC AUC (DEFAULT PARAMETERS)")
print("=" * 80)
print(f"\n{'Rank':<5} {'Model':<25} {'ROC AUC':<18} {'F1 Score':<18} {'PR AUC':<10} {'vs RF':<8}")
print("-" * 84)
for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
    marker = " <<BEST>>" if rank == 1 else ""
    print(f"{rank:<5} {row['Model']:<25} {row['ROC_AUC_mean']:.4f} +/- {row['ROC_AUC_std']:.4f}  "
          f"{row['F1_mean']:.4f} +/- {row['F1_std']:.4f}  {row['PR_AUC_mean']:.4f}    "
          f"{row['vs_RF_baseline']:+.2f}%{marker}")

summary_path = os.path.join(OUTPUT_DIR, 'model_comparison_summary.csv')
summary_df.to_csv(summary_path, index=False)

best = summary_df.iloc[0]
print(f"\n[WINNER] {best['Model']}")
print(f"   ROC AUC: {best['ROC_AUC_mean']:.4f} +/- {best['ROC_AUC_std']:.4f}")
print(f"   F1 Score: {best['F1_mean']:.4f} +/- {best['F1_std']:.4f}")
print(f"   vs RF Baseline: {best['vs_RF_baseline']:+.2f}%")

# ============================================================================
# REPORT
# ============================================================================
report_path = os.path.join(OUTPUT_DIR, 'model_comparison_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("STEP 10: DEFAULT MODEL COMPARISON REPORT\n")
    f.write("Professor's Workflow Step 6\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now()}\n")
    f.write(f"Data: Raw Data Direct, {X.shape[0]} samples, {X.shape[1]} features\n")
    f.write(f"Evaluation: {N_FOLDS}-fold Stratified CV x {len(SEEDS)} seeds = {N_FOLDS * len(SEEDS)} evaluations\n\n")
    f.write("RANKING:\n")
    f.write("-" * 80 + "\n")
    for rank, (_, row) in enumerate(summary_df.iterrows(), 1):
        f.write(f"{rank}. {row['Model']:<25} ROC: {row['ROC_AUC_mean']:.4f} +/- {row['ROC_AUC_std']:.4f}  "
                f"F1: {row['F1_mean']:.4f}  vs_RF: {row['vs_RF_baseline']:+.2f}%\n")
    f.write(f"\nWINNER: {best['Model']}\n")
    f.write(f"ROC AUC: {best['ROC_AUC_mean']:.4f} +/- {best['ROC_AUC_std']:.4f}\n")
    f.write(f"F1 Score: {best['F1_mean']:.4f} +/- {best['F1_std']:.4f}\n")
    lr_score = summary_df[summary_df['Model'] == '1_LogisticRegression']['ROC_AUC_mean'].values[0]
    rf_score = summary_df[summary_df['Model'] == '5_RandomForest']['ROC_AUC_mean'].values[0]
    best_score = best['ROC_AUC_mean']
    f.write(f"\n\nSTAIRCASE PROGRESS (Step 10):\n")
    f.write(f"  Logistic Regression (baseline): {lr_score:.4f}\n")
    f.write(f"  Random Forest (default):        {rf_score:.4f} ({((rf_score-lr_score)/lr_score*100):+.2f}%)\n")
    f.write(f"  Best Model (default):           {best_score:.4f} ({((best_score-lr_score)/lr_score*100):+.2f}%)\n")

print(f"\n[+] Results saved to: {OUTPUT_DIR}")
print(f"   - model_comparison_detailed.csv")
print(f"   - model_comparison_summary.csv")
print(f"   - model_comparison_report.txt")

print("\n" + "=" * 80)
print("[OK] STEP 10 COMPLETE -- Proceed to Step 11 (Hyperparameter Tuning)")
print("=" * 80)
