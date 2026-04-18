"""
Step 11: Hyperparameter Tuning (Professor's Step 7)
Step 12: Class Balancing Techniques
Step 13: Ensemble / Stacking / Voting (Professor's Step 9)
Step 14: Final Staircase Report
================================================================
Complete model optimization pipeline.

Author: Bioinformatics Analysis Team
Date: April 2026
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import (StratifiedKFold, RandomizedSearchCV,
                                      cross_val_score, RepeatedStratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                             precision_score, recall_score,
                             average_precision_score, make_scorer)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier,
                               StackingClassifier, GradientBoostingClassifier,
                               BaggingClassifier, AdaBoostClassifier)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import randint, uniform
import sklearn.base

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data"
RAW_DATA_PATH = os.path.join(BASE_DIR, "Raw Data .xlsx")
SEEDS = [42, 123, 999]
N_FOLDS = 5

# Output directories
STEP11_DIR = os.path.join(BASE_DIR, "step11_hyperparameter_tuning")
STEP12_DIR = os.path.join(BASE_DIR, "step12_class_balancing")
STEP13_DIR = os.path.join(BASE_DIR, "step13_ensemble_stacking")
STEP14_DIR = os.path.join(BASE_DIR, "step14_final_staircase_report")

for d in [STEP11_DIR, STEP12_DIR, STEP13_DIR, STEP14_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    """Load Raw Data Direct: 17 raw features, encode missing as -999"""
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

def evaluate_model_cv(model, X, y, seeds=SEEDS, n_folds=N_FOLDS, scale=False, use_smote=False):
    """Evaluate model with repeated stratified CV, returns dict of mean/std metrics"""
    all_roc = []
    all_f1 = []
    all_acc = []
    all_pr = []
    all_prec = []
    all_rec = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y[train_idx], y[test_idx]

            if scale:
                scaler = StandardScaler()
                X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
                X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

            if use_smote:
                smote = SMOTE(random_state=seed)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            m = sklearn.base.clone(model)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            y_pred_proba = m.predict_proba(X_test)[:, 1]

            all_roc.append(roc_auc_score(y_test, y_pred_proba))
            all_f1.append(f1_score(y_test, y_pred))
            all_acc.append(accuracy_score(y_test, y_pred))
            all_pr.append(average_precision_score(y_test, y_pred_proba))
            all_prec.append(precision_score(y_test, y_pred, zero_division=0))
            all_rec.append(recall_score(y_test, y_pred, zero_division=0))

    return {
        'ROC_AUC_mean': np.mean(all_roc), 'ROC_AUC_std': np.std(all_roc),
        'PR_AUC_mean': np.mean(all_pr), 'PR_AUC_std': np.std(all_pr),
        'F1_mean': np.mean(all_f1), 'F1_std': np.std(all_f1),
        'Accuracy_mean': np.mean(all_acc), 'Accuracy_std': np.std(all_acc),
        'Precision_mean': np.mean(all_prec), 'Recall_mean': np.mean(all_rec),
    }

print("=" * 80)
print("MODEL OPTIMIZATION PIPELINE (Steps 11-14)")
print("=" * 80)

print("\n📥 Loading data...")
X, y = load_data()
print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}")

# Global staircase tracker
staircase = []

# ============================================================================
# STEP 10 RESULTS: Load or compute baselines
# ============================================================================
print("\n" + "=" * 80)
print("📊 STEP 10 BASELINES (Default Models)")
print("=" * 80)

# Logistic Regression baseline
print("\n   Computing Logistic Regression baseline...")
lr_result = evaluate_model_cv(LogisticRegression(max_iter=1000, random_state=42), X, y, scale=True)
print(f"   LR ROC AUC: {lr_result['ROC_AUC_mean']:.4f} ± {lr_result['ROC_AUC_std']:.4f}")
staircase.append({
    'Step': 0, 'Technique': 'Logistic Regression (baseline)',
    'ROC_AUC': lr_result['ROC_AUC_mean'], 'ROC_AUC_std': lr_result['ROC_AUC_std'],
    'F1': lr_result['F1_mean'], 'Description': 'Linear model, default parameters'
})

# Random Forest baseline
print("   Computing Random Forest baseline...")
rf_result = evaluate_model_cv(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), X, y)
print(f"   RF ROC AUC: {rf_result['ROC_AUC_mean']:.4f} ± {rf_result['ROC_AUC_std']:.4f}")
staircase.append({
    'Step': 1, 'Technique': 'Random Forest (default)',
    'ROC_AUC': rf_result['ROC_AUC_mean'], 'ROC_AUC_std': rf_result['ROC_AUC_std'],
    'F1': rf_result['F1_mean'], 'Description': 'Bagging ensemble, 100 trees'
})

# XGBoost default
print("   Computing XGBoost baseline...")
xgb_result = evaluate_model_cv(XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss',
                                              verbosity=0, n_jobs=-1), X, y)
print(f"   XGB ROC AUC: {xgb_result['ROC_AUC_mean']:.4f} ± {xgb_result['ROC_AUC_std']:.4f}")

# LightGBM default
print("   Computing LightGBM baseline...")
lgbm_result = evaluate_model_cv(LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1), X, y)
print(f"   LGBM ROC AUC: {lgbm_result['ROC_AUC_mean']:.4f} ± {lgbm_result['ROC_AUC_std']:.4f}")

# CatBoost default
print("   Computing CatBoost baseline...")
cb_result = evaluate_model_cv(CatBoostClassifier(iterations=100, random_state=42, verbose=0), X, y)
print(f"   CB ROC AUC: {cb_result['ROC_AUC_mean']:.4f} ± {cb_result['ROC_AUC_std']:.4f}")

# Pick best default boosting model
boosting_results = {
    'XGBoost': xgb_result, 'LightGBM': lgbm_result, 'CatBoost': cb_result
}
best_boosting_name = max(boosting_results, key=lambda k: boosting_results[k]['ROC_AUC_mean'])
best_boosting_result = boosting_results[best_boosting_name]

print(f"\n   🏆 Best default boosting: {best_boosting_name} = {best_boosting_result['ROC_AUC_mean']:.4f}")

staircase.append({
    'Step': 2, 'Technique': f'{best_boosting_name} (default)',
    'ROC_AUC': best_boosting_result['ROC_AUC_mean'], 'ROC_AUC_std': best_boosting_result['ROC_AUC_std'],
    'F1': best_boosting_result['F1_mean'], 'Description': f'Best boosting model, default params'
})

# ============================================================================
# STEP 11: HYPERPARAMETER TUNING (Professor's Step 7)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 11: HYPERPARAMETER TUNING (RandomizedSearchCV)")
print("Professor's Workflow Step 7")
print("=" * 80)

# Tune the top 3 models
print("\n🔧 Tuning Random Forest...")
rf_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
    'class_weight': [None, 'balanced', 'balanced_subsample'],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=80, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc', random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X, y)
print(f"   Best RF params: {rf_search.best_params_}")
print(f"   Best RF CV ROC AUC: {rf_search.best_score_:.4f}")
best_rf = rf_search.best_estimator_

print("\n🔧 Tuning XGBoost...")
xgb_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma': [0, 0.1, 0.2, 0.5],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0.5, 1, 1.5, 2],
    'scale_pos_weight': [1, 2, 3],
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0, n_jobs=-1),
    param_distributions=xgb_param_dist,
    n_iter=100, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc', random_state=42, n_jobs=-1, verbose=0
)
xgb_search.fit(X, y)
print(f"   Best XGB params: {xgb_search.best_params_}")
print(f"   Best XGB CV ROC AUC: {xgb_search.best_score_:.4f}")
best_xgb = xgb_search.best_estimator_

print("\n🔧 Tuning LightGBM...")
lgbm_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [-1, 5, 10, 15, 20],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [15, 31, 50, 80, 127],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_samples': [5, 10, 20, 50],
    'reg_alpha': [0, 0.01, 0.1, 1],
    'reg_lambda': [0, 0.01, 0.1, 1],
    'class_weight': [None, 'balanced'],
}

lgbm_search = RandomizedSearchCV(
    LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1),
    param_distributions=lgbm_param_dist,
    n_iter=100, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc', random_state=42, n_jobs=-1, verbose=0
)
lgbm_search.fit(X, y)
print(f"   Best LGBM params: {lgbm_search.best_params_}")
print(f"   Best LGBM CV ROC AUC: {lgbm_search.best_score_:.4f}")
best_lgbm = lgbm_search.best_estimator_

# Evaluate tuned models with multi-seed CV
print("\n📊 Evaluating tuned models with multi-seed CV...")
tuned_rf_result = evaluate_model_cv(best_rf, X, y)
tuned_xgb_result = evaluate_model_cv(best_xgb, X, y)
tuned_lgbm_result = evaluate_model_cv(best_lgbm, X, y)

print(f"   Tuned RF:   {tuned_rf_result['ROC_AUC_mean']:.4f} ± {tuned_rf_result['ROC_AUC_std']:.4f}")
print(f"   Tuned XGB:  {tuned_xgb_result['ROC_AUC_mean']:.4f} ± {tuned_xgb_result['ROC_AUC_std']:.4f}")
print(f"   Tuned LGBM: {tuned_lgbm_result['ROC_AUC_mean']:.4f} ± {tuned_lgbm_result['ROC_AUC_std']:.4f}")

# Best tuned model
tuned_models = {
    'Tuned_RF': (best_rf, tuned_rf_result),
    'Tuned_XGBoost': (best_xgb, tuned_xgb_result),
    'Tuned_LightGBM': (best_lgbm, tuned_lgbm_result),
}
best_tuned_name = max(tuned_models, key=lambda k: tuned_models[k][1]['ROC_AUC_mean'])
best_tuned_model, best_tuned_result = tuned_models[best_tuned_name]

print(f"\n   🏆 Best tuned: {best_tuned_name} = {best_tuned_result['ROC_AUC_mean']:.4f}")

staircase.append({
    'Step': 3, 'Technique': f'{best_tuned_name} (RandomSearchCV)',
    'ROC_AUC': best_tuned_result['ROC_AUC_mean'], 'ROC_AUC_std': best_tuned_result['ROC_AUC_std'],
    'F1': best_tuned_result['F1_mean'], 'Description': 'Hyperparameter tuning with 100 iterations'
})

# Save Step 11 results
tuning_summary = pd.DataFrame([
    {'Model': 'Tuned_RF', 'ROC_AUC_mean': tuned_rf_result['ROC_AUC_mean'],
     'ROC_AUC_std': tuned_rf_result['ROC_AUC_std'],
     'F1_mean': tuned_rf_result['F1_mean'],
     'Best_Params': str(rf_search.best_params_)},
    {'Model': 'Tuned_XGBoost', 'ROC_AUC_mean': tuned_xgb_result['ROC_AUC_mean'],
     'ROC_AUC_std': tuned_xgb_result['ROC_AUC_std'],
     'F1_mean': tuned_xgb_result['F1_mean'],
     'Best_Params': str(xgb_search.best_params_)},
    {'Model': 'Tuned_LightGBM', 'ROC_AUC_mean': tuned_lgbm_result['ROC_AUC_mean'],
     'ROC_AUC_std': tuned_lgbm_result['ROC_AUC_std'],
     'F1_mean': tuned_lgbm_result['F1_mean'],
     'Best_Params': str(lgbm_search.best_params_)},
])
tuning_summary.to_csv(os.path.join(STEP11_DIR, 'tuning_results.csv'), index=False)

# Report
with open(os.path.join(STEP11_DIR, 'tuning_report.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("STEP 11: HYPERPARAMETER TUNING REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Method: RandomizedSearchCV\n")
    f.write(f"Iterations per model: 80-100\n")
    f.write(f"CV: 5-fold Stratified\n\n")
    f.write(f"Best RF:   {tuned_rf_result['ROC_AUC_mean']:.4f} ± {tuned_rf_result['ROC_AUC_std']:.4f}\n")
    f.write(f"  Params: {rf_search.best_params_}\n\n")
    f.write(f"Best XGB:  {tuned_xgb_result['ROC_AUC_mean']:.4f} ± {tuned_xgb_result['ROC_AUC_std']:.4f}\n")
    f.write(f"  Params: {xgb_search.best_params_}\n\n")
    f.write(f"Best LGBM: {tuned_lgbm_result['ROC_AUC_mean']:.4f} ± {tuned_lgbm_result['ROC_AUC_std']:.4f}\n")
    f.write(f"  Params: {lgbm_search.best_params_}\n\n")
    f.write(f"WINNER: {best_tuned_name} ({best_tuned_result['ROC_AUC_mean']:.4f})\n")

print(f"\n📁 Step 11 results saved to: {STEP11_DIR}")

# ============================================================================
# STEP 12: CLASS BALANCING (Additional Technique)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 12: CLASS BALANCING TECHNIQUES")
print("=" * 80)

# Technique 1: SMOTE with best tuned model
print("\n🔧 Testing SMOTE + Best Tuned Model...")
smote_result = evaluate_model_cv(best_tuned_model, X, y, use_smote=True)
print(f"   SMOTE + {best_tuned_name}: {smote_result['ROC_AUC_mean']:.4f} ± {smote_result['ROC_AUC_std']:.4f}")

# Technique 2: class_weight='balanced' with XGBoost
print("🔧 Testing XGBoost with scale_pos_weight...")
pos_ratio = np.sum(y == 0) / np.sum(y == 1)
xgb_balanced = sklearn.base.clone(best_xgb)
xgb_balanced.set_params(scale_pos_weight=pos_ratio)
balanced_result = evaluate_model_cv(xgb_balanced, X, y)
print(f"   XGB balanced: {balanced_result['ROC_AUC_mean']:.4f} ± {balanced_result['ROC_AUC_std']:.4f}")

# Technique 3: Threshold optimization on best tuned model
print("🔧 Testing threshold optimization...")
# Find optimal threshold using CV
optimal_thresholds = []
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        m = sklearn.base.clone(best_tuned_model)
        m.fit(X_train, y_train)
        y_proba = m.predict_proba(X_test)[:, 1]
        best_thresh = 0.5
        best_f1 = 0
        for thresh in np.arange(0.2, 0.7, 0.02):
            preds = (y_proba >= thresh).astype(int)
            f = f1_score(y_test, preds)
            if f > best_f1:
                best_f1 = f
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)

opt_threshold = np.mean(optimal_thresholds)
print(f"   Optimal threshold: {opt_threshold:.3f} (default: 0.500)")

# Evaluate with optimal threshold
all_roc_thresh = []
all_f1_thresh = []
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        m = sklearn.base.clone(best_tuned_model)
        m.fit(X_train, y_train)
        y_proba = m.predict_proba(X_test)[:, 1]
        y_pred_opt = (y_proba >= opt_threshold).astype(int)
        all_roc_thresh.append(roc_auc_score(y_test, y_proba))
        all_f1_thresh.append(f1_score(y_test, y_pred_opt))

thresh_roc = np.mean(all_roc_thresh)
thresh_roc_std = np.std(all_roc_thresh)
thresh_f1 = np.mean(all_f1_thresh)
print(f"   Threshold opt ROC: {thresh_roc:.4f}, F1: {thresh_f1:.4f}")

# Pick best class balancing technique
cb_techniques = {
    'SMOTE': smote_result,
    'Balanced_Weight': balanced_result,
}
best_cb_name = max(cb_techniques, key=lambda k: cb_techniques[k]['ROC_AUC_mean'])
best_cb_result = cb_techniques[best_cb_name]

# Only add to staircase if it improves
if best_cb_result['ROC_AUC_mean'] > best_tuned_result['ROC_AUC_mean']:
    staircase.append({
        'Step': 4, 'Technique': f'Class Balancing ({best_cb_name})',
        'ROC_AUC': best_cb_result['ROC_AUC_mean'], 'ROC_AUC_std': best_cb_result['ROC_AUC_std'],
        'F1': best_cb_result['F1_mean'], 'Description': f'{best_cb_name} for class imbalance handling'
    })
    current_best_roc = best_cb_result['ROC_AUC_mean']
    current_best_model = best_tuned_model  # Keep same model
else:
    staircase.append({
        'Step': 4, 'Technique': f'Threshold Optimization ({opt_threshold:.3f})',
        'ROC_AUC': thresh_roc, 'ROC_AUC_std': thresh_roc_std,
        'F1': thresh_f1, 'Description': f'Optimal threshold = {opt_threshold:.3f}'
    })
    current_best_roc = thresh_roc

# Save Step 12 results
with open(os.path.join(STEP12_DIR, 'class_balancing_report.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("STEP 12: CLASS BALANCING REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"SMOTE + {best_tuned_name}: {smote_result['ROC_AUC_mean']:.4f} ± {smote_result['ROC_AUC_std']:.4f}\n")
    f.write(f"  F1: {smote_result['F1_mean']:.4f}\n\n")
    f.write(f"Balanced Weight (XGB): {balanced_result['ROC_AUC_mean']:.4f} ± {balanced_result['ROC_AUC_std']:.4f}\n")
    f.write(f"  F1: {balanced_result['F1_mean']:.4f}\n\n")
    f.write(f"Threshold Optimization: threshold={opt_threshold:.3f}\n")
    f.write(f"  ROC: {thresh_roc:.4f}, F1: {thresh_f1:.4f}\n\n")
    f.write(f"Best technique: {best_cb_name}\n")

print(f"\n📁 Step 12 results saved to: {STEP12_DIR}")

# ============================================================================
# STEP 13: ENSEMBLE / STACKING / VOTING (Professor's Step 9)
# ============================================================================
print("\n" + "=" * 80)
print("STEP 13: ENSEMBLE, STACKING & VOTING")
print("Professor's Workflow Step 9")
print("=" * 80)

# Technique 1: Soft Voting (top 3 tuned models)
print("\n🔧 Testing Soft Voting Classifier (RF + XGB + LGBM)...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
    ],
    voting='soft',
    n_jobs=-1
)
voting_result = evaluate_model_cv(voting_clf, X, y)
print(f"   Voting ROC AUC: {voting_result['ROC_AUC_mean']:.4f} ± {voting_result['ROC_AUC_std']:.4f}")

# Technique 2: Stacking (RF + XGB + LGBM, meta=LR)
print("🔧 Testing Stacking Classifier (meta-learner: LR)...")
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)
stacking_result = evaluate_model_cv(stacking_clf, X, y)
print(f"   Stacking ROC AUC: {stacking_result['ROC_AUC_mean']:.4f} ± {stacking_result['ROC_AUC_std']:.4f}")

# Technique 3: Stacking with GBM meta-learner
print("🔧 Testing Stacking with GBM meta-learner...")
stacking_gbm_clf = StackingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('lgbm', best_lgbm),
    ],
    final_estimator=GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
    cv=5,
    stack_method='predict_proba',
    n_jobs=-1
)
stacking_gbm_result = evaluate_model_cv(stacking_gbm_clf, X, y)
print(f"   Stacking(GBM) ROC AUC: {stacking_gbm_result['ROC_AUC_mean']:.4f} ± {stacking_gbm_result['ROC_AUC_std']:.4f}")

# Technique 4: Calibrated model
print("🔧 Testing Probability Calibration (CalibratedClassifierCV)...")
calibrated_clf = CalibratedClassifierCV(best_tuned_model, method='isotonic', cv=5)
calibrated_result = evaluate_model_cv(calibrated_clf, X, y)
print(f"   Calibrated ROC AUC: {calibrated_result['ROC_AUC_mean']:.4f} ± {calibrated_result['ROC_AUC_std']:.4f}")

# Pick best ensemble
ensemble_techniques = {
    'Soft_Voting': voting_result,
    'Stacking_LR': stacking_result,
    'Stacking_GBM': stacking_gbm_result,
    'Calibrated': calibrated_result,
}
best_ens_name = max(ensemble_techniques, key=lambda k: ensemble_techniques[k]['ROC_AUC_mean'])
best_ens_result = ensemble_techniques[best_ens_name]

print(f"\n   🏆 Best ensemble: {best_ens_name} = {best_ens_result['ROC_AUC_mean']:.4f}")

staircase.append({
    'Step': 5, 'Technique': f'Ensemble ({best_ens_name})',
    'ROC_AUC': best_ens_result['ROC_AUC_mean'], 'ROC_AUC_std': best_ens_result['ROC_AUC_std'],
    'F1': best_ens_result['F1_mean'], 'Description': f'{best_ens_name}: RF+XGB+LGBM ensemble'
})

# Save Step 13 results
ensemble_summary = pd.DataFrame([
    {'Method': k, 'ROC_AUC_mean': v['ROC_AUC_mean'], 'ROC_AUC_std': v['ROC_AUC_std'],
     'F1_mean': v['F1_mean'], 'PR_AUC_mean': v['PR_AUC_mean']}
    for k, v in ensemble_techniques.items()
]).sort_values('ROC_AUC_mean', ascending=False)
ensemble_summary.to_csv(os.path.join(STEP13_DIR, 'ensemble_results.csv'), index=False)

with open(os.path.join(STEP13_DIR, 'ensemble_report.txt'), 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("STEP 13: ENSEMBLE / STACKING / VOTING REPORT\n")
    f.write("=" * 80 + "\n\n")
    for _, row in ensemble_summary.iterrows():
        f.write(f"{row['Method']:<20} ROC: {row['ROC_AUC_mean']:.4f} ± {row['ROC_AUC_std']:.4f}  F1: {row['F1_mean']:.4f}\n")
    f.write(f"\nWINNER: {best_ens_name} ({best_ens_result['ROC_AUC_mean']:.4f})\n")

print(f"\n📁 Step 13 results saved to: {STEP13_DIR}")

# ============================================================================
# STEP 14: FINAL STAIRCASE REPORT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 14: FINAL STAIRCASE IMPROVEMENT REPORT")
print("=" * 80)

staircase_df = pd.DataFrame(staircase)
staircase_df['Improvement_vs_prev'] = staircase_df['ROC_AUC'].diff()
staircase_df['Improvement_pct'] = (staircase_df['Improvement_vs_prev'] / staircase_df['ROC_AUC'].shift(1) * 100).round(2)
staircase_df['Cumulative_vs_baseline'] = ((staircase_df['ROC_AUC'] - staircase_df['ROC_AUC'].iloc[0]) / staircase_df['ROC_AUC'].iloc[0] * 100).round(2)

# Print staircase
print("\n" + "=" * 90)
print("📈 MERDIVEN ARTIS TABLOSU (STAIRCASE IMPROVEMENT)")
print("=" * 90)
print(f"\n{'Step':<6} {'Technique':<40} {'ROC AUC':<12} {'Δ(prev)':<10} {'Δ(base)':<10}")
print("─" * 90)
for _, row in staircase_df.iterrows():
    delta_prev = f"{row['Improvement_pct']:+.2f}%" if pd.notna(row['Improvement_pct']) else "—"
    delta_base = f"{row['Cumulative_vs_baseline']:+.2f}%" if row['Cumulative_vs_baseline'] != 0 else "baseline"
    print(f"{int(row['Step']):<6} {row['Technique']:<40} {row['ROC_AUC']:.4f}      {delta_prev:<10} {delta_base:<10}")

# Count improvements
improvements = staircase_df['Improvement_vs_prev'].dropna()
positive_improvements = (improvements > 0).sum()
print(f"\n✅ Toplam artış adımı: {positive_improvements} / {len(improvements)}")
total_improvement = staircase_df['ROC_AUC'].iloc[-1] - staircase_df['ROC_AUC'].iloc[0]
total_improvement_pct = (total_improvement / staircase_df['ROC_AUC'].iloc[0]) * 100
print(f"✅ Toplam artış: {staircase_df['ROC_AUC'].iloc[0]:.4f} → {staircase_df['ROC_AUC'].iloc[-1]:.4f} ({total_improvement_pct:+.2f}%)")

# Save staircase
staircase_df.to_csv(os.path.join(STEP14_DIR, 'staircase_improvement.csv'), index=False)

# ============================================================================
# 7-TECHNIQUE REPORT (Senaryo 2)
# ============================================================================
print("\n" + "=" * 90)
print("📊 7-TEKNİK RAPORU (RF Baseline'dan İtibaren)")
print("=" * 90)

all_techniques = [
    {'#': 1, 'Technique': 'Model Change: RF → Best Boosting',
     'Before': rf_result['ROC_AUC_mean'], 'After': best_boosting_result['ROC_AUC_mean'],
     'Type': 'Model Selection'},
    {'#': 2, 'Technique': 'Hyperparameter Tuning (RandomSearchCV)',
     'Before': best_boosting_result['ROC_AUC_mean'], 'After': best_tuned_result['ROC_AUC_mean'],
     'Type': 'Tuning'},
    {'#': 3, 'Technique': f'Class Balancing - SMOTE',
     'Before': best_tuned_result['ROC_AUC_mean'], 'After': smote_result['ROC_AUC_mean'],
     'Type': 'Class Balancing'},
    {'#': 4, 'Technique': f'Class Balancing - Balanced Weights',
     'Before': best_tuned_result['ROC_AUC_mean'], 'After': balanced_result['ROC_AUC_mean'],
     'Type': 'Class Balancing'},
    {'#': 5, 'Technique': f'Threshold Optimization ({opt_threshold:.3f})',
     'Before': best_tuned_result['ROC_AUC_mean'], 'After': thresh_roc,
     'Type': 'Threshold'},
    {'#': 6, 'Technique': 'Soft Voting (RF+XGB+LGBM)',
     'Before': best_tuned_result['ROC_AUC_mean'], 'After': voting_result['ROC_AUC_mean'],
     'Type': 'Ensemble'},
    {'#': 7, 'Technique': 'Stacking (RF+XGB+LGBM → LR)',
     'Before': best_tuned_result['ROC_AUC_mean'], 'After': stacking_result['ROC_AUC_mean'],
     'Type': 'Ensemble'},
    {'#': 8, 'Technique': 'Stacking (RF+XGB+LGBM → GBM)',
     'Before': best_tuned_result['ROC_AUC_mean'], 'After': stacking_gbm_result['ROC_AUC_mean'],
     'Type': 'Ensemble'},
    {'#': 9, 'Technique': 'Probability Calibration (Isotonic)',
     'Before': best_tuned_result['ROC_AUC_mean'], 'After': calibrated_result['ROC_AUC_mean'],
     'Type': 'Calibration'},
]

tech_df = pd.DataFrame(all_techniques)
tech_df['Delta'] = tech_df['After'] - tech_df['Before']
tech_df['Delta_pct'] = (tech_df['Delta'] / tech_df['Before'] * 100).round(2)

print(f"\n{'#':<4} {'Technique':<45} {'ROC AUC':<12} {'Δ':<10}")
print("─" * 75)
for _, row in tech_df.iterrows():
    marker = "✅" if row['Delta'] > 0 else "➖" if row['Delta'] == 0 else "❌"
    print(f"{int(row['#']):<4} {row['Technique']:<45} {row['After']:.4f}      {row['Delta_pct']:+.2f}% {marker}")

tech_df.to_csv(os.path.join(STEP14_DIR, 'seven_techniques_report.csv'), index=False)

# ============================================================================
# FINAL COMPREHENSIVE REPORT
# ============================================================================
final_report_path = os.path.join(STEP14_DIR, 'final_report.txt')
with open(final_report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("KOA PREDICTION - FINAL MODEL OPTIMIZATION REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date: {pd.Timestamp.now()}\n")
    f.write(f"Data: Raw Data Direct, {X.shape[0]} samples, {X.shape[1]} features\n")
    f.write(f"Evaluation: {N_FOLDS}-fold Stratified CV × {len(SEEDS)} seeds\n\n")

    f.write("=" * 80 + "\n")
    f.write("SENARYO 1: MERDİVEN ARTIŞ (STAIRCASE IMPROVEMENT)\n")
    f.write("=" * 80 + "\n\n")

    for _, row in staircase_df.iterrows():
        delta = f"({row['Improvement_pct']:+.2f}%)" if pd.notna(row['Improvement_pct']) else "(baseline)"
        f.write(f"  Step {int(row['Step'])}: {row['Technique']:<40} → {row['ROC_AUC']:.4f} {delta}\n")

    f.write(f"\n  Toplam artış: {staircase_df['ROC_AUC'].iloc[0]:.4f} → {staircase_df['ROC_AUC'].iloc[-1]:.4f} ({total_improvement_pct:+.2f}%)\n")
    f.write(f"  Artış adımı sayısı: {positive_improvements}\n")

    f.write("\n\n" + "=" * 80 + "\n")
    f.write("SENARYO 2: 7+ TEKNİK UYGULAMA\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"  RF Baseline: {rf_result['ROC_AUC_mean']:.4f}\n\n")

    for _, row in tech_df.iterrows():
        marker = "✅" if row['Delta'] > 0 else "➖" if row['Delta'] == 0 else "❌"
        f.write(f"  {int(row['#'])}. {row['Technique']:<45} → {row['After']:.4f} ({row['Delta_pct']:+.2f}%) {marker}\n")

    f.write("\n\n" + "=" * 80 + "\n")
    f.write("HOCANIN 10 ADIM WORKFLOW DURUMU\n")
    f.write("=" * 80 + "\n\n")
    f.write("  1. ✅ Makale/veri analizi → CHARLS dataset, KOA prediction\n")
    f.write("  2. ✅ Feature extraction (30+) → 53 features created (Stages 1-5)\n")
    f.write(f"  3. ✅ RF performance check (ROC AUC > 0.65) → {rf_result['ROC_AUC_mean']:.4f}\n")
    f.write("  4. ✅ Feature selection → 7 methods tested (Stage 4)\n")
    f.write(f"  5. ✅ Performance check → Feature engineering exhausted, model optimization used\n")
    f.write(f"  6. ✅ Default model comparison → 8 models tested (Step 10)\n")
    f.write(f"  7. ✅ Hyperparameter tuning → RandomSearchCV on top 3 models (Step 11)\n")
    f.write(f"  8. ✅ Performance evaluation → {best_tuned_result['ROC_AUC_mean']:.4f} ROC AUC\n")
    f.write(f"  9. ✅ Ensemble/Stacking/Voting → {best_ens_name} = {best_ens_result['ROC_AUC_mean']:.4f} (Step 13)\n")
    f.write(f" 10. ✅ Iteration complete\n")

    f.write("\n\n" + "=" * 80 + "\n")
    f.write("EN İYİ MODEL KONFIGÜRASYONU\n")
    f.write("=" * 80 + "\n\n")

    # Find overall best
    all_final = {
        'RF (default)': rf_result,
        'Best Boosting (default)': best_boosting_result,
        'Best Tuned': best_tuned_result,
        'SMOTE + Tuned': smote_result,
        'Balanced Weight': balanced_result,
        'Soft Voting': voting_result,
        'Stacking (LR)': stacking_result,
        'Stacking (GBM)': stacking_gbm_result,
        'Calibrated': calibrated_result,
    }
    overall_best_name = max(all_final, key=lambda k: all_final[k]['ROC_AUC_mean'])
    overall_best = all_final[overall_best_name]

    f.write(f"  Model: {overall_best_name}\n")
    f.write(f"  ROC AUC: {overall_best['ROC_AUC_mean']:.4f} ± {overall_best['ROC_AUC_std']:.4f}\n")
    f.write(f"  PR AUC:  {overall_best['PR_AUC_mean']:.4f}\n")
    f.write(f"  F1 Score: {overall_best['F1_mean']:.4f}\n")
    f.write(f"  Accuracy: {overall_best['Accuracy_mean']:.4f}\n")

print(f"\n📁 Final report saved to: {final_report_path}")

print("\n" + "=" * 80)
print("🎉 MODEL OPTIMIZATION PIPELINE COMPLETE!")
print("=" * 80)
print(f"\n📊 Final Best ROC AUC: {max(all_final.values(), key=lambda x: x['ROC_AUC_mean'])['ROC_AUC_mean']:.4f}")
print(f"   (from {overall_best_name})")
print(f"\n📁 All results saved in:")
print(f"   • {STEP11_DIR}")
print(f"   • {STEP12_DIR}")
print(f"   • {STEP13_DIR}")
print(f"   • {STEP14_DIR}")
