"""
STEP 02: Baseline Models (KOA Target, Sheet1 Data)
===================================================
Train baseline models on Dataset A and Dataset B with:
- 5-fold Stratified CV x 3 seeds
- Models: LR, RF, XGBoost, LightGBM, CatBoost
- Feature sets: raw17 (no biomarkers), raw17+biomarkers, raw17+BA, raw17+biomarkers+BA
- Compare with Phase 1 results (Arthritis, AUROC 0.71)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STEP_DIR = os.path.join(BASE_DIR, 'step_02_baseline')
STEP01_DIR = os.path.join(BASE_DIR, 'step_01_data_prep')

SEEDS = [42, 123, 999]
N_FOLDS = 5
EARLY_SPLIT_TEST_SIZE = 0.20
EARLY_SPLIT_RANDOM_STATE = 42

def get_feature_sets(df):
    raw17 = ['wave', 'Time', 'Gender', 'Age_New', 'Marital', 'Education',
              'Residence', 'Hypertension', 'Dyslipidemia', 'Diabetes',
              'Cancer', 'CVD', 'Smoke', 'Drink', 'BMI', 'BMI_New',
              'Biological Age']
    
    biomarkers_log = ['plt_10.9.L', 'crp_mg.L', 'Hb A1c',
                      'creatinine_mg.d L', 'bun_mg.d L',
                      'TC_mg.d L', 'TG_mg.d L', 'sbp.mean']
    
    biomarkers_orig = ['platelet_original', 'crp_original', 'hba1c_original',
                       'creatinine_original', 'bun_original',
                       'total_cholesterol_original', 'triglycerides_original',
                       'sbp_original']
    
    feature_sets = {
        'raw17': raw17,
        'raw17_biomarkers_log': raw17 + biomarkers_log,
        'raw17_biomarkers_orig': raw17 + biomarkers_orig,
        'raw17_BA': raw17,
        'raw17_biomarkers_log_BA': raw17 + biomarkers_log + ['Biological Age'],
        'raw17_biomarkers_orig_BA': raw17 + biomarkers_orig + ['Biological Age'],
    }
    
    return feature_sets


def train_fitted_impute_and_scale(X_train_df, X_test_df):
    X_train_proc = X_train_df.copy()
    X_test_proc = X_test_df.copy()

    numeric_cols = X_train_proc.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_train_proc.columns if c not in numeric_cols]

    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X_train_proc[numeric_cols] = num_imputer.fit_transform(X_train_proc[numeric_cols])
        X_test_proc[numeric_cols] = num_imputer.transform(X_test_proc[numeric_cols])

    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
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


def load_dataset_partitions(dataset_df, dataset_key):
    train_path = os.path.join(STEP01_DIR, f'{dataset_key}_internal_train.csv')
    unseen_path = os.path.join(STEP01_DIR, f'{dataset_key}_unseen_test.csv')

    if os.path.exists(train_path) and os.path.exists(unseen_path):
        train_df = pd.read_csv(train_path)
        unseen_df = pd.read_csv(unseen_path)
        if 'KOA' in train_df.columns and 'KOA' in unseen_df.columns:
            return train_df, unseen_df, 'step01_early_split'

    train_df, unseen_df = train_test_split(
        dataset_df,
        test_size=EARLY_SPLIT_TEST_SIZE,
        random_state=EARLY_SPLIT_RANDOM_STATE,
        stratify=dataset_df['KOA'],
    )
    return train_df.reset_index(drop=True), unseen_df.reset_index(drop=True), 'step02_fallback_split'


def evaluate_unseen_holdout(model, X_internal, y_internal, X_unseen, y_unseen):
    X_internal_scaled, X_unseen_scaled = train_fitted_impute_and_scale(X_internal, X_unseen)

    model_clone = type(model)(**model.get_params())
    if isinstance(model_clone, CatBoostClassifier):
        model_clone.fit(X_internal_scaled, y_internal, verbose=0)
    elif isinstance(model_clone, LGBMClassifier):
            model_clone.fit(X_internal_scaled, y_internal)
    else:
        model_clone.fit(X_internal_scaled, y_internal)

    y_proba = model_clone.predict_proba(X_unseen_scaled)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        'roc_auc': roc_auc_score(y_unseen, y_proba),
        'f1': f1_score(y_unseen, y_pred),
        'accuracy': accuracy_score(y_unseen, y_pred),
        'precision': precision_score(y_unseen, y_pred, zero_division=0),
        'recall': recall_score(y_unseen, y_pred),
    }

def evaluate_model(model, X, y, seeds=SEEDS, n_folds=N_FOLDS):
    results = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            X_train_scaled, X_test_scaled = train_fitted_impute_and_scale(X_train, X_test)
            
            model_clone = type(model)(**model.get_params())
            
            if isinstance(model_clone, CatBoostClassifier):
                model_clone.fit(X_train_scaled, y_train, verbose=0)
            elif isinstance(model_clone, LGBMClassifier):
                    model_clone.fit(X_train_scaled, y_train)
            else:
                model_clone.fit(X_train_scaled, y_train)
            
            y_pred_proba = model_clone.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            results.append({
                'seed': seed,
                'fold': fold,
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'f1': f1_score(y_test, y_pred),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred),
            })
    
    df_results = pd.DataFrame(results)
    summary = {
        'roc_auc_mean': df_results['roc_auc'].mean(),
        'roc_auc_std': df_results['roc_auc'].std(),
        'f1_mean': df_results['f1'].mean(),
        'f1_std': df_results['f1'].std(),
        'accuracy_mean': df_results['accuracy'].mean(),
        'accuracy_std': df_results['accuracy'].std(),
    }
    return summary, df_results

def main():
    print("=" * 80)
    print("STEP 02: BASELINE MODELS (KOA Target, Sheet1 Data)")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    dataset_a = pd.read_csv(os.path.join(STEP01_DIR, 'dataset_A_all.csv'))
    dataset_b = pd.read_csv(os.path.join(STEP01_DIR, 'dataset_B_ba55.csv'))
    
    if not os.path.exists(STEP_DIR):
        os.makedirs(STEP_DIR)
    
    feature_sets = get_feature_sets(dataset_a)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
    }
    
    all_results = []
    
    dataset_configs = [
        ('Dataset_A_12329', 'dataset_A', dataset_a),
        ('Dataset_B_7635', 'dataset_B', dataset_b),
    ]

    for dataset_name, dataset_key, df in dataset_configs:
        internal_df, unseen_df, split_source = load_dataset_partitions(df, dataset_key)

        print(f"\n{'='*80}")
        print(
            f"  {dataset_name} | split={split_source} | "
            f"internal={len(internal_df)} (KOA={internal_df['KOA'].mean()*100:.1f}%) | "
            f"unseen={len(unseen_df)} (KOA={unseen_df['KOA'].mean()*100:.1f}%)"
        )
        print(f"{'='*80}")

        y_internal = internal_df['KOA']
        y_unseen = unseen_df['KOA']
        
        for fs_name, fs_cols in feature_sets.items():
            if not all(col in internal_df.columns for col in fs_cols):
                print(f"  SKIP {fs_name}: missing columns")
                continue

            X_internal = internal_df[fs_cols]
            X_unseen = unseen_df[fs_cols]
            
            for model_name, model in models.items():
                config_name = f"{dataset_name}|{fs_name}|{model_name}"
                print(f"\n  Evaluating: {fs_name} + {model_name}")
                
                try:
                    summary, _ = evaluate_model(model, X_internal, y_internal)
                    unseen_metrics = evaluate_unseen_holdout(
                        model,
                        X_internal,
                        y_internal,
                        X_unseen,
                        y_unseen,
                    )

                    summary['dataset'] = dataset_name
                    summary['feature_set'] = fs_name
                    summary['model'] = model_name
                    summary['n_features'] = len(fs_cols)
                    summary['split_source'] = split_source
                    summary['n_internal_samples'] = len(internal_df)
                    summary['n_unseen_samples'] = len(unseen_df)
                    summary['internal_koa_rate'] = internal_df['KOA'].mean()
                    summary['unseen_koa_rate'] = unseen_df['KOA'].mean()
                    summary['unseen_roc_auc'] = unseen_metrics['roc_auc']
                    summary['unseen_f1'] = unseen_metrics['f1']
                    summary['unseen_accuracy'] = unseen_metrics['accuracy']
                    summary['unseen_precision'] = unseen_metrics['precision']
                    summary['unseen_recall'] = unseen_metrics['recall']
                    all_results.append(summary)
                    
                    print(f"    Internal ROC AUC: {summary['roc_auc_mean']:.4f} +/- {summary['roc_auc_std']:.4f}")
                    print(f"    Internal F1:      {summary['f1_mean']:.4f} +/- {summary['f1_std']:.4f}")
                    print(f"    Internal Acc:     {summary['accuracy_mean']:.4f} +/- {summary['accuracy_std']:.4f}")
                    print(f"    Unseen ROC AUC:   {summary['unseen_roc_auc']:.4f}")
                    print(f"    Unseen F1:        {summary['unseen_f1']:.4f}")
                except Exception as e:
                    print(f"    ERROR: {e}")
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('roc_auc_mean', ascending=False)
    results_df.to_csv(os.path.join(STEP_DIR, 'baseline_results.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 80)
    print("\nTop 10 configurations by internal ROC AUC:")
    print(
        results_df[
            [
                'dataset', 'feature_set', 'model', 'n_features',
                'roc_auc_mean', 'roc_auc_std', 'f1_mean', 'unseen_roc_auc', 'unseen_f1'
            ]
        ].head(10).to_string(index=False)
    )
    
    print("\n\nComparison: Feature set impact (averaged across datasets and models):")
    fs_comparison = results_df.groupby('feature_set').agg({
        'roc_auc_mean': 'mean',
        'f1_mean': 'mean',
        'n_features': 'first'
    }).sort_values('roc_auc_mean', ascending=False)
    print(fs_comparison.to_string())
    
    print("\n\nComparison: Dataset impact (averaged across feature sets and models):")
    ds_comparison = results_df.groupby('dataset').agg({
        'roc_auc_mean': 'mean',
        'f1_mean': 'mean'
    }).sort_values('roc_auc_mean', ascending=False)
    print(ds_comparison.to_string())
    
    print("\n\nPhase 1 reference: AUROC = 0.7058 (Arthritis target, Sheet data)")
    print("Phase 2 target: AUROC > 0.85 (with KDM-BA)")
    
    report = []
    report.append("=" * 80)
    report.append("STEP 02: BASELINE MODEL REPORT")
    report.append("=" * 80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("Validation protocol:")
    report.append("  - Internal: 5-fold CV x 3 seeds on internal partition")
    report.append("  - Unseen: early 80/20 holdout (Step 01) when available, deterministic fallback otherwise")
    report.append("  - Missing data: train-fitted SimpleImputer (median/mode)")
    report.append("")
    report.append("Datasets:")
    report.append(f"  Dataset A: {len(dataset_a)} rows, KOA rate={dataset_a['KOA'].mean()*100:.1f}%")
    report.append(f"  Dataset B: {len(dataset_b)} rows, KOA rate={dataset_b['KOA'].mean()*100:.1f}%")
    report.append("")
    report.append("Top 10 configurations:")
    for _, row in results_df.head(10).iterrows():
        report.append(
            f"  {row['dataset']} | {row['feature_set']} | {row['model']} | "
            f"AUC_internal={row['roc_auc_mean']:.4f} +/- {row['roc_auc_std']:.4f} | "
            f"F1_internal={row['f1_mean']:.4f} | "
            f"AUC_unseen={row['unseen_roc_auc']:.4f} | F1_unseen={row['unseen_f1']:.4f}"
        )
    report.append("")
    report.append("Feature set comparison (avg across datasets & models):")
    for _, row in fs_comparison.iterrows():
        report.append(f"  {row.name}: AUC={row['roc_auc_mean']:.4f}, F1={row['f1_mean']:.4f}, n_feat={int(row['n_features'])}")
    report.append("")
    report.append("Phase 1 reference: AUROC = 0.7058 (Arthritis target, Sheet data)")
    report.append("Best Phase 2 baseline: see above")
    
    report_text = "\n".join(report)
    with open(os.path.join(STEP_DIR, 'step02_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nReport saved to {os.path.join(STEP_DIR, 'step02_report.txt')}")
    print(f"Results saved to {os.path.join(STEP_DIR, 'baseline_results.csv')}")

if __name__ == '__main__':
    main()