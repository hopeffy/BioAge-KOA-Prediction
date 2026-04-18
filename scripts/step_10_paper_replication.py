"""
STEP 10: Paper Replication - Exact Feature Space
==================================================
Replicate Fu et al. (2025) PLOS ONE paper methodology:

1. Feature Space: Paper's SHAP plot 13 features
   - Remove: Age_New, wave, Time, 8 raw biomarkers
   - Keep: Biological Age, Gender, Marital, Education, Residence,
           Hypertension, Dyslipidemia, Diabetes, Cancer, CVD,
           Smoke, Drink, BMI (continuous)
   - BMI_New (category) optional

2. Standardization: Z-score on BA and BMI (continuous variables)

3. One-Hot Encoding: Categorical variables (Gender, Marital, Education, Residence)

4. Model: XGBoost, 70/30 stratified train/test split
   - Also test with Optuna optimization

5. Evaluate: AUROC, PR-AUC, F1-Macro, F1-Binary
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve, roc_curve, classification_report
)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = r'C:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data'
STEP_DIR = os.path.join(BASE_DIR, 'step_10_paper_replication')
os.makedirs(STEP_DIR, exist_ok=True)


def find_thresh_youden(y, p):
    fpr, tpr, th = roc_curve(y, p)
    return th[np.argmax(tpr - fpr)]


def find_thresh_f1(y, p):
    prec, rec, th = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    return th[np.argmax(f1[:-1])]


def evaluate_all(y_true, y_proba, label=""):
    auc = roc_auc_score(y_true, y_proba)
    prauc = average_precision_score(y_true, y_proba)

    results = {'roc_auc': auc, 'pr_auc': prauc}

    for th_name, th_val in [('0.5', 0.5),
                             ('youden', find_thresh_youden(y_true, y_proba)),
                             ('f1_optimal', find_thresh_f1(y_true, y_proba))]:
        y_pred = (y_proba >= th_val).astype(int)
        results[f'f1_macro_{th_name}'] = f1_score(y_true, y_pred, average='macro')
        results[f'f1_binary_{th_name}'] = f1_score(y_true, y_pred)
        results[f'accuracy_{th_name}'] = accuracy_score(y_true, y_pred)
        results[f'precision_{th_name}'] = precision_score(y_true, y_pred, zero_division=0)
        results[f'recall_{th_name}'] = recall_score(y_true, y_pred)
        results[f'threshold_{th_name}'] = th_val

    if label:
        print(f"\n{label}")
        print(f"  AUROC={auc:.4f}  PR-AUC={prauc:.4f}")
        for th_name in ['0.5', 'youden', 'f1_optimal']:
            print(f"  [{th_name:>12s}] F1-Macro={results[f'f1_macro_{th_name}']:.4f}  "
                  f"F1={results[f'f1_binary_{th_name}']:.4f}  "
                  f"Acc={results[f'accuracy_{th_name}']:.4f}  "
                  f"Recall={results[f'recall_{th_name}']:.4f}  "
                  f"th={results[f'threshold_{th_name}']:.3f}")
    return results


def main():
    t0 = datetime.now()
    print("=" * 80)
    print("STEP 10: PAPER REPLICATION - EXACT FEATURE SPACE")
    print("=" * 80)
    print(f"Start: {t0.strftime('%Y-%m-%d %H:%M:%S')}\n")

    df = pd.read_csv(os.path.join(BASE_DIR, 'step_01_data_prep', 'dataset_A_all.csv'))
    y = df['KOA'].values
    spw = (y == 0).sum() / (y == 1).sum()
    print(f"Dataset: {len(df)}, KOA={y.mean()*100:.1f}%, spw={spw:.2f}")

    # =====================================================================
    # FEATURE SETS - Paper's approach
    # =====================================================================

    # Paper's LASSO-selected 11 features:
    # BA, Gender, Education, Residence, Hypertension, Dyslipidemia,
    # CVD, Smoke, Drink, BMI Category, Cancer
    # Plus 2 more from SHAP: Marital, Diabetes
    # Total: 13 features (removing wave, Time, Age_New, 8 biomarkers)

    # SET A: Paper's exact 11 features (LASSO-selected)
    paper_11 = ['Biological Age', 'Gender', 'Education', 'Residence',
                 'Hypertension', 'Dyslipidemia', 'CVD', 'Smoke',
                 'Drink', 'BMI_New', 'Cancer']

    # SET B: Paper 11 + Marital + Diabetes = 13 (full SHAP)
    paper_13 = ['Biological Age', 'Gender', 'Marital', 'Education', 'Residence',
                 'Hypertension', 'Dyslipidemia', 'Diabetes', 'Cancer', 'CVD',
                 'Smoke', 'Drink', 'BMI_New']

    # SET C: Paper 13 BUT use continuous BMI instead of BMI_New
    paper_13_cont_bmi = ['Biological Age', 'Gender', 'Marital', 'Education', 'Residence',
                          'Hypertension', 'Dyslipidemia', 'Diabetes', 'Cancer', 'CVD',
                          'Smoke', 'Drink', 'BMI']

    # SET D: Paper 13 with continuous BMI (BA + BMI Z-scored)
    # Same as C but with Z-score applied to BA and BMI

    all_results = []

    # =====================================================================
    # APPROACH 1: 70/30 Train/Test Split (Paper's exact methodology)
    # =====================================================================
    print("\n" + "=" * 80)
    print("APPROACH 1: 70/30 Train/Test Split (Paper's methodology)")
    print("=" * 80)

    for set_name, feature_list in [('paper_11', paper_11), ('paper_13', paper_13), ('paper_13_cont_BMI', paper_13_cont_bmi)]:
        print(f"\n--- Feature Set: {set_name} ({len(feature_list)} features) ---")
        print(f"    Features: {feature_list}")

        X_raw = df[feature_list].values

        cat_cols_in_set = [c for c in feature_list if c in ['Gender', 'Marital', 'Education', 'Residence']]
        num_cols_in_set = [c for c in feature_list if c not in cat_cols_in_set]

        # One-hot encode categorical variables (paper's approach)
        if cat_cols_in_set:
            cat_indices = [feature_list.index(c) for c in cat_cols_in_set]
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            X_cat = encoder.fit_transform(X_raw[:, cat_indices])
            X_num = np.delete(X_raw, cat_indices, axis=1)
            X = np.hstack([X_num, X_cat])
            ohe_names = encoder.get_feature_names_out(cat_cols_in_set).tolist()
            feat_names = num_cols_in_set + ohe_names
        else:
            X = X_raw
            feat_names = feature_list.copy()

        # Z-score standardization (paper's approach)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print(f"    After encoding: {X.shape[1]} features")

        # 70/30 stratified split (paper's approach)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        spw_train = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"    Train: {len(y_train)}, KOA={y_train.mean()*100:.1f}%, spw={spw_train:.2f}")
        print(f"    Test:  {len(y_test)}, KOA={y_test.mean()*100:.1f}%")

        # XGBoost (paper's model)
        models = {
            'XGB_default': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'XGB_spw': XGBClassifier(scale_pos_weight=spw_train, random_state=42,
                                      eval_metric='logloss', use_label_encoder=False),
            'XGB_tuned': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                        subsample=0.8, colsample_bytree=0.7,
                                        scale_pos_weight=spw_train, random_state=42,
                                        eval_metric='logloss', use_label_encoder=False),
            'RF': RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            res = evaluate_all(y_test, y_proba, f"{set_name} | {model_name} | 70/30 split")
            res['feature_set'] = set_name
            res['model'] = model_name
            res['split'] = '70/30'
            res['n_features'] = X.shape[1]
            all_results.append(res)

    # =====================================================================
    # APPROACH 2: 5-Fold CV × 3 Seeds (our robust approach)
    # =====================================================================
    print("\n" + "=" * 80)
    print("APPROACH 2: 5-Fold CV × 3 Seeds (robust evaluation)")
    print("=" * 80)

    for set_name, feature_list in [('paper_11', paper_11), ('paper_13', paper_13), ('paper_13_cont_BMI', paper_13_cont_bmi)]:
        print(f"\n--- Feature Set: {set_name} (CV evaluation) ---")

        X_raw = df[feature_list].values

        cat_cols_in_set = [c for c in feature_list if c in ['Gender', 'Marital', 'Education', 'Residence']]
        num_cols_in_set = [c for c in feature_list if c not in cat_cols_in_set]

        if cat_cols_in_set:
            cat_indices = [feature_list.index(c) for c in cat_cols_in_set]
            encoder = OneHotEncoder(sparse_output=False, drop='first')
            X_cat = encoder.fit_transform(X_raw[:, cat_indices])
            X_num = np.delete(X_raw, cat_indices, axis=1)
            X = np.hstack([X_num, X_cat])
        else:
            X = X_raw.copy()

        models_cv = {
            'XGB_spw': lambda: XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                              subsample=0.8, colsample_bytree=0.7,
                                              scale_pos_weight=spw, random_state=42,
                                              eval_metric='logloss', use_label_encoder=False),
            'XGB_default': lambda: XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'RF': lambda: RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        }

        for model_name, model_fn in models_cv.items():
            cv_rows = []
            for seed in [42, 123, 999]:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for fold, (tri, tei) in enumerate(skf.split(X, y)):
                    Xtr, Xte = X[tri], X[tei]
                    ytr, yte = y[tri], y[tei]
                    sc = StandardScaler()
                    Xtr_s = sc.fit_transform(Xtr)
                    Xte_s = sc.transform(Xte)
                    m = model_fn()
                    m.fit(Xtr_s, ytr)
                    proba = m.predict_proba(Xte_s)[:, 1]
                    th_f1 = find_thresh_f1(yte, proba)
                    for tn, tv in [('0.5', 0.5), ('f1_optimal', th_f1)]:
                        pred = (proba >= tv).astype(int)
                        cv_rows.append({
                            'roc_auc': roc_auc_score(yte, proba),
                            'pr_auc': average_precision_score(yte, proba),
                            'f1_macro': f1_score(yte, pred, average='macro'),
                            'f1_binary': f1_score(yte, pred),
                            'accuracy': accuracy_score(yte, pred),
                            'threshold_method': tn,
                        })

            cv_df = pd.DataFrame(cv_rows)
            for th in ['0.5', 'f1_optimal']:
                sub = cv_df[cv_df['threshold_method'] == th]
                res = {
                    'feature_set': set_name, 'model': model_name, 'split': '5fold_CV',
                    'n_features': X.shape[1],
                    'roc_auc': sub['roc_auc'].mean(), 'pr_auc': sub['pr_auc'].mean(),
                    'f1_macro': sub['f1_macro'].mean(), 'f1_binary': sub['f1_binary'].mean(),
                    'accuracy': sub['accuracy'].mean(),
                    'threshold_method': th,
                    'roc_auc_std': sub['roc_auc'].std(),
                    'f1_macro_std': sub['f1_macro'].std(),
                }
                all_results.append(res)

            for th in ['0.5', 'f1_optimal']:
                sub = cv_df[cv_df['threshold_method'] == th]
                print(f"  {set_name} | {model_name} | CV | {th}: "
                      f"AUROC={sub['roc_auc'].mean():.4f}+/-{sub['roc_auc'].std():.4f} "
                      f"PR-AUC={sub['pr_auc'].mean():.4f} "
                      f"F1-Macro={sub['f1_macro'].mean():.4f}+/-{sub['f1_macro'].std():.4f}")

    # =====================================================================
    # APPROACH 3: Optuna XGBoost on paper_13_cont_BMI + Z-score
    # =====================================================================
    print("\n" + "=" * 80)
    print("APPROACH 3: Optuna XGBoost Optimization on Paper Features")
    print("=" * 80)

    feature_list = paper_13_cont_bmi
    X_raw = df[feature_list].values
    cat_cols_in_set = [c for c in feature_list if c in ['Gender', 'Marital', 'Education', 'Residence']]
    cat_indices = [feature_list.index(c) for c in cat_cols_in_set]
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_cat = encoder.fit_transform(X_raw[:, cat_indices])
    X_num = np.delete(X_raw, cat_indices, axis=1)
    X_paper = np.hstack([X_num, X_cat])
    print(f"Paper features ({len(feature_list)}): {feature_list}")
    print(f"After one-hot: {X_paper.shape[1]} features")

    print("\nOptuna XGBoost (100 trials)...")
    def xgb_obj(trial):
        p = {
            'n_estimators': trial.suggest_int('n', 100, 600),
            'max_depth': trial.suggest_int('md', 3, 7),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('ss', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('cs', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_float('spw', 1, spw),
            'reg_alpha': trial.suggest_float('ra', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('rl', 1e-8, 10, log=True),
            'min_child_weight': trial.suggest_int('mcw', 1, 10),
            'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False,
        }
        scores = []
        for seed in [42, 123]:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            for tri, vai in skf.split(X_paper, y):
                sc = StandardScaler()
                Xt = sc.fit_transform(X_paper[tri]); Xv = sc.transform(X_paper[vai])
                m = XGBClassifier(**p)
                m.fit(Xt, y[tri], verbose=False)
                pr = m.predict_proba(Xv)[:, 1]
                scores.append(0.5 * average_precision_score(y[vai], pr) + 0.5 * roc_auc_score(y[vai], pr))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(xgb_obj, n_trials=100, show_progress_bar=False)
    print(f"  Best value: {study.best_value:.6f}")
    print(f"  Best params: {study.best_params}")

    best_xgb_params = {
        'n_estimators': study.best_params['n'], 'max_depth': study.best_params['md'],
        'learning_rate': study.best_params['lr'], 'subsample': study.best_params['ss'],
        'colsample_bytree': study.best_params['cs'], 'scale_pos_weight': study.best_params['spw'],
        'reg_alpha': study.best_params['ra'], 'reg_lambda': study.best_params['rl'],
        'min_child_weight': study.best_params['mcw'],
        'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False,
    }

    # Evaluate with 70/30 split
    print("\nOptuna XGBoost (70/30 split):")
    X_train, X_test, y_train, y_test = train_test_split(X_paper, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)
    best_xgb = XGBClassifier(**best_xgb_params)
    best_xgb.fit(X_train_s, y_train, verbose=False)
    y_proba = best_xgb.predict_proba(X_test_s)[:, 1]
    res = evaluate_all(y_test, y_proba, "paper_13_contBMI | XGB_optuna | 70/30")
    res['feature_set'] = 'paper_13_contBMI'; res['model'] = 'XGB_optuna'; res['split'] = '70/30'
    res['n_features'] = X_paper.shape[1]
    all_results.append(res)

    # Evaluate with CV
    print("\nOptuna XGBoost (5-fold CV × 3 seeds):")
    cv_rows = []
    for seed in [42, 123, 999]:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (tri, tei) in enumerate(skf.split(X_paper, y)):
            Xtr, Xte = X_paper[tri], X_paper[tei]; ytr, yte = y[tri], y[tei]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
            m = XGBClassifier(**best_xgb_params)
            m.fit(Xtr_s, ytr, verbose=False)
            proba = m.predict_proba(Xte_s)[:, 1]
            for tn, tv in [('0.5', 0.5), ('f1_optimal', find_thresh_f1(yte, proba))]:
                pred = (proba >= tv).astype(int)
                cv_rows.append({
                    'roc_auc': roc_auc_score(yte, proba), 'pr_auc': average_precision_score(yte, proba),
                    'f1_macro': f1_score(yte, pred, average='macro'), 'f1_binary': f1_score(yte, pred),
                    'accuracy': accuracy_score(yte, pred), 'threshold_method': tn,
                })
    cv_df = pd.DataFrame(cv_rows)
    for th in ['0.5', 'f1_optimal']:
        sub = cv_df[cv_df['threshold_method'] == th]
        res = {
            'feature_set': 'paper_13_contBMI', 'model': 'XGB_optuna', 'split': '5fold_CV',
            'n_features': X_paper.shape[1], 'threshold_method': th,
            'roc_auc': sub['roc_auc'].mean(), 'pr_auc': sub['pr_auc'].mean(),
            'f1_macro': sub['f1_macro'].mean(), 'f1_binary': sub['f1_binary'].mean(),
            'accuracy': sub['accuracy'].mean(),
            'roc_auc_std': sub['roc_auc'].std(), 'f1_macro_std': sub['f1_macro'].std(),
        }
        all_results.append(res)
        print(f"  paper_13_contBMI | XGB_optuna | CV | {th}: "
              f"AUROC={sub['roc_auc'].mean():.4f}+/-{sub['roc_auc'].std():.4f} "
              f"PR-AUC={sub['pr_auc'].mean():.4f} F1-Macro={sub['f1_macro'].mean():.4f}")

    # =====================================================================
    # RESULTS SUMMARY
    # =====================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS - PAPER REPLICATION")
    print("=" * 80)

    results_df = pd.DataFrame(all_results).sort_values('roc_auc', ascending=False)
    results_df.to_csv(os.path.join(STEP_DIR, 'step10_results.csv'), index=False)

    print("\nAll results sorted by AUROC:")
    for _, r in results_df.head(25).iterrows():
        fs = r.get('feature_set', '?')
        model = r.get('model', '?')
        split = r.get('split', '?')
        auc = r.get('roc_auc', 0)
        prauc = r.get('pr_auc', 0)
        f1m = r.get('f1_macro', 0)
        f1b = r.get('f1_binary', 0)
        print(f"  {fs:<22s} {model:<18s} {split:<8s} AUC={auc:.4f} PRAUC={prauc:.4f} F1M={f1m:.4f} F1={f1b:.4f}")

    paper_auc = 0.9078
    best_auc = results_df['roc_auc'].max()
    print(f"\nPaper AUROC: {paper_auc:.4f}")
    print(f"Our best AUROC: {best_auc:.4f}")
    print(f"Gap: {paper_auc - best_auc:.4f}")

    best = results_df.iloc[0]
    print(f"\nBest config: {best.get('feature_set', '?')} | {best.get('model', '?')} | {best.get('split', '?')}")

    report = [
        "=" * 80,
        "STEP 10: PAPER REPLICATION - EXACT FEATURE SPACE",
        "=" * 80,
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Duration: {datetime.now() - t0}",
        "",
        "PAPER'S METHODOLOGY:",
        "  - LASSO-selected 11 features (BA + 10 sociodemographic/disease)",
        "  - One-hot encoding for categorical variables",
        "  - Z-score standardization",
        "  - XGBoost, 70/30 stratified split",
        "  - Paper AUROC: 0.9078",
        "",
        "FEATURE SETS TESTED:",
        f"  paper_11: {paper_11}",
        f"  paper_13: {paper_13}",
        f"  paper_13_contBMI: {paper_13_cont_bmi}",
        "",
        f"OPTUNA BEST PARAMS:",
        f"  {study.best_params}",
        "",
        "TOP 25 RESULTS:",
    ]
    for _, r in results_df.head(25).iterrows():
        report.append(f"  {r.get('feature_set', '?'):<22s} {r.get('model', '?'):<18s} {r.get('split', '?'):<8s} "
                      f"AUC={r.get('roc_auc', 0):.4f} PRAUC={r.get('pr_auc', 0):.4f} "
                      f"F1M={r.get('f1_macro', 0):.4f} F1={r.get('f1_binary', 0):.4f}")
    report.extend(["",
                    "COMPARISON:",
                    f"  Paper AUROC: {paper_auc:.4f}",
                    f"  Our best AUROC: {best_auc:.4f}",
                    f"  Gap: {paper_auc - best_auc:.4f}",
                    f"  Previous best (Step 09): 0.8967",
                    f"  Improvement over previous: {best_auc - 0.8967:+.4f}",
                    ])

    with open(os.path.join(STEP_DIR, 'step10_report.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

    print(f"\nSaved to {STEP_DIR}/")
    print(f"Total time: {datetime.now() - t0}")


if __name__ == '__main__':
    main()