"""
STEP 09: Final Push - Optuna LightGBM + Stacking Ensemble
==========================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
    average_precision_score, precision_recall_curve, roc_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = r'C:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data'
STEP_DIR = os.path.join(BASE_DIR, 'step_09_advanced')
os.makedirs(STEP_DIR, exist_ok=True)

def find_thresh_youden(y, p):
    fpr, tpr, th = roc_curve(y, p)
    return th[np.argmax(tpr - fpr)]

def find_thresh_f1(y, p):
    prec, rec, th = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    return th[np.argmax(f1[:-1])]

def eval_model(model_fn, X, y, seeds=[42, 123, 999], n_folds=5):
    rows = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tri, tei) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
            m = model_fn(); m.fit(Xtr_s, ytr)
            proba = m.predict_proba(Xte_s)[:, 1]
            auc = roc_auc_score(yte, proba); prauc = average_precision_score(yte, proba)
            for tn, tv in [('d_0.5', 0.5), ('youden', find_thresh_youden(yte, proba)), ('f1opt', find_thresh_f1(yte, proba))]:
                pred = (proba >= tv).astype(int)
                rows.append({'seed': seed, 'fold': fold, 'th': tn, 'roc_auc': auc, 'pr_auc': prauc,
                             'f1_macro': f1_score(yte, pred, average='macro'), 'f1_binary': f1_score(yte, pred),
                             'accuracy': accuracy_score(yte, pred), 'precision': precision_score(yte, pred, zero_division=0),
                             'recall': recall_score(yte, pred)})
    df = pd.DataFrame(rows)
    out = {}
    for th in df['th'].unique():
        s = df[df['th'] == th]
        out[th] = {c: s[c].mean() for c in ['roc_auc','pr_auc','f1_macro','f1_binary','accuracy','precision','recall']}
        out[th]['roc_auc_std'] = s['roc_auc'].std(); out[th]['f1_macro_std'] = s['f1_macro'].std()
    return out

def stacking_eval(X, y, model_fns, seeds=[42, 123, 999], n_folds=5):
    rows = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tri, tei) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
            probas_tr = []; probas_te = []
            for name, mfn in model_fns:
                m = mfn(); m.fit(Xtr_s, ytr)
                probas_tr.append(m.predict_proba(Xtr_s)[:, 1:2])
                probas_te.append(m.predict_proba(Xte_s)[:, 1:2])
            meta_tr = np.hstack(probas_tr); meta_te = np.hstack(probas_te)
            lr = LogisticRegression(max_iter=1000, random_state=42); lr.fit(meta_tr, ytr)
            proba = lr.predict_proba(meta_te)[:, 1]
            auc = roc_auc_score(yte, proba); prauc = average_precision_score(yte, proba)
            for tn, tv in [('d_0.5', 0.5), ('youden', find_thresh_youden(yte, proba)), ('f1opt', find_thresh_f1(yte, proba))]:
                pred = (proba >= tv).astype(int)
                rows.append({'seed': seed, 'fold': fold, 'th': tn, 'roc_auc': auc, 'pr_auc': prauc,
                             'f1_macro': f1_score(yte, pred, average='macro'), 'f1_binary': f1_score(yte, pred),
                             'accuracy': accuracy_score(yte, pred), 'precision': precision_score(yte, pred, zero_division=0),
                             'recall': recall_score(yte, pred)})
    df = pd.DataFrame(rows)
    out = {}
    for th in df['th'].unique():
        s = df[df['th'] == th]
        out[th] = {c: s[c].mean() for c in ['roc_auc','pr_auc','f1_macro','f1_binary','accuracy','precision','recall']}
        out[th]['roc_auc_std'] = s['roc_auc'].std(); out[th]['f1_macro_std'] = s['f1_macro'].std()
    return out

def main():
    t0 = datetime.now()
    print("=" * 80)
    print("STEP 09: FINAL PUSH - OPTUNA LGBM + STACKING")
    print(f"Start: {t0.strftime('%H:%M:%S')}")
    print("=" * 80)

    df = pd.read_csv(os.path.join(BASE_DIR, 'step_01_data_prep', 'dataset_A_all.csv'))
    df_kdm = pd.read_csv(os.path.join(BASE_DIR, 'step_03_kdm_ba', 'dataset_A_with_kdm_ba.csv'))
    y = df['KOA'].values; spw = (y == 0).sum() / (y == 1).sum()
    raw17 = ['wave', 'Time', 'Gender', 'Age_New', 'Marital', 'Education',
              'Residence', 'Hypertension', 'Dyslipidemia', 'Diabetes',
              'Cancer', 'CVD', 'Smoke', 'Drink', 'BMI', 'BMI_New', 'Biological Age']
    X = df[raw17].values

    # Load previous results if they exist
    prev_results_path = os.path.join(STEP_DIR, 'step09_results.csv')
    all_results = []
    if os.path.exists(prev_results_path):
        all_results = pd.read_csv(prev_results_path).to_dict('records')
        print(f"Loaded {len(all_results)} previous results")

    print(f"Dataset: {len(df)}, KOA={y.mean()*100:.1f}%, spw={spw:.2f}")

    # =====================================================================
    # Optuna LightGBM on raw17 (50 trials, 3-fold, 1 seed)
    # =====================================================================
    print("\n--- Optuna LightGBM (50 trials, 3-fold) ---")
    def lgbm_obj(trial):
        p = {'n_estimators': trial.suggest_int('n', 200, 600),
             'max_depth': trial.suggest_int('md', 3, 12),
             'num_leaves': trial.suggest_int('nl', 15, 63),
             'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
             'subsample': trial.suggest_float('ss', 0.6, 1.0),
             'colsample_bytree': trial.suggest_float('cs', 0.5, 1.0),
             'scale_pos_weight': trial.suggest_float('spw', 1, spw),
             'reg_alpha': trial.suggest_float('ra', 1e-8, 10, log=True),
             'reg_lambda': trial.suggest_float('rl', 1e-8, 10, log=True),
             'min_child_samples': trial.suggest_int('mcs', 5, 50),
             'class_weight': trial.suggest_categorical('cw', [None, 'balanced']),
             'random_state': 42, 'verbose': -1}
        scores = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        for tri, vai in skf.split(X, y):
            sc = StandardScaler()
            Xt = sc.fit_transform(X[tri]); Xv = sc.transform(X[vai])
            m = LGBMClassifier(**p); m.fit(Xt, y[tri])
            pr = m.predict_proba(Xv)[:, 1]
            scores.append(0.5 * average_precision_score(y[vai], pr) + 0.5 * roc_auc_score(y[vai], pr))
        return np.mean(scores)

    sl = optuna.create_study(direction='maximize')
    sl.optimize(lgbm_obj, n_trials=50, show_progress_bar=False)
    print(f"  Best value: {sl.best_value:.6f}")
    print(f"  Best params: {sl.best_params}")

    lgbm_params = {
        'n_estimators': sl.best_params['n'], 'max_depth': sl.best_params['md'],
        'num_leaves': sl.best_params['nl'], 'learning_rate': sl.best_params['lr'],
        'subsample': sl.best_params['ss'], 'colsample_bytree': sl.best_params['cs'],
        'scale_pos_weight': sl.best_params['spw'], 'reg_alpha': sl.best_params['ra'],
        'reg_lambda': sl.best_params['rl'], 'min_child_samples': sl.best_params['mcs'],
        'class_weight': sl.best_params['cw'], 'random_state': 42, 'verbose': -1,
    }

    print("\n  Evaluating optimized LightGBM...")
    lgbm_res = eval_model(lambda: LGBMClassifier(**lgbm_params), X, y)
    for th, metrics in lgbm_res.items():
        all_results.append({'phase': '2_optuna', 'feature_set': 'raw17', 'model': 'LGBM_optuna',
                            'threshold': th, 'n_features': 17, **metrics})
    print(f"  LGBM_optuna: AUROC={lgbm_res['d_0.5']['roc_auc']:.4f} PR-AUC={lgbm_res['d_0.5']['pr_auc']:.4f} "
          f"F1-Macro={lgbm_res['d_0.5']['f1_macro']:.4f}@0.5 | F1-Macro={lgbm_res['f1opt']['f1_macro']:.4f}@f1")

    # =====================================================================
    # Optuna XGBoost results (from previous run)
    # =====================================================================
    xgb_params = {
        'n_estimators': 504, 'max_depth': 7, 'learning_rate': 0.2423516592308228,
        'subsample': 0.780557784485733, 'colsample_bytree': 0.6498853913935047,
        'scale_pos_weight': 5.863002181650144, 'reg_alpha': 0.03546538020511568,
        'reg_lambda': 3.2948720321207057e-07, 'min_child_weight': 1,
        'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False,
    }

    # =====================================================================
    # STACKING ENSEMBLES
    # =====================================================================
    rf_params = {'n_estimators': 500, 'random_state': 42, 'n_jobs': -1}

    print("\n--- Stacking: RF + XGBoost_optuna -> LR ---")
    stack2_res = stacking_eval(X, y, [('RF', lambda: RandomForestClassifier(**rf_params)),
                                         ('XGB_opt', lambda: XGBClassifier(**xgb_params))])
    for th, metrics in stack2_res.items():
        all_results.append({'phase': '3_stacking', 'feature_set': 'raw17', 'model': 'Stack_RF+XGB_LR',
                            'threshold': th, 'n_features': 17, **metrics})
    print(f"  AUROC={stack2_res['d_0.5']['roc_auc']:.4f} PR-AUC={stack2_res['d_0.5']['pr_auc']:.4f} "
          f"F1-Macro={stack2_res['d_0.5']['f1_macro']:.4f}@0.5 | F1-Macro={stack2_res['f1opt']['f1_macro']:.4f}@f1")

    print("\n--- Stacking: RF + XGBoost_opt + LGBM_opt -> LR ---")
    stack3_res = stacking_eval(X, y, [('RF', lambda: RandomForestClassifier(**rf_params)),
                                          ('XGB_opt', lambda: XGBClassifier(**xgb_params)),
                                          ('LGBM_opt', lambda: LGBMClassifier(**lgbm_params))])
    for th, metrics in stack3_res.items():
        all_results.append({'phase': '3_stacking', 'feature_set': 'raw17', 'model': 'Stack_RF+XGB+LGBM_LR',
                            'threshold': th, 'n_features': 17, **metrics})
    print(f"  AUROC={stack3_res['d_0.5']['roc_auc']:.4f} PR-AUC={stack3_res['d_0.5']['pr_auc']:.4f} "
          f"F1-Macro={stack3_res['d_0.5']['f1_macro']:.4f}@0.5 | F1-Macro={stack3_res['f1opt']['f1_macro']:.4f}@f1")

    # Also try stacking with balanced class weight for RF
    rf_balanced = {'n_estimators': 500, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
    print("\n--- Stacking: RF_balanced + XGBoost_opt -> LR ---")
    stack_b_res = stacking_eval(X, y, [('RF_bal', lambda: RandomForestClassifier(**rf_balanced)),
                                            ('XGB_opt', lambda: XGBClassifier(**xgb_params))])
    for th, metrics in stack_b_res.items():
        all_results.append({'phase': '3_stacking', 'feature_set': 'raw17', 'model': 'Stack_RFbal+XGB_LR',
                            'threshold': th, 'n_features': 17, **metrics})
    print(f"  AUROC={stack_b_res['d_0.5']['roc_auc']:.4f} PR-AUC={stack_b_res['d_0.5']['pr_auc']:.4f} "
          f"F1-Macro={stack_b_res['d_0.5']['f1_macro']:.4f}@0.5 | F1-Macro={stack_b_res['f1opt']['f1_macro']:.4f}@f1")

    # =====================================================================
    # RESULTS
    # =====================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    results_df = pd.DataFrame(all_results).sort_values('roc_auc', ascending=False)
    results_df.to_csv(os.path.join(STEP_DIR, 'step09_results.csv'), index=False)

    cols = ['phase', 'feature_set', 'model', 'threshold', 'n_features', 'roc_auc', 'roc_auc_std',
            'pr_auc', 'f1_macro', 'f1_binary', 'accuracy', 'precision', 'recall']
    print("\nTop 20 by AUROC:")
    print(results_df[cols].head(20).to_string(index=False))

    best = results_df.iloc[0]
    prev = 0.8967; paper = 0.9078
    print(f"\nPrevious best: {prev:.4f} | Paper: {paper:.4f}")
    print(f"This step best: AUC={best['roc_auc']:.4f} ({best['roc_auc']-prev:+.4f}), gap={paper-best['roc_auc']:.4f}")
    print(f"Config: {best['phase']} | {best['feature_set']} | {best['model']} | {best['threshold']}")

    report = ["=" * 80, "STEP 09: FINAL PUSH RESULTS", "=" * 80,
              f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"Duration: {datetime.now()-t0}", ""]
    report.extend(["OPTUNA LGBM PARAMS:", f"  {sl.best_params}", ""])
    for _, r in results_df.head(30).iterrows():
        report.append(f"  {r['phase']:<12s} {r['feature_set']:<25s} {r['model']:<22s} {r['threshold']:<8s} "
                      f"AUC={r['roc_auc']:.4f} PRAUC={r['pr_auc']:.4f} F1M={r['f1_macro']:.4f}")
    report += ["", f"COMPARISON:", f"  Previous: {prev:.4f}", f"  This step: {best['roc_auc']:.4f}",
               f"  Paper: {paper:.4f}", f"  Gap: {paper-best['roc_auc']:.4f}"]
    with open(os.path.join(STEP_DIR, 'step09_report.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print(f"\nResults saved. Total time: {datetime.now()-t0}")

if __name__ == '__main__':
    main()