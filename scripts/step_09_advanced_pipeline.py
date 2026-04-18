"""
STEP 09: Advanced Pipeline - Closing the Gap (LEAN VERSION)
=============================================================
Focused: Optuna on raw17 + stacking. Minimal scope to finish fast.
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
from sklearn.preprocessing import StandardScaler
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

def quick_eval(model_fn, X, y, seeds=[42, 123, 999], n_folds=5):
    rows = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tri, tei) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tri], X[tei]
            ytr, yte = y[tri], y[tei]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            m = model_fn()
            m.fit(Xtr_s, ytr)
            proba = m.predict_proba(Xte_s)[:, 1]
            auc = roc_auc_score(yte, proba)
            prauc = average_precision_score(yte, proba)
            for tn, tv in [('d_0.5', 0.5), ('youden', find_thresh_youden(yte, proba)), ('f1opt', find_thresh_f1(yte, proba))]:
                pred = (proba >= tv).astype(int)
                rows.append({'seed': seed, 'fold': fold, 'th': tn,
                             'roc_auc': auc, 'pr_auc': prauc,
                             'f1_macro': f1_score(yte, pred, average='macro'),
                             'f1_binary': f1_score(yte, pred),
                             'accuracy': accuracy_score(yte, pred),
                             'precision': precision_score(yte, pred, zero_division=0),
                             'recall': recall_score(yte, pred)})
    df = pd.DataFrame(rows)
    out = {}
    for th in df['th'].unique():
        s = df[df['th'] == th]
        out[th] = {c: s[c].mean() for c in ['roc_auc','pr_auc','f1_macro','f1_binary','accuracy','precision','recall']}
        out[th]['roc_auc_std'] = s['roc_auc'].std()
        out[th]['f1_macro_std'] = s['f1_macro'].std()
        out[th]['pr_auc_std'] = s['pr_auc'].std()
    return out

def stacking_eval(X, y, rf_p, xgb_p, seeds=[42, 123, 999], n_folds=5):
    rows = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tri, tei) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tri], X[tei]
            ytr, yte = y[tri], y[tei]
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            rf = RandomForestClassifier(**rf_p)
            xgb = XGBClassifier(**xgb_p)
            rf.fit(Xtr_s, ytr); xgb.fit(Xtr_s, ytr)
            meta_tr = np.hstack([rf.predict_proba(Xtr_s)[:,1:2], xgb.predict_proba(Xtr_s)[:,1:2]])
            meta_te = np.hstack([rf.predict_proba(Xte_s)[:,1:2], xgb.predict_proba(Xte_s)[:,1:2]])
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(meta_tr, ytr)
            proba = lr.predict_proba(meta_te)[:, 1]
            auc = roc_auc_score(yte, proba)
            prauc = average_precision_score(yte, proba)
            for tn, tv in [('d_0.5', 0.5), ('youden', find_thresh_youden(yte, proba)), ('f1opt', find_thresh_f1(yte, proba))]:
                pred = (proba >= tv).astype(int)
                rows.append({'seed': seed, 'fold': fold, 'th': tn,
                             'roc_auc': auc, 'pr_auc': prauc,
                             'f1_macro': f1_score(yte, pred, average='macro'),
                             'f1_binary': f1_score(yte, pred),
                             'accuracy': accuracy_score(yte, pred),
                             'precision': precision_score(yte, pred, zero_division=0),
                             'recall': recall_score(yte, pred)})
    df = pd.DataFrame(rows)
    out = {}
    for th in df['th'].unique():
        s = df[df['th'] == th]
        out[th] = {c: s[c].mean() for c in ['roc_auc','pr_auc','f1_macro','f1_binary','accuracy','precision','recall']}
        out[th]['roc_auc_std'] = s['roc_auc'].std()
        out[th]['f1_macro_std'] = s['f1_macro'].std()
    return out

def main():
    t0 = datetime.now()
    print("=" * 80)
    print("STEP 09: ADVANCED PIPELINE - CLOSING THE GAP")
    print(f"Start: {t0.strftime('%H:%M:%S')}")
    print("=" * 80)

    df = pd.read_csv(os.path.join(BASE_DIR, 'step_01_data_prep', 'dataset_A_all.csv'))
    df_kdm = pd.read_csv(os.path.join(BASE_DIR, 'step_03_kdm_ba', 'dataset_A_with_kdm_ba.csv'))
    y = df['KOA'].values
    spw = (y == 0).sum() / (y == 1).sum()
    print(f"Dataset: {len(df)}, KOA={y.mean()*100:.1f}%, spw={spw:.2f}")

    raw17 = ['wave', 'Time', 'Gender', 'Age_New', 'Marital', 'Education',
              'Residence', 'Hypertension', 'Dyslipidemia', 'Diabetes',
              'Cancer', 'CVD', 'Smoke', 'Drink', 'BMI', 'BMI_New', 'Biological Age']
    X_raw17 = df[raw17].values

    from sklearn.preprocessing import OneHotEncoder
    cat_cols = ['Gender', 'Age_New', 'Marital', 'Education', 'Residence', 'BMI_New']
    num_cols = [c for c in raw17 if c not in cat_cols]
    enc = OneHotEncoder(sparse_output=False, drop='first')
    X_cat = enc.fit_transform(df[cat_cols])
    X_ohe = np.hstack([df[num_cols].values, X_cat])

    df_kdm['delta_BA'] = df_kdm['BA_KDM_orig'] - df_kdm['Biological Age']
    df_kdm['CVD_x_KDM_BA'] = df_kdm['CVD'] * df_kdm['BA_KDM_orig']
    df_kdm['CVD_x_delta_BA'] = df_kdm['CVD'] * df_kdm['delta_BA']
    interact = ['BA_KDM_orig', 'delta_BA', 'CVD_x_KDM_BA', 'CVD_x_delta_BA']
    X_ohe_int = np.hstack([X_ohe, df_kdm[interact].values])

    all_results = []

    # =======================================================================
    # 1. BASELINE: RF + XGBoost (+spw) + XGBoost + LightGBM (+spw)
    # =======================================================================
    print("\n--- PHASE 1: BASELINE ---")
    models = {
        'RF': lambda: RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
        'XGB_spw': lambda: XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.7,
                                           scale_pos_weight=spw, random_state=42,
                                           eval_metric='logloss', use_label_encoder=False),
        'XGB': lambda: XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.7,
                                      random_state=42, eval_metric='logloss', use_label_encoder=False),
        'LGBM_spw': lambda: LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                                              num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                                              scale_pos_weight=spw, random_state=42, verbose=-1),
    }

    for fs_name, X_fs in [('raw17', X_raw17), ('raw17_ohe', X_ohe), ('raw17_ohe_interact', X_ohe_int)]:
        for mname, mfn in models.items():
            print(f"  {fs_name} + {mname}...", end='', flush=True)
            res = quick_eval(mfn, X_fs, y)
            for th, metrics in res.items():
                all_results.append({'phase': '1_baseline', 'feature_set': fs_name, 'model': mname,
                                    'threshold': th, 'n_features': X_fs.shape[1], **metrics})
            print(f" AUROC={res['d_0.5']['roc_auc']:.4f} PR-AUC={res['d_0.5']['pr_auc']:.4f} "
                  f"F1-Macro={res['d_0.5']['f1_macro']:.4f} | F1={res['f1opt']['f1_macro']:.4f}@f1 "
                  f"AUROC={res['youden']['roc_auc']:.4f}@youden")

    # =======================================================================
    # 2. OPTUNA on best feature set
    # =======================================================================
    print("\n--- PHASE 2: OPTUNA ---")
    # Determine best feature set from baseline
    best_fs = max(all_results, key=lambda r: r['roc_auc'])['feature_set']
    X_best = {'raw17': X_raw17, 'raw17_ohe': X_ohe, 'raw17_ohe_interact': X_ohe_int}[best_fs]
    print(f"Best feature set from baseline: {best_fs} ({X_best.shape[1]} features)")

    # XGBoost Optuna
    print("  Optuna XGBoost (80 trials)...", flush=True)
    def xgb_obj(trial):
        p = {'n_estimators': trial.suggest_int('n', 200, 600),
             'max_depth': trial.suggest_int('md', 3, 7),
             'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
             'subsample': trial.suggest_float('ss', 0.6, 1.0),
             'colsample_bytree': trial.suggest_float('cs', 0.5, 1.0),
             'scale_pos_weight': trial.suggest_float('spw', 1, spw),
             'reg_alpha': trial.suggest_float('ra', 1e-8, 10, log=True),
             'reg_lambda': trial.suggest_float('rl', 1e-8, 10, log=True),
             'min_child_weight': trial.suggest_int('mcw', 1, 10),
             'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False}
        scores = []
        for seed in [42]:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            for tri, vai in skf.split(X_best, y):
                sc = StandardScaler()
                Xt = sc.fit_transform(X_best[tri]); Xv = sc.transform(X_best[vai])
                m = XGBClassifier(**p); m.fit(Xt, y[tri], verbose=False)
                pr = m.predict_proba(Xv)[:, 1]
                scores.append(0.5 * average_precision_score(y[vai], pr) + 0.5 * roc_auc_score(y[vai], pr))
        return np.mean(scores)
    sx = optuna.create_study(direction='maximize')
    sx.optimize(xgb_obj, n_trials=80, show_progress_bar=False)
    xgb_best = {**sx.best_params, 'random_state': 42, 'eval_metric': 'logloss', 'use_label_encoder': False}
    xgb_best['n_estimators'] = sx.best_params['n']
    xgb_best['max_depth'] = sx.best_params['md']
    xgb_best['learning_rate'] = sx.best_params['lr']
    xgb_best['subsample'] = sx.best_params['ss']
    xgb_best['colsample_bytree'] = sx.best_params['cs']
    xgb_best['scale_pos_weight'] = sx.best_params['spw']
    xgb_best['reg_alpha'] = sx.best_params['ra']
    xgb_best['reg_lambda'] = sx.best_params['rl']
    xgb_best['min_child_weight'] = sx.best_params['mcw']
    # Clean up param names
    xgb_clean = {k: v for k, v in xgb_best.items() if k not in ['n', 'md', 'lr', 'ss', 'cs', 'spw_val', 'ra', 'rl', 'mcw']}
    print(f"  XGBoost best: {sx.best_value:.6f}, params: {sx.best_params}")

    res_xgb = quick_eval(lambda: XGBClassifier(**xgb_clean), X_best, y)
    for th, metrics in res_xgb.items():
        all_results.append({'phase': '2_optuna', 'feature_set': best_fs, 'model': 'XGB_optuna',
                            'threshold': th, 'n_features': X_best.shape[1], **metrics})
    print(f"  XGB_optuna: AUROC={res_xgb['d_0.5']['roc_auc']:.4f} PR-AUC={res_xgb['d_0.5']['pr_auc']:.4f} "
          f"F1-Macro={res_xgb['f1opt']['f1_macro']:.4f}@f1")

    # LightGBM Optuna
    print("  Optuna LightGBM (80 trials)...", flush=True)
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
        for seed in [42]:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            for tri, vai in skf.split(X_best, y):
                sc = StandardScaler()
                Xt = sc.fit_transform(X_best[tri]); Xv = sc.transform(X_best[vai])
                m = LGBMClassifier(**p); m.fit(Xt, y[tri])
                pr = m.predict_proba(Xv)[:, 1]
                scores.append(0.5 * average_precision_score(y[vai], pr) + 0.5 * roc_auc_score(y[vai], pr))
        return np.mean(scores)
    sl = optuna.create_study(direction='maximize')
    sl.optimize(lgbm_obj, n_trials=80, show_progress_bar=False)
    lgbm_best = {**sl.best_params, 'random_state': 42, 'verbose': -1}
    lgbm_best['n_estimators'] = sl.best_params['n']
    lgbm_best['max_depth'] = sl.best_params['md']
    lgbm_best['num_leaves'] = sl.best_params['nl']
    lgbm_best['learning_rate'] = sl.best_params['lr']
    lgbm_best['subsample'] = sl.best_params['ss']
    lgbm_best['colsample_bytree'] = sl.best_params['cs']
    lgbm_best['scale_pos_weight'] = sl.best_params['spw']
    lgbm_best['reg_alpha'] = sl.best_params['ra']
    lgbm_best['reg_lambda'] = sl.best_params['rl']
    lgbm_best['min_child_samples'] = sl.best_params['mcs']
    lgbm_best['class_weight'] = sl.best_params['cw']
    lgbm_clean = {k: v for k, v in lgbm_best.items() if k not in ['n', 'md', 'nl', 'lr', 'ss', 'cs', 'spw_val', 'ra', 'rl', 'mcs', 'cw']}
    print(f"  LightGBM best: {sl.best_value:.6f}, params: {sl.best_params}")

    res_lgbm = quick_eval(lambda: LGBMClassifier(**lgbm_clean), X_best, y)
    for th, metrics in res_lgbm.items():
        all_results.append({'phase': '2_optuna', 'feature_set': best_fs, 'model': 'LGBM_optuna',
                            'threshold': th, 'n_features': X_best.shape[1], **metrics})
    print(f"  LGBM_optuna: AUROC={res_lgbm['d_0.5']['roc_auc']:.4f} PR-AUC={res_lgbm['d_0.5']['pr_auc']:.4f} "
          f"F1-Macro={res_lgbm['f1opt']['f1_macro']:.4f}@f1")

    # =======================================================================
    # 3. STACKING
    # =======================================================================
    print("\n--- PHASE 3: STACKING ---")
    rf_p = {'n_estimators': 500, 'random_state': 42, 'n_jobs': -1}
    stack_res = stacking_eval(X_best, y, rf_p, xgb_clean)
    for th, metrics in stack_res.items():
        all_results.append({'phase': '3_stacking', 'feature_set': best_fs, 'model': 'Stack_RF+XGB_LR',
                            'threshold': th, 'n_features': X_best.shape[1], **metrics})
    print(f"  Stacking: AUROC={stack_res['d_0.5']['roc_auc']:.4f} PR-AUC={stack_res['d_0.5']['pr_auc']:.4f} "
          f"F1-Macro={stack_res['f1opt']['f1_macro']:.4f}@f1")

    # Stacking with LGBM too
    def stacking_3model(X, y, rf_p, xgb_p, lgbm_p, seeds=[42, 123, 999], n_folds=5):
        rows = []
        for seed in seeds:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            for fold, (tri, tei) in enumerate(skf.split(X, y)):
                Xtr, Xte = X[tri], X[tei]; ytr, yte = y[tri], y[tei]
                sc = StandardScaler()
                Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
                rf = RandomForestClassifier(**rf_p); rf.fit(Xtr_s, ytr)
                xgb = XGBClassifier(**xgb_p); xgb.fit(Xtr_s, ytr)
                lgbm = LGBMClassifier(**lgbm_p); lgbm.fit(Xtr_s, ytr)
                meta_tr = np.hstack([rf.predict_proba(Xtr_s)[:,1:2], xgb.predict_proba(Xtr_s)[:,1:2], lgbm.predict_proba(Xtr_s)[:,1:2]])
                meta_te = np.hstack([rf.predict_proba(Xte_s)[:,1:2], xgb.predict_proba(Xte_s)[:,1:2], lgbm.predict_proba(Xte_s)[:,1:2]])
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

    stack3_res = stacking_3model(X_best, y, rf_p, xgb_clean, lgbm_clean)
    for th, metrics in stack3_res.items():
        all_results.append({'phase': '3_stacking', 'feature_set': best_fs, 'model': 'Stack_RF+XGB+LGBM_LR',
                            'threshold': th, 'n_features': X_best.shape[1], **metrics})
    print(f"  Stacking3: AUROC={stack3_res['d_0.5']['roc_auc']:.4f} PR-AUC={stack3_res['d_0.5']['pr_auc']:.4f} "
          f"F1-Macro={stack3_res['f1opt']['f1_macro']:.4f}@f1")

    # =======================================================================
    # RESULTS
    # =======================================================================
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
    print(f"Best config: {best['phase']} | {best['feature_set']} | {best['model']} | {best['threshold']}")

    report = ["=" * 80, "STEP 09: ADVANCED PIPELINE RESULTS", "=" * 80,
              f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"Duration: {datetime.now()-t0}", "",
              "OPTUNA BEST PARAMS:",
              f"  XGBoost: {sx.best_params}", f"  LightGBM: {sl.best_params}", "",
              "TOP 30:", ""]
    for _, r in results_df.head(30).iterrows():
        report.append(f"  {r['phase']:<12s} {r['feature_set']:<25s} {r['model']:<22s} {r['threshold']:<8s} "
                      f"AUC={r['roc_auc']:.4f} PRAUC={r['pr_auc']:.4f} F1M={r['f1_macro']:.4f}")
    report += ["", "COMPARISON:",
               f"  Previous best: {prev:.4f}", f"  This step best: {best['roc_auc']:.4f} ({best['roc_auc']-prev:+.4f})",
               f"  Paper: {paper:.4f}", f"  Gap: {paper-best['roc_auc']:.4f}"]
    with open(os.path.join(STEP_DIR, 'step09_report.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print(f"\nSaved to {STEP_DIR}")
    print(f"Total: {datetime.now()-t0}")

if __name__ == '__main__':
    main()