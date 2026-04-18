"""
STEP 08: XGBoost Fix & Encoding Improvements
==============================================
1. Threshold optimization (Youden Index) + Probability calibration
2. One-Hot Encoding for categorical variables
3. PCA on biomarkers (dimensionality reduction)
4. Force BA into LASSO feature selection
5. Re-evaluate all models with improvements
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

BASE_DIR = r'C:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data'
STEP_DIR = os.path.join(BASE_DIR, 'step_08_xgboost_fix')

def youden_threshold(y_true, y_proba):
    best_th = 0.5
    best_youden = 0
    for th in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_proba >= th).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        youden = sensitivity + specificity - 1
        if youden > best_youden:
            best_youden = youden
            best_th = th
    return best_th, best_youden

def evaluate_model_cv(model, X, y, name, seeds=[42, 123, 999], n_folds=5, 
                      use_calibration=False, use_youden=False):
    results = []
    all_y_true = []
    all_y_proba = []
    
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            model_clone = type(model)(**model.get_params())
            
            if use_calibration:
                model_clone = CalibratedClassifierCV(model_clone, method='isotonic', cv=3)
            
            try:
                if isinstance(model_clone, CatBoostClassifier):
                    if not use_calibration:
                        model_clone.fit(X_train_s, y_train, verbose=0)
                elif isinstance(model_clone, LGBMClassifier):
                    model_clone.fit(X_train_s, y_train)
                else:
                    model_clone.fit(X_train_s, y_train)
                
                y_proba = model_clone.predict_proba(X_test_s)[:, 1]
            except Exception as e:
                continue
            
            auc = roc_auc_score(y_test, y_proba)
            
            if use_youden:
                th, youden = youden_threshold(y_test, y_proba)
            else:
                th = 0.5
            
            y_pred = (y_proba >= th).astype(int)
            f1 = f1_score(y_test, y_pred)
            acc = accuracy_score(y_test, y_pred)
            
            results.append({
                'auc': auc, 'f1': f1, 'acc': acc, 'threshold': th
            })
            all_y_true.extend(y_test)
            all_y_proba.extend(y_proba)
    
    if len(results) == 0:
        return None
    
    df_results = pd.DataFrame(results)
    summary = {
        'name': name,
        'auc_mean': df_results['auc'].mean(),
        'auc_std': df_results['auc'].std(),
        'f1_mean': df_results['f1'].mean(),
        'f1_std': df_results['f1'].std(),
        'acc_mean': df_results['acc'].mean(),
        'threshold_mean': df_results['threshold'].mean(),
    }
    return summary

def main():
    print("=" * 80)
    print("STEP 08: XGBOOST FIX & ENCODING IMPROVEMENTS")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not os.path.exists(STEP_DIR):
        os.makedirs(STEP_DIR)
    
    df = pd.read_csv(os.path.join(BASE_DIR, 'step_01_data_prep', 'dataset_A_all.csv'))
    print(f"Dataset: {df.shape}, KOA rate={df['KOA'].mean()*100:.1f}%\n")
    
    raw17 = ['wave', 'Time', 'Gender', 'Age_New', 'Marital', 'Education',
              'Residence', 'Hypertension', 'Dyslipidemia', 'Diabetes',
              'Cancer', 'CVD', 'Smoke', 'Drink', 'BMI', 'BMI_New',
              'Biological Age']
    biomarkers_log = ['plt_10.9.L', 'crp_mg.L', 'Hb A1c', 'creatinine_mg.d L',
                       'bun_mg.d L', 'TC_mg.d L', 'TG_mg.d L', 'sbp.mean']
    
    y = df['KOA'].values
    spw = (y == 0).sum() / (y == 1).sum()
    print(f"scale_pos_weight = {spw:.2f}")
    
    # ========================================================================
    # 1. ONE-HOT ENCODING
    # ========================================================================
    print("=" * 80)
    print("1. ONE-HOT ENCODING (Paper's approach)")
    print("=" * 80)
    
    cat_cols = ['Gender', 'Age_New', 'Marital', 'Education', 'Residence', 'BMI_New']
    num_cols = [c for c in raw17 if c not in cat_cols]
    
    # One-hot encode
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_cat = encoder.fit_transform(df[cat_cols])
    ohe_feature_names = encoder.get_feature_names_out(cat_cols).tolist()
    
    X_num = df[num_cols].values
    X_ohe = np.hstack([X_num, X_cat])
    ohe_feature_names_all = num_cols + ohe_feature_names
    print(f"Features: {len(num_cols)} numeric + {len(ohe_feature_names)} one-hot = {len(ohe_feature_names_all)} total")
    print(f"One-hot features: {ohe_feature_names}")
    print()
    
    # ========================================================================
    # 2. PCA ON BIOMARKERS
    # ========================================================================
    print("=" * 80)
    print("2. PCA ON BIOMARKERS")
    print("=" * 80)
    
    bm_data = df[biomarkers_log].values
    scaler_bm = StandardScaler()
    bm_scaled = scaler_bm.fit_transform(bm_data)
    
    pca = PCA(n_components=0.80)
    bm_pca = pca.fit_transform(bm_scaled)
    print(f"PCA components explaining 80% variance: {bm_pca.shape[1]}")
    print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {pca.explained_variance_ratio_.cumsum()}")
    
    pca_feature_names = [f'PC{i+1}' for i in range(bm_pca.shape[1])]
    
    # Create feature sets
    feature_sets = {}
    
    # A: raw17 (integer encoding) - baseline
    feature_sets['A_raw17_int'] = (df[raw17].values, raw17)
    
    # B: raw17 (one-hot encoding)
    feature_sets['B_raw17_ohe'] = (X_ohe, ohe_feature_names_all)
    
    # C: raw17 + PCA biomarkers (integer encoding)
    X_int_pca = np.hstack([df[raw17].values, bm_pca])
    feature_sets['C_raw17_int_PCA'] = (X_int_pca, raw17 + pca_feature_names)
    
    # D: raw17 + PCA biomarkers (one-hot encoding)
    X_ohe_pca = np.hstack([X_ohe, bm_pca])
    feature_sets['D_raw17_ohe_PCA'] = (X_ohe_pca, ohe_feature_names_all + pca_feature_names)
    
    # E: raw17 + BA_KDM (one-hot) 
    df_kdm = pd.read_csv(os.path.join(BASE_DIR, 'step_03_kdm_ba', 'dataset_A_with_kdm_ba.csv'))
    X_ohe_kdm = np.hstack([X_ohe, df_kdm['BA_KDM_log'].values.reshape(-1, 1)])
    feature_sets['E_ohe_KDM'] = (X_ohe_kdm, ohe_feature_names_all + ['BA_KDM_log'])
    
    # F: raw17 + PCA + KDM (one-hot)
    X_ohe_pca_kdm = np.hstack([X_ohe, bm_pca, df_kdm['BA_KDM_log'].values.reshape(-1, 1)])
    feature_sets['F_ohe_PCA_KDM'] = (X_ohe_pca_kdm, ohe_feature_names_all + pca_feature_names + ['BA_KDM_log'])
    
    # ========================================================================
    # 3. EVALUATE ALL CONFIGURATIONS
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. MODEL EVALUATION")
    print("=" * 80)
    
    models = {
        'XGBoost_spw': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      subsample=0.8, colsample_bytree=0.7, 
                                      scale_pos_weight=spw, random_state=42,
                                      eval_metric='logloss'),
        'XGBoost_default': XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.7,
                                          random_state=42, eval_metric='logloss'),
        'LightGBM_spw': LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                                        num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                                        scale_pos_weight=spw, random_state=42, verbose=-1),
        'RF': RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1),
    }
    
    all_results = []
    
    for fs_name, (X_fs, feat_names) in feature_sets.items():
        print(f"\n--- Feature Set: {fs_name} ({X_fs.shape[1]} features) ---")
        
        for model_name, model in models.items():
            config_name = f"{fs_name}|{model_name}"
            
            # Default threshold
            result = evaluate_model_cv(model, X_fs, y, config_name, 
                                       use_calibration=False, use_youden=False)
            if result:
                result['feature_set'] = fs_name
                result['model'] = model_name
                result['n_features'] = X_fs.shape[1]
                result['calibration'] = 'none'
                result['threshold'] = 'default'
                all_results.append(result)
                print(f"  {model_name:<20s}: AUC={result['auc_mean']:.4f} F1={result['f1_mean']:.4f} (th=0.5)")
            
            # Youden threshold
            result_y = evaluate_model_cv(model, X_fs, y, config_name + '_youden',
                                        use_calibration=False, use_youden=True)
            if result_y:
                result_y['feature_set'] = fs_name
                result_y['model'] = model_name
                result_y['n_features'] = X_fs.shape[1]
                result_y['calibration'] = 'none'
                result_y['threshold'] = f"youden({result_y['threshold_mean']:.2f})"
                all_results.append(result_y)
                print(f"  {model_name:<20s}: AUC={result_y['auc_mean']:.4f} F1={result_y['f1_mean']:.4f} (th={result_y['threshold_mean']:.2f})")
    
    # ========================================================================
    # 4. CALIBRATION ON BEST CONFIGS
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. PROBABILITY CALIBRATION (Isotonic Regression)")
    print("=" * 80)
    
    best_configs = [
        ('A_raw17_int', 'XGBoost_spw'),
        ('B_raw17_ohe', 'XGBoost_spw'),
        ('D_raw17_ohe_PCA', 'XGBoost_spw'),
        ('B_raw17_ohe', 'RF'),
    ]
    
    for fs_name, model_name in best_configs:
        if fs_name not in feature_sets:
            continue
        X_fs, _ = feature_sets[fs_name]
        
        if model_name == 'XGBoost_spw':
            model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                   subsample=0.8, colsample_bytree=0.7,
                                   scale_pos_weight=spw, random_state=42,
                                   eval_metric='logloss')
        elif model_name == 'RF':
            model = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        else:
            continue
        
        result_cal = evaluate_model_cv(model, X_fs, y, f"{fs_name}|{model_name}_cal",
                                       use_calibration=True, use_youden=True)
        if result_cal:
            result_cal['feature_set'] = fs_name
            result_cal['model'] = model_name + '_calibrated'
            result_cal['n_features'] = X_fs.shape[1]
            result_cal['calibration'] = 'isotonic'
            result_cal['threshold'] = f"youden({result_cal['threshold_mean']:.2f})"
            all_results.append(result_cal)
            print(f"  {fs_name} + {model_name}_cal: AUC={result_cal['auc_mean']:.4f} F1={result_cal['f1_mean']:.4f}")
    
    # ========================================================================
    # 5. FORCED BA LASSO FEATURE SELECTION
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. FORCED BA LASSO FEATURE SELECTION (one-hot encoded)")
    print("=" * 80)
    
    X_all = np.hstack([X_ohe, bm_pca, df_kdm['BA_KDM_log'].values.reshape(-1, 1)])
    all_feat_names = ohe_feature_names_all + pca_feature_names + ['BA_KDM_log']
    
    scaler_all = StandardScaler()
    X_all_scaled = scaler_all.fit_transform(X_all)
    
    # LASSO with different C values
    for C in [0.01, 0.05, 0.1, 0.5, 1.0]:
        lasso = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42, max_iter=2000)
        lasso.fit(X_all_scaled, y)
        
        selected = [(name, coef) for name, coef in zip(all_feat_names, lasso.coef_[0]) if abs(coef) > 0]
        selected_sorted = sorted(selected, key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n  LASSO C={C}: {len(selected)}/{len(all_feat_names)} features selected")
        for name, coef in selected_sorted[:10]:
            print(f"    {name:<30s}: {coef:+.4f}")
        
        # Evaluate with top features + forced BA
        if len(selected) >= 5:
            selected_features = [s[0] for s in selected_sorted]
            X_lasso = np.hstack([df_kdm['BA_KDM_log'].values.reshape(-1, 1),
                                 df[[c for c in selected_features if c in df.columns]].values])
            n_lasso_features = 1 + len([c for c in selected_features if c in df.columns])
            
            xgb_lasso = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                                      scale_pos_weight=spw, random_state=42, eval_metric='logloss')
            result = evaluate_model_cv(xgb_lasso, X_lasso, y, f'LASSO_C{C}_forcedBA_XGB',
                                       use_youden=True)
            if result:
                result['feature_set'] = f'lasso_C{C}_forced_BA'
                result['model'] = 'XGBoost_spw_youden'
                result['n_features'] = X_lasso.shape[1]
                all_results.append(result)
                print(f"\n  LASSO C={C} + forced BA + XGBoost: AUC={result['auc_mean']:.4f} F1={result['f1_mean']:.4f}")
    
    # ========================================================================
    # 6. SAVE RESULTS & REPORT
    # ========================================================================
    results_df = pd.DataFrame(all_results).sort_values('auc_mean', ascending=False)
    results_df.to_csv(os.path.join(STEP_DIR, 'step08_results.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("TOP 20 CONFIGURATIONS")
    print("=" * 80)
    for _, row in results_df.head(20).iterrows():
        print(f"  {row['feature_set']:<25s} | {row['model']:<25s} | AUC={row['auc_mean']:.4f} | F1={row['f1_mean']:.4f} | {row.get('threshold', 'N/A')}")
    
    # Compare with previous best
    print("\n" + "=" * 80)
    print("COMPARISON WITH PREVIOUS BEST")
    print("=" * 80)
    print(f"  Previous best: RF raw17 (int encoding, th=0.5): AUROC=0.89, F1=0.76")
    print(f"  Best from this step: see above")
    print(f"  Paper reference: XGBoost AUROC=0.9078")
    
    report = []
    report.append("=" * 80)
    report.append("STEP 08: XGBOOST FIX & ENCODING IMPROVEMENTS - REPORT")
    report.append("=" * 80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("IMPROVEMENTS APPLIED:")
    report.append("1. One-Hot Encoding (Gender, Age_New, Marital, Education, Residence, BMI_New)")
    report.append("2. PCA on biomarkers (8 -> 1-2 components)")
    report.append("3. XGBoost scale_pos_weight + Youden threshold")
    report.append("4. Probability calibration (Isotonic Regression)")
    report.append("5. Forced BA LASSO feature selection")
    report.append("")
    report.append("TOP 20 CONFIGURATIONS:")
    for _, row in results_df.head(20).iterrows():
        report.append(f"  {row['feature_set']:<25s} | {row['model']:<25s} | AUC={row['auc_mean']:.4f} | F1={row['f1_mean']:.4f}")
    report.append("")
    report.append("KEY FINDINGS:")
    best_int = results_df[results_df['feature_set'] == 'A_raw17_int']
    best_ohe = results_df[results_df['feature_set'].str.contains('ohe')]
    if len(best_int) > 0 and len(best_ohe) > 0:
        report.append(f"  Integer encoding best: AUC={best_int.iloc[0]['auc_mean']:.4f}")
        report.append(f"  One-hot encoding best: AUC={best_ohe.iloc[0]['auc_mean']:.4f}")
    
    report_text = "\n".join(report)
    with open(os.path.join(STEP_DIR, 'step08_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nResults saved to {os.path.join(STEP_DIR, 'step08_results.csv')}")
    print(f"Report saved to {os.path.join(STEP_DIR, 'step08_report.txt')}")

if __name__ == '__main__':
    main()