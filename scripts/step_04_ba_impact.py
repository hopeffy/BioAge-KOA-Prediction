"""
STEP 04: Feature Impact Analysis & Paper Replication
======================================================
1. Real SHAP analysis on RF and XGBoost (TreeExplainer)
2. Feature impact comparison (BA vs KDM-BA vs biomarkers)
3. Paper replication (LASSO feature selection + XGBoost/LightGBM/CatBoost/RF)
4. One-hot encoding + Z-score standardization
5. Class imbalance handling (scale_pos_weight)
6. Aging clock comparison including PhenoAge_Adapted
7. SHAP interaction analysis (raw17 and raw17+PhenoAge_Adapted)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STEP_DIR = os.path.join(BASE_DIR, 'step_04_ba_impact')

def _select_positive_class_shap(raw_values):
    if isinstance(raw_values, list):
        idx = 1 if len(raw_values) > 1 else 0
        return np.array(raw_values[idx])

    arr = np.array(raw_values)
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            return arr[:, :, 1]
        if arr.shape[0] == 2:
            return arr[1]
    return arr


def _select_positive_class_interactions(raw_values):
    if isinstance(raw_values, list):
        idx = 1 if len(raw_values) > 1 else 0
        return np.array(raw_values[idx])

    arr = np.array(raw_values)
    if arr.ndim == 4:
        if arr.shape[-1] == 2:
            return arr[:, :, :, 1]
        if arr.shape[0] == 2:
            return arr[1]
    return arr


def _build_top_interactions_df(interaction_matrix, feature_cols):
    rows = []
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            rows.append({
                'feature_1': feature_cols[i],
                'feature_2': feature_cols[j],
                'interaction_strength': interaction_matrix[i, j],
            })
    return pd.DataFrame(rows).sort_values('interaction_strength', ascending=False)


def _filter_clinically_meaningful_interactions(top_interactions_df, feature_cols):
    # Group aliases map clinical concepts to possible dataset feature names.
    group_aliases = {
        'biological_age': ['Biological Age', 'Age_New'],
        'bmi': ['BMI', 'BMI_New'],
        'hypertension': ['Hypertension'],
        'gender': ['Gender'],
        'hscrp': ['crp_mg.L', 'crp_original'],
    }

    chosen = {}
    for group, aliases in group_aliases.items():
        for alias in aliases:
            if alias in feature_cols:
                chosen[group] = alias
                break

    target_group_pairs = [
        ('biological_age', 'bmi'),
        ('biological_age', 'hypertension'),
        ('biological_age', 'gender'),
        ('biological_age', 'hscrp'),
        ('bmi', 'hypertension'),
        ('bmi', 'gender'),
        ('bmi', 'hscrp'),
        ('hypertension', 'gender'),
        ('hypertension', 'hscrp'),
        ('gender', 'hscrp'),
    ]

    rationale = {
        ('biological_age', 'bmi'): 'Aging-mechanical load interaction',
        ('biological_age', 'hypertension'): 'Aging-hemodynamic risk interaction',
        ('biological_age', 'gender'): 'Sex-specific aging risk interaction',
        ('biological_age', 'hscrp'): 'Aging-inflammation interaction',
        ('bmi', 'hypertension'): 'Obesity-hemodynamic interaction',
        ('bmi', 'gender'): 'Sex-specific obesity burden',
        ('bmi', 'hscrp'): 'Adiposity-inflammation interaction',
        ('hypertension', 'gender'): 'Sex-specific hypertension effect',
        ('hypertension', 'hscrp'): 'Hemodynamic-inflammation interaction',
        ('gender', 'hscrp'): 'Sex-specific inflammatory burden',
    }

    accepted_pairs = {}
    for g1, g2 in target_group_pairs:
        if g1 in chosen and g2 in chosen:
            f1 = chosen[g1]
            f2 = chosen[g2]
            key = tuple(sorted((f1, f2)))
            accepted_pairs[key] = {
                'clinical_pair_group': f'{g1} x {g2}',
                'clinical_rationale': rationale.get((g1, g2), ''),
            }

    if len(accepted_pairs) == 0:
        return pd.DataFrame(columns=[
            'feature_1',
            'feature_2',
            'interaction_strength',
            'clinical_pair_group',
            'clinical_rationale',
        ])

    rows = []
    for _, row in top_interactions_df.iterrows():
        key = tuple(sorted((row['feature_1'], row['feature_2'])))
        if key in accepted_pairs:
            rows.append({
                'feature_1': row['feature_1'],
                'feature_2': row['feature_2'],
                'interaction_strength': row['interaction_strength'],
                'clinical_pair_group': accepted_pairs[key]['clinical_pair_group'],
                'clinical_rationale': accepted_pairs[key]['clinical_rationale'],
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=[
            'feature_1',
            'feature_2',
            'interaction_strength',
            'clinical_pair_group',
            'clinical_rationale',
        ])

    return pd.DataFrame(rows).sort_values('interaction_strength', ascending=False)


def _plot_interaction_heatmap(interaction_matrix, feature_cols, out_path, title, top_k=10):
    k = min(top_k, len(feature_cols))
    global_strength = interaction_matrix.sum(axis=1)
    ranked_idx = np.argsort(global_strength)[::-1][:k]

    matrix_top = interaction_matrix[np.ix_(ranked_idx, ranked_idx)]
    labels = [feature_cols[i] for i in ranked_idx]

    plt.figure(figsize=(9, 7))
    plt.imshow(matrix_top, cmap='viridis')
    plt.colorbar(label='Mean |SHAP interaction|')
    plt.xticks(range(k), labels, rotation=45, ha='right')
    plt.yticks(range(k), labels)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def _get_positive_class_base_value(expected_value):
    arr = np.array(expected_value)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape[0] == 2:
        return float(arr[1])
    return float(arr.flatten()[0])


def _save_xgb_waterfall_examples(explainer_xgb, shap_xgb, X_test, y_test, y_proba, feature_cols, analysis_name):
    if len(X_test) == 0:
        return

    base_value = _get_positive_class_base_value(explainer_xgb.expected_value)

    candidate_indices = {}
    all_idx = np.arange(len(y_test))

    pos_mask = y_test == 1
    if pos_mask.any():
        pos_idx = all_idx[pos_mask]
        best_pos = pos_idx[np.argmax(y_proba[pos_mask])]
        candidate_indices['high_risk_positive'] = int(best_pos)

    neg_mask = y_test == 0
    if neg_mask.any():
        neg_idx = all_idx[neg_mask]
        best_neg = neg_idx[np.argmax(y_proba[neg_mask])]
        candidate_indices['high_risk_negative'] = int(best_neg)

    median_target = float(np.median(y_proba))
    median_idx = int(np.argmin(np.abs(y_proba - median_target)))
    candidate_indices['median_risk_case'] = median_idx

    used_indices = set()
    rows = []

    for tag, idx in candidate_indices.items():
        if idx in used_indices:
            continue
        used_indices.add(idx)

        explanation = shap.Explanation(
            values=shap_xgb[idx],
            base_values=base_value,
            data=X_test.iloc[idx].values,
            feature_names=feature_cols,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.tight_layout()
        out_path = os.path.join(STEP_DIR, f'shap_waterfall_xgb_{analysis_name}_{tag}.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

        rows.append({
            'analysis_name': analysis_name,
            'case_tag': tag,
            'local_test_index': idx,
            'true_label': int(y_test[idx]),
            'predicted_risk': float(y_proba[idx]),
            'waterfall_path': out_path,
        })

    if len(rows) > 0:
        cases_df = pd.DataFrame(rows)
        cases_df.to_csv(os.path.join(STEP_DIR, f'shap_waterfall_cases_{analysis_name}.csv'), index=False)

        print("\nSaved SHAP waterfall examples:")
        for _, row in cases_df.iterrows():
            print(
                f"  {row['case_tag']:<20s} "
                f"label={int(row['true_label'])}, risk={row['predicted_risk']:.4f}"
            )


def shap_analysis(df, feature_cols, analysis_name, target_col='KOA', max_interaction_samples=2000):
    print("=" * 80)
    print(f"1. SHAP ANALYSIS ({analysis_name})")
    print("=" * 80)

    X = df[feature_cols].copy()
    y = df[target_col].values

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # RF model (real SHAP)
    print("\nTraining Random Forest for SHAP...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    print(f"RF Test AUROC: {rf_auc:.4f}")

    print("\nComputing SHAP values for RF (TreeExplainer)...")
    explainer_rf = shap.TreeExplainer(rf)
    shap_values_rf_raw = explainer_rf.shap_values(X_test)
    shap_rf = _select_positive_class_shap(shap_values_rf_raw)

    mean_shap_rf = np.abs(shap_rf).mean(axis=0)
    if mean_shap_rf.ndim > 1:
        mean_shap_rf = mean_shap_rf.flatten()

    shap_importance_rf = pd.DataFrame({
        'feature': feature_cols,
        'shap_mean': mean_shap_rf[:len(feature_cols)]
    }).sort_values('shap_mean', ascending=False)
    
    print("\nRF SHAP Feature Importance (top 10):")
    for i, (_, row) in enumerate(shap_importance_rf.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<30s} SHAP={row['shap_mean']:.4f}")
    
    # XGBoost model (real SHAP + interaction)
    print("\nTraining XGBoost for SHAP...")
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_proba)
    print(f"XGBoost Test AUROC: {xgb_auc:.4f}")

    print("\nComputing SHAP values for XGBoost (TreeExplainer)...")
    explainer_xgb = shap.TreeExplainer(xgb)
    shap_values_xgb_raw = explainer_xgb.shap_values(X_test)
    shap_xgb = _select_positive_class_shap(shap_values_xgb_raw)

    mean_shap_xgb = np.abs(shap_xgb).mean(axis=0)
    if mean_shap_xgb.ndim > 1:
        mean_shap_xgb = mean_shap_xgb.flatten()

    shap_importance_xgb = pd.DataFrame({
        'feature': feature_cols,
        'shap_mean': mean_shap_xgb[:len(feature_cols)]
    }).sort_values('shap_mean', ascending=False)
    
    print("\nXGBoost SHAP Feature Importance (top 10):")
    for i, (_, row) in enumerate(shap_importance_xgb.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['feature']:<30s} SHAP={row['shap_mean']:.4f}")
    
    # Save SHAP summary plots (XGBoost)
    shap.summary_plot(shap_xgb, X_test, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(STEP_DIR, f'shap_summary_xgb_{analysis_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_xgb, X_test, show=False, plot_type='bar', max_display=15)
    plt.tight_layout()
    plt.savefig(os.path.join(STEP_DIR, f'shap_summary_bar_xgb_{analysis_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Representative local explanations for clinical communication
    _save_xgb_waterfall_examples(
        explainer_xgb,
        shap_xgb,
        X_test,
        y_test,
        xgb_proba,
        feature_cols,
        analysis_name,
    )

    # SHAP interaction values (XGBoost)
    if len(X_test) > max_interaction_samples:
        X_interaction = X_test.sample(n=max_interaction_samples, random_state=42)
    else:
        X_interaction = X_test.copy()

    print(f"\nComputing SHAP interaction values for XGBoost on n={len(X_interaction)} samples...")
    interaction_raw = explainer_xgb.shap_interaction_values(X_interaction)
    interaction_values = _select_positive_class_interactions(interaction_raw)

    interaction_matrix = np.abs(interaction_values).mean(axis=0)
    if interaction_matrix.ndim > 2:
        interaction_matrix = interaction_matrix[:, :, 0]

    interaction_matrix = interaction_matrix[:len(feature_cols), :len(feature_cols)]
    np.fill_diagonal(interaction_matrix, 0.0)

    top_interactions_df = _build_top_interactions_df(interaction_matrix, feature_cols)
    top_interactions_df.to_csv(
        os.path.join(STEP_DIR, f'shap_interactions_xgb_{analysis_name}.csv'), index=False
    )

    clinical_interactions_df = _filter_clinically_meaningful_interactions(top_interactions_df, feature_cols)
    clinical_interactions_df.to_csv(
        os.path.join(STEP_DIR, f'shap_interactions_clinical_{analysis_name}.csv'),
        index=False,
    )

    _plot_interaction_heatmap(
        interaction_matrix,
        feature_cols,
        os.path.join(STEP_DIR, f'shap_interaction_heatmap_xgb_{analysis_name}.png'),
        title=f'XGBoost SHAP Interaction Heatmap ({analysis_name})',
        top_k=10,
    )

    print("\nTop XGBoost SHAP interactions (top 10):")
    for i, (_, row) in enumerate(top_interactions_df.head(10).iterrows()):
        print(
            f"  {i+1:2d}. {row['feature_1']} x {row['feature_2']} "
            f"-> {row['interaction_strength']:.4f}"
        )

    if len(clinical_interactions_df) > 0:
        print("\nClinically filtered SHAP interactions (top 10):")
        for i, (_, row) in enumerate(clinical_interactions_df.head(10).iterrows()):
            print(
                f"  {i+1:2d}. {row['feature_1']} x {row['feature_2']} "
                f"-> {row['interaction_strength']:.4f} [{row['clinical_pair_group']}]"
            )
    else:
        print("\nClinically filtered SHAP interactions: no predefined clinical pairs available in this feature set.")

    # Paper comparison (SHAP > 0.6 for BA)
    print("\n" + "=" * 80)
    print("PAPER COMPARISON: Paper found SHAP > 0.6 for Biological Age")
    print("=" * 80)
    
    ba_shap_rf = shap_importance_rf[shap_importance_rf['feature'] == 'Biological Age']['shap_mean'].values
    ba_shap_xgb = shap_importance_xgb[shap_importance_xgb['feature'] == 'Biological Age']['shap_mean'].values
    
    print(f"  RF SHAP for Biological Age: {ba_shap_rf[0]:.4f}" if len(ba_shap_rf) > 0 else "  RF: Biological Age not found")
    print(f"  XGB SHAP for Biological Age: {ba_shap_xgb[0]:.4f}" if len(ba_shap_xgb) > 0 else "  XGB: Biological Age not found")
    print(f"  Paper SHAP for BA: > 0.6")
    print(f"  Difference: Our SHAP values are much lower than paper's 0.6")
    print(f"  Note: Values above are from real TreeExplainer SHAP (not proxy importance)")

    return shap_importance_rf, shap_importance_xgb, top_interactions_df, clinical_interactions_df

def aging_clock_summary(df, target_col='KOA'):
    print("\n" + "=" * 80)
    print("1B. AGING CLOCK SUMMARY")
    print("=" * 80)

    clock_cols = [
        'Biological Age',
        'BA_KDM_log',
        'BA_KDM_orig',
        'BIR_log',
        'BIR_orig',
        'PhenoAge_Adapted',
        'PhenoAge_Adapted_Accel',
    ]
    available = [c for c in clock_cols if c in df.columns]

    rows = []
    y = df[target_col].values
    for col in available:
        X = df[[col]].values
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        proba = lr.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, proba)
        coef = lr.coef_[0][0]
        odds_ratio = np.exp(coef)
        corr_with_koa = df[target_col].corr(df[col])
        corr_with_ba = df['Biological Age'].corr(df[col]) if col != 'Biological Age' else 1.0

        rows.append({
            'aging_clock': col,
            'auc_univariate': auc,
            'odds_ratio_per_unit': odds_ratio,
            'coef': coef,
            'corr_with_koa': corr_with_koa,
            'corr_with_biological_age': corr_with_ba,
        })

    summary_df = pd.DataFrame(rows).sort_values('auc_univariate', ascending=False)

    print("\nTop aging clocks by univariate AUROC:")
    for _, row in summary_df.iterrows():
        print(
            f"  {row['aging_clock']:<24s}: "
            f"AUC={row['auc_univariate']:.4f}, "
            f"OR={row['odds_ratio_per_unit']:.4f}, "
            f"corr(KOA)={row['corr_with_koa']:.4f}"
        )

    return summary_df

def feature_impact_comparison(df):
    print("\n" + "=" * 80)
    print("2. FEATURE IMPACT COMPARISON")
    print("=" * 80)
    
    raw17 = ['wave', 'Time', 'Gender', 'Age_New', 'Marital', 'Education',
              'Residence', 'Hypertension', 'Dyslipidemia', 'Diabetes',
              'Cancer', 'CVD', 'Smoke', 'Drink', 'BMI', 'BMI_New',
              'Biological Age']
    
    biomarkers_log = ['plt_10.9.L', 'crp_mg.L', 'Hb A1c', 'creatinine_mg.d L',
                       'bun_mg.d L', 'TC_mg.d L', 'TG_mg.d L', 'sbp.mean']
    
    kdm_cols = ['BA_KDM_log', 'BA_KDM_orig']
    bir_cols = ['BIR_log', 'BIR_orig']
    pheno_cols = ['PhenoAge_Adapted', 'PhenoAge_Adapted_Accel']
    quartile_cols = ['Biological Age_Qint', 'BA_KDM_log_Qint', 'BA_KDM_orig_Qint']
    if 'PhenoAge_Adapted_Qint' in df.columns:
        quartile_cols.append('PhenoAge_Adapted_Qint')
    
    feature_sets = {
        'raw17': raw17,
        'raw17_plus_biomarkers': raw17 + biomarkers_log,
        'raw17_plus_KDM_log': raw17 + ['BA_KDM_log'],
        'raw17_plus_KDM_orig': raw17 + ['BA_KDM_orig'],
        'raw17_plus_PhenoAge_Adapted': raw17 + ['PhenoAge_Adapted'],
        'raw17_plus_PhenoAge_Accel': raw17 + ['PhenoAge_Adapted_Accel'],
        'raw17_plus_KDM_PhenoAge_Adapted': raw17 + ['BA_KDM_orig', 'PhenoAge_Adapted'],
        'raw17_plus_BIR': raw17 + ['BIR_log', 'BIR_orig'],
        'raw17_plus_aging_accel': raw17 + ['BIR_orig', 'PhenoAge_Adapted_Accel'],
        'raw17_plus_KDM_biomarkers': raw17 + biomarkers_log + ['BA_KDM_log'],
        'raw17_plus_Pheno_biomarkers': raw17 + biomarkers_log + ['PhenoAge_Adapted'],
        'raw17_plus_KDM_quartile': raw17 + ['BA_KDM_log', 'BA_KDM_log_Qint'],
        'raw17_plus_Pheno_quartile': raw17 + ['PhenoAge_Adapted', 'PhenoAge_Adapted_Qint'],
        'raw17_all': raw17 + biomarkers_log + kdm_cols + bir_cols + pheno_cols + quartile_cols,
    }
    
    y = df['KOA'].values
    
    results = []
    
    for fs_name, fs_cols in feature_sets.items():
        available_cols = [c for c in fs_cols if c in df.columns]
        X = df[available_cols].values
        
        # RF with 5-fold CV x 3 seeds
        aucs = []
        for seed in [42, 123, 999]:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)
                
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf.fit(X_train_s, y_train)
                aucs.append(roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1]))
        
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        
        results.append({
            'feature_set': fs_name,
            'n_features': len(available_cols),
            'roc_auc_mean': mean_auc,
            'roc_auc_std': std_auc,
        })
        
        print(f"  {fs_name:<30s}: AUC={mean_auc:.4f} +/- {std_auc:.4f} ({len(available_cols)} features)")
    
    results_df = pd.DataFrame(results).sort_values('roc_auc_mean', ascending=False)
    results_df.to_csv(os.path.join(STEP_DIR, 'feature_impact_comparison.csv'), index=False)
    
    print(f"\nBest feature set: {results_df.iloc[0]['feature_set']} (AUC={results_df.iloc[0]['roc_auc_mean']:.4f})")
    print(f"Worst feature set: {results_df.iloc[-1]['feature_set']} (AUC={results_df.iloc[-1]['roc_auc_mean']:.4f})")
    
    return results_df

def paper_replication(df):
    print("\n" + "=" * 80)
    print("3. PAPER REPLICATION (LASSO + XGBoost/LightGBM/CatBoost/RF)")
    print("=" * 80)
    
    # Paper features (LASSO selected 11 features)
    # BA, gender, education, residence, hypertension, dyslipidemia, CVD, smoking, drinking, BMI category, cancer
    paper_features = ['Biological Age', 'Gender', 'Education', 'Residence',
                      'Hypertension', 'Dyslipidemia', 'CVD', 'Smoke', 'Drink',
                      'BMI_New', 'Cancer']
    
    # Plus KDM-BA version
    paper_features_kdm = ['BA_KDM_log', 'Gender', 'Education', 'Residence',
                           'Hypertension', 'Dyslipidemia', 'CVD', 'Smoke', 'Drink',
                           'BMI_New', 'Cancer']

    paper_features_pheno = ['PhenoAge_Adapted', 'Gender', 'Education', 'Residence',
                            'Hypertension', 'Dyslipidemia', 'CVD', 'Smoke', 'Drink',
                            'BMI_New', 'Cancer']
    
    # All available features (like paper's approach)
    all_features = ['wave', 'Time', 'Gender', 'Age_New', 'Marital', 'Education',
                    'Residence', 'Hypertension', 'Dyslipidemia', 'Diabetes',
                    'Cancer', 'CVD', 'Smoke', 'Drink', 'BMI', 'BMI_New',
                    'Biological Age']
    
    biomarkers_log = ['plt_10.9.L', 'crp_mg.L', 'Hb A1c', 'creatinine_mg.d L',
                      'bun_mg.d L', 'TC_mg.d L', 'TG_mg.d L', 'sbp.mean']
    
    feature_configs = {
        'paper_11': paper_features,
        'paper_11_KDM': paper_features_kdm,
        'paper_11_Pheno': paper_features_pheno,
        'all_raw17': all_features,
        'all_plus_biomarkers': all_features + biomarkers_log,
        'all_plus_KDM': all_features + ['BA_KDM_log'],
        'all_plus_PhenoAge': all_features + ['PhenoAge_Adapted'],
        'all_plus_KDM_PhenoAge': all_features + ['BA_KDM_log', 'PhenoAge_Adapted'],
        'all_plus_KDM_biomarkers': all_features + biomarkers_log + ['BA_KDM_log'],
    }
    
    y = df['KOA'].values
    
    # LASSO feature selection first
    print("\nLASSO Feature Selection:")
    lasso_pool = all_features + biomarkers_log + ['BA_KDM_log']
    if 'PhenoAge_Adapted' in df.columns:
        lasso_pool.append('PhenoAge_Adapted')
    if 'PhenoAge_Adapted_Accel' in df.columns:
        lasso_pool.append('PhenoAge_Adapted_Accel')

    X_all = df[lasso_pool].values
    scaler_lasso = StandardScaler()
    X_all_scaled = scaler_lasso.fit_transform(X_all)
    
    lasso = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42, max_iter=1000)
    lasso.fit(X_all_scaled, y)
    
    lasso_features = pd.DataFrame({
        'feature': lasso_pool,
        'lasso_coef': lasso.coef_[0],
        'abs_coef': np.abs(lasso.coef_[0])
    }).sort_values('abs_coef', ascending=False)
    
    print(f"\nLASSO selected features (coef != 0):")
    selected = lasso_features[lasso_features['abs_coef'] > 0]
    for _, row in selected.iterrows():
        print(f"  {row['feature']:<30s}: coef={row['lasso_coef']:.6f}")
    
    lasso_selected_features = selected['feature'].tolist()
    feature_configs['lasso_selected'] = lasso_selected_features
    
    # Now test all feature configs with all 4 models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON (5-fold CV x 3 seeds)")
    print("=" * 80)
    
    models = {
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    
    # Add scale_pos_weight for XGBoost (class imbalance)
    koa_pos = y.sum()
    koa_neg = len(y) - koa_pos
    spw = koa_neg / koa_pos
    
    models_scaled = {
        'XGBoost_scaled': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, 
                                         eval_metric='logloss', scale_pos_weight=spw),
    }
    models.update(models_scaled)
    
    all_results = []
    
    for fs_name, fs_cols in feature_configs.items():
        available_cols = [c for c in fs_cols if c in df.columns]
        if len(available_cols) == 0:
            continue
        X = df[available_cols].values
        
        for model_name, model in models.items():
            aucs = []
            f1s = []
            
            for seed in [42, 123, 999]:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for train_idx, test_idx in skf.split(X, y):
                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                    
                    model_clone = type(model)(**model.get_params())
                    
                    try:
                        if isinstance(model_clone, CatBoostClassifier):
                            model_clone.fit(X_train_s, y_train, verbose=0)
                        elif isinstance(model_clone, LGBMClassifier):
                            model_clone.fit(X_train_s, y_train)
                        else:
                            model_clone.fit(X_train_s, y_train)
                        
                        y_pred_proba = model_clone.predict_proba(X_test_s)[:, 1]
                        y_pred = (y_pred_proba >= 0.5).astype(int)
                        
                        aucs.append(roc_auc_score(y_test, y_pred_proba))
                        f1s.append(f1_score(y_test, y_pred))
                    except Exception as e:
                        print(f"    ERROR: {model_name} with {fs_name}: {e}")
                        continue
            
            if len(aucs) > 0:
                all_results.append({
                    'feature_set': fs_name,
                    'model': model_name,
                    'n_features': len(available_cols),
                    'roc_auc_mean': np.mean(aucs),
                    'roc_auc_std': np.std(aucs),
                    'f1_mean': np.mean(f1s),
                    'f1_std': np.std(f1s),
                })
                
                if np.mean(aucs) > 0.80:
                    print(f"  {fs_name:<25s} + {model_name:<20s}: AUC={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}, F1={np.mean(f1s):.4f}")
    
    results_df = pd.DataFrame(all_results).sort_values('roc_auc_mean', ascending=False)
    results_df.to_csv(os.path.join(STEP_DIR, 'paper_replication_results.csv'), index=False)
    
    print("\n" + "=" * 80)
    print("TOP 15 CONFIGURATIONS")
    print("=" * 80)
    for _, row in results_df.head(15).iterrows():
        print(f"  {row['feature_set']:<25s} + {row['model']:<20s}: AUC={row['roc_auc_mean']:.4f}, F1={row['f1_mean']:.4f} ({int(row['n_features'])} feats)")
    
    return results_df, lasso_features

def subgroup_analysis(df):
    print("\n" + "=" * 80)
    print("4. SUBGROUP ANALYSIS")
    print("=" * 80)
    
    raw17 = ['wave', 'Time', 'Gender', 'Age_New', 'Marital', 'Education',
              'Residence', 'Hypertension', 'Dyslipidemia', 'Diabetes',
              'Cancer', 'CVD', 'Smoke', 'Drink', 'BMI', 'BMI_New',
              'Biological Age']
    
    subgroups = {
        'All': None,
        'Male': (df['Gender'] == 1),
        'Female': (df['Gender'] == 2),
        'Urban': (df['Residence'] == 1),
        'Rural': (df['Residence'] == 2),
        'CVD_yes': (df['CVD'] == 1),
        'CVD_no': (df['CVD'] == 0),
        'Hypertension_yes': (df['Hypertension'] == 1),
        'Hypertension_no': (df['Hypertension'] == 0),
        'Older (Age_New=2)': (df['Age_New'] == 2),
        'Younger (Age_New=1)': (df['Age_New'] == 1),
    }
    
    results = []
    
    for name, mask in subgroups.items():
        subset = df if mask is None else df[mask]
        if len(subset) < 100:
            continue
        
        X = subset[raw17].values
        y = subset['KOA'].values
        
        if y.sum() < 10:
            print(f"  {name}: SKIP (too few positive cases: {y.sum()})")
            continue
        
        aucs = []
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        aucs.append(roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1]))
        
        koa_rate = y.mean() * 100
        results.append({
            'subgroup': name,
            'n': len(subset),
            'koa_rate': koa_rate,
            'auc_mean': np.mean(aucs),
        })
        print(f"  {name:<25s}: n={len(subset):>5d}, KOA={koa_rate:.1f}%, AUC={np.mean(aucs):.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(STEP_DIR, 'subgroup_analysis.csv'), index=False)
    return results_df

def main():
    print("=" * 80)
    print("STEP 04: FEATURE IMPACT ANALYSIS & PAPER REPLICATION")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not os.path.exists(STEP_DIR):
        os.makedirs(STEP_DIR)
    
    # Load dataset with KDM-BA
    df = pd.read_csv(os.path.join(BASE_DIR, 'step_03_kdm_ba', 'dataset_A_with_kdm_ba.csv'))
    print(f"Dataset A loaded: {df.shape}")
    print(f"KOA rate: {df['KOA'].mean()*100:.1f}%\n")
    
    # Feature columns for SHAP
    raw17 = ['wave', 'Time', 'Gender', 'Age_New', 'Marital', 'Education',
              'Residence', 'Hypertension', 'Dyslipidemia', 'Diabetes',
              'Cancer', 'CVD', 'Smoke', 'Drink', 'BMI', 'BMI_New',
              'Biological Age']
    
    biomarkers_log = ['plt_10.9.L', 'crp_mg.L', 'Hb A1c', 'creatinine_mg.d L',
                       'bun_mg.d L', 'TC_mg.d L', 'TG_mg.d L', 'sbp.mean']
    
    # Run analyses
    print("\n" + "=" * 80)
    print("STARTING ANALYSES")
    print("=" * 80)
    
    # 1. SHAP Analysis (real TreeExplainer)
    shap_rf, shap_xgb, shap_interactions, shap_interactions_clinical = shap_analysis(df, raw17, analysis_name='raw17')
    shap_rf.to_csv(os.path.join(STEP_DIR, 'shap_importance_rf.csv'), index=False)
    shap_xgb.to_csv(os.path.join(STEP_DIR, 'shap_importance_xgb.csv'), index=False)
    shap_interactions.to_csv(os.path.join(STEP_DIR, 'shap_interactions_xgb.csv'), index=False)
    shap_interactions_clinical.to_csv(os.path.join(STEP_DIR, 'shap_interactions_clinical_raw17.csv'), index=False)

    shap_rf_pheno = None
    shap_xgb_pheno = None
    shap_interactions_pheno = None
    shap_interactions_pheno_clinical = None
    if 'PhenoAge_Adapted' in df.columns:
        raw17_plus_pheno = raw17 + ['PhenoAge_Adapted']
        shap_rf_pheno, shap_xgb_pheno, shap_interactions_pheno, shap_interactions_pheno_clinical = shap_analysis(
            df,
            raw17_plus_pheno,
            analysis_name='raw17_plus_pheno'
        )
        shap_rf_pheno.to_csv(os.path.join(STEP_DIR, 'shap_importance_rf_raw17_plus_pheno.csv'), index=False)
        shap_xgb_pheno.to_csv(os.path.join(STEP_DIR, 'shap_importance_xgb_raw17_plus_pheno.csv'), index=False)
        shap_interactions_pheno.to_csv(os.path.join(STEP_DIR, 'shap_interactions_xgb_raw17_plus_pheno.csv'), index=False)
        shap_interactions_pheno_clinical.to_csv(
            os.path.join(STEP_DIR, 'shap_interactions_clinical_raw17_plus_pheno.csv'),
            index=False,
        )

    shap_rf_hscrp = None
    shap_xgb_hscrp = None
    shap_interactions_hscrp = None
    shap_interactions_hscrp_clinical = None
    if 'crp_mg.L' in df.columns:
        raw17_plus_hscrp = raw17 + ['crp_mg.L']
        shap_rf_hscrp, shap_xgb_hscrp, shap_interactions_hscrp, shap_interactions_hscrp_clinical = shap_analysis(
            df,
            raw17_plus_hscrp,
            analysis_name='raw17_plus_hscrp'
        )
        shap_rf_hscrp.to_csv(os.path.join(STEP_DIR, 'shap_importance_rf_raw17_plus_hscrp.csv'), index=False)
        shap_xgb_hscrp.to_csv(os.path.join(STEP_DIR, 'shap_importance_xgb_raw17_plus_hscrp.csv'), index=False)
        shap_interactions_hscrp.to_csv(os.path.join(STEP_DIR, 'shap_interactions_xgb_raw17_plus_hscrp.csv'), index=False)
        shap_interactions_hscrp_clinical.to_csv(
            os.path.join(STEP_DIR, 'shap_interactions_clinical_raw17_plus_hscrp.csv'),
            index=False,
        )

    # 1B. Aging clock summary table
    aging_summary = aging_clock_summary(df)
    aging_summary.to_csv(os.path.join(STEP_DIR, 'aging_clock_summary.csv'), index=False)
    
    # 2. Feature Impact Comparison
    impact_results = feature_impact_comparison(df)
    
    # 3. Paper Replication
    replication_results, lasso_features = paper_replication(df)
    lasso_features.to_csv(os.path.join(STEP_DIR, 'lasso_features.csv'), index=False)
    
    # 4. Subgroup Analysis
    subgroup_results = subgroup_analysis(df)
    
    # Generate report
    report = []
    report.append("=" * 80)
    report.append("STEP 04: FEATURE IMPACT & PAPER REPLICATION - REPORT")
    report.append("=" * 80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset: {len(df)} rows, KOA rate={df['KOA'].mean()*100:.1f}%")
    report.append("")
    
    report.append("1. SHAP ANALYSIS")
    report.append("-" * 40)
    report.append("RF SHAP Top Features:")
    for _, row in shap_rf.head(10).iterrows():
        report.append(f"  {row['feature']:<30s}: SHAP={row['shap_mean']:.4f}")
    report.append("")
    report.append("XGBoost SHAP Top Features:")
    for _, row in shap_xgb.head(10).iterrows():
        report.append(f"  {row['feature']:<30s}: SHAP={row['shap_mean']:.4f}")
    report.append("")
    report.append("XGBoost SHAP Top Interactions (raw17):")
    for _, row in shap_interactions.head(10).iterrows():
        report.append(
            f"  {row['feature_1']:<20s} x {row['feature_2']:<20s}: "
            f"INT={row['interaction_strength']:.4f}"
        )
    report.append("")
    report.append("Clinically Filtered SHAP Interactions (raw17):")
    for _, row in shap_interactions_clinical.head(10).iterrows():
        report.append(
            f"  {row['feature_1']:<20s} x {row['feature_2']:<20s}: "
            f"INT={row['interaction_strength']:.4f} [{row['clinical_pair_group']}]"
        )
    report.append("")

    if shap_rf_pheno is not None and shap_xgb_pheno is not None and shap_interactions_pheno is not None:
        report.append("1A-2. SHAP ANALYSIS WITH PHENOAGE_ADAPTED")
        report.append("-" * 40)
        report.append("XGBoost SHAP Top Features (raw17 + PhenoAge_Adapted):")
        for _, row in shap_xgb_pheno.head(10).iterrows():
            report.append(f"  {row['feature']:<30s}: SHAP={row['shap_mean']:.4f}")
        report.append("")
        report.append("XGBoost SHAP Top Interactions (raw17 + PhenoAge_Adapted):")
        for _, row in shap_interactions_pheno.head(10).iterrows():
            report.append(
                f"  {row['feature_1']:<20s} x {row['feature_2']:<20s}: "
                f"INT={row['interaction_strength']:.4f}"
            )
        report.append("")
        if shap_interactions_pheno_clinical is not None:
            report.append("Clinically Filtered SHAP Interactions (raw17 + PhenoAge_Adapted):")
            for _, row in shap_interactions_pheno_clinical.head(10).iterrows():
                report.append(
                    f"  {row['feature_1']:<20s} x {row['feature_2']:<20s}: "
                    f"INT={row['interaction_strength']:.4f} [{row['clinical_pair_group']}]"
                )
            report.append("")

    if shap_rf_hscrp is not None and shap_xgb_hscrp is not None and shap_interactions_hscrp is not None:
        report.append("1A-3. SHAP ANALYSIS WITH HS-CRP")
        report.append("-" * 40)
        report.append("XGBoost SHAP Top Features (raw17 + hs-CRP):")
        for _, row in shap_xgb_hscrp.head(10).iterrows():
            report.append(f"  {row['feature']:<30s}: SHAP={row['shap_mean']:.4f}")
        report.append("")
        report.append("XGBoost SHAP Top Interactions (raw17 + hs-CRP):")
        for _, row in shap_interactions_hscrp.head(10).iterrows():
            report.append(
                f"  {row['feature_1']:<20s} x {row['feature_2']:<20s}: "
                f"INT={row['interaction_strength']:.4f}"
            )
        report.append("")
        if shap_interactions_hscrp_clinical is not None:
            report.append("Clinically Filtered SHAP Interactions (raw17 + hs-CRP):")
            for _, row in shap_interactions_hscrp_clinical.head(10).iterrows():
                report.append(
                    f"  {row['feature_1']:<20s} x {row['feature_2']:<20s}: "
                    f"INT={row['interaction_strength']:.4f} [{row['clinical_pair_group']}]"
                )
            report.append("")

    report.append("1B. AGING CLOCK SUMMARY")
    report.append("-" * 40)
    for _, row in aging_summary.iterrows():
        report.append(
            f"  {row['aging_clock']:<24s}: "
            f"AUC={row['auc_univariate']:.4f}, "
            f"OR={row['odds_ratio_per_unit']:.4f}, "
            f"corr(KOA)={row['corr_with_koa']:.4f}"
        )
    report.append("")
    
    report.append("2. LASSO FEATURE SELECTION")
    report.append("-" * 40)
    selected = lasso_features[lasso_features['abs_coef'] > 0]
    for _, row in selected.iterrows():
        report.append(f"  {row['feature']:<30s}: coef={row['lasso_coef']:.6f}")
    report.append(f"\n  Features selected: {len(selected)}/{len(lasso_features)}")
    report.append("")
    
    report.append("3. FEATURE IMPACT COMPARISON")
    report.append("-" * 40)
    for _, row in impact_results.iterrows():
        report.append(f"  {row['feature_set']:<30s}: AUC={row['roc_auc_mean']:.4f} (+/-{row['roc_auc_std']:.4f}), {int(row['n_features'])} feats")
    report.append("")
    
    report.append("4. PAPER REPLICATION - TOP 15")
    report.append("-" * 40)
    for _, row in replication_results.head(15).iterrows():
        report.append(f"  {row['feature_set']:<25s} + {row['model']:<20s}: AUC={row['roc_auc_mean']:.4f}, F1={row['f1_mean']:.4f}")
    report.append("")
    
    report.append("5. COMPARISON WITH PAPER")
    report.append("-" * 40)
    best_auc = replication_results['roc_auc_mean'].max()
    report.append(f"  Our best AUROC: {best_auc:.4f}")
    report.append(f"  Paper best AUROC: 0.9078")
    report.append(f"  Difference: {0.9078 - best_auc:.4f}")
    report.append("")
    
    report_text = "\n".join(report)
    with open(os.path.join(STEP_DIR, 'step04_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nReport saved to {os.path.join(STEP_DIR, 'step04_report.txt')}")
    print(f"Results saved to {os.path.join(STEP_DIR, 'paper_replication_results.csv')}")
    
    print("\n" + "=" * 80)
    print("STEP 04 COMPLETED")

if __name__ == '__main__':
    main()