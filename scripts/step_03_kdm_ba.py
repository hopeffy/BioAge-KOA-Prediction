"""
STEP 03: KDM-Biological Age Calculation
=========================================
Calculate Biological Age using Klemera-Doubal Method (KDM) from 8 biomarkers.
Compare KDM-BA with existing Biological Age column.
Calculate BIR (Biological Illness Risk) and BA quartiles.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = r'C:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data'
STEP_DIR = os.path.join(BASE_DIR, 'step_03_kdm_ba')

def calculate_kdm_ba(df, biomarker_cols, age_col='Biological Age'):
    """
    Klemera-Doubal Method (KDM) for Biological Age calculation.
    
    KDM formula:
    BA_kdm = (sum_i[(x_i - a_i) * b_i / sigma_i^2] + CA / sigma_CA^2) / (sum_i[b_i^2 / sigma_i^2] + 1 / sigma_CA^2)
    
    For each biomarker x_i:
    - Fit linear regression: x_i = a_i + b_i * CA + epsilon_i
    - a_i = intercept
    - b_i = slope (regression coefficient)
    - sigma_i = residual standard error
    - r_i = correlation with age
    
    CA = chronological age (using Biological Age as proxy since we don't have exact CA)
    """
    print("=" * 80)
    print("KDM-BIOLOGICAL AGE CALCULATION")
    print("=" * 80)
    
    ca = df[age_col].values.copy()
    
    regression_results = {}
    
    print("\nStep 1: Regression of each biomarker on chronological age")
    print("-" * 80)
    print(f"{'Biomarker':<25} {'Intercept':>10} {'Slope':>10} {'R':>8} {'R^2':>8} {'SE_resid':>10} {'p-value':>12}")
    print("-" * 80)
    
    for biomarker in biomarker_cols:
        x = df[biomarker].values
        
        # Linear regression: x = a + b * CA
        slope, intercept, r_value, p_value, std_err = stats.linregress(ca, x)
        
        # Residual standard error
        predicted = intercept + slope * ca
        residuals = x - predicted
        se_residual = np.std(residuals)
        
        regression_results[biomarker] = {
            'intercept': intercept,
            'slope': slope,
            'r': r_value,
            'r_squared': r_value**2,
            'se_residual': se_residual,
            'p_value': p_value,
            'std_err': std_err,
        }
        
        print(f"{biomarker:<25} {intercept:>10.4f} {slope:>10.6f} {r_value:>8.4f} {r_value**2:>8.4f} {se_residual:>10.4f} {p_value:>12.2e}")
    
    print("\nStep 2: KDM-BA calculation")
    print("-" * 80)
    
    sigma_ca = np.std(ca)
    
    numerator = 0.0
    denominator = 0.0
    
    for biomarker, res in regression_results.items():
        x_i = df[biomarker].values
        a_i = res['intercept']
        b_i = res['slope']
        sigma_i = res['se_residual']
        
        numerator += ((x_i - a_i) * b_i) / (sigma_i ** 2)
        denominator += (b_i ** 2) / (sigma_i ** 2)
    
    # Add CA term
    numerator_total = numerator + (ca / sigma_ca ** 2)
    denominator_total = denominator + (1 / sigma_ca ** 2)
    
    ba_kdm = numerator_total / denominator_total
    
    return ba_kdm, regression_results, sigma_ca

def robust_zscore(values):
    """Return winsorized z-scores to reduce outlier sensitivity."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() == 0:
        return np.zeros_like(arr)

    median_val = np.nanmedian(arr[finite])
    arr = np.where(finite, arr, median_val)

    q_low, q_high = np.percentile(arr, [1, 99])
    arr = np.clip(arr, q_low, q_high)

    std_val = np.std(arr)
    if std_val < 1e-8:
        return np.zeros_like(arr)
    return (arr - np.mean(arr)) / std_val

def calculate_phenoage_adapted(df, age_col='Biological Age'):
    """
    Build an adapted phenotypic-age surrogate from available biomarkers.

    This is not canonical Levine PhenoAge. It is a weighted biomarker-derived
    age surrogate for settings where full PhenoAge inputs are unavailable.
    """
    required_cols = [
        'crp_original', 'hba1c_original', 'creatinine_original', 'bun_original',
        'total_cholesterol_original', 'triglycerides_original', 'sbp_original',
        'platelet_original',
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns for PhenoAge_Adapted: {missing_cols}")

    # Positive weight: higher values indicate higher age-related burden.
    # Negative weight for platelet reflects lower platelet tendency with aging burden.
    weights = {
        'crp_original': 1.40,
        'hba1c_original': 1.20,
        'creatinine_original': 1.00,
        'bun_original': 0.90,
        'sbp_original': 1.00,
        'total_cholesterol_original': 0.60,
        'triglycerides_original': 0.60,
        'platelet_original': -0.40,
    }

    weighted_risk = np.zeros(len(df), dtype=float)
    for col, w in weights.items():
        weighted_risk += w * robust_zscore(df[col].values)

    # Normalize composite risk before mapping to age units.
    risk_std = np.std(weighted_risk)
    if risk_std < 1e-8:
        risk_z = np.zeros_like(weighted_risk)
    else:
        risk_z = (weighted_risk - np.mean(weighted_risk)) / risk_std

    age_vals = df[age_col].values.astype(float)
    age_std = np.std(age_vals)
    scale = 0.45 * age_std

    pheno_age_adapted = age_vals + (risk_z * scale)
    pheno_age_adapted = np.clip(pheno_age_adapted, age_vals.min() - 15, age_vals.max() + 15)
    pheno_age_accel = pheno_age_adapted - age_vals

    details = {
        'weights': weights,
        'risk_mean': float(np.mean(weighted_risk)),
        'risk_std': float(np.std(weighted_risk)),
        'age_scale': float(scale),
    }
    return pheno_age_adapted, pheno_age_accel, details

def main():
    print("=" * 80)
    print("STEP 03: KDM-BIOLOGICAL AGE CALCULATION")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    if not os.path.exists(STEP_DIR):
        os.makedirs(STEP_DIR)
    
    # Load Dataset A
    df = pd.read_csv(os.path.join(BASE_DIR, 'step_01_data_prep', 'dataset_A_all.csv'))
    print(f"Dataset A loaded: {df.shape}")
    print(f"KOA rate: {df['KOA'].mean()*100:.1f}%\n")
    
    # Define biomarkers (log-transformed versions)
    biomarkers_log = ['plt_10.9.L', 'crp_mg.L', 'Hb A1c', 'creatinine_mg.d L',
                       'bun_mg.d L', 'TC_mg.d L', 'TG_mg.d L', 'sbp.mean']
    
    biomarkers_orig = ['platelet_original', 'crp_original', 'hba1c_original',
                        'creatinine_original', 'bun_original',
                        'total_cholesterol_original', 'triglycerides_original',
                        'sbp_original']
    
    # ==================================================================
    # KDM-BA from LOG-TRANSFORMED biomarkers
    # ==================================================================
    print("Calculating KDM-BA from log-transformed biomarkers...")
    ba_kdm_log, reg_log, sigma_ca = calculate_kdm_ba(df, biomarkers_log, 'Biological Age')
    df['BA_KDM_log'] = ba_kdm_log
    
    # ==================================================================
    # KDM-BA from ORIGINAL SCALE biomarkers
    # ==================================================================
    print("\n\nCalculating KDM-BA from original-scale biomarkers...")
    ba_kdm_orig, reg_orig, _ = calculate_kdm_ba(df, biomarkers_orig, 'Biological Age')
    df['BA_KDM_orig'] = ba_kdm_orig

    # ==================================================================
    # Adapted Phenotypic Age surrogate from available biomarkers
    # ==================================================================
    print("\n\nCalculating PhenoAge_Adapted from available biomarkers...")
    pheno_age_adapted, pheno_age_accel, pheno_details = calculate_phenoage_adapted(df, 'Biological Age')
    df['PhenoAge_Adapted'] = pheno_age_adapted
    df['PhenoAge_Adapted_Accel'] = pheno_age_accel

    print("\nPhenoAge_Adapted statistics:")
    print(f"  Mean: {df['PhenoAge_Adapted'].mean():.2f}")
    print(f"  Std:  {df['PhenoAge_Adapted'].std():.2f}")
    print(f"  Min:  {df['PhenoAge_Adapted'].min():.2f}")
    print(f"  Max:  {df['PhenoAge_Adapted'].max():.2f}")
    print("PhenoAge_Adapted_Accel statistics:")
    print(f"  Mean: {df['PhenoAge_Adapted_Accel'].mean():.4f}")
    print(f"  Std:  {df['PhenoAge_Adapted_Accel'].std():.4f}")
    print(f"  Min:  {df['PhenoAge_Adapted_Accel'].min():.4f}")
    print(f"  Max:  {df['PhenoAge_Adapted_Accel'].max():.4f}")
    
    # ==================================================================
    # BIR (Biological Illness Risk) = KDM-BA - Chronological Age
    # ==================================================================
    print("\n" + "=" * 80)
    print("BIR CALCULATION (Biological Illness Risk)")
    print("=" * 80)
    
    df['BIR_log'] = df['BA_KDM_log'] - df['Biological Age']
    df['BIR_orig'] = df['BA_KDM_orig'] - df['Biological Age']
    
    print(f"\nBIR_log statistics:")
    print(f"  Mean: {df['BIR_log'].mean():.4f}")
    print(f"  Std:  {df['BIR_log'].std():.4f}")
    print(f"  Min:  {df['BIR_log'].min():.4f}")
    print(f"  Max:  {df['BIR_log'].max():.4f}")
    
    print(f"\nBIR_orig statistics:")
    print(f"  Mean: {df['BIR_orig'].mean():.4f}")
    print(f"  Std:  {df['BIR_orig'].std():.4f}")
    print(f"  Min:  {df['BIR_orig'].min():.4f}")
    print(f"  Max:  {df['BIR_orig'].max():.4f}")
    
    # ==================================================================
    # BA Quartiles
    # ==================================================================
    print("\n" + "=" * 80)
    print("BA QUARTILES")
    print("=" * 80)
    
    for ba_col in ['Biological Age', 'BA_KDM_log', 'BA_KDM_orig', 'PhenoAge_Adapted']:
        df[f'{ba_col}_Q'] = pd.qcut(df[ba_col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        print(f"\n{ba_col} quartiles:")
        for q in ['Q1', 'Q2', 'Q3', 'Q4']:
            subset = df[df[f'{ba_col}_Q'] == q]
            koa_rate = subset['KOA'].mean() * 100
            print(f"  {q}: n={len(subset)}, BA range=[{subset[ba_col].min():.1f}, {subset[ba_col].max():.1f}], KOA rate={koa_rate:.1f}%")
    
    # ==================================================================
    # Compare BA_KDM with existing Biological Age
    # ==================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: AGING CLOCKS vs Existing Biological Age")
    print("=" * 80)
    
    for ba_type, col in [
        ('KDM-BA (log)', 'BA_KDM_log'),
        ('KDM-BA (orig)', 'BA_KDM_orig'),
        ('PhenoAge_Adapted', 'PhenoAge_Adapted'),
    ]:
        print(f"\n{ba_type}:")
        print(f"  Correlation with Biological Age: {df['Biological Age'].corr(df[col]):.4f}")
        print(f"  Mean: {df[col].mean():.2f} (CA mean: {df['Biological Age'].mean():.2f})")
        print(f"  Std:  {df[col].std():.2f} (CA std:  {df['Biological Age'].std():.2f})")
        print(f"  KOA correlation: {df['KOA'].corr(df[col]):.4f} (BA: {df['KOA'].corr(df['Biological Age']):.4f})")
    
    # ==================================================================
    # Logistic Regression for BA-KOA relationship
    # ==================================================================
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION: BA-KOA RELATIONSHIP")
    print("=" * 80)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    
    for ba_col in ['Biological Age', 'BA_KDM_log', 'BA_KDM_orig', 'PhenoAge_Adapted', 'PhenoAge_Adapted_Accel']:
        X = df[[ba_col]].values
        y = df['KOA'].values
        
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        y_pred_proba = lr.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_pred_proba)
        coef = lr.coef_[0][0]
        or_val = np.exp(coef)
        
        print(f"\n{ba_col}:")
        print(f"  AUROC (univariate): {auc:.4f}")
        print(f"  Coefficient: {coef:.4f}")
        print(f"  Odds Ratio (per 1 year): {or_val:.4f}")
    
    # ==================================================================
    # RCS analysis (simplified cubic spline)
    # ==================================================================
    print("\n" + "=" * 80)
    print("NONLINEAR BA-KOA RELATIONSHIP CHECK")
    print("=" * 80)
    
    for ba_col in ['Biological Age', 'BA_KDM_log', 'BA_KDM_orig', 'PhenoAge_Adapted', 'PhenoAge_Adapted_Accel']:
        # Split by median and check KOA rate above/below
        median_ba = df[ba_col].median()
        below = df[df[ba_col] <= median_ba]
        above = df[df[ba_col] > median_ba]
        
        from scipy.stats import chi2_contingency
        ct = pd.crosstab(df[ba_col] > median_ba, df['KOA'])
        chi2, p_val, dof, expected = chi2_contingency(ct)
        
        print(f"\n{ba_col}:")
        print(f"  Median: {median_ba:.2f}")
        print(f"  Below median: KOA rate = {below['KOA'].mean()*100:.2f}%")
        print(f"  Above median: KOA rate = {above['KOA'].mean()*100:.2f}%")
        print(f"  Chi-squared: {chi2:.2f}, p-value: {p_val:.2e}")
    
    # ==================================================================
    # Save dataset with KDM-BA
    # ==================================================================
    # Create quartile columns as integers for ML
    ba_q_col_map = {}
    for ba_col in ['Biological Age', 'BA_KDM_log', 'BA_KDM_orig', 'PhenoAge_Adapted']:
        q_col = f'{ba_col}_Q'
        df[f'{ba_col}_Qint'] = df[q_col].map({'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4})
    
    # Drop quartile string columns
    for ba_col in ['Biological Age', 'BA_KDM_log', 'BA_KDM_orig', 'PhenoAge_Adapted']:
        df.drop(f'{ba_col}_Q', axis=1, inplace=True)
    
    # Save
    df.to_csv(os.path.join(STEP_DIR, 'dataset_A_with_kdm_ba.csv'), index=False)
    
    # Also create Dataset B with BA >= 55 filter
    df_b = df[df['Biological Age'] >= 55].copy()
    df_b.to_csv(os.path.join(STEP_DIR, 'dataset_B_with_kdm_ba.csv'), index=False)
    
    # ==================================================================
    # Generate report
    # ==================================================================
    report = []
    report.append("=" * 80)
    report.append("STEP 03: KDM-BIOLOGICAL AGE CALCULATION - REPORT")
    report.append("=" * 80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("1. KDM-BA CALCULATION METHOD")
    report.append("-" * 40)
    report.append("  Method: Klemera-Doubal Method (KDM)")
    report.append("  Chronological Age proxy: Biological Age column")
    report.append("  Biomarkers (log-transformed): 8 markers")
    report.append("  Biomarkers (original scale): 8 markers")
    report.append("  PhenoAge_Adapted: weighted biomarker-derived surrogate (non-canonical)")
    report.append("")
    report.append("2. REGRESSION RESULTS (log-transformed)")
    report.append("-" * 40)
    for biomarker, res in reg_log.items():
        report.append(f"  {biomarker}: b={res['slope']:.6f}, a={res['intercept']:.4f}, r={res['r']:.4f}, R^2={res['r_squared']:.4f}")
    report.append("")
    report.append("3. AGING CLOCKS vs EXISTING BIOLOGICAL AGE")
    report.append("-" * 40)
    for ba_type, col in [
        ('KDM-BA (log)', 'BA_KDM_log'),
        ('KDM-BA (orig)', 'BA_KDM_orig'),
        ('PhenoAge_Adapted', 'PhenoAge_Adapted'),
    ]:
        corr = df['Biological Age'].corr(df[col])
        koa_corr = df['KOA'].corr(df[col])
        ba_koa_corr = df['KOA'].corr(df['Biological Age'])
        report.append(f"  {ba_type}:")
        report.append(f"    Correlation with Biological Age: {corr:.4f}")
        report.append(f"    Correlation with KOA: {koa_corr:.4f} (BA: {ba_koa_corr:.4f})")
        report.append(f"    Mean: {df[col].mean():.2f} (BA mean: {df['Biological Age'].mean():.2f})")
    report.append("")
    report.append("4. LOGISTIC REGRESSION (Univariate aging clock-KOA)")
    report.append("-" * 40)
    for ba_col in ['Biological Age', 'BA_KDM_log', 'BA_KDM_orig', 'PhenoAge_Adapted', 'PhenoAge_Adapted_Accel']:
        X = df[[ba_col]].values
        y = df['KOA'].values
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X, y)
        auc = roc_auc_score(y, lr.predict_proba(X)[:, 1])
        report.append(f"  {ba_col}: AUROC={auc:.4f}, OR={np.exp(lr.coef_[0][0]):.4f}")
    report.append("")
    report.append("5. PHENOAGE_ADAPTED DETAILS")
    report.append("-" * 40)
    report.append("  This score is adapted and should not be interpreted as canonical Levine PhenoAge.")
    report.append(f"  Composite risk mean/std: {pheno_details['risk_mean']:.4f} / {pheno_details['risk_std']:.4f}")
    report.append(f"  Age mapping scale: {pheno_details['age_scale']:.4f}")
    report.append("  Weights:")
    for biomarker_name, weight_val in pheno_details['weights'].items():
        report.append(f"    {biomarker_name}: {weight_val:+.2f}")
    report.append("")
    report.append("6. DATASETS SAVED")
    report.append("-" * 40)
    report.append(f"  Dataset A (n={len(df)}): dataset_A_with_kdm_ba.csv")
    report.append(f"  Dataset B (n={len(df_b)}): dataset_B_with_kdm_ba.csv")
    report.append("  KDM-BA columns added: BA_KDM_log, BA_KDM_orig, BIR_log, BIR_orig")
    report.append("  Adapted clock columns added: PhenoAge_Adapted, PhenoAge_Adapted_Accel")
    report.append("  Quartile columns added: Biological Age_Qint, BA_KDM_log_Qint, BA_KDM_orig_Qint, PhenoAge_Adapted_Qint")
    
    report_text = "\n".join(report)
    with open(os.path.join(STEP_DIR, 'step03_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + "=" * 80)
    print("STEP 03 COMPLETED")
    print("=" * 80)
    print(f"Files saved in: {STEP_DIR}")
    print(f"  dataset_A_with_kdm_ba.csv ({len(df)} rows)")
    print(f"  dataset_B_with_kdm_ba.csv ({len(df_b)} rows)")
    print(f"  step03_report.txt")

if __name__ == '__main__':
    main()