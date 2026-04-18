"""
Step 6: Feature Quality Audit - Deep Dive Analysis
===================================================
Forensic analysis of domain-specific features to identify WHY they failed

MOTIVATION (User Question):
"feature kötü mü, implementation mı kötü?"
"belki 16 feature'ın 3 tanesi iyi, 13 tanesi zararlı"

AUDIT CHECKLIST (for each engineered feature):
1. Missing rate (% NaN/inf)
2. Distribution (mean, std, min, max, skewness, kurtosis)
3. Outliers (IQR method, Z-score > 3)
4. Target separation (boxplot target=0 vs target=1)
5. Correlation with target (Spearman, Pearson)
6. Correlation with raw parents (multicollinearity check)
7. Train/Test distribution shift (KS test)
8. Unique values (is it informative?)

DELIVERABLES:
- Feature quality score (0-100)
- Red flags identification
- Actionable recommendations

Author: Bioinformatics Analysis Team
Date: March 2026
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import ks_2samp, zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STAGE 6: FEATURE QUALITY AUDIT - FORENSIC ANALYSIS")
print("="*80)
print("\n🔬 OBJECTIVE: Identify WHY domain features failed")
print("   → Audit each of 16 features individually")
print("   → Find: good features vs bad features vs implementation issues\n")
print("="*80)

# Configuration
SEED = 42
raw_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step6_feature_quality_audit"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)

def binary_encode(series):
    """Convert yes/no to 0/1"""
    if series.dtype == 'object' or series.dtype == 'string':
        series_str = series.astype(str).str.lower().str.strip()
        result = pd.Series(0, index=series.index, dtype=float)
        result[series_str == 'yes'] = 1
        result[series_str == 'no'] = 0
        result[series_str == '1'] = 1
        result[series_str == '1.0'] = 1
        result[series_str == '0'] = 0
        result[series_str == '0.0'] = 0
        result[series_str.isna() | (series_str == 'nan')] = 0
        return result
    else:
        return series.fillna(0).astype(float)

def create_domain_specific_features(df):
    """Create 16 domain-specific features (same as Stage 5)"""
    df = df.copy()
    
    # Pre-compute values
    age = df['Biological Age'].fillna(df['Biological Age'].median())
    bmi = df['bmi_kg.m2'].fillna(df['bmi_kg.m2'].median())
    
    sex_col = 'sex' if 'sex' in df.columns else 'Sex'
    sex_numeric = df[sex_col].map({'male': 1, 'female': 0, 'Male': 1, 'Female': 0}).fillna(0.5)
    is_female = (sex_numeric == 0).astype(int)
    
    # Comorbidities
    heart_disease = binary_encode(df['doctor_diagnose_heart_disease']) if 'doctor_diagnose_heart_disease' in df.columns else binary_encode(df.get('Heart_Disease', pd.Series(0, index=df.index)))
    hypertension = binary_encode(df['doctor_diagnose_hypertension']) if 'doctor_diagnose_hypertension' in df.columns else binary_encode(df.get('Hypertension', pd.Series(0, index=df.index)))
    diabetes = binary_encode(df['doctor_diagnose_diabetes']) if 'doctor_diagnose_diabetes' in df.columns else binary_encode(df.get('Diabetes', pd.Series(0, index=df.index)))
    stroke = binary_encode(df.get('doctor_diagnose_stroke', df.get('Stroke', pd.Series(0, index=df.index))))
    lung_disease = binary_encode(df.get('doctor_diagnose_lung_disease', df.get('Lung_Disease', pd.Series(0, index=df.index))))
    cancer = binary_encode(df['doctor_diagnose_cancer']) if 'doctor_diagnose_cancer' in df.columns else binary_encode(df.get('Cancer', pd.Series(0, index=df.index)))
    
    met = df['MET'].fillna(df['MET'].median())
    
    # Create features
    df['OA_Age_Risk_Score'] = np.where(age < 50, (age / 50), 1 + ((age - 50) / 10) ** 1.5)
    df['OA_Mechanical_Load'] = (bmi / 25) * (age / 60)
    df['OA_Gender_Risk'] = np.where(is_female == 1, np.where(age >= 50, 2.5, 1.5), 1.0)
    
    is_obese = (bmi >= 30).astype(int)
    df['OA_Metabolic_Syndrome'] = (diabetes * 2 + hypertension * 1.5 + is_obese * 1.5) / 5.0
    df['OA_Inflammatory_Score'] = (heart_disease * 2 + hypertension * 1.5 + diabetes * 2 + stroke * 2.5 + lung_disease * 1.5 + cancer * 1) / 10.0
    
    low_activity = (met < met.quantile(0.33)).astype(int)
    df['OA_Joint_Stress_Index'] = (bmi / 25) * (1 + low_activity * 2)
    df['OA_CV_Risk_Cluster'] = (heart_disease + hypertension + stroke) / 3.0
    df['OA_Systemic_Burden'] = (heart_disease * 2 + stroke * 2.5 + diabetes * 2 + cancer * 1.5 + hypertension * 1 + lung_disease * 1) / 10.0
    
    activity_ratio = met / met.quantile(0.75)
    bmi_ratio = bmi / 30
    df['OA_Activity_Balance'] = bmi_ratio / (activity_ratio + 0.1)
    df['OA_Obesity_Severity'] = np.where(bmi < 25, 0, np.where(bmi < 30, 1, np.where(bmi < 35, 2, 3)))
    df['OA_Age_Gender_Interaction'] = is_female * (age / 60) * np.where(age >= 50, 2, 1)
    df['OA_Sedentary_Load'] = low_activity * (bmi / 25) * (age / 60)
    
    chronic_conditions = heart_disease + hypertension + diabetes + lung_disease
    df['OA_Chronic_Disease_Load'] = (chronic_conditions / 4.0) * (age / 60)
    df['OA_Lifestyle_Risk'] = ((bmi / 35) * 0.4 + (1 - activity_ratio) * 0.3 + (age / 80) * 0.3)
    df['OA_Biomechanical_Stress'] = (age / 60) * (bmi / 25) * (1 + is_female * 0.5)
    
    total_comorbidities = heart_disease + hypertension + diabetes + stroke + lung_disease + cancer
    df['OA_Multi_System_Degradation'] = (age / 60) * (total_comorbidities / 6) * (bmi / 25)
    
    return df

def load_data():
    """Load and prepare data"""
    df_raw = pd.read_excel(raw_data_path)
    df = df_raw[df_raw['Arthritis'].notna()].copy()
    return df

def compute_feature_stats(feature_series, feature_name, target_series):
    """Compute comprehensive statistics for a feature"""
    
    stats_dict = {
        'Feature': feature_name,
        'Type': str(feature_series.dtype)
    }
    
    # 1. MISSING RATE
    total = len(feature_series)
    missing = feature_series.isna().sum()
    inf_count = np.isinf(feature_series.replace([np.inf, -np.inf], np.nan)).sum()
    stats_dict['Missing_Count'] = missing
    stats_dict['Missing_Pct'] = (missing / total) * 100
    stats_dict['Inf_Count'] = inf_count
    
    # Work with non-missing values
    valid = feature_series.dropna()
    valid = valid.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid) == 0:
        stats_dict['Status'] = 'EMPTY'
        return stats_dict
    
    # 2. DISTRIBUTION STATS
    stats_dict['Mean'] = valid.mean()
    stats_dict['Std'] = valid.std()
    stats_dict['Min'] = valid.min()
    stats_dict['Q25'] = valid.quantile(0.25)
    stats_dict['Median'] = valid.median()
    stats_dict['Q75'] = valid.quantile(0.75)
    stats_dict['Max'] = valid.max()
    stats_dict['Skewness'] = valid.skew()
    stats_dict['Kurtosis'] = valid.kurtosis()
    
    # 3. UNIQUE VALUES
    stats_dict['Unique_Count'] = valid.nunique()
    stats_dict['Unique_Pct'] = (valid.nunique() / len(valid)) * 100
    
    # 4. OUTLIERS (IQR method)
    Q1 = valid.quantile(0.25)
    Q3 = valid.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = ((valid < lower_bound) | (valid > upper_bound)).sum()
    stats_dict['Outlier_Count'] = outliers
    stats_dict['Outlier_Pct'] = (outliers / len(valid)) * 100
    
    # 5. Z-SCORE OUTLIERS (|z| > 3)
    z_scores = np.abs(zscore(valid))
    extreme_outliers = (z_scores > 3).sum()
    stats_dict['Extreme_Outlier_Count'] = extreme_outliers
    stats_dict['Extreme_Outlier_Pct'] = (extreme_outliers / len(valid)) * 100
    
    # 6. CORRELATION WITH TARGET
    valid_with_target = pd.DataFrame({
        'feature': feature_series,
        'target': target_series
    }).dropna()
    
    if len(valid_with_target) > 10:
        try:
            pearson_corr, pearson_p = scipy_stats.pearsonr(valid_with_target['feature'], valid_with_target['target'])
            spearman_corr, spearman_p = scipy_stats.spearmanr(valid_with_target['feature'], valid_with_target['target'])
            stats_dict['Pearson_Corr'] = pearson_corr
            stats_dict['Pearson_P'] = pearson_p
            stats_dict['Spearman_Corr'] = spearman_corr
            stats_dict['Spearman_P'] = spearman_p
        except:
            stats_dict['Pearson_Corr'] = np.nan
            stats_dict['Pearson_P'] = np.nan
            stats_dict['Spearman_Corr'] = np.nan
            stats_dict['Spearman_P'] = np.nan
    
    # 7. TARGET SEPARATION (effect size)
    target_0 = feature_series[target_series == 0].dropna()
    target_1 = feature_series[target_series == 1].dropna()
    
    if len(target_0) > 0 and len(target_1) > 0:
        stats_dict['Mean_Target0'] = target_0.mean()
        stats_dict['Mean_Target1'] = target_1.mean()
        stats_dict['Mean_Diff'] = target_1.mean() - target_0.mean()
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(((len(target_0) - 1) * target_0.std()**2 + (len(target_1) - 1) * target_1.std()**2) / (len(target_0) + len(target_1) - 2))
        stats_dict['Cohens_d'] = (target_1.mean() - target_0.mean()) / (pooled_std + 1e-10)
        
        # T-test
        try:
            t_stat, t_p = scipy_stats.ttest_ind(target_0, target_1)
            stats_dict['T_Statistic'] = t_stat
            stats_dict['T_P_Value'] = t_p
        except:
            stats_dict['T_Statistic'] = np.nan
            stats_dict['T_P_Value'] = np.nan
    
    return stats_dict

def assess_train_test_shift(feature_train, feature_test, feature_name):
    """Check distribution shift between train and test"""
    
    valid_train = feature_train.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    valid_test = feature_test.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(valid_train) < 10 or len(valid_test) < 10:
        return {'Feature': feature_name, 'KS_Statistic': np.nan, 'KS_P_Value': np.nan, 'Shift_Detected': 'INSUFFICIENT_DATA'}
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = ks_2samp(valid_train, valid_test)
    
    shift_detected = 'YES' if ks_p < 0.05 else 'NO'
    
    return {
        'Feature': feature_name,
        'KS_Statistic': ks_stat,
        'KS_P_Value': ks_p,
        'Shift_Detected': shift_detected,
        'Train_Mean': valid_train.mean(),
        'Test_Mean': valid_test.mean(),
        'Train_Std': valid_train.std(),
        'Test_Std': valid_test.std()
    }

def compute_quality_score(stats_row):
    """Compute overall quality score (0-100) for a feature"""
    
    score = 100
    issues = []
    
    # Penalty 1: Missing data
    if stats_row['Missing_Pct'] > 10:
        penalty = min(30, stats_row['Missing_Pct'] * 2)
        score -= penalty
        issues.append(f"High missing ({stats_row['Missing_Pct']:.1f}%)")
    
    # Penalty 2: Infinite values
    if stats_row.get('Inf_Count', 0) > 0:
        score -= 20
        issues.append(f"Infinite values ({stats_row['Inf_Count']})")
    
    # Penalty 3: Low variance (not informative)
    if stats_row['Std'] < 0.01:
        score -= 15
        issues.append("Near-zero variance")
    
    # Penalty 4: Extreme outliers
    if stats_row['Extreme_Outlier_Pct'] > 5:
        score -= 10
        issues.append(f"Extreme outliers ({stats_row['Extreme_Outlier_Pct']:.1f}%)")
    
    # Penalty 5: Weak correlation with target
    if abs(stats_row.get('Spearman_Corr', 0)) < 0.05:
        score -= 15
        issues.append("Weak target correlation")
    
    # Penalty 6: Single value dominance
    if stats_row['Unique_Pct'] < 5:
        score -= 10
        issues.append(f"Low diversity ({stats_row['Unique_Pct']:.1f}% unique)")
    
    # Penalty 7: Extreme skewness
    if abs(stats_row['Skewness']) > 3:
        score -= 5
        issues.append(f"Extreme skew ({stats_row['Skewness']:.2f})")
    
    return max(0, score), issues

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOAD DATA & CREATE FEATURES")
print("="*80)

df = load_data()
print(f"✓ Loaded {len(df)} samples")

# Create domain features
df_with_domain = create_domain_specific_features(df.drop('Arthritis', axis=1))
target = df['Arthritis']

# Encode target
le_target = LabelEncoder()
target_encoded = le_target.fit_transform(target)

# Get domain feature names
domain_features = [
    'OA_Age_Risk_Score', 'OA_Mechanical_Load', 'OA_Gender_Risk',
    'OA_Metabolic_Syndrome', 'OA_Inflammatory_Score', 'OA_Joint_Stress_Index',
    'OA_CV_Risk_Cluster', 'OA_Systemic_Burden', 'OA_Activity_Balance',
    'OA_Obesity_Severity', 'OA_Age_Gender_Interaction', 'OA_Sedentary_Load',
    'OA_Chronic_Disease_Load', 'OA_Lifestyle_Risk', 'OA_Biomechanical_Stress',
    'OA_Multi_System_Degradation'
]

print(f"✓ Created {len(domain_features)} domain-specific features")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df_with_domain[domain_features], 
    target_encoded, 
    test_size=0.2, 
    random_state=SEED, 
    stratify=target_encoded
)

print(f"✓ Split: Train={len(X_train)}, Test={len(X_test)}")

# ============================================================================
# STEP 2: COMPREHENSIVE FEATURE AUDIT
# ============================================================================

print("\n" + "="*80)
print("STEP 2: FEATURE-BY-FEATURE AUDIT")
print("="*80)

all_stats = []
shift_stats = []

for idx, feat in enumerate(domain_features, 1):
    print(f"\n[{idx}/16] Analyzing: {feat}")
    
    # Full dataset stats
    stats = compute_feature_stats(
        df_with_domain[feat], 
        feat, 
        target_encoded
    )
    all_stats.append(stats)
    
    # Train/test shift
    shift = assess_train_test_shift(
        X_train[feat],
        X_test[feat],
        feat
    )
    shift_stats.append(shift)
    
    # Print key findings
    print(f"   Missing: {stats['Missing_Pct']:.1f}%")
    print(f"   Mean: {stats['Mean']:.4f}, Std: {stats['Std']:.4f}")
    print(f"   Correlation (Spearman): {stats.get('Spearman_Corr', np.nan):.4f} (p={stats.get('Spearman_P', np.nan):.3e})")
    print(f"   Target separation (Cohen's d): {stats.get('Cohens_d', np.nan):.4f}")
    print(f"   Train/Test shift: {shift['Shift_Detected']}")

# Convert to DataFrames
stats_df = pd.DataFrame(all_stats)
shift_df = pd.DataFrame(shift_stats)

# ============================================================================
# STEP 3: COMPUTE QUALITY SCORES
# ============================================================================

print("\n" + "="*80)
print("STEP 3: QUALITY SCORING")
print("="*80)

quality_scores = []
for idx, row in stats_df.iterrows():
    score, issues = compute_quality_score(row)
    quality_scores.append({
        'Feature': row['Feature'],
        'Quality_Score': score,
        'Issues': '; '.join(issues) if issues else 'NONE',
        'Issue_Count': len(issues)
    })

quality_df = pd.DataFrame(quality_scores)
stats_df = stats_df.merge(quality_df, on='Feature')

# Sort by quality score
stats_df_sorted = stats_df.sort_values('Quality_Score', ascending=False)

print("\n🏆 QUALITY RANKING (Top 5):")
print("="*80)
for idx, row in stats_df_sorted.head(5).iterrows():
    print(f"{row['Feature']:35s} | Score: {row['Quality_Score']:3.0f} | Issues: {row['Issue_Count']}")

print("\n❌ WORST QUALITY (Bottom 5):")
print("="*80)
for idx, row in stats_df_sorted.tail(5).iterrows():
    print(f"{row['Feature']:35s} | Score: {row['Quality_Score']:3.0f} | Issues: {row['Issues']}")

# ============================================================================
# STEP 4: IDENTIFY RED FLAGS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: RED FLAG IDENTIFICATION")
print("="*80)

red_flags = []

# Flag 1: High missing rate
high_missing = stats_df[stats_df['Missing_Pct'] > 10]
if len(high_missing) > 0:
    red_flags.append(f"🚩 HIGH MISSING: {len(high_missing)} features > 10% missing")
    for feat in high_missing['Feature']:
        print(f"   • {feat}: {high_missing[high_missing['Feature']==feat]['Missing_Pct'].values[0]:.1f}%")

# Flag 2: Infinite values
inf_features = stats_df[stats_df['Inf_Count'] > 0]
if len(inf_features) > 0:
    red_flags.append(f"🚩 INFINITE VALUES: {len(inf_features)} features")
    for feat in inf_features['Feature']:
        print(f"   • {feat}: {inf_features[inf_features['Feature']==feat]['Inf_Count'].values[0]} inf values")

# Flag 3: Weak target correlation
weak_corr = stats_df[abs(stats_df['Spearman_Corr']) < 0.05]
if len(weak_corr) > 0:
    print(f"\n🚩 WEAK TARGET CORRELATION: {len(weak_corr)} features < 0.05")
    for feat in weak_corr['Feature']:
        corr_val = weak_corr[weak_corr['Feature']==feat]['Spearman_Corr'].values[0]
        print(f"   • {feat}: {corr_val:.4f}")

# Flag 4: Train/test distribution shift
shifted = shift_df[shift_df['Shift_Detected'] == 'YES']
if len(shifted) > 0:
    print(f"\n🚩 TRAIN/TEST SHIFT: {len(shifted)} features (KS test p < 0.05)")
    for feat in shifted['Feature']:
        print(f"   • {feat}")

# Flag 5: Near-zero variance
low_var = stats_df[stats_df['Std'] < 0.01]
if len(low_var) > 0:
    print(f"\n🚩 LOW VARIANCE: {len(low_var)} features (std < 0.01)")
    for feat in low_var['Feature']:
        print(f"   • {feat}: std={low_var[low_var['Feature']==feat]['Std'].values[0]:.6f}")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SAVE RESULTS")
print("="*80)

# Save comprehensive stats
stats_path = os.path.join(output_dir, 'feature_quality_stats.csv')
stats_df_sorted.to_csv(stats_path, index=False)
print(f"✓ Stats: {stats_path}")

# Save shift analysis
shift_path = os.path.join(output_dir, 'train_test_shift_analysis.csv')
shift_df.to_csv(shift_path, index=False)
print(f"✓ Shift analysis: {shift_path}")

# Create summary report
summary_report = []
summary_report.append("="*80)
summary_report.append("FEATURE QUALITY AUDIT - SUMMARY REPORT")
summary_report.append("="*80)
summary_report.append(f"\nTotal Features Analyzed: {len(domain_features)}")
summary_report.append(f"Average Quality Score: {stats_df['Quality_Score'].mean():.1f}/100")
summary_report.append(f"\nQuality Distribution:")
summary_report.append(f"  Excellent (>80): {len(stats_df[stats_df['Quality_Score'] > 80])}")
summary_report.append(f"  Good (60-80):    {len(stats_df[(stats_df['Quality_Score'] >= 60) & (stats_df['Quality_Score'] <= 80)])}")
summary_report.append(f"  Fair (40-60):    {len(stats_df[(stats_df['Quality_Score'] >= 40) & (stats_df['Quality_Score'] < 60)])}")
summary_report.append(f"  Poor (<40):      {len(stats_df[stats_df['Quality_Score'] < 40])}")
summary_report.append(f"\nRed Flags Identified: {len(red_flags)}")
for flag in red_flags:
    summary_report.append(f"  {flag}")

summary_report.append(f"\n{'='*80}")
summary_report.append("TOP 5 QUALITY FEATURES (Candidates for Stage 7 Group Testing)")
summary_report.append("="*80)
for i, (idx, row) in enumerate(stats_df_sorted.head(5).iterrows(), 1):
    summary_report.append(f"\n{i}. {row['Feature']}")
    summary_report.append(f"   Quality Score: {row['Quality_Score']:.0f}/100")
    summary_report.append(f"   Spearman Corr: {row['Spearman_Corr']:.4f} (p={row['Spearman_P']:.3e})")
    summary_report.append(f"   Cohen's d: {row.get('Cohens_d', np.nan):.4f}")
    summary_report.append(f"   Issues: {row['Issues']}")

summary_report.append(f"\n{'='*80}")
summary_report.append("BOTTOM 5 QUALITY FEATURES (Likely Harmful)")
summary_report.append("="*80)
for i, (idx, row) in enumerate(stats_df_sorted.tail(5).iterrows(), 1):
    summary_report.append(f"\n{i}. {row['Feature']}")
    summary_report.append(f"   Quality Score: {row['Quality_Score']:.0f}/100")
    summary_report.append(f"   Issues: {row['Issues']}")

summary_report.append(f"\n{'='*80}")
summary_report.append("RECOMMENDATIONS FOR STAGE 7 (Group-wise Ablation)")
summary_report.append("="*80)

# Group features by quality
excellent = stats_df_sorted[stats_df_sorted['Quality_Score'] > 80]['Feature'].tolist()
good = stats_df_sorted[(stats_df_sorted['Quality_Score'] >= 60) & (stats_df_sorted['Quality_Score'] <= 80)]['Feature'].tolist()
poor = stats_df_sorted[stats_df_sorted['Quality_Score'] < 40]['Feature'].tolist()

summary_report.append(f"\nGroup A (Excellent): {len(excellent)} features")
for feat in excellent:
    summary_report.append(f"  • {feat}")

summary_report.append(f"\nGroup B (Good): {len(good)} features")
for feat in good:
    summary_report.append(f"  • {feat}")

summary_report.append(f"\nGroup C (Poor - Consider Dropping): {len(poor)} features")
for feat in poor:
    summary_report.append(f"  • {feat}")

summary_report.append(f"\n{'='*80}")
summary_report.append("END OF AUDIT")
summary_report.append("="*80)

# Save summary
summary_path = os.path.join(output_dir, 'audit_summary_report.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_report))

print(f"✓ Summary report: {summary_path}")

print("\n" + "="*80)
print("✅ STAGE 6 COMPLETE - FEATURE QUALITY AUDIT")
print("="*80)
print(f"\n📊 VERDICT:")
print(f"   • {len(excellent)} features: EXCELLENT quality (>80)")
print(f"   • {len(good)} features: GOOD quality (60-80)")
print(f"   • {len(poor)} features: POOR quality (<40)")
print(f"\n➡️  NEXT: Stage 7 - Group-wise ablation testing with quality-sorted groups")
print("="*80)
