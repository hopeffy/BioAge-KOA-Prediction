"""
Step 8: Repeated Stratified K-Fold Cross-Validation
====================================================
Robust validation of Stage 7 winner: Raw + OA_Multi_System_Degradation

MOTIVATION:
Stage 7 showed +0.22% improvement with single train/test split (3 seeds)
Need robust validation with Repeated K-Fold CV to confirm:
1. Improvement is real (not lucky split)
2. Performance is stable across folds
3. No overfitting

METHODOLOGY:
- 5-fold Stratified K-Fold
- 5 repetitions (different random splits)
- Total: 5 × 5 = 25 evaluations per configuration

CONFIGURATIONS TESTED:
1. Baseline: Raw features (17)
2. Winner: Raw + OA_Multi_System_Degradation (18)

Author: Bioinformatics Analysis Team
Date: March 2026
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score
)
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STAGE 8: REPEATED STRATIFIED K-FOLD CROSS-VALIDATION")
print("="*80)
print("\n🎯 OBJECTIVE: Robust validation of Stage 7 winner")
print("   → Test Baseline vs Winner with Repeated K-Fold CV")
print("   → Confirm +0.22% improvement is real, not lucky split\n")
print("="*80)

# Configuration
N_SPLITS = 5
N_REPEATS = 5
TOTAL_FOLDS = N_SPLITS * N_REPEATS
raw_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step8_repeated_cv"
os.makedirs(output_dir, exist_ok=True)

def binary_encode(series):
    """Convert yes/no to 0/1"""
    if series.dtype == 'object' or pd.api.types.is_string_dtype(series):
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

def create_oa_multi_system_degradation(df):
    """Create the winning feature from Stage 7"""
    df = df.copy()
    
    age = df['Biological Age'].fillna(df['Biological Age'].median())
    bmi = df['bmi_kg.m2'].fillna(df['bmi_kg.m2'].median())
    
    # Comorbidities
    heart_disease = binary_encode(df['doctor_diagnose_heart_disease']) if 'doctor_diagnose_heart_disease' in df.columns else binary_encode(df.get('Heart_Disease', pd.Series(0, index=df.index)))
    hypertension = binary_encode(df['doctor_diagnose_hypertension']) if 'doctor_diagnose_hypertension' in df.columns else binary_encode(df.get('Hypertension', pd.Series(0, index=df.index)))
    diabetes = binary_encode(df['doctor_diagnose_diabetes']) if 'doctor_diagnose_diabetes' in df.columns else binary_encode(df.get('Diabetes', pd.Series(0, index=df.index)))
    stroke = binary_encode(df.get('doctor_diagnose_stroke', df.get('Stroke', pd.Series(0, index=df.index))))
    lung_disease = binary_encode(df.get('doctor_diagnose_lung_disease', df.get('Lung_Disease', pd.Series(0, index=df.index))))
    cancer = binary_encode(df['doctor_diagnose_cancer']) if 'doctor_diagnose_cancer' in df.columns else binary_encode(df.get('Cancer', pd.Series(0, index=df.index)))
    
    total_comorbidities = heart_disease + hypertension + diabetes + stroke + lung_disease + cancer
    
    # Formula: (Age/60) × (Comorbidities/6) × (BMI/25)
    df['OA_Multi_System_Degradation'] = (age / 60) * (total_comorbidities / 6) * (bmi / 25)
    
    return df

def load_and_prepare_data():
    """Load data and prepare both configurations"""
    df_raw = pd.read_excel(raw_data_path)
    df = df_raw[df_raw['Arthritis'].notna()].copy()
    
    # Save target
    target = df['Arthritis'].copy()
    df = df.drop('Arthritis', axis=1)
    
    # Get raw column names before creating feature
    raw_columns = df.columns.tolist()
    
    # Create the winning feature
    df_with_feature = create_oa_multi_system_degradation(df)
    
    # Encode all columns
    le = LabelEncoder()
    for col in df_with_feature.columns:
        if df_with_feature[col].dtype == 'object' or pd.api.types.is_string_dtype(df_with_feature[col]):
            df_with_feature[col] = le.fit_transform(df_with_feature[col].astype(str))
    
    # Encode target
    le_target = LabelEncoder()
    target_encoded = le_target.fit_transform(target)
    
    return df_with_feature, target_encoded, raw_columns

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train RF and compute metrics"""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'ROC_AUC': roc_auc_score(y_test, y_proba),
        'PR_AUC': average_precision_score(y_test, y_proba),
        'F1_Score': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0)
    }
    
    return metrics

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

print("\n" + "="*80)
print("STEP 1: LOAD DATA & PREPARE CONFIGURATIONS")
print("="*80)

df_with_feature, target_encoded, raw_columns = load_and_prepare_data()

print(f"✓ Loaded {len(df_with_feature)} samples")
print(f"✓ Raw features: {len(raw_columns)}")
print(f"✓ Winner features: {len(raw_columns) + 1} (raw + OA_Multi_System_Degradation)")

# Define configurations
configs = {
    'Baseline_Raw': raw_columns,
    'Winner_Raw_Plus_MultiSystem': raw_columns + ['OA_Multi_System_Degradation']
}

print(f"\n✓ Configurations: {len(configs)}")
print(f"✓ CV Strategy: {N_SPLITS}-fold × {N_REPEATS} repeats = {TOTAL_FOLDS} evaluations per config")

# ============================================================================
# STEP 2: REPEATED STRATIFIED K-FOLD CV
# ============================================================================

print("\n" + "="*80)
print("STEP 2: REPEATED STRATIFIED K-FOLD CROSS-VALIDATION")
print("="*80)

all_results = []

rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)

for config_name, feature_cols in configs.items():
    print(f"\n[{list(configs.keys()).index(config_name)+1}/{len(configs)}] Testing: {config_name}")
    print(f"   Features: {len(feature_cols)}")
    
    X = df_with_feature[feature_cols]
    y = target_encoded
    
    fold_idx = 0
    for train_idx, test_idx in rskf.split(X, y):
        fold_idx += 1
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        metrics = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        result = {
            'Configuration': config_name,
            'Fold': fold_idx,
            'Num_Features': len(feature_cols),
            **metrics
        }
        all_results.append(result)
        
        if fold_idx % 5 == 0:
            avg_roc = np.mean([r['ROC_AUC'] for r in all_results if r['Configuration'] == config_name])
            print(f"   Fold {fold_idx}/{TOTAL_FOLDS}: Avg ROC AUC = {avg_roc:.4f}")

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# ============================================================================
# STEP 3: STATISTICAL ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: STATISTICAL ANALYSIS")
print("="*80)

summary_stats = []
for config_name in results_df['Configuration'].unique():
    config_data = results_df[results_df['Configuration'] == config_name]
    
    summary = {
        'Configuration': config_name,
        'Num_Features': config_data['Num_Features'].iloc[0],
        'ROC_AUC_Mean': config_data['ROC_AUC'].mean(),
        'ROC_AUC_Std': config_data['ROC_AUC'].std(),
        'ROC_AUC_Min': config_data['ROC_AUC'].min(),
        'ROC_AUC_Max': config_data['ROC_AUC'].max(),
        'ROC_AUC_CV_Pct': (config_data['ROC_AUC'].std() / config_data['ROC_AUC'].mean()) * 100,
        'F1_Mean': config_data['F1_Score'].mean(),
        'F1_Std': config_data['F1_Score'].std(),
        'PR_AUC_Mean': config_data['PR_AUC'].mean(),
        'Accuracy_Mean': config_data['Accuracy'].mean(),
        'Precision_Mean': config_data['Precision'].mean(),
        'Recall_Mean': config_data['Recall'].mean()
    }
    summary_stats.append(summary)

summary_df = pd.DataFrame(summary_stats)

# Compute improvement
baseline_row = summary_df[summary_df['Configuration'] == 'Baseline_Raw'].iloc[0]
winner_row = summary_df[summary_df['Configuration'] == 'Winner_Raw_Plus_MultiSystem'].iloc[0]

baseline_roc = baseline_row['ROC_AUC_Mean']
winner_roc = winner_row['ROC_AUC_Mean']
improvement_pct = ((winner_roc - baseline_roc) / baseline_roc) * 100
improvement_abs = winner_roc - baseline_roc

print(f"\n📊 RESULTS:")
print("="*80)
print(f"\nBaseline (Raw 17 features):")
print(f"  ROC AUC: {baseline_roc:.4f} ± {baseline_row['ROC_AUC_Std']:.4f}")
print(f"  CV%: {baseline_row['ROC_AUC_CV_Pct']:.2f}%")
print(f"  Range: [{baseline_row['ROC_AUC_Min']:.4f}, {baseline_row['ROC_AUC_Max']:.4f}]")
print(f"  F1: {baseline_row['F1_Mean']:.4f} ± {baseline_row['F1_Std']:.4f}")

print(f"\nWinner (Raw +1 feature = 18 total):")
print(f"  ROC AUC: {winner_roc:.4f} ± {winner_row['ROC_AUC_Std']:.4f}")
print(f"  CV%: {winner_row['ROC_AUC_CV_Pct']:.2f}%")
print(f"  Range: [{winner_row['ROC_AUC_Min']:.4f}, {winner_row['ROC_AUC_Max']:.4f}]")
print(f"  F1: {winner_row['F1_Mean']:.4f} ± {winner_row['F1_Std']:.4f}")

print(f"\n🎯 IMPROVEMENT:")
print(f"  Absolute: {improvement_abs:+.4f} ROC AUC points")
print(f"  Relative: {improvement_pct:+.2f}%")

# Statistical significance test (paired t-test)
from scipy import stats as scipy_stats

baseline_scores = results_df[results_df['Configuration'] == 'Baseline_Raw']['ROC_AUC'].values
winner_scores = results_df[results_df['Configuration'] == 'Winner_Raw_Plus_MultiSystem']['ROC_AUC'].values

t_stat, p_value = scipy_stats.ttest_rel(winner_scores, baseline_scores)

print(f"\n📈 STATISTICAL SIGNIFICANCE (Paired t-test):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
if p_value < 0.05:
    print(f"  ✅ SIGNIFICANT (p < 0.05) - Improvement is statistically reliable")
else:
    print(f"  ⚠️  NOT SIGNIFICANT (p ≥ 0.05) - Improvement may be due to chance")

# Effect size (Cohen's d)
pooled_std = np.sqrt((baseline_row['ROC_AUC_Std']**2 + winner_row['ROC_AUC_Std']**2) / 2)
cohens_d = (winner_roc - baseline_roc) / pooled_std

print(f"\n📏 EFFECT SIZE (Cohen's d):")
print(f"  d = {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_interp = "Very small"
elif abs(cohens_d) < 0.5:
    effect_interp = "Small"
elif abs(cohens_d) < 0.8:
    effect_interp = "Medium"
else:
    effect_interp = "Large"
print(f"  Interpretation: {effect_interp} effect")

# ============================================================================
# STEP 4: STABILITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: STABILITY ANALYSIS")
print("="*80)

print(f"\nCoefficient of Variation (lower = more stable):")
print(f"  Baseline: {baseline_row['ROC_AUC_CV_Pct']:.2f}%")
print(f"  Winner:   {winner_row['ROC_AUC_CV_Pct']:.2f}%")

if winner_row['ROC_AUC_CV_Pct'] < baseline_row['ROC_AUC_CV_Pct']:
    print(f"  ✅ Winner is MORE stable")
else:
    print(f"  ⚠️  Winner is LESS stable")

print(f"\nFold-to-fold consistency:")
print(f"  Baseline wins: {sum(1 for i in range(TOTAL_FOLDS) if baseline_scores[i] > winner_scores[i])} / {TOTAL_FOLDS}")
print(f"  Winner wins:   {sum(1 for i in range(TOTAL_FOLDS) if winner_scores[i] > baseline_scores[i])} / {TOTAL_FOLDS}")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SAVE RESULTS")
print("="*80)

# Save detailed results
detailed_path = os.path.join(output_dir, 'repeated_cv_detailed.csv')
results_df.to_csv(detailed_path, index=False)
print(f"✓ Detailed: {detailed_path}")

# Save summary
summary_path = os.path.join(output_dir, 'repeated_cv_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"✓ Summary: {summary_path}")

# Create report
report_lines = []
report_lines.append("="*80)
report_lines.append("STAGE 8: REPEATED STRATIFIED K-FOLD CV - VALIDATION REPORT")
report_lines.append("="*80)
report_lines.append(f"\nMethodology: {N_SPLITS}-fold × {N_REPEATS} repeats = {TOTAL_FOLDS} evaluations")
report_lines.append(f"\nBaseline (17 features): {baseline_roc:.4f} ± {baseline_row['ROC_AUC_Std']:.4f}")
report_lines.append(f"Winner (18 features):   {winner_roc:.4f} ± {winner_row['ROC_AUC_Std']:.4f}")
report_lines.append(f"\nImprovement: {improvement_abs:+.4f} ({improvement_pct:+.2f}%)")
report_lines.append(f"P-value: {p_value:.4f} {'(SIGNIFICANT)' if p_value < 0.05 else '(not significant)'}")
report_lines.append(f"Cohen's d: {cohens_d:.4f} ({effect_interp} effect)")
report_lines.append(f"\nWinner feature: OA_Multi_System_Degradation")
report_lines.append(f"Formula: (Age/60) × (Comorbidities/6) × (BMI/25)")
report_lines.append(f"\nConclusion:")
if p_value < 0.05 and improvement_pct > 0:
    report_lines.append("✅ Winner configuration validated with statistical significance")
    report_lines.append("   Recommended for production deployment")
else:
    report_lines.append("⚠️  Improvement not statistically significant")
    report_lines.append("   Consider using baseline or further optimization")
report_lines.append("="*80)

report_path = os.path.join(output_dir, 'validation_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"✓ Report: {report_path}")

print("\n" + "="*80)
print("✅ STAGE 8 COMPLETE - REPEATED CV VALIDATION")
print("="*80)

if p_value < 0.05 and improvement_pct > 0:
    print(f"\n🎉 VALIDATED: +{improvement_pct:.2f}% improvement is statistically significant!")
    print(f"   Winner configuration ready for production")
else:
    print(f"\n⚠️  NOT VALIDATED: Improvement not statistically reliable")
    print(f"   May be due to random chance")

print("\n" + "="*80)
