"""
Step 7: Group-wise Ablation Testing
====================================
Incremental feature addition to find optimal subset from Stage 6 audit

MOTIVATION:
Stage 6 identified:
- 8 STRONG features (Spearman > 0.07)
- 8 WEAK features (Spearman < 0.05)

HYPOTHESIS:
If we add ONLY strong features → Performance improves
If we add weak features → Performance degrades (proof they're harmful)

TEST STRATEGY:
1. Baseline: Raw features only (17) → 0.6655 ROC AUC
2. Raw + Top 3 strong
3. Raw + Top 5 strong
4. Raw + All 8 strong
5. Raw + 8 weak features (control - should fail)
6. Raw + each strong feature individually (1-by-1)

SEEDS: 3 seeds (42, 123, 999) for reproducibility

Author: Bioinformatics Analysis Team
Date: March 2026
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score
)
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STAGE 7: GROUP-WISE ABLATION TESTING")
print("="*80)
print("\n🎯 OBJECTIVE: Find optimal feature subset from Stage 6 audit")
print("   → Test strong features (Spearman > 0.07)")
print("   → Test weak features (Spearman < 0.05)")
print("   → Prove: Strong features help, weak features harm\n")
print("="*80)

# Configuration
SEEDS = [42, 123, 999]
raw_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step7_groupwise_ablation"
os.makedirs(output_dir, exist_ok=True)

# Feature groups from Stage 6 audit
STRONG_FEATURES = [
    'OA_Age_Gender_Interaction',      # 0.0931 - BEST
    'OA_Biomechanical_Stress',        # 0.0871
    'OA_Gender_Risk',                 # 0.0844
    'OA_Chronic_Disease_Load',        # 0.0786
    'OA_CV_Risk_Cluster',             # 0.0782
    'OA_Inflammatory_Score',          # 0.0773
    'OA_Multi_System_Degradation',    # 0.0769
    'OA_Systemic_Burden'              # 0.0768
]

WEAK_FEATURES = [
    'OA_Age_Risk_Score',              # 0.0476
    'OA_Metabolic_Syndrome',          # 0.0443
    'OA_Mechanical_Load',             # 0.0388
    'OA_Obesity_Severity',            # 0.0379
    'OA_Joint_Stress_Index',          # 0.0052
    'OA_Sedentary_Load',              # 0.0010
    'OA_Lifestyle_Risk',              # -0.0014
    'OA_Activity_Balance'             # -0.0172
]

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
    """Create 16 domain-specific features (same as Stage 5/6)"""
    df = df.copy()
    
    # Pre-compute values
    age = df['Biological Age'].fillna(df['Biological Age'].median())
    bmi = df['bmi_kg.m2'].fillna(df['bmi_kg.m2'].median())
    
    # Handle sex column
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

def load_and_encode_data():
    """Load data with Raw Data Direct preprocessing"""
    df_raw = pd.read_excel(raw_data_path)
    df = df_raw[df_raw['Arthritis'].notna()].copy()
    
    # Save target
    target = df['Arthritis'].copy()
    df = df.drop('Arthritis', axis=1)
    
    # Get raw columns BEFORE any modifications
    raw_features_list = df.columns.tolist()
    
    # Create domain features (BEFORE encoding - function handles its own mappings)
    df_with_domain = create_domain_specific_features(df)
    
    # NOW encode ALL columns (raw + domain features)
    le = LabelEncoder()
    for col in df_with_domain.columns:
        if df_with_domain[col].dtype == 'object' or pd.api.types.is_string_dtype(df_with_domain[col]):
            df_with_domain[col] = le.fit_transform(df_with_domain[col].astype(str))
    
    # Encode target
    le_target = LabelEncoder()
    target_encoded = le_target.fit_transform(target)
    
    return df_with_domain, target_encoded, raw_features_list

def train_and_evaluate(X_train, X_test, y_train, y_test, seed):
    """Train RF and compute metrics"""
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
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
print("STEP 1: LOAD DATA & PREPARE FEATURES")
print("="*80)

df_with_domain, target_encoded, raw_features = load_and_encode_data()
print(f"✓ Loaded {len(df_with_domain)} samples")
print(f"✓ Raw features: {len(raw_features)}")
print(f"✓ Strong domain features: {len(STRONG_FEATURES)}")
print(f"✓ Weak domain features: {len(WEAK_FEATURES)}")

# ============================================================================
# STEP 2: GROUP-WISE ABLATION EXPERIMENTS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: GROUP-WISE ABLATION EXPERIMENTS")
print("="*80)

all_results = []

# Define test configurations
test_configs = [
    {
        'name': 'Baseline_Raw',
        'description': 'Raw features only (17)',
        'features': raw_features,
        'group': 'Baseline'
    },
    {
        'name': 'Raw_Plus_Top3_Strong',
        'description': 'Raw + Top 3 strong',
        'features': raw_features + STRONG_FEATURES[:3],
        'group': 'Strong_Incremental'
    },
    {
        'name': 'Raw_Plus_Top5_Strong',
        'description': 'Raw + Top 5 strong',
        'features': raw_features + STRONG_FEATURES[:5],
        'group': 'Strong_Incremental'
    },
    {
        'name': 'Raw_Plus_All8_Strong',
        'description': 'Raw + All 8 strong',
        'features': raw_features + STRONG_FEATURES,
        'group': 'Strong_Full'
    },
    {
        'name': 'Raw_Plus_All8_Weak',
        'description': 'Raw + All 8 weak (control)',
        'features': raw_features + WEAK_FEATURES,
        'group': 'Weak_Control'
    }
]

# Add individual strong feature tests
for i, feat in enumerate(STRONG_FEATURES, 1):
    test_configs.append({
        'name': f'Raw_Plus_{feat}',
        'description': f'Raw + {feat} only',
        'features': raw_features + [feat],
        'group': 'Individual_Strong'
    })

print(f"\nTotal configurations: {len(test_configs)}")
print(f"Total runs: {len(test_configs) * len(SEEDS)}\n")

# Run experiments
for idx, config in enumerate(test_configs, 1):
    config_name = config['name']
    features = config['features']
    
    print(f"\n[{idx}/{len(test_configs)}] Testing: {config_name}")
    print(f"   Description: {config['description']}")
    print(f"   Features: {len(features)}")
    
    for seed in SEEDS:
        # Prepare data
        X = df_with_domain[features]
        y = target_encoded
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        
        # Train and evaluate
        metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
        
        # Store results
        result = {
            'Configuration': config_name,
            'Description': config['description'],
            'Group': config['group'],
            'Num_Features': len(features),
            'Seed': seed,
            **metrics
        }
        all_results.append(result)
    
    # Print current average
    config_results = [r for r in all_results if r['Configuration'] == config_name]
    avg_roc = np.mean([r['ROC_AUC'] for r in config_results])
    std_roc = np.std([r['ROC_AUC'] for r in config_results])
    avg_f1 = np.mean([r['F1_Score'] for r in config_results])
    
    print(f"   → ROC AUC: {avg_roc:.4f} ± {std_roc:.4f}")
    print(f"   → F1: {avg_f1:.4f}")

# Convert to DataFrame
results_df = pd.DataFrame(all_results)

# ============================================================================
# STEP 3: COMPUTE SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("STEP 3: SUMMARY & ANALYSIS")
print("="*80)

# Group-level summary
summary_stats = []
for config_name in results_df['Configuration'].unique():
    config_data = results_df[results_df['Configuration'] == config_name]
    
    summary = {
        'Configuration': config_name,
        'Description': config_data['Description'].iloc[0],
        'Group': config_data['Group'].iloc[0],
        'Num_Features': config_data['Num_Features'].iloc[0],
        'ROC_AUC_Mean': config_data['ROC_AUC'].mean(),
        'ROC_AUC_Std': config_data['ROC_AUC'].std(),
        'ROC_AUC_CV_Pct': (config_data['ROC_AUC'].std() / config_data['ROC_AUC'].mean()) * 100,
        'F1_Mean': config_data['F1_Score'].mean(),
        'F1_Std': config_data['F1_Score'].std(),
        'PR_AUC_Mean': config_data['PR_AUC'].mean(),
        'Accuracy_Mean': config_data['Accuracy'].mean()
    }
    summary_stats.append(summary)

summary_df = pd.DataFrame(summary_stats)

# Compute delta from baseline
baseline_roc = summary_df[summary_df['Configuration'] == 'Baseline_Raw']['ROC_AUC_Mean'].values[0]
summary_df['Delta_from_Baseline_Pct'] = ((summary_df['ROC_AUC_Mean'] - baseline_roc) / baseline_roc) * 100

# Sort by ROC AUC
summary_df_sorted = summary_df.sort_values('ROC_AUC_Mean', ascending=False)

print("\n🏆 RANKING BY ROC AUC:")
print("="*80)
for idx, row in summary_df_sorted.iterrows():
    delta_symbol = "✅" if row['Delta_from_Baseline_Pct'] > 0 else "❌" if row['Delta_from_Baseline_Pct'] < -0.5 else "➖"
    print(f"{delta_symbol} {row['Configuration']:40s} | "
          f"ROC: {row['ROC_AUC_Mean']:.4f} ± {row['ROC_AUC_Std']:.4f} | "
          f"Δ: {row['Delta_from_Baseline_Pct']:+.2f}%")

# ============================================================================
# STEP 4: KEY INSIGHTS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: KEY INSIGHTS")
print("="*80)

# Find best configuration
best_config = summary_df_sorted.iloc[0]
print(f"\n✨ BEST CONFIGURATION:")
print(f"   Name: {best_config['Configuration']}")
print(f"   ROC AUC: {best_config['ROC_AUC_Mean']:.4f} ± {best_config['ROC_AUC_Std']:.4f}")
print(f"   Improvement: {best_config['Delta_from_Baseline_Pct']:+.2f}%")
print(f"   Features: {best_config['Num_Features']}")

# Compare groups
print("\n📊 GROUP COMPARISON:")
for group in summary_df['Group'].unique():
    group_data = summary_df[summary_df['Group'] == group]
    avg_roc = group_data['ROC_AUC_Mean'].mean()
    avg_delta = group_data['Delta_from_Baseline_Pct'].mean()
    print(f"   {group:25s}: ROC {avg_roc:.4f}, Avg Δ {avg_delta:+.2f}%")

# Weak features performance
weak_config = summary_df[summary_df['Configuration'] == 'Raw_Plus_All8_Weak']
if len(weak_config) > 0:
    weak_delta = weak_config['Delta_from_Baseline_Pct'].values[0]
    print(f"\n⚠️  WEAK FEATURES (Control):")
    print(f"   Raw + All 8 weak features: {weak_delta:+.2f}%")
    if weak_delta < 0:
        print(f"   ✅ PROOF: Weak features ARE harmful (as predicted)")
    else:
        print(f"   ⚠️  UNEXPECTED: Weak features not harmful (investigate)")

# Strong features performance
strong_configs = summary_df[summary_df['Group'].str.contains('Strong')]
if len(strong_configs) > 0:
    best_strong = strong_configs.loc[strong_configs['ROC_AUC_Mean'].idxmax()]
    print(f"\n✅ STRONG FEATURES:")
    print(f"   Best: {best_strong['Configuration']}")
    print(f"   ROC: {best_strong['ROC_AUC_Mean']:.4f}")
    print(f"   Improvement: {best_strong['Delta_from_Baseline_Pct']:+.2f}%")

# Individual feature analysis
individual_configs = summary_df[summary_df['Group'] == 'Individual_Strong'].sort_values('ROC_AUC_Mean', ascending=False)
if len(individual_configs) > 0:
    print(f"\n🔍 BEST INDIVIDUAL FEATURES (Top 3):")
    for idx, row in individual_configs.head(3).iterrows():
        feat_name = row['Configuration'].replace('Raw_Plus_', '')
        print(f"   {feat_name:35s}: {row['ROC_AUC_Mean']:.4f} ({row['Delta_from_Baseline_Pct']:+.2f}%)")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: SAVE RESULTS")
print("="*80)

# Save detailed results
detailed_path = os.path.join(output_dir, 'groupwise_ablation_detailed.csv')
results_df.to_csv(detailed_path, index=False)
print(f"✓ Detailed: {detailed_path}")

# Save summary
summary_path = os.path.join(output_dir, 'groupwise_ablation_summary.csv')
summary_df_sorted.to_csv(summary_path, index=False)
print(f"✓ Summary: {summary_path}")

# Create text report
report_lines = []
report_lines.append("="*80)
report_lines.append("STAGE 7: GROUP-WISE ABLATION - RESULTS SUMMARY")
report_lines.append("="*80)
report_lines.append(f"\nBaseline (Raw features): {baseline_roc:.4f} ROC AUC")
report_lines.append(f"\nBest Configuration: {best_config['Configuration']}")
report_lines.append(f"  ROC AUC: {best_config['ROC_AUC_Mean']:.4f} ± {best_config['ROC_AUC_Std']:.4f}")
report_lines.append(f"  Improvement: {best_config['Delta_from_Baseline_Pct']:+.2f}%")
report_lines.append(f"  Features: {best_config['Num_Features']}")
report_lines.append("\n" + "="*80)
report_lines.append("FULL RANKING:")
report_lines.append("="*80)
for idx, row in summary_df_sorted.iterrows():
    report_lines.append(f"{row['Configuration']:40s} | "
                       f"ROC: {row['ROC_AUC_Mean']:.4f} ± {row['ROC_AUC_Std']:.4f} | "
                       f"Δ: {row['Delta_from_Baseline_Pct']:+.2f}%")

report_path = os.path.join(output_dir, 'ablation_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
print(f"✓ Report: {report_path}")

print("\n" + "="*80)
print("✅ STAGE 7 COMPLETE - GROUP-WISE ABLATION")
print("="*80)
print(f"\n📈 VERDICT:")
print(f"   Best improvement: {best_config['Delta_from_Baseline_Pct']:+.2f}%")
if best_config['Delta_from_Baseline_Pct'] > 0:
    print(f"   ✅ SUCCESS: Found feature combination that improves baseline!")
else:
    print(f"   ❌ FAILURE: No feature combination beats baseline")
print(f"\n➡️  NEXT: Stage 8 - Repeated CV validation on best configuration")
print("="*80)
