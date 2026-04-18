"""
Step 3: Robust Feature Engineering Evaluation
==============================================
Test feature engineering with multiple seeds on Raw Data Direct (most stable preprocessing)

CONTEXT:
- Stage 2 showed Raw Data Direct is most stable (1.61% CV)
- Stage 1 suggested FE hurts clean data, but only tested seed=42
- This stage validates FE impact with multiple seeds (42, 123, 999)

OBJECTIVE: 
Determine if feature engineering provides stable improvement when 
evaluated with proper reproducibility testing

Author: Bioinformatics Analysis Team
Date: March 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score, 
                             precision_score, recall_score, 
                             average_precision_score)
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ROBUST FEATURE ENGINEERING EVALUATION")
print("="*80)
print("\n🎯 OBJECTIVE: Test feature engineering with multiple seeds")
print("   Base Strategy: Raw Data Direct (most stable from Stage 2)")
print("   Seeds: 42, 123, 999")
print("   Comparison: Raw Features vs Engineered Features\n")
print("="*80)

# Configuration
SEEDS = [42, 123, 999]
raw_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step3_robust_feature_engineering"
os.makedirs(output_dir, exist_ok=True)

def engineer_features(df):
    """
    Create 37 new features based on clinical knowledge.
    Same function used in Stage 1 (step1_comprehensive_comparison.py)
    """
    df = df.copy()
    
    # 1. Biological Age Features (5 features)
    if 'Biological Age' in df.columns:
        df['BioAge_60_Plus'] = (df['Biological Age'] >= 60).astype(int)
        df['BioAge_70_Plus'] = (df['Biological Age'] >= 70).astype(int)
        df['BioAge_50_60'] = ((df['Biological Age'] >= 50) & (df['Biological Age'] < 60)).astype(int)
        df['Log_Biological_Age'] = np.log1p(df['Biological Age'].fillna(0))
        df['BioAge_Squared'] = df['Biological Age'].fillna(0) ** 2
    
    # 2. BMI Features (6 features)
    if 'bmi_kg.m2' in df.columns:
        df['BMI'] = df['bmi_kg.m2']
        df['BMI_Obese'] = (df['BMI'] >= 30).astype(int)
        df['BMI_Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
        df['BMI_Normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
        df['BMI_Underweight'] = (df['BMI'] < 18.5).astype(int)
        df['BMI_Squared'] = df['BMI'].fillna(0) ** 2
        
        df['BMI_Category'] = 1  # Default Normal
        df.loc[df['BMI'] < 18.5, 'BMI_Category'] = 0
        df.loc[df['BMI'] >= 25, 'BMI_Category'] = 2
        df.loc[df['BMI'] >= 30, 'BMI_Category'] = 3
    
    # 3. Comorbidity Count (1 feature)
    comorbidity_cols = ['Heart_Disease', 'Hypertension', 'Lung_Disease', 
                        'Diabetes', 'Cancer', 'Stroke']
    available_comorbidity = [col for col in comorbidity_cols if col in df.columns]
    if available_comorbidity:
        df['Comorbidity_Count'] = df[available_comorbidity].fillna(0).sum(axis=1)
    
    # 4. Metabolic Syndrome Features (3 features)
    if 'Hypertension' in df.columns and 'BMI_Obese' in df.columns:
        df['Metabolic_Syndrome_Risk'] = (
            df['Hypertension'].fillna(0) + 
            df['BMI_Obese'].fillna(0) + 
            df['Diabetes'].fillna(0)
        )
    
    if 'MET' in df.columns:
        df['Low_Physical_Activity'] = (df['MET'] < df['MET'].median()).astype(int)
        df['High_Physical_Activity'] = (df['MET'] > df['MET'].quantile(0.75)).astype(int)
    
    # 5. Gender-specific features (2 features)
    if 'Sex' in df.columns:
        df['Sex_Numeric'] = df['Sex'].map({'male': 1, 'female': 0, 'Male': 1, 'Female': 0}).fillna(0)
        if 'Biological Age' in df.columns:
            df['Female_PostMenopause'] = ((df['Sex_Numeric'] == 0) & 
                                          (df['Biological Age'] >= 50)).astype(int)
    
    # 6. Interaction Features (8 features)
    if 'Biological Age' in df.columns and 'BMI' in df.columns:
        df['Age_x_BMI'] = df['Biological Age'].fillna(0) * df['BMI'].fillna(0)
    
    if 'Biological Age' in df.columns and 'Comorbidity_Count' in df.columns:
        df['Age_x_Comorbidity'] = df['Biological Age'].fillna(0) * df['Comorbidity_Count']
    
    if 'BMI' in df.columns and 'Comorbidity_Count' in df.columns:
        df['BMI_x_Comorbidity'] = df['BMI'].fillna(0) * df['Comorbidity_Count']
    
    if 'Sex_Numeric' in df.columns and 'BMI' in df.columns:
        df['Sex_x_BMI'] = df['Sex_Numeric'] * df['BMI'].fillna(0)
    
    if 'MET' in df.columns and 'BMI' in df.columns:
        df['MET_x_BMI'] = df['MET'].fillna(0) * df['BMI'].fillna(0)
    
    if 'MET' in df.columns and 'Biological Age' in df.columns:
        df['MET_x_Age'] = df['MET'].fillna(0) * df['Biological Age'].fillna(0)
    
    if 'Comorbidity_Count' in df.columns and 'Low_Physical_Activity' in df.columns:
        df['Comorbidity_x_LowActivity'] = df['Comorbidity_Count'] * df['Low_Physical_Activity']
    
    if 'Sex_Numeric' in df.columns and 'Comorbidity_Count' in df.columns:
        df['Sex_x_Comorbidity'] = df['Sex_Numeric'] * df['Comorbidity_Count']
    
    # 7. Residence Level Features (2 features)
    if 'Residence Level' in df.columns:
        df['Residence_Level'] = df['Residence Level'].map({
            'Village': 0, 'District': 1, 'Province': 2, 'Capital': 3
        }).fillna(0)
        df['Urban'] = (df['Residence_Level'] >= 2).astype(int)
    
    # 8. Position Knees Encoding (1 feature)
    if 'position_knees' in df.columns:
        position_map = {'never': 0, 'rarely': 1, 'sometimes': 2, 
                        'often': 3, 'always': 4}
        df['Position_Knees_Encoded'] = df['position_knees'].map(position_map).fillna(-1)
    
    # 9. Age Risk Categories (3 features)
    if 'Biological Age' in df.columns:
        df['Age_Risk_Low'] = (df['Biological Age'] < 50).astype(int)
        df['Age_Risk_Medium'] = ((df['Biological Age'] >= 50) & 
                                 (df['Biological Age'] < 65)).astype(int)
        df['Age_Risk_High'] = (df['Biological Age'] >= 65).astype(int)
    
    # 10. Diabetes-BMI Interaction (1 feature)
    if 'Diabetes' in df.columns and 'BMI_Obese' in df.columns:
        df['Diabetes_Obese'] = (df['Diabetes'].fillna(0) * df['BMI_Obese'])
    
    # 11. Cardiovascular Risk Score (1 feature)
    cv_risk_cols = ['Heart_Disease', 'Hypertension', 'Stroke']
    available_cv = [col for col in cv_risk_cols if col in df.columns]
    if available_cv:
        df['CV_Risk_Score'] = df[available_cv].fillna(0).sum(axis=1)
    
    # 12. High-Risk Profile (1 feature)
    if all(col in df.columns for col in ['BioAge_60_Plus', 'BMI_Obese', 'Comorbidity_Count']):
        df['High_Risk_Profile'] = (
            (df['BioAge_60_Plus'] == 1) & 
            (df['BMI_Obese'] == 1) & 
            (df['Comorbidity_Count'] >= 2)
        ).astype(int)
    
    return df

def load_and_prepare_raw_data_direct(seed):
    """
    Load using Raw Data Direct strategy (most stable from Stage 2)
    Following Stage 1's preprocessing approach
    """
    # Load
    df_raw = pd.read_excel(raw_data_path)
    df = df_raw[df_raw['Arthritis'].notna()].copy()
    
    # Separate target
    y = df['Arthritis']
    X = df.drop('Arthritis', axis=1)
    
    # Encode categorical (string categories) as -999 for missing
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        # Fill missing with special marker
        X[col] = X[col].fillna('__MISSING__')
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode numeric missing as -999
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        X[col] = X[col].fillna(-999)
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    return X, y

def load_for_feature_engineering(seed):
    """
    Load data preserving original column names for feature engineering.
    Encode only after feature engineering is complete.
    """
    # Load
    df_raw = pd.read_excel(raw_data_path)
    df = df_raw[df_raw['Arthritis'].notna()].copy()
    
    return df

# Store results
all_results = []

# ============================================================================
# TEST 1: RAW FEATURES (Baseline from Stage 2)
# ============================================================================

print("\n" + "="*80)
print("TEST 1: RAW FEATURES (17 features)")
print("="*80)

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\n[Run {seed_idx}/3] Seed = {seed}")
    print("-" * 40)
    
    # Load data
    X, y = load_and_prepare_raw_data_direct(seed)
    feature_count = X.shape[1]
    
    print(f"   Samples: {len(X):,}, Features: {feature_count}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    
    # Train
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    elapsed = time.time() - start_time
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"   ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
    print(f"   F1: {f1:.4f} | Acc: {accuracy:.4f}")
    
    all_results.append({
        'Feature_Set': 'Raw_Features',
        'Seed': seed,
        'Samples': len(X),
        'Features': feature_count,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'F1_Score': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Train_Size': len(X_train),
        'Test_Size': len(X_test),
        'Time_sec': elapsed
    })

# Calculate statistics
print("\n" + "="*80)
print("RAW FEATURES SUMMARY (Mean ± Std)")
print("="*80)
raw_df = pd.DataFrame([r for r in all_results if r['Feature_Set'] == 'Raw_Features'])
metrics = ['ROC_AUC', 'PR_AUC', 'F1_Score', 'Accuracy', 'Precision', 'Recall']
for metric in metrics:
    mean = raw_df[metric].mean()
    std = raw_df[metric].std()
    print(f"   {metric:15s}: {mean:.4f} ± {std:.4f}")

# ============================================================================
# TEST 2: ENGINEERED FEATURES (37 additional = 54 total)
# ============================================================================

print("\n" + "="*80)
print("TEST 2: ENGINEERED FEATURES (37 additional = ~54 total)")
print("="*80)

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\n[Run {seed_idx}/3] Seed = {seed}")
    print("-" * 40)
    
    # Load data (before encoding, to preserve column names for FE)
    df = load_for_feature_engineering(seed)
    
    # Separate target
    y = df['Arthritis']
    df_features = df.drop('Arthritis', axis=1)
    
    # Apply feature engineering BEFORE encoding
    df_eng = engineer_features(df_features)
    
    # NOW encode everything
    # Categorical
    categorical_cols = df_eng.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df_eng[col] = df_eng[col].fillna('__MISSING__')
        df_eng[col] = le.fit_transform(df_eng[col].astype(str))
    
    # Numeric missing as -999
    numeric_cols = df_eng.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df_eng[col] = df_eng[col].fillna(-999)
    
    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    X_eng = df_eng
    feature_count = X_eng.shape[1]
    print(f"   Samples: {len(X_eng):,}, Features: {feature_count}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_eng, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded
    )
    
    # Train
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    elapsed = time.time() - start_time
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"   ROC AUC: {roc_auc:.4f} | PR AUC: {pr_auc:.4f}")
    print(f"   F1: {f1:.4f} | Acc: {accuracy:.4f}")
    
    all_results.append({
        'Feature_Set': 'Engineered_Features',
        'Seed': seed,
        'Samples': len(X_eng),
        'Features': feature_count,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'F1_Score': f1,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'Train_Size': len(X_train),
        'Test_Size': len(X_test),
        'Time_sec': elapsed
    })

# Calculate statistics
print("\n" + "="*80)
print("ENGINEERED FEATURES SUMMARY (Mean ± Std)")
print("="*80)
eng_df = pd.DataFrame([r for r in all_results if r['Feature_Set'] == 'Engineered_Features'])
for metric in metrics:
    mean = eng_df[metric].mean()
    std = eng_df[metric].std()
    print(f"   {metric:15s}: {mean:.4f} ± {std:.4f}")

# ============================================================================
# COMPARISON & STABILITY ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("COMPARATIVE ANALYSIS: Raw vs Engineered Features")
print("="*80)

all_results_df = pd.DataFrame(all_results)

# Save detailed results
detailed_path = os.path.join(output_dir, 'robust_feature_engineering_detailed.csv')
all_results_df.to_csv(detailed_path, index=False)
print(f"\n✓ Detailed results saved: {detailed_path}")

# Summary statistics
summary_data = []
for feature_set in ['Raw_Features', 'Engineered_Features']:
    subset = all_results_df[all_results_df['Feature_Set'] == feature_set]
    summary_data.append({
        'Feature_Set': feature_set,
        'Features': int(subset['Features'].mean()),
        'ROC_AUC_mean': subset['ROC_AUC'].mean(),
        'ROC_AUC_std': subset['ROC_AUC'].std(),
        'PR_AUC_mean': subset['PR_AUC'].mean(),
        'PR_AUC_std': subset['PR_AUC'].std(),
        'F1_mean': subset['F1_Score'].mean(),
        'F1_std': subset['F1_Score'].std(),
        'Accuracy_mean': subset['Accuracy'].mean(),
        'Accuracy_std': subset['Accuracy'].std()
    })

summary_df = pd.DataFrame(summary_data)

print("\n┌────────────────────────┬──────────┬─────────────────┬─────────────────┬──────────┐")
print("│ Feature Set            │ Features │ ROC AUC         │ PR AUC          │ CV (%)   │")
print("├────────────────────────┼──────────┼─────────────────┼─────────────────┼──────────┤")
for _, row in summary_df.iterrows():
    cv_roc = (row['ROC_AUC_std'] / row['ROC_AUC_mean']) * 100
    print(f"│ {row['Feature_Set']:22s} │ {row['Features']:8d} │ {row['ROC_AUC_mean']:.4f} ± {row['ROC_AUC_std']:.4f} │ "
          f"{row['PR_AUC_mean']:.4f} ± {row['PR_AUC_std']:.4f} │ {cv_roc:8.2f} │")
print("└────────────────────────┴──────────┴─────────────────┴─────────────────┴──────────┘")

# Calculate impact
raw_roc = summary_df[summary_df['Feature_Set'] == 'Raw_Features']['ROC_AUC_mean'].values[0]
eng_roc = summary_df[summary_df['Feature_Set'] == 'Engineered_Features']['ROC_AUC_mean'].values[0]
delta = eng_roc - raw_roc
pct_change = (delta / raw_roc) * 100

print(f"\n🎯 FEATURE ENGINEERING IMPACT:")
print(f"   Baseline (Raw): {raw_roc:.4f}")
print(f"   Engineered:     {eng_roc:.4f}")
print(f"   Delta:          {delta:+.4f} ({pct_change:+.2f}%)")

if delta > 0.005:
    verdict = "✅ BENEFICIAL - Feature engineering improves performance"
elif delta < -0.005:
    verdict = "❌ HARMFUL - Feature engineering degrades performance"
else:
    verdict = "⚖️ NEUTRAL - No significant impact"

print(f"   Verdict: {verdict}")

# Stability comparison
raw_cv = (summary_df[summary_df['Feature_Set'] == 'Raw_Features']['ROC_AUC_std'].values[0] / raw_roc) * 100
eng_cv = (summary_df[summary_df['Feature_Set'] == 'Engineered_Features']['ROC_AUC_std'].values[0] / eng_roc) * 100

print(f"\n📊 STABILITY COMPARISON:")
print(f"   Raw Features CV:        {raw_cv:.2f}%")
print(f"   Engineered Features CV: {eng_cv:.2f}%")

if raw_cv < eng_cv:
    stability_verdict = "Raw features more stable"
else:
    stability_verdict = "Engineered features more stable"
print(f"   Stability: {stability_verdict}")

# Save summary
summary_path = os.path.join(output_dir, 'robust_feature_engineering_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary saved: {summary_path}")

# ============================================================================
# FINAL RECOMMENDATION
# ============================================================================

print("\n" + "="*80)
print("FINAL RECOMMENDATION")
print("="*80)

if delta > 0.005 and eng_cv <= raw_cv * 1.5:
    recommendation = "Use ENGINEERED FEATURES"
    reason = "Significant performance gain with acceptable stability"
elif delta < -0.005:
    recommendation = "Use RAW FEATURES"
    reason = "Feature engineering degrades performance (confirmed with multiple seeds)"
elif raw_cv < eng_cv:
    recommendation = "Use RAW FEATURES"
    reason = "More stable and simpler model"
else:
    recommendation = "Use RAW FEATURES (default)"
    reason = "No clear benefit from feature engineering"

print(f"\n🏆 {recommendation}")
print(f"   Reason: {reason}")
print(f"\n   Expected Performance:")
if recommendation == "Use RAW FEATURES":
    print(f"   - ROC AUC: {raw_roc:.4f} ± {summary_df[summary_df['Feature_Set'] == 'Raw_Features']['ROC_AUC_std'].values[0]:.4f}")
    print(f"   - Features: {int(summary_df[summary_df['Feature_Set'] == 'Raw_Features']['Features'].values[0])}")
    print(f"   - CV: {raw_cv:.2f}% (stable)")
else:
    print(f"   - ROC AUC: {eng_roc:.4f} ± {summary_df[summary_df['Feature_Set'] == 'Engineered_Features']['ROC_AUC_std'].values[0]:.4f}")
    print(f"   - Features: {int(summary_df[summary_df['Feature_Set'] == 'Engineered_Features']['Features'].values[0])}")
    print(f"   - CV: {eng_cv:.2f}%")

print("\n" + "="*80)
print("✅ ROBUST FEATURE ENGINEERING EVALUATION COMPLETE!")
print("="*80)
