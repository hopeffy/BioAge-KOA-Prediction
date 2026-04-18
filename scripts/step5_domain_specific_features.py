"""
Step 5: Domain-Specific Medical Feature Engineering
===================================================
CREATE OSTEOARTHRITIS-SPECIFIC FEATURES BASED ON MEDICAL LITERATURE

CONTEXT:
- Stage 4 showed: Generic FE hurts performance (-1.17%)
- Generic features: squared, log, interactions (no medical meaning)
- NOW: Create 16 MEDICALLY-INFORMED features based on OA risk factors

OBJECTIVE:
Achieve +5% performance boost (0.6655 → 0.6988) using domain knowledge

MEDICAL LITERATURE REFERENCES:
1. Age + BMI = Primary mechanical load factor
2. Gender-specific risk (females > males, especially post-menopause)
3. Metabolic syndrome = Inflammatory pathway
4. Physical activity balance = Joint health
5. Comorbidity burden = System deterioration
6. Cardiovascular disease cluster = Systemic inflammation

NEW FEATURES (16 Domain-Specific):
1. OA_Age_Risk_Score - Age-weighted risk (exponential after 50)
2. OA_Mechanical_Load - BMI × Age interaction (joint stress)
3. OA_Gender_Risk - Female-specific risk multiplier
4. OA_Metabolic_Syndrome - DM + HTN + Obesity cluster
5. OA_Inflammatory_Score - Comorbidity inflammatory burden
6. OA_Joint_Stress_Index - BMI × Low_Activity (sedentary obesity)
7. OA_CV_Risk_Cluster - Heart disease + HTN + Stroke
8. OA_Systemic_Burden - Total disease count weighted by severity
9. OA_Activity_Balance - MET vs Age-BMI balance
10. OA_Obesity_Severity - BMI categories with exponential weight
11. OA_Age_Gender_Interaction - Female × Age (menopause effect)
12. OA_Sedentary_Load - Low activity × High BMI × Age
13. OA_Chronic_Disease_Load - Long-term conditions weighted
14. OA_Lifestyle_Risk - Combined lifestyle factors
15. OA_Biomechanical_Stress - Age × BMI × Gender
16. OA_Multi_System_Degradation - Age × Comorbidities × BMI

Author: Bioinformatics Analysis Team - Medical Domain Expert Review
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
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STAGE 5: DOMAIN-SPECIFIC MEDICAL FEATURE ENGINEERING")
print("="*80)
print("\n🎯 OBJECTIVE: Achieve +5% performance using OA-specific features")
print("   Current: 0.6655 ROC AUC")
print("   Target:  0.6988 ROC AUC (+0.033)")
print("\n📚 APPROACH: Evidence-based medical features from OA literature")
print("   16 domain-specific features replacing 37 generic features\n")
print("="*80)

# Configuration
SEEDS = [42, 123, 999]
raw_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step5_domain_specific_features"
os.makedirs(output_dir, exist_ok=True)

def create_domain_specific_features(df):
    """
    Create 16 medically-informed features for osteoarthritis prediction
    Based on published OA risk factors and pathophysiology
    """
    df = df.copy()
    
    # Pre-compute commonly used values
    age = df['Biological Age'].fillna(df['Biological Age'].median())
    bmi = df['bmi_kg.m2'].fillna(df['bmi_kg.m2'].median())
    
    # Gender encoding
    sex_col = 'sex' if 'sex' in df.columns else 'Sex'
    sex_numeric = df[sex_col].map({'male': 1, 'female': 0, 'Male': 1, 'Female': 0}).fillna(0.5)
    is_female = (sex_numeric == 0).astype(int)
    
    # Helper function to convert yes/no to 0/1
    def binary_encode(series):
        """Convert yes/no/1/0 to numeric"""
        if series.dtype == 'object' or series.dtype == 'string':
            # Convert to lowercase string first
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
    
    # Comorbidities - check both naming conventions and encode
    heart_disease = binary_encode(df['doctor_diagnose_heart_disease']) if 'doctor_diagnose_heart_disease' in df.columns else binary_encode(df.get('Heart_Disease', pd.Series(0, index=df.index)))
    hypertension = binary_encode(df['doctor_diagnose_hypertension']) if 'doctor_diagnose_hypertension' in df.columns else binary_encode(df.get('Hypertension', pd.Series(0, index=df.index)))
    diabetes = binary_encode(df['doctor_diagnose_diabetes']) if 'doctor_diagnose_diabetes' in df.columns else binary_encode(df.get('Diabetes', pd.Series(0, index=df.index)))
    stroke = binary_encode(df.get('doctor_diagnose_stroke', df.get('Stroke', pd.Series(0, index=df.index))))
    lung_disease = binary_encode(df.get('doctor_diagnose_lung_disease', df.get('Lung_Disease', pd.Series(0, index=df.index))))
    cancer = binary_encode(df['doctor_diagnose_cancer']) if 'doctor_diagnose_cancer' in df.columns else binary_encode(df.get('Cancer', pd.Series(0, index=df.index)))
    
    # Physical activity
    met = df['MET'].fillna(df['MET'].median())
    
    print("Creating 16 domain-specific features...")
    
    # ========================================================================
    # FEATURE 1: OA_Age_Risk_Score
    # Age is THE primary risk factor - exponential increase after 50
    # ========================================================================
    df['OA_Age_Risk_Score'] = np.where(
        age < 50, 
        (age / 50),  # Linear until 50
        1 + ((age - 50) / 10) ** 1.5  # Exponential after 50
    )
    
    # ========================================================================
    # FEATURE 2: OA_Mechanical_Load
    # BMI × Age = Cumulative joint stress over lifetime
    # Higher BMI + older age = exponentially higher risk
    # ========================================================================
    df['OA_Mechanical_Load'] = (bmi / 25) * (age / 60)  # Normalized to reference
    
    # ========================================================================
    # FEATURE 3: OA_Gender_Risk
    # Females have 2-3x higher risk, especially post-menopause (age > 50)
    # ========================================================================
    df['OA_Gender_Risk'] = np.where(
        is_female == 1,
        np.where(age >= 50, 2.5, 1.5),  # Post-menopause multiplier
        1.0  # Baseline male risk
    )
    
    # ========================================================================
    # FEATURE 4: OA_Metabolic_Syndrome
    # DM + HTN + Obesity = Inflammatory cascade
    # ========================================================================
    is_obese = (bmi >= 30).astype(int)
    df['OA_Metabolic_Syndrome'] = (
        diabetes * 2 +  # DM strongest inflammatory marker
        hypertension * 1.5 +
        is_obese * 1.5
    ) / 5.0  # Normalize to 0-1
    
    # ========================================================================
    # FEATURE 5: OA_Inflammatory_Score
    # Comorbidities = chronic inflammation = cartilage degradation
    # ========================================================================
    df['OA_Inflammatory_Score'] = (
        heart_disease * 2 +  # CV disease = high inflammation
        hypertension * 1.5 +
        diabetes * 2 +
        stroke * 2.5 +  # Stroke = severe vascular inflammation
        lung_disease * 1.5 +
        cancer * 1  # Cancer weaker association
    ) / 10.0  # Normalize
    
    # ========================================================================
    # FEATURE 6: OA_Joint_Stress_Index
    # High BMI + Low activity = Maximum joint stress without muscle support
    # ========================================================================
    low_activity = (met < met.quantile(0.33)).astype(int)
    df['OA_Joint_Stress_Index'] = (bmi / 25) * (1 + low_activity * 2)
    
    # ========================================================================
    # FEATURE 7: OA_CV_Risk_Cluster
    # Cardiovascular cluster = systemic inflammation
    # ========================================================================
    df['OA_CV_Risk_Cluster'] = (heart_disease + hypertension + stroke) / 3.0
    
    # ========================================================================
    # FEATURE 8: OA_Systemic_Burden
    # Total disease count weighted by severity
    # ========================================================================
    df['OA_Systemic_Burden'] = (
        heart_disease * 2 +
        stroke * 2.5 +
        diabetes * 2 +
        cancer * 1.5 +
        hypertension * 1 +
        lung_disease * 1
    ) / 10.0
    
    # ========================================================================
    # FEATURE 9: OA_Activity_Balance
    # Physical activity vs mechanical load balance
    # High MET + reasonable BMI = protective
    # Low MET + high BMI = risk
    # ========================================================================
    activity_ratio = met / met.quantile(0.75)  # Normalize to 75th percentile
    bmi_ratio = bmi / 30  # Normalize to obesity threshold
    df['OA_Activity_Balance'] = bmi_ratio / (activity_ratio + 0.1)  # Avoid div by 0
    
    # ========================================================================
    # FEATURE 10: OA_Obesity_Severity
    # BMI categories with exponential weighting
    # ========================================================================
    df['OA_Obesity_Severity'] = np.where(
        bmi < 25, 0,  # Normal
        np.where(bmi < 30, 1, np.where(bmi < 35, 2, 3))  # Overweight, Obese, Severe
    )
    
    # ========================================================================
    # FEATURE 11: OA_Age_Gender_Interaction
    # Female × Age interaction (menopause effect)
    # ========================================================================
    df['OA_Age_Gender_Interaction'] = is_female * (age / 60) * np.where(age >= 50, 2, 1)
    
    # ========================================================================
    # FEATURE 12: OA_Sedentary_Load
    # Low activity × High BMI × Age = Triple threat
    # ========================================================================
    df['OA_Sedentary_Load'] = low_activity * (bmi / 25) * (age / 60)
    
    # ========================================================================
    # FEATURE 13: OA_Chronic_Disease_Load
    # Long-term conditions = cumulative damage
    # ========================================================================
    chronic_conditions = heart_disease + hypertension + diabetes + lung_disease
    df['OA_Chronic_Disease_Load'] = (chronic_conditions / 4.0) * (age / 60)
    
    # ========================================================================
    # FEATURE 14: OA_Lifestyle_Risk
    # Combined lifestyle factors (BMI, activity, age)
    # ========================================================================
    df['OA_Lifestyle_Risk'] = (
        (bmi / 35) * 0.4 +  # Obesity component
        (1 - activity_ratio) * 0.3 +  # Inactivity component
        (age / 80) * 0.3  # Age component
    )
    
    # ========================================================================
    # FEATURE 15: OA_Biomechanical_Stress
    # Age × BMI × Gender (females have different biomechanics)
    # ========================================================================
    df['OA_Biomechanical_Stress'] = (age / 60) * (bmi / 25) * (1 + is_female * 0.5)
    
    # ========================================================================
    # FEATURE 16: OA_Multi_System_Degradation
    # Age × Comorbidities × BMI = Multi-system failure cascade
    # ========================================================================
    total_comorbidities = heart_disease + hypertension + diabetes + stroke + lung_disease + cancer
    df['OA_Multi_System_Degradation'] = (age / 60) * (total_comorbidities / 6) * (bmi / 25)
    
    print(f"✓ Created 16 domain-specific features")
    
    # List all new features
    new_features = [
        'OA_Age_Risk_Score', 'OA_Mechanical_Load', 'OA_Gender_Risk',
        'OA_Metabolic_Syndrome', 'OA_Inflammatory_Score', 'OA_Joint_Stress_Index',
        'OA_CV_Risk_Cluster', 'OA_Systemic_Burden', 'OA_Activity_Balance',
        'OA_Obesity_Severity', 'OA_Age_Gender_Interaction', 'OA_Sedentary_Load',
        'OA_Chronic_Disease_Load', 'OA_Lifestyle_Risk', 'OA_Biomechanical_Stress',
        'OA_Multi_System_Degradation'
    ]
    
    print("\nFeature Summary:")
    for i, feat in enumerate(new_features, 1):
        print(f"  {i:2d}. {feat}")
    
    return df, new_features

def load_and_encode_data():
    """Load raw data, encode, return X and y"""
    df_raw = pd.read_excel(raw_data_path)
    df = df_raw[df_raw['Arthritis'].notna()].copy()
    
    y = df['Arthritis']
    X = df.drop('Arthritis', axis=1)
    
    # Encode categorical
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('__MISSING__')
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode numeric missing
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        X[col] = X[col].fillna(-999)
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    return X, y

def load_and_prepare_with_domain_features():
    """Load data, create domain features, return df"""
    df_raw = pd.read_excel(raw_data_path)
    df = df_raw[df_raw['Arthritis'].notna()].copy()
    return df

def encode_dataframe(df):
    """Encode after feature engineering"""
    # Categorical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna('__MISSING__')
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(-999)
    
    return df

def train_and_evaluate(X_train, X_test, y_train, y_test, seed):
    """Train RF and return metrics"""
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    return {
        'ROC_AUC': roc_auc_score(y_test, y_pred_proba),
        'PR_AUC': average_precision_score(y_test, y_pred_proba),
        'F1_Score': f1_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }

# Store all results
all_results = []

# ============================================================================
# TEST 1: BASELINE (Raw Features - 17)
# ============================================================================

print("\n" + "="*80)
print("TEST 1: BASELINE - RAW FEATURES (17 features)")
print("="*80)

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    X, y = load_and_encode_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
    
    all_results.append({
        'Method': 'Baseline_Raw',
        'Seed': seed,
        'Features': X.shape[1],
        **metrics
    })

# ============================================================================
# TEST 2: RAW + DOMAIN-SPECIFIC FEATURES (17 + 16 = 33)
# ============================================================================

print("\n" + "="*80)
print("TEST 2: RAW + DOMAIN-SPECIFIC FEATURES (33 features)")
print("="*80)

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    df = load_and_prepare_with_domain_features()
    y = df['Arthritis']
    df_features = df.drop('Arthritis', axis=1)
    
    # Create domain-specific features
    df_domain, new_feat_names = create_domain_specific_features(df_features)
    
    # Encode everything
    df_domain = encode_dataframe(df_domain)
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(df_domain, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
    
    all_results.append({
        'Method': 'Domain_Specific',
        'Seed': seed,
        'Features': df_domain.shape[1],
        **metrics
    })

# ============================================================================
# TEST 3: ONLY DOMAIN-SPECIFIC FEATURES (16 features)
# ============================================================================

print("\n" + "="*80)
print("TEST 3: ONLY DOMAIN-SPECIFIC FEATURES (16 features)")
print("="*80)

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    df = load_and_prepare_with_domain_features()
    y = df['Arthritis']
    df_features = df.drop('Arthritis', axis=1)
    
    # Create domain-specific features
    df_domain, new_feat_names = create_domain_specific_features(df_features)
    
    # Keep ONLY new features
    df_domain_only = df_domain[new_feat_names].copy()
    df_domain_only = encode_dataframe(df_domain_only)
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(df_domain_only, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
    
    all_results.append({
        'Method': 'Domain_Only',
        'Seed': seed,
        'Features': df_domain_only.shape[1],
        **metrics
    })

# ============================================================================
# ANALYSIS & SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS")
print("="*80)

all_results_df = pd.DataFrame(all_results)

# Save detailed
detailed_path = os.path.join(output_dir, 'domain_specific_features_detailed.csv')
all_results_df.to_csv(detailed_path, index=False)
print(f"\n✓ Detailed results: {detailed_path}")

# Summary by method
summary_data = []
for method in all_results_df['Method'].unique():
    subset = all_results_df[all_results_df['Method'] == method]
    summary_data.append({
        'Method': method,
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
summary_df['CV_ROC'] = (summary_df['ROC_AUC_std'] / summary_df['ROC_AUC_mean']) * 100
summary_df = summary_df.sort_values('ROC_AUC_mean', ascending=False)

print("\n" + "="*80)
print("RANKING BY PERFORMANCE (Mean ROC AUC)")
print("="*80)
print("\n┌────────────────────────────┬────────┬─────────────────┬─────────────────┬─────────┐")
print("│ Method                     │ Feats  │ ROC AUC         │ F1 Score        │ CV (%)  │")
print("├────────────────────────────┼────────┼─────────────────┼─────────────────┼─────────┤")
for idx, row in summary_df.iterrows():
    print(f"│ {row['Method']:26s} │ {row['Features']:6d} │ {row['ROC_AUC_mean']:.4f} ± {row['ROC_AUC_std']:.4f} │ "
          f"{row['F1_mean']:.4f} ± {row['F1_std']:.4f} │ {row['CV_ROC']:7.2f} │")
print("└────────────────────────────┴────────┴─────────────────┴─────────────────┴─────────┘")

# Save summary
summary_path = os.path.join(output_dir, 'domain_specific_features_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary saved: {summary_path}")

# Calculate improvement
baseline_roc = summary_df[summary_df['Method'] == 'Baseline_Raw']['ROC_AUC_mean'].values[0]
best_roc = summary_df.iloc[0]['ROC_AUC_mean']
improvement = ((best_roc - baseline_roc) / baseline_roc) * 100

print("\n" + "="*80)
print("🎯 PERFORMANCE IMPROVEMENT")
print("="*80)
print(f"\nBaseline (Raw):            {baseline_roc:.4f}")
print(f"Best Method:               {best_roc:.4f}")
print(f"Improvement:               {improvement:+.2f}%")
print(f"Target (+5%):              {baseline_roc * 1.05:.4f}")
print(f"Target Achieved:           {'✅ YES!' if improvement >= 5 else '❌ Not yet'}")

print("\n" + "="*80)
print("✅ DOMAIN-SPECIFIC FEATURE ENGINEERING COMPLETE!")
print("="*80)
