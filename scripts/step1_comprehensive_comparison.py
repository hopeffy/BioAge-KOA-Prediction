"""
Step 1: Comprehensive Preprocessing × Feature Engineering Comparison
=====================================================================
3 Preprocessing Strategies × Full Feature Engineering Pipeline

RESEARCH QUESTION: 
How does feature engineering performance vary across different 
missing value handling strategies?

PREPROCESSING STRATEGIES:
0. Raw Data Direct: Arthritis filter only, missing=-999, 18,046 samples
1. Listwise Deletion: Complete cases only, 2,742 samples
2. Mean Imputation: Imputed values, 9,505 samples

FEATURE ENGINEERING (Applied to all 3):
- 37 new features (aging, BMI, comorbidity, interactions)
- Total: ~55 features per dataset

MODELS TESTED:
- Random Forest (n_estimators=100)
- Feature importance analysis
- ANOVA F-test ranking

Author: Bioinformatics Analysis Team
Date: March 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE COMPARISON: 3 Preprocessing × Feature Engineering")
print("="*80)
print("\nRESEARCH QUESTION: Does feature engineering benefit differ by preprocessing?\n")
print("="*80)

# Paths
raw_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step1_comprehensive_comparison"
os.makedirs(output_dir, exist_ok=True)

results = []

# ============================================================================
# FEATURE ENGINEERING FUNCTION (Applied to all datasets)
# ============================================================================

def engineer_features(df):
    """
    Create 37 new features based on clinical knowledge.
    Same function applied to all preprocessing strategies.
    """
    df = df.copy()
    
    print("   [FE] Creating 37 new features...")
    
    # 1. Biological Age Features (5 features)
    if 'Biological Age' in df.columns:
        df['BioAge_60_Plus'] = (df['Biological Age'] >= 60).astype(int)
        df['BioAge_70_Plus'] = (df['Biological Age'] >= 70).astype(int)
        df['BioAge_50_60'] = ((df['Biological Age'] >= 50) & (df['Biological Age'] < 60)).astype(int)
        # Handle missing values in log transform
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
        
        # BMI Category (0=Underweight, 1=Normal, 2=Overweight, 3=Obese)
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
    
    # 12. Physical Activity Status (1 feature - already done in metabolic section)
    # 13. High-Risk Profile (1 feature)
    if all(col in df.columns for col in ['BioAge_60_Plus', 'BMI_Obese', 'Comorbidity_Count']):
        df['High_Risk_Profile'] = (
            (df['BioAge_60_Plus'] == 1) & 
            (df['BMI_Obese'] == 1) & 
            (df['Comorbidity_Count'] >= 2)
        ).astype(int)
    
    initial_features = len([col for col in df.columns if col != 'Arthritis'])
    print(f"   [FE] ✓ Feature engineering complete: {initial_features} total features")
    
    return df

# ============================================================================
# PREPROCESSING STRATEGY 0: RAW DATA DIRECT
# ============================================================================

print("\n" + "="*80)
print("PREPROCESSING 0: RAW DATA DIRECT")
print("="*80)

print("[1/5] Loading Raw Data.xlsx...")
df_raw = pd.read_excel(raw_data_path)
print(f"   ✓ Loaded: {df_raw.shape[0]:,} rows")

print("[2/5] Filtering Arthritis available only...")
df0 = df_raw[df_raw['Arthritis'].notna()].copy()
print(f"   ✓ Filtered: {df0.shape[0]:,} rows with target")

print("[3/5] Encoding missing values as -999...")
# Prepare features first
X0_raw = df0.drop('Arthritis', axis=1)
y0 = df0['Arthritis']

# Encode categorical
categorical_cols = X0_raw.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X0_raw[col] = X0_raw[col].fillna('__MISSING__')
    X0_raw[col] = le.fit_transform(X0_raw[col].astype(str))

# Encode numeric missing
numeric_cols = X0_raw.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    X0_raw[col] = X0_raw[col].fillna(-999)

# Reconstruct dataframe for feature engineering
df0_processed = X0_raw.copy()
df0_processed['Arthritis'] = y0.values
print(f"   ✓ Missing values encoded")

print("[4/5] Applying feature engineering...")
df0_features = engineer_features(df0_processed)

print("[5/5] Training Random Forest...")
X0 = df0_features.drop('Arthritis', axis=1)
y0 = df0_features['Arthritis']

# Encode target
le_target = LabelEncoder()
y0 = le_target.fit_transform(y0)

# Train/test split
X0_train, X0_test, y0_train, y0_test = train_test_split(
    X0, y0, test_size=0.2, random_state=42, stratify=y0
)

start_time = time.time()
rf0 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf0.fit(X0_train, y0_train)
y0_pred = rf0.predict(X0_test)
y0_pred_proba = rf0.predict_proba(X0_test)[:, 1]
elapsed0 = time.time() - start_time

# Metrics
roc_auc0 = roc_auc_score(y0_test, y0_pred_proba)
f1_0 = f1_score(y0_test, y0_pred)
acc0 = accuracy_score(y0_test, y0_pred)
prec0 = precision_score(y0_test, y0_pred)
rec0 = recall_score(y0_test, y0_pred)

print(f"\n   📊 Results (Raw Data Direct + Feature Engineering):")
print(f"      ROC AUC: {roc_auc0:.4f}")
print(f"      F1 Score: {f1_0:.4f}")
print(f"      Accuracy: {acc0:.4f}")
print(f"      Features: {X0.shape[1]}")
print(f"      Train size: {len(X0_train):,}")
print(f"      Test size: {len(X0_test):,}")

results.append({
    'Preprocessing': '0_Raw_Data_Direct',
    'Samples': len(df0_features),
    'Features': X0.shape[1],
    'ROC_AUC': roc_auc0,
    'F1_Score': f1_0,
    'Accuracy': acc0,
    'Precision': prec0,
    'Recall': rec0,
    'Train_Size': len(X0_train),
    'Test_Size': len(X0_test),
    'Time_sec': elapsed0
})

# ============================================================================
# PREPROCESSING STRATEGY 1: LISTWISE DELETION
# ============================================================================

print("\n" + "="*80)
print("PREPROCESSING 1: LISTWISE DELETION")
print("="*80)

print("[1/4] Loading and filtering Arthritis...")
df1 = df_raw[df_raw['Arthritis'].notna()].copy()
print(f"   ✓ Target available: {df1.shape[0]:,} rows")

print("[2/4] Applying listwise deletion...")
df1_clean = df1.dropna()
print(f"   ✓ After dropna: {df1_clean.shape[0]:,} rows")
print(f"   ✓ Data loss: {len(df1) - len(df1_clean):,} rows ({((len(df1) - len(df1_clean))/len(df1))*100:.1f}%)")

print("[3/4] Applying feature engineering...")
df1_features = engineer_features(df1_clean)

print("[4/4] Training Random Forest...")
X1 = df1_features.drop('Arthritis', axis=1)
y1 = df1_features['Arthritis']

# Encode categorical
categorical_cols = X1.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X1[col] = le.fit_transform(X1[col].astype(str))

# Encode target
le_target = LabelEncoder()
y1 = le_target.fit_transform(y1)

# Train/test split
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42, stratify=y1
)

start_time = time.time()
rf1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf1.fit(X1_train, y1_train)
y1_pred = rf1.predict(X1_test)
y1_pred_proba = rf1.predict_proba(X1_test)[:, 1]
elapsed1 = time.time() - start_time

# Metrics
roc_auc1 = roc_auc_score(y1_test, y1_pred_proba)
f1_1 = f1_score(y1_test, y1_pred)
acc1 = accuracy_score(y1_test, y1_pred)
prec1 = precision_score(y1_test, y1_pred)
rec1 = recall_score(y1_test, y1_pred)

print(f"\n   📊 Results (Listwise Deletion + Feature Engineering):")
print(f"      ROC AUC: {roc_auc1:.4f}")
print(f"      F1 Score: {f1_1:.4f}")
print(f"      Accuracy: {acc1:.4f}")
print(f"      Features: {X1.shape[1]}")
print(f"      Train size: {len(X1_train):,}")
print(f"      Test size: {len(X1_test):,}")

results.append({
    'Preprocessing': '1_Listwise_Deletion',
    'Samples': len(df1_features),
    'Features': X1.shape[1],
    'ROC_AUC': roc_auc1,
    'F1_Score': f1_1,
    'Accuracy': acc1,
    'Precision': prec1,
    'Recall': rec1,
    'Train_Size': len(X1_train),
    'Test_Size': len(X1_test),
    'Time_sec': elapsed1
})

# ============================================================================
# PREPROCESSING STRATEGY 2: MEAN IMPUTATION
# ============================================================================

print("\n" + "="*80)
print("PREPROCESSING 2: MEAN IMPUTATION")
print("="*80)

print("[1/4] Loading and filtering Arthritis...")
df2 = df_raw[df_raw['Arthritis'].notna()].copy()
print(f"   ✓ Target available: {df2.shape[0]:,} rows")

print("[2/4] Applying mean/mode imputation...")
X2_raw = df2.drop('Arthritis', axis=1)
y2 = df2['Arthritis']

# Separate numeric and categorical
numeric_cols = X2_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X2_raw.select_dtypes(include=['object']).columns.tolist()

# Impute numeric with mean
if numeric_cols:
    imputer_num = SimpleImputer(strategy='mean')
    X2_raw[numeric_cols] = imputer_num.fit_transform(X2_raw[numeric_cols])

# Impute categorical with most_frequent
if categorical_cols:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X2_raw[categorical_cols] = imputer_cat.fit_transform(X2_raw[categorical_cols])

df2_imputed = X2_raw.copy()
df2_imputed['Arthritis'] = y2.values
print(f"   ✓ Imputation complete: 0 missing values")

print("[3/4] Applying feature engineering...")
df2_features = engineer_features(df2_imputed)

print("[4/4] Training Random Forest...")
X2 = df2_features.drop('Arthritis', axis=1)
y2 = df2_features['Arthritis']

# Encode categorical
categorical_cols = X2.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X2[col] = le.fit_transform(X2[col].astype(str))

# Encode target
le_target = LabelEncoder()
y2 = le_target.fit_transform(y2)

# Train/test split
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42, stratify=y2
)

start_time = time.time()
rf2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf2.fit(X2_train, y2_train)
y2_pred = rf2.predict(X2_test)
y2_pred_proba = rf2.predict_proba(X2_test)[:, 1]
elapsed2 = time.time() - start_time

# Metrics
roc_auc2 = roc_auc_score(y2_test, y2_pred_proba)
f1_2 = f1_score(y2_test, y2_pred)
acc2 = accuracy_score(y2_test, y2_pred)
prec2 = precision_score(y2_test, y2_pred)
rec2 = recall_score(y2_test, y2_pred)

print(f"\n   📊 Results (Mean Imputation + Feature Engineering):")
print(f"      ROC AUC: {roc_auc2:.4f}")
print(f"      F1 Score: {f1_2:.4f}")
print(f"      Accuracy: {acc2:.4f}")
print(f"      Features: {X2.shape[1]}")
print(f"      Train size: {len(X2_train):,}")
print(f"      Test size: {len(X2_test):,}")

results.append({
    'Preprocessing': '2_Mean_Imputation',
    'Samples': len(df2_features),
    'Features': X2.shape[1],
    'ROC_AUC': roc_auc2,
    'F1_Score': f1_2,
    'Accuracy': acc2,
    'Precision': prec2,
    'Recall': rec2,
    'Train_Size': len(X2_train),
    'Test_Size': len(X2_test),
    'Time_sec': elapsed2
})

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("COMPARATIVE RESULTS: Preprocessing × Feature Engineering")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC_AUC', ascending=False)
print("\n" + results_df.to_string(index=False))

# Save results
results_path = os.path.join(output_dir, 'comprehensive_comparison_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\n✓ Results saved: {results_path}")

# ============================================================================
# KEY FINDINGS
# ============================================================================

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

best = results_df.iloc[0]
print(f"\n🏆 Best Method: {best['Preprocessing']}")
print(f"   • ROC AUC: {best['ROC_AUC']:.4f}")
print(f"   • F1 Score: {best['F1_Score']:.4f}")
print(f"   • Samples: {best['Samples']:,.0f}")
print(f"   • Features: {best['Features']:.0f}")

print("\n📊 Performance Ranking:")
for idx, row in results_df.iterrows():
    print(f"\n   {row['Preprocessing']}:")
    print(f"      ROC AUC: {row['ROC_AUC']:.4f} | F1: {row['F1_Score']:.4f}")
    print(f"      Samples: {row['Samples']:,.0f} | Features: {row['Features']:.0f}")

# Feature Engineering Impact Analysis
print("\n" + "="*80)
print("FEATURE ENGINEERING IMPACT (vs Stage 0 Baseline)")
print("="*80)

baseline_results = {
    '0_Raw_Data_Direct': 0.6767,
    '1_Listwise_Deletion': 0.7262,
    '2_Mean_Imputation': 0.6249
}

print("\nBaseline (Raw Features) vs FE (Engineered Features):")
for idx, row in results_df.iterrows():
    method = row['Preprocessing']
    if method in baseline_results:
        baseline_auc = baseline_results[method]
        fe_auc = row['ROC_AUC']
        delta = fe_auc - baseline_auc
        pct_change = (delta / baseline_auc) * 100
        
        symbol = "📈" if delta > 0 else "📉"
        print(f"\n   {method}:")
        print(f"      Baseline: {baseline_auc:.4f} → FE: {fe_auc:.4f}")
        print(f"      {symbol} Change: {delta:+.4f} ({pct_change:+.1f}%)")

print("\n" + "="*80)
print("✅ COMPREHENSIVE COMPARISON COMPLETE!")
print("="*80)
print(f"\nResults saved to: {output_dir}")
print("Next: Analyze which preprocessing benefits most from feature engineering")
