"""
Step 4: Comprehensive Feature Selection Methods Comparison
==========================================================
Test multiple feature selection/engineering approaches with multiple seeds

CONTEXT:
- Stage 3 showed: FE hurts performance (-1.17%)
- But we only tested ONE FE approach (37 features)
- Previous experiments (Step 3-4) tested different selection methods on different preprocessing
- NOW: Test ALL feature approaches on Raw Data Direct (most stable) with multiple seeds

OBJECTIVE: 
Find optimal feature set that balances:
1. Performance (ROC AUC, F1)
2. Stability (low CV across seeds)
3. Simplicity (fewer features)

METHODS TO TEST:
1. Raw Features (17) - baseline
2. All Engineered Features (38) - full FE
3. ANOVA Top-K (statistical significance)
4. Hierarchical Clustering (5 representative)
5. L1 LASSO Selection (embedded)
6. Tree-based Importance (RF)
7. Mutual Information (information gain)

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
from sklearn.feature_selection import (SelectKBest, f_classif, 
                                       mutual_info_classif, RFE)
from sklearn.linear_model import LassoCV
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import time
import os
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE FEATURE SELECTION COMPARISON")
print("="*80)
print("\n🎯 OBJECTIVE: Find optimal feature set across multiple approaches")
print("   Base: Raw Data Direct (most stable preprocessing)")
print("   Seeds: 42, 123, 999")
print("   Methods: 7 different feature selection/engineering approaches\n")
print("="*80)

# Configuration
SEEDS = [42, 123, 999]
raw_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step4_comprehensive_feature_selection"
os.makedirs(output_dir, exist_ok=True)

def engineer_features(df):
    """Apply comprehensive feature engineering (38 features total)"""
    df = df.copy()
    
    # Biological Age Features
    if 'Biological Age' in df.columns:
        df['BioAge_60_Plus'] = (df['Biological Age'] >= 60).astype(int)
        df['BioAge_70_Plus'] = (df['Biological Age'] >= 70).astype(int)
        df['BioAge_50_60'] = ((df['Biological Age'] >= 50) & (df['Biological Age'] < 60)).astype(int)
        df['Log_Biological_Age'] = np.log1p(df['Biological Age'].fillna(0))
        df['BioAge_Squared'] = df['Biological Age'].fillna(0) ** 2
    
    # BMI Features
    if 'bmi_kg.m2' in df.columns:
        df['BMI'] = df['bmi_kg.m2']
        df['BMI_Obese'] = (df['BMI'] >= 30).astype(int)
        df['BMI_Overweight'] = ((df['BMI'] >= 25) & (df['BMI'] < 30)).astype(int)
        df['BMI_Normal'] = ((df['BMI'] >= 18.5) & (df['BMI'] < 25)).astype(int)
        df['BMI_Underweight'] = (df['BMI'] < 18.5).astype(int)
        df['BMI_Squared'] = df['BMI'].fillna(0) ** 2
        df['BMI_Category'] = 1
        df.loc[df['BMI'] < 18.5, 'BMI_Category'] = 0
        df.loc[df['BMI'] >= 25, 'BMI_Category'] = 2
        df.loc[df['BMI'] >= 30, 'BMI_Category'] = 3
    
    # Comorbidity
    comorbidity_cols = ['Heart_Disease', 'Hypertension', 'Lung_Disease', 
                        'Diabetes', 'Cancer', 'Stroke']
    available = [col for col in comorbidity_cols if col in df.columns]
    if available:
        df['Comorbidity_Count'] = df[available].fillna(0).sum(axis=1)
    
    # Metabolic
    if 'Hypertension' in df.columns and 'BMI_Obese' in df.columns:
        df['Metabolic_Syndrome_Risk'] = (
            df['Hypertension'].fillna(0) + df['BMI_Obese'].fillna(0) + df['Diabetes'].fillna(0)
        )
    
    if 'MET' in df.columns:
        df['Low_Physical_Activity'] = (df['MET'] < df['MET'].median()).astype(int)
        df['High_Physical_Activity'] = (df['MET'] > df['MET'].quantile(0.75)).astype(int)
    
    # Gender
    if 'Sex' in df.columns:
        df['Sex_Numeric'] = df['Sex'].map({'male': 1, 'female': 0, 'Male': 1, 'Female': 0}).fillna(0)
        if 'Biological Age' in df.columns:
            df['Female_PostMenopause'] = ((df['Sex_Numeric'] == 0) & (df['Biological Age'] >= 50)).astype(int)
    
    # Interactions
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
    
    # Residence
    if 'Residence Level' in df.columns:
        df['Residence_Level'] = df['Residence Level'].map({
            'Village': 0, 'District': 1, 'Province': 2, 'Capital': 3
        }).fillna(0)
        df['Urban'] = (df['Residence_Level'] >= 2).astype(int)
    
    # Position Knees
    if 'position_knees' in df.columns:
        position_map = {'never': 0, 'rarely': 1, 'sometimes': 2, 'often': 3, 'always': 4}
        df['Position_Knees_Encoded'] = df['position_knees'].map(position_map).fillna(-1)
    
    # Age Risk
    if 'Biological Age' in df.columns:
        df['Age_Risk_Low'] = (df['Biological Age'] < 50).astype(int)
        df['Age_Risk_Medium'] = ((df['Biological Age'] >= 50) & (df['Biological Age'] < 65)).astype(int)
        df['Age_Risk_High'] = (df['Biological Age'] >= 65).astype(int)
    
    # Diabetes-BMI
    if 'Diabetes' in df.columns and 'BMI_Obese' in df.columns:
        df['Diabetes_Obese'] = df['Diabetes'].fillna(0) * df['BMI_Obese']
    
    # CV Risk
    cv_risk_cols = ['Heart_Disease', 'Hypertension', 'Stroke']
    available_cv = [col for col in cv_risk_cols if col in df.columns]
    if available_cv:
        df['CV_Risk_Score'] = df[available_cv].fillna(0).sum(axis=1)
    
    # High Risk Profile
    if all(col in df.columns for col in ['BioAge_60_Plus', 'BMI_Obese', 'Comorbidity_Count']):
        df['High_Risk_Profile'] = (
            (df['BioAge_60_Plus'] == 1) & (df['BMI_Obese'] == 1) & (df['Comorbidity_Count'] >= 2)
        ).astype(int)
    
    return df

def load_and_encode_data():
    """Load data, encode, return X and y"""
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

def load_for_feature_engineering():
    """Load without encoding for FE"""
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
        'Recall': recall_score(y_test, y_pred),
        'model': rf
    }

# Store all results
all_results = []

# ============================================================================
# METHOD 1: RAW FEATURES (Baseline)
# ============================================================================

print("\n" + "="*80)
print("METHOD 1: RAW FEATURES (17 features) - Baseline")
print("="*80)

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    X, y = load_and_encode_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
    
    all_results.append({
        'Method': 'Raw_Features',
        'Seed': seed,
        'Features': X.shape[1],
        **{k: v for k, v in metrics.items() if k != 'model'}
    })

# ============================================================================
# METHOD 2: ALL ENGINEERED FEATURES
# ============================================================================

print("\n" + "="*80)
print("METHOD 2: ALL ENGINEERED FEATURES (38 features)")
print("="*80)

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    df = load_for_feature_engineering()
    y = df['Arthritis']
    df_features = df.drop('Arthritis', axis=1)
    df_eng = engineer_features(df_features)
    df_eng = encode_dataframe(df_eng)
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(df_eng, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
    
    all_results.append({
        'Method': 'All_Engineered',
        'Seed': seed,
        'Features': df_eng.shape[1],
        **{k: v for k, v in metrics.items() if k != 'model'}
    })

# ============================================================================
# METHOD 3: ANOVA Top-K (K=15)
# ============================================================================

print("\n" + "="*80)
print("METHOD 3: ANOVA Top-K (K=15) - Statistical Significance")
print("="*80)

K_ANOVA = 15

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    df = load_for_feature_engineering()
    y = df['Arthritis']
    df_features = df.drop('Arthritis', axis=1)
    df_eng = engineer_features(df_features)
    df_eng = encode_dataframe(df_eng)
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Select top K by ANOVA F-score
    selector = SelectKBest(f_classif, k=K_ANOVA)
    X_selected = selector.fit_transform(df_eng, y_encoded)
    selected_features = df_eng.columns[selector.get_support()].tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
    
    all_results.append({
        'Method': f'ANOVA_Top{K_ANOVA}',
        'Seed': seed,
        'Features': K_ANOVA,
        **{k: v for k, v in metrics.items() if k != 'model'}
    })

# ============================================================================
# METHOD 4: HIERARCHICAL CLUSTERING (5 representative features)
# ============================================================================

print("\n" + "="*80)
print("METHOD 4: HIERARCHICAL CLUSTERING (5 representative) - Feature Deduplication")
print("="*80)

N_CLUSTERS = 5

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    df = load_for_feature_engineering()
    y = df['Arthritis']
    df_features = df.drop('Arthritis', axis=1)
    df_eng = engineer_features(df_features)
    df_eng = encode_dataframe(df_eng)
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Hierarchical clustering
    corr_matrix = df_eng.corr().abs()
    distance_matrix = (1 - corr_matrix).to_numpy().copy()
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Symmetrize
    np.fill_diagonal(distance_matrix, 0)
    
    condensed_dist = squareform(distance_matrix, checks=False)
    linkage_matrix = linkage(condensed_dist, method='ward')
    clusters = fcluster(linkage_matrix, N_CLUSTERS, criterion='maxclust')
    
    # Select one feature per cluster (highest correlation with target)
    selected_features = []
    for cluster_id in range(1, N_CLUSTERS + 1):
        cluster_features = df_eng.columns[clusters == cluster_id]
        if len(cluster_features) > 0:
            correlations = [abs(np.corrcoef(df_eng[f], y_encoded)[0, 1]) for f in cluster_features]
            best_feature = cluster_features[np.argmax(correlations)]
            selected_features.append(best_feature)
    
    X_selected = df_eng[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
    
    all_results.append({
        'Method': f'Hierarchical_{N_CLUSTERS}',
        'Seed': seed,
        'Features': len(selected_features),
        **{k: v for k, v in metrics.items() if k != 'model'}
    })

# ============================================================================
# METHOD 5: TREE-BASED IMPORTANCE (Top 15)
# ============================================================================

print("\n" + "="*80)
print("METHOD 5: TREE-BASED IMPORTANCE (Top 15) - Random Forest Feature Importance")
print("="*80)

K_TREE = 15

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    df = load_for_feature_engineering()
    y = df['Arthritis']
    df_features = df.drop('Arthritis', axis=1)
    df_eng = engineer_features(df_features)
    df_eng = encode_dataframe(df_eng)
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Train RF to get importance
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf_selector.fit(df_eng, y_encoded)
    
    importances = rf_selector.feature_importances_
    top_indices = np.argsort(importances)[::-1][:K_TREE]
    selected_features = df_eng.columns[top_indices].tolist()
    
    X_selected = df_eng[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
    
    all_results.append({
        'Method': f'TreeBased_Top{K_TREE}',
        'Seed': seed,
        'Features': K_TREE,
        **{k: v for k, v in metrics.items() if k != 'model'}
    })

# ============================================================================
# METHOD 6: MUTUAL INFORMATION (Top 15)
# ============================================================================

print("\n" + "="*80)
print("METHOD 6: MUTUAL INFORMATION (Top 15) - Information Gain")
print("="*80)

K_MI = 15

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    df = load_for_feature_engineering()
    y = df['Arthritis']
    df_features = df.drop('Arthritis', axis=1)
    df_eng = engineer_features(df_features)
    df_eng = encode_dataframe(df_eng)
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # Select top K by mutual information
    selector = SelectKBest(mutual_info_classif, k=K_MI)
    X_selected = selector.fit_transform(df_eng, y_encoded)
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}")
    
    all_results.append({
        'Method': f'MutualInfo_Top{K_MI}',
        'Seed': seed,
        'Features': K_MI,
        **{k: v for k, v in metrics.items() if k != 'model'}
    })

# ============================================================================
# METHOD 7: L1 LASSO (Embedded Selection)
# ============================================================================

print("\n" + "="*80)
print("METHOD 7: L1 LASSO - Embedded Feature Selection")
print("="*80)

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"[Seed {seed}] ", end="")
    
    df = load_for_feature_engineering()
    y = df['Arthritis']
    df_features = df.drop('Arthritis', axis=1)
    df_eng = engineer_features(df_features)
    df_eng = encode_dataframe(df_eng)
    
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    # L1 regularization
    lasso = LassoCV(cv=5, random_state=seed, max_iter=5000, n_jobs=-1)
    lasso.fit(df_eng, y_encoded)
    
    selected_mask = np.abs(lasso.coef_) > 1e-5
    selected_features = df_eng.columns[selected_mask].tolist()
    
    if len(selected_features) == 0:
        selected_features = df_eng.columns[:15].tolist()  # Fallback
    
    X_selected = df_eng[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.2, random_state=seed, stratify=y_encoded)
    
    metrics = train_and_evaluate(X_train, X_test, y_train, y_test, seed)
    print(f"ROC: {metrics['ROC_AUC']:.4f}, F1: {metrics['F1_Score']:.4f}, Features: {len(selected_features)}")
    
    all_results.append({
        'Method': 'L1_LASSO',
        'Seed': seed,
        'Features': len(selected_features),
        **{k: v for k, v in metrics.items() if k != 'model'}
    })

# ============================================================================
# ANALYSIS & SUMMARY
# ============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS")
print("="*80)

all_results_df = pd.DataFrame(all_results)

# Save detailed
detailed_path = os.path.join(output_dir, 'comprehensive_feature_selection_detailed.csv')
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
summary_path = os.path.join(output_dir, 'comprehensive_feature_selection_summary.csv')
summary_df.to_csv(summary_path, index=False)
print(f"\n✓ Summary saved: {summary_path}")

# Find winner
best = summary_df.iloc[0]
print("\n" + "="*80)
print("🏆 WINNER")
print("="*80)
print(f"\nMethod: {best['Method']}")
print(f"ROC AUC: {best['ROC_AUC_mean']:.4f} ± {best['ROC_AUC_std']:.4f} (CV: {best['CV_ROC']:.2f}%)")
print(f"F1 Score: {best['F1_mean']:.4f} ± {best['F1_std']:.4f}")
print(f"Features: {best['Features']}")

print("\n" + "="*80)
print("✅ COMPREHENSIVE FEATURE SELECTION COMPLETE!")
print("="*80)
