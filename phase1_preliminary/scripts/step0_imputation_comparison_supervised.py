"""
Step 0: Imputation Methods Comparison (Supervised Data Only)
============================================================
Supervised learning data (target mevcut) üzerinde farklı imputation yöntemlerini test eder.
6 farklı yöntem deneyip performanslarını karşılaştırır.

Imputation Methods:
1. Listwise Deletion - Missing rows silinir (baseline)
2. Mean Imputation - Numeric: mean, Categorical: mode
3. Median Imputation - Numeric: median, Categorical: mode  
4. KNN Imputation - K-Nearest Neighbors
5. Iterative Imputation - MICE benzeri
6. Forward Fill + Backward Fill - Zaman serisi mantığı

Author: Bioinformatics Analysis Team
Date: March 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import time
import warnings
import os
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 0: IMPUTATION METHODS COMPARISON (SUPERVISED DATA)")
print("="*80)

# Paths
supervised_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_unsupervised_data\supervised_data_with_target.csv"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_imputation_comparison"
os.makedirs(output_dir, exist_ok=True)

# Load supervised data
print("\n[1/8] Loading supervised data (target available)...")
df = pd.read_csv(supervised_data_path)
print(f"   ✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"   ✓ Target (Arthritis) is available for all rows")

# Missing data analysis
print("\n[2/8] Analyzing missing data...")
missing_counts = df.isnull().sum()
missing_features = missing_counts[missing_counts > 0].sort_values(ascending=False)

if len(missing_features) > 0:
    print(f"   • Features with missing values: {len(missing_features)}")
    for col, count in missing_features.items():
        pct = (count / len(df)) * 100
        print(f"      {col:35s}: {count:5,} ({pct:5.1f}%)")
    total_missing = missing_counts.sum()
    print(f"   • Total missing cells: {total_missing:,}")
else:
    print("   ✓ No missing values!")

# Target distribution
print("\n[3/8] Target distribution...")
target_dist = df['Arthritis'].value_counts().sort_index()
print(f"   • Class 0/no (No Arthritis): {target_dist.iloc[0]:,} ({target_dist.iloc[0]/len(df)*100:.1f}%)")
print(f"   • Class 1/yes (Arthritis): {target_dist.iloc[1]:,} ({target_dist.iloc[1]/len(df)*100:.1f}%)")

# Prepare feature and target
X = df.drop('Arthritis', axis=1)
y = df['Arthritis']

# Encode target to 0/1
print("\n[4/8] Encoding target variable...")
print(f"   • Target dtype: {y.dtype}")
print(f"   • Target unique values: {y.unique()[:5]}")  # Show first 5
if y.dtype == 'object' or str(y.dtype) == 'string' or y.dtype.name == 'string':
    print(f"   • Target is categorical, encoding to numeric...")
    y = y.map({'no': 0, 'yes': 1})
    print(f"   ✓ Target encoded: 'no'→0, 'yes'→1")
    print(f"   ✓ Distribution: 0={(y==0).sum()}, 1={(y==1).sum()}")
elif set(y.unique()) == {'no', 'yes'}:
    print(f"   • Target values are 'no'/'yes', encoding...")
    y = y.map({'no': 0, 'yes': 1})
    print(f"   ✓ Target encoded to 0/1")
else:
    print(f"   ✓ Target is already numeric: min={y.min()}, max={y.max()}")

# Encode categorical features
print("\n[5/8] Encoding categorical features...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}

if len(categorical_cols) > 0:
    print(f"   • Encoding {len(categorical_cols)} categorical columns...")
    for col in categorical_cols:
        le = LabelEncoder()
        # Save NaN locations
        nan_mask = X[col].isna()
        # Fill temporarily and encode
        X[col] = X[col].fillna('__MISSING__')
        X[col] = le.fit_transform(X[col].astype(str))
        # Restore NaN
        X.loc[nan_mask, col] = np.nan
        label_encoders[col] = le
    print(f"   ✓ Encoded successfully")

# Split data ONCE (same split for all methods)
print("\n[6/8] Creating train/test split (70/30)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"   ✓ Train: {len(X_train):,} samples")
print(f"   ✓ Test: {len(X_test):,} samples")

# Define imputation methods
print("\n[7/8] Defining imputation methods...")
imputation_methods = {
    '1_Listwise_Deletion': 'listwise',  # Special marker
    '2_Mean_Imputation': SimpleImputer(strategy='mean'),
    '3_Median_Imputation': SimpleImputer(strategy='median'),
    '4_Most_Frequent': SimpleImputer(strategy='most_frequent'),
    '5_KNN_k5': KNNImputer(n_neighbors=5),
    '6_KNN_k10': KNNImputer(n_neighbors=10),
    '7_Iterative_MICE': IterativeImputer(max_iter=10, random_state=42),
}
print(f"   ✓ {len(imputation_methods)} methods ready to test")

# Test each method
print("\n[8/8] Testing imputation methods...")
results = []

for method_name, imputer in imputation_methods.items():
    print(f"\n   {'-'*70}")
    print(f"   Testing: {method_name}")
    print(f"   {'-'*70}")
    
    try:
        start_time = time.time()
        
        # Special handling for listwise deletion
        if imputer == 'listwise':
            # Remove rows with ANY missing values
            train_complete_mask = X_train.notna().all(axis=1)
            test_complete_mask = X_test.notna().all(axis=1)
            
            X_train_clean = X_train[train_complete_mask]
            y_train_clean = y_train[train_complete_mask]
            X_test_clean = X_test[test_complete_mask]
            y_test_clean = y_test[test_complete_mask]
            
            print(f"      • Train before: {len(X_train):,} → after: {len(X_train_clean):,} (removed {len(X_train) - len(X_train_clean)})")
            print(f"      • Test before: {len(X_test):,} → after: {len(X_test_clean):,} (removed {len(X_test) - len(X_test_clean)})")
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_test_scaled = scaler.transform(X_test_clean)
            
            # Train model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train_clean)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            roc_auc = roc_auc_score(y_test_clean, y_pred_proba)
            f1 = f1_score(y_test_clean, y_pred)
            accuracy = accuracy_score(y_test_clean, y_pred)
            precision = precision_score(y_test_clean, y_pred)
            recall = recall_score(y_test_clean, y_pred)
            
            data_loss_pct = ((len(X_train) + len(X_test) - len(X_train_clean) - len(X_test_clean)) / (len(X_train) + len(X_test))) * 100
            
        else:
            # Apply imputation
            X_train_imputed = imputer.fit_transform(X_train)
            X_test_imputed = imputer.transform(X_test)
            
            # Scale  
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imputed)
            X_test_scaled = scaler.transform(X_test_imputed)
            
            # Train model
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Metrics
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            data_loss_pct = 0.0
        
        elapsed_time = time.time() - start_time
        
        # Store results
        results.append({
            'Method': method_name,
            'ROC_AUC': roc_auc,
            'F1_Score': f1,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'Data_Loss_%': data_loss_pct,
            'Time_sec': elapsed_time
        })
        
        print(f"      ✓ ROC AUC: {roc_auc:.4f}")
        print(f"      ✓ F1 Score: {f1:.4f}")
        print(f"      ✓ Accuracy: {accuracy:.4f}")
        print(f"      ✓ Time: {elapsed_time:.2f}s")
        
    except Exception as e:
        print(f"      ✗ FAILED: {str(e)}")
        results.append({
            'Method': method_name,
            'ROC_AUC': np.nan,
            'F1_Score': np.nan,
            'Accuracy': np.nan,
            'Precision': np.nan,
            'Recall': np.nan,
            'Data_Loss_%': np.nan,
            'Time_sec': np.nan,
            'Error': str(e)
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC_AUC', ascending=False)

# Save results
print("\n[9/9] Saving results...")
results_path = os.path.join(output_dir, 'imputation_comparison_results.csv')
results_df.to_csv(results_path, index=False)
print(f"   ✓ Results saved: {results_path}")

# Display summary
print("\n" + "="*80)
print("RESULTS SUMMARY (sorted by ROC AUC)")
print("="*80)
print(results_df.to_string(index=False))

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
best_method = results_df.iloc[0]
print(f"\n🏆 Best Method: {best_method['Method']}")
print(f"   • ROC AUC: {best_method['ROC_AUC']:.4f}")
print(f"   • F1 Score: {best_method['F1_Score']:.4f}")
print(f"   • Accuracy: {best_method['Accuracy']:.4f}")
print(f"   • Precision: {best_method['Precision']:.4f}")
print(f"   • Recall: {best_method['Recall']:.4f}")
print(f"   • Data Loss: {best_method['Data_Loss_%']:.1f}%")
print(f"   • Time: {best_method['Time_sec']:.2f}s")

# Compare with listwise deletion
listwise_result = results_df[results_df['Method'] == '1_Listwise_Deletion']
if not listwise_result.empty and best_method['Method'] != '1_Listwise_Deletion':
    auc_improvement = ((best_method['ROC_AUC'] - listwise_result['ROC_AUC'].values[0]) / listwise_result['ROC_AUC'].values[0]) * 100
    f1_improvement = ((best_method['F1_Score'] - listwise_result['F1_Score'].values[0]) / listwise_result['F1_Score'].values[0]) * 100
    
    print(f"\n📈 Improvement vs Listwise Deletion:")
    print(f"   • ROC AUC: {auc_improvement:+.2f}%")
    print(f"   • F1 Score: {f1_improvement:+.2f}%")

print("\n" + "="*80)
print("✅ IMPUTATION COMPARISON COMPLETE!")
print("="*80)
print(f"\n💡 Next Steps:")
print(f"   1. Review results in: {results_path}")
print(f"   2. Select best imputation method")
print(f"   3. Use it in Step 2 (Feature Engineering)")
print(f"   4. Continue with Steps 3-4")
print("\n" + "="*80)
