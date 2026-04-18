"""
Step 0: Baseline Comparison
============================
Dört farklı data preprocessing yaklaşımını karşılaştırır:

0. Raw Data Direct: Sadece Arthritis filter, diğer hiçbir işlem yok (-999 encoding)
1. Listwise Deletion (Orijinal): Missing rows silinir
2. Mean Imputation: Missing values doldurulur  
3. Raw AS-IS: Missing values -999 ile encode edilir (RF için)

Her dördü de RAW features (feature engineering YOK) + Random Forest ile test edilir.

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
import time
import os

print("="*80)
print("STEP 0: BASELINE COMPARISON (4 Preprocessing Strategies)")
print("="*80)

# Paths
raw_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
supervised_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_unsupervised_data\supervised_data_with_target.csv"
imputed_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_imputed_data\data_imputed_mean.csv"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_baseline_comparison"
os.makedirs(output_dir, exist_ok=True)

results = []

# ============================================================================
# BASELINE 0: RAW DATA DIRECT (No preprocessing at all)
# ============================================================================

print("\n" + "="*80)
print("BASELINE 0: RAW DATA DIRECT (Arthritis filter only, no other preprocessing)")
print("="*80)

print("[1/4] Loading Raw Data.xlsx directly...")
df_raw_direct = pd.read_excel(raw_data_path)
print(f"   ✓ Loaded: {df_raw_direct.shape[0]:,} rows total")

print("[2/4] Filtering rows with Arthritis target only...")
df_raw_direct = df_raw_direct[df_raw_direct['Arthritis'].notna()]
print(f"   ✓ After filtering: {df_raw_direct.shape[0]:,} rows")
print(f"   ✓ Missing values in features: {df_raw_direct.drop('Arthritis', axis=1).isnull().sum().sum():,}")

# Prepare data
X = df_raw_direct.drop('Arthritis', axis=1)
y = df_raw_direct['Arthritis']

# Encode target with LabelEncoder for robust handling
le_target = LabelEncoder()
y = le_target.fit_transform(y)

print("[3/4] Encoding categorical and missing values...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    # Fill NaN temporarily for encoding
    X[col] = X[col].fillna('__MISSING__')
    X[col] = le.fit_transform(X[col].astype(str))

# Encode missing values as -999
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    X[col] = X[col].fillna(-999)

print(f"   ✓ All missing values encoded as -999")
print(f"   ✓ Total samples: {len(X):,}")

print(f"[4/4] Training Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
elapsed = time.time() - start_time

# Metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n   📊 Results:")
print(f"      ROC AUC: {roc_auc:.4f}")
print(f"      F1 Score: {f1:.4f}")
print(f"      Accuracy: {accuracy:.4f}")
print(f"      Train size: {len(X_train):,}")
print(f"      Test size: {len(X_test):,}")
print(f"      Time: {elapsed:.2f}s")

results.append({
    'Method': '0_Raw_Data_Direct_No_Preprocessing',
    'ROC_AUC': roc_auc,
    'F1_Score': f1,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'Train_Size': len(X_train),
    'Test_Size': len(X_test),
    'Total_Samples': len(df_raw_direct),
    'Data_Loss_%': 0.0,
    'Time_sec': elapsed
})

# ============================================================================
# BASELINE 1: LISTWISE DELETION (Original Approach)
# ============================================================================

print("\n" + "="*80)
print("BASELINE 1: LISTWISE DELETION (Original)")
print("="*80)

print("[1/3] Loading data with target...")
df_supervised = pd.read_csv(supervised_data_path)
print(f"   ✓ Loaded: {df_supervised.shape[0]:,} rows (target available)")

print("[2/3] Applying listwise deletion...")
df_clean = df_supervised.dropna()
print(f"   ✓ After dropna(): {df_clean.shape[0]:,} rows")
print(f"   ✓ Data loss: {df_supervised.shape[0] - df_clean.shape[0]:,} rows ({(1 - df_clean.shape[0]/df_supervised.shape[0])*100:.1f}%)")

# Prepare data
X = df_clean.drop('Arthritis', axis=1)
y = df_clean['Arthritis']

# Encode target with LabelEncoder for robust handling
print(f"   Target unique before encoding: {y.unique()}")
le_target = LabelEncoder()
y = le_target.fit_transform(y)
print(f"   Target unique after encoding: {np.unique(y)}")

# Encode categorical
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print(f"[3/3] Training Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
elapsed = time.time() - start_time

# Metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n   📊 Results:")
print(f"      ROC AUC: {roc_auc:.4f}")
print(f"      F1 Score: {f1:.4f}")
print(f"      Accuracy: {accuracy:.4f}")
print(f"      Train size: {len(X_train):,}")
print(f"      Test size: {len(X_test):,}")
print(f"      Time: {elapsed:.2f}s")

results.append({
    'Method': '1_Listwise_Deletion',
    'ROC_AUC': roc_auc,
    'F1_Score': f1,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'Train_Size': len(X_train),
    'Test_Size': len(X_test),
    'Total_Samples': len(df_clean),
    'Data_Loss_%': (1 - len(df_clean)/df_supervised.shape[0])*100,
    'Time_sec': elapsed
})

# ============================================================================
# BASELINE 2: MEAN IMPUTATION
# ============================================================================

print("\n" + "="*80)
print("BASELINE 2: MEAN IMPUTATION")
print("="*80)

print("[1/3] Loading imputed data...")
df_imputed = pd.read_csv(imputed_data_path)
print(f"   ✓ Loaded: {df_imputed.shape[0]:,} rows")

# Prepare data
X = df_imputed.drop('Arthritis', axis=1)
y = df_imputed['Arthritis']

# Encode target with LabelEncoder for robust handling
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Encode categorical
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

print(f"[2/3] Verifying no missing values...")
if X.isnull().sum().sum() == 0:
    print(f"   ✓ No missing values")
else:
    print(f"   ⚠️  {X.isnull().sum().sum()} missing values found")

print(f"[3/3] Training Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
elapsed = time.time() - start_time

# Metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n   📊 Results:")
print(f"      ROC AUC: {roc_auc:.4f}")
print(f"      F1 Score: {f1:.4f}")
print(f"      Accuracy: {accuracy:.4f}")
print(f"      Train size: {len(X_train):,}")
print(f"      Test size: {len(X_test):,}")
print(f"      Time: {elapsed:.2f}s")

results.append({
    'Method': '2_Mean_Imputation',
    'ROC_AUC': roc_auc,
    'F1_Score': f1,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'Train_Size': len(X_train),
    'Test_Size': len(X_test),
    'Total_Samples': len(df_imputed),
    'Data_Loss_%': 0.0,
    'Time_sec': elapsed
})

# ============================================================================
# BASELINE 3: RAW AS-IS (Missing → -999 for RF)
# ============================================================================

print("\n" + "="*80)
print("BASELINE 3: RAW AS-IS (Missing values encoded as -999)")
print("="*80)

print("[1/4] Loading supervised data (no dropna)...")
df_raw = pd.read_csv(supervised_data_path)
print(f"   ✓ Loaded: {df_raw.shape[0]:,} rows")
print(f"   ✓ Missing values: {df_raw.isnull().sum().sum():,}")

# Prepare data
X = df_raw.drop('Arthritis', axis=1)
y = df_raw['Arthritis']

# Encode target with LabelEncoder for robust handling
le_target = LabelEncoder()
y = le_target.fit_transform(y)

print("[2/4] Encoding categorical features...")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    # Fill NaN temporarily for encoding
    X[col] = X[col].fillna('__MISSING__')
    X[col] = le.fit_transform(X[col].astype(str))

print("[3/4] Encoding missing values as -999...")
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    X[col] = X[col].fillna(-999)

print(f"   ✓ All missing values encoded")
print(f"   ✓ Total samples: {len(X):,}")

print(f"[4/4] Training Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
elapsed = time.time() - start_time

# Metrics
roc_auc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"\n   📊 Results:")
print(f"      ROC AUC: {roc_auc:.4f}")
print(f"      F1 Score: {f1:.4f}")
print(f"      Accuracy: {accuracy:.4f}")
print(f"      Train size: {len(X_train):,}")
print(f"      Test size: {len(X_test):,}")
print(f"      Time: {elapsed:.2f}s")

results.append({
    'Method': '3_Raw_AsIs_Missing999',
    'ROC_AUC': roc_auc,
    'F1_Score': f1,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'Train_Size': len(X_train),
    'Test_Size': len(X_test),
    'Total_Samples': len(df_raw),
    'Data_Loss_%': 0.0,
    'Time_sec': elapsed
})

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('ROC_AUC', ascending=False)
print(results_df.to_string(index=False))

# Save results
results_path = os.path.join(output_dir, 'baseline_comparison_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\n✓ Results saved: {results_path}")

# Best method
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)
best = results_df.iloc[0]
print(f"\n🏆 Best Method: {best['Method']}")
print(f"   • ROC AUC: {best['ROC_AUC']:.4f}")
print(f"   • F1 Score: {best['F1_Score']:.4f}")
print(f"   • Samples: {best['Total_Samples']:,.0f}")
print(f"   • Data Loss: {best['Data_Loss_%']:.1f}%")

print("\n📊 Trade-offs:")
for _, row in results_df.iterrows():
    print(f"\n   {row['Method']}:")
    print(f"      Samples: {row['Total_Samples']:,.0f} | Loss: {row['Data_Loss_%']:.1f}%")
    print(f"      ROC AUC: {row['ROC_AUC']:.4f} | F1: {row['F1_Score']:.4f}")

print("\n" + "="*80)
print("✅ BASELINE COMPARISON COMPLETE!")
print("="*80)
