"""
Step 0: Prepare Imputed Dataset for Steps 2-4
==============================================
Mean Imputation kullanarak tam dataset hazırlar.
Bu dataset Step 2 (Feature Engineering), Step 3 ve Step 4'te kullanılacak.

Strategy: Mean/Mode Imputation
- Numeric features: Mean
- Categorical features: Mode
- No data loss (0%)
- ROC AUC: 0.6792 (baseline'a çok yakın)

Author: Bioinformatics Analysis Team
Date: March 2026
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

print("="*80)
print("STEP 0: PREPARE IMPUTED DATASET (MEAN IMPUTATION)")
print("="*80)

# Paths
supervised_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_unsupervised_data\supervised_data_with_target.csv"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_imputed_data"
os.makedirs(output_dir, exist_ok=True)

# Load data
print("\n[1/6] Loading supervised data...")
df = pd.read_csv(supervised_data_path)
print(f"   ✓ Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Check missing values
print("\n[2/6] Analyzing missing values...")
missing_counts = df.isnull().sum()
missing_features = missing_counts[missing_counts > 0]
total_missing = missing_counts.sum()
print(f"   • Total missing cells: {total_missing:,}")
print(f"   • Features with missing values: {len(missing_features)}")

for col, count in missing_features.items():
    pct = (count / len(df)) * 100
    print(f"      {col:35s}: {count:5,} ({pct:5.1f}%)")

# Separate features and target
print("\n[3/6] Separating features and target...")
X = df.drop('Arthritis', axis=1)
y = df['Arthritis']

# Encode target to 0/1 if needed
if y.dtype == 'object' or str(y.dtype) == 'string':
    print(f"   • Encoding target: {y.unique()} → [0, 1]")
    y = y.map({'no': 0, 'yes': 1})
    print(f"   ✓ Target encoded")

# Identify column types
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"   • Numeric features: {len(numeric_cols)}")
print(f"   • Categorical features: {len(categorical_cols)}")

# Apply Mean/Mode Imputation
print("\n[4/6] Applying Mean/Mode Imputation...")

# Numeric features: Mean imputation
if len(numeric_cols) > 0:
    print(f"   • Imputing {len(numeric_cols)} numeric features with MEAN...")
    numeric_missing = X[numeric_cols].isnull().sum().sum()
    if numeric_missing > 0:
        imputer_numeric = SimpleImputer(strategy='mean')
        X[numeric_cols] = imputer_numeric.fit_transform(X[numeric_cols])
        print(f"      ✓ {numeric_missing:,} values imputed")
    else:
        print(f"      ✓ No missing values in numeric features")

# Categorical features: Mode imputation
if len(categorical_cols) > 0:
    print(f"   • Imputing {len(categorical_cols)} categorical features with MODE...")
    categorical_missing = X[categorical_cols].isnull().sum().sum()
    if categorical_missing > 0:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = imputer_categorical.fit_transform(X[categorical_cols])
        print(f"      ✓ {categorical_missing:,} values imputed")
    else:
        print(f"      ✓ No missing values in categorical features")

# Verify no missing values remain
print("\n[5/6] Verifying imputation...")
remaining_missing = X.isnull().sum().sum()
if remaining_missing == 0:
    print(f"   ✓ SUCCESS: No missing values remain!")
    print(f"   ✓ Dataset is complete: {df.shape[0]:,} rows × {X.shape[1]} features")
else:
    print(f"   ⚠️  WARNING: {remaining_missing} missing values still remain")

# Combine features and target
df_imputed = X.copy()
df_imputed['Arthritis'] = y

# Save imputed dataset
print("\n[6/6] Saving imputed dataset...")
output_path = os.path.join(output_dir, 'data_imputed_mean.csv')
df_imputed.to_csv(output_path, index=False)
print(f"   ✓ Saved: {output_path}")

# Create metadata file
metadata_path = os.path.join(output_dir, 'imputation_metadata.txt')
with open(metadata_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("IMPUTED DATASET METADATA\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. IMPUTATION METHOD\n")
    f.write("-" * 40 + "\n")
    f.write("   Method: Mean/Mode Imputation\n")
    f.write("   - Numeric features: Mean\n")
    f.write("   - Categorical features: Mode\n")
    f.write("   - Rationale: Best balance of performance and data retention\n\n")
    
    f.write("2. PERFORMANCE (from Step 0 comparison)\n")
    f.write("-" * 40 + "\n")
    f.write("   ROC AUC: 0.6792\n")
    f.write("   F1 Score: 0.3837\n")
    f.write("   Accuracy: 0.6981\n")
    f.write("   Data Loss: 0% (vs 71.2% with listwise deletion)\n\n")
    
    f.write("3. DATASET DETAILS\n")
    f.write("-" * 40 + "\n")
    f.write(f"   Total rows: {df_imputed.shape[0]:,}\n")
    f.write(f"   Total features: {df_imputed.shape[1] - 1} (+ 1 target)\n")
    f.write(f"   Missing values imputed: {total_missing:,}\n")
    f.write(f"   Target distribution:\n")
    f.write(f"      Class 0 (No Arthritis): {(y==0).sum():,} ({(y==0).sum()/len(y)*100:.1f}%)\n")
    f.write(f"      Class 1 (Arthritis): {(y==1).sum():,} ({(y==1).sum()/len(y)*100:.1f}%)\n\n")
    
    f.write("4. COMPARISON WITH ORIGINAL APPROACH\n")
    f.write("-" * 40 + "\n")
    f.write("   Original (Listwise Deletion):\n")
    f.write("      - Rows: 2,742\n")
    f.write("      - ROC AUC: 0.6832\n")
    f.write("      - Data Loss: 71.2%\n\n")
    f.write("   New (Mean Imputation):\n")
    f.write(f"      - Rows: {df_imputed.shape[0]:,}\n")
    f.write("      - ROC AUC: 0.6792 (-0.6%)\n")
    f.write("      - Data Loss: 0%\n")
    f.write(f"      - Additional samples: +{df_imputed.shape[0] - 2742:,} (+{((df_imputed.shape[0] - 2742)/2742)*100:.1f}%)\n\n")
    
    f.write("5. USAGE\n")
    f.write("-" * 40 + "\n")
    f.write("   This dataset should be used for:\n")
    f.write("   - Step 2: Feature Engineering\n")
    f.write("   - Step 3: Position Knees Experiments\n")
    f.write("   - Step 4: Feature Selection Comparison\n\n")
    
    f.write("6. NEXT STEPS\n")
    f.write("-" * 40 + "\n")
    f.write("   1. Update Step 2-4 scripts to use this dataset\n")
    f.write("   2. Re-run feature engineering pipeline\n")
    f.write("   3. Compare results with original baseline\n")
    f.write("   4. Update experiment log and memory\n")

print(f"   ✓ Metadata saved: {metadata_path}")

# Summary
print("\n" + "="*80)
print("✅ IMPUTED DATASET READY!")
print("="*80)
print(f"\n📊 Summary:")
print(f"   • Original data: {df.shape[0]:,} rows")
print(f"   • Missing cells imputed: {total_missing:,}")
print(f"   • Final dataset: {df_imputed.shape[0]:,} rows × {df_imputed.shape[1]} columns")
print(f"   • Data recovery vs listwise deletion: +{df_imputed.shape[0] - 2742:,} rows (+{((df_imputed.shape[0] - 2742)/2742)*100:.1f}%)")
print(f"\n📁 Files:")
print(f"   • Dataset: {output_path}")
print(f"   • Metadata: {metadata_path}")
print(f"\n💡 Next: Use this dataset in Step 2 (Feature Engineering)")
print("="*80)
