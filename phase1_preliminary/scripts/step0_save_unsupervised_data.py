"""
Step 0: Save Unsupervised Data (Target Missing Cases)
=====================================================
Target'ı (Arthritis) missing olan satırları ayrı bir dataset olarak kaydeder.
Bu veriler ileride unsupervised learning için kullanılabilir.

Author: Bioinformatics Analysis Team
Date: March 2026
"""

import pandas as pd
import numpy as np
import os

print("="*80)
print("STEP 0: SEPARATING UNSUPERVISED DATA (TARGET MISSING CASES)")
print("="*80)

# Paths
raw_data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
output_dir = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_unsupervised_data"
os.makedirs(output_dir, exist_ok=True)

# Load raw data
print("\n[1/5] Loading raw data...")
df_raw = pd.read_excel(raw_data_path)
print(f"   ✓ Loaded: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

# Analyze missing data
print("\n[2/5] Analyzing missing data pattern...")
total_cells = df_raw.shape[0] * df_raw.shape[1]
missing_cells = df_raw.isnull().sum().sum()
missing_pct = (missing_cells / total_cells) * 100

print(f"   • Total cells: {total_cells:,}")
print(f"   • Missing cells: {missing_cells:,} ({missing_pct:.2f}%)")
print(f"   • Complete cases: {df_raw.dropna().shape[0]:,} ({(df_raw.dropna().shape[0]/df_raw.shape[0])*100:.2f}%)")

# Most missing columns
print("\n   Top 5 columns with missing values:")
missing_per_col = df_raw.isnull().sum().sort_values(ascending=False).head(5)
for col, count in missing_per_col.items():
    if count > 0:
        pct = (count / df_raw.shape[0]) * 100
        print(f"      {col:35s}: {count:6,} ({pct:5.1f}%)")

# Apply Age >= 18 filter
print("\n[3/5] Applying Age >= 18 filter...")
df_filtered = df_raw[df_raw['Biological Age'] >= 18].copy()
removed_age = df_raw.shape[0] - df_filtered.shape[0]
print(f"   ✓ After age filter: {df_filtered.shape[0]:,} rows")
print(f"   ✓ Removed (age < 18): {removed_age:,} rows")

# Separate based on target availability
print("\n[4/5] Separating data based on target availability...")
df_target_missing = df_filtered[df_filtered['Arthritis'].isnull()].copy()
df_with_target = df_filtered[df_filtered['Arthritis'].notnull()].copy()

print(f"\n   📊 Data Distribution:")
print(f"      • WITH target (for supervised learning): {df_with_target.shape[0]:,} rows")
print(f"      • WITHOUT target (for unsupervised learning): {df_target_missing.shape[0]:,} rows")
print(f"      • Target missing percentage: {(df_target_missing.shape[0]/df_filtered.shape[0])*100:.2f}%")

# Analyze what features are available in target-missing data
print(f"\n   📊 Feature availability in target-missing data:")
feature_completeness = {}
for col in df_target_missing.columns:
    if col != 'Arthritis':
        non_missing = df_target_missing[col].notna().sum()
        pct = (non_missing / len(df_target_missing)) * 100
        feature_completeness[col] = pct

# Sort by completeness
feature_completeness = dict(sorted(feature_completeness.items(), key=lambda x: x[1], reverse=True))

print(f"\n   Top 10 most complete features in unsupervised data:")
for i, (col, pct) in enumerate(list(feature_completeness.items())[:10]):
    print(f"      {i+1:2d}. {col:35s}: {pct:5.1f}% complete")

# Save datasets
print("\n[5/5] Saving datasets...")

# Save supervised learning data (with target)
supervised_path = os.path.join(output_dir, 'supervised_data_with_target.csv')
df_with_target.to_csv(supervised_path, index=False)
print(f"   ✓ Supervised data: {supervised_path}")
print(f"      - {df_with_target.shape[0]:,} rows × {df_with_target.shape[1]} columns")

# Save unsupervised learning data (target missing, but keep other features)
unsupervised_path = os.path.join(output_dir, 'unsupervised_data_target_missing.csv')
df_target_missing.to_csv(unsupervised_path, index=False)
print(f"   ✓ Unsupervised data: {unsupervised_path}")
print(f"      - {df_target_missing.shape[0]:,} rows × {df_target_missing.shape[1]} columns")
print(f"      - ⚠️  This data has NO target labels (Arthritis is missing)")
print(f"      - 💡 Can be used for: clustering, anomaly detection, dimensionality reduction")

# Create a summary report
summary_path = os.path.join(output_dir, 'data_separation_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DATA SEPARATION SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. ORIGINAL DATA\n")
    f.write("-" * 40 + "\n")
    f.write(f"   Total rows: {df_raw.shape[0]:,}\n")
    f.write(f"   Total columns: {df_raw.shape[1]}\n")
    f.write(f"   Missing cells: {missing_cells:,} ({missing_pct:.2f}%)\n\n")
    
    f.write("2. AFTER AGE FILTER (Age >= 18)\n")
    f.write("-" * 40 + "\n")
    f.write(f"   Rows: {df_filtered.shape[0]:,}\n")
    f.write(f"   Removed: {removed_age:,}\n\n")
    
    f.write("3. DATA SPLIT BY TARGET AVAILABILITY\n")
    f.write("-" * 40 + "\n")
    f.write(f"   a) WITH Target (Supervised Learning):\n")
    f.write(f"      - File: supervised_data_with_target.csv\n")
    f.write(f"      - Rows: {df_with_target.shape[0]:,}\n")
    f.write(f"      - Usage: Classification models (Step 1-4)\n\n")
    
    f.write(f"   b) WITHOUT Target (Unsupervised Learning):\n")
    f.write(f"      - File: unsupervised_data_target_missing.csv\n")
    f.write(f"      - Rows: {df_target_missing.shape[0]:,}\n")
    f.write(f"      - Usage: Clustering, Anomaly Detection, etc.\n")
    f.write(f"      - Note: Arthritis column is missing/null\n\n")
    
    f.write("4. FEATURE COMPLETENESS IN UNSUPERVISED DATA\n")
    f.write("-" * 40 + "\n")
    for col, pct in feature_completeness.items():
        f.write(f"   {col:35s}: {pct:5.1f}% complete\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("="*80 + "\n")
    f.write("1. Use 'supervised_data_with_target.csv' for classification (Steps 1-4)\n")
    f.write("2. Keep 'unsupervised_data_target_missing.csv' for future analysis:\n")
    f.write("   - Clustering patients by health characteristics\n")
    f.write("   - Anomaly detection (unusual patient profiles)\n")
    f.write("   - Dimensionality reduction (PCA, t-SNE)\n")
    f.write("   - Semi-supervised learning experiments\n")
    f.write("3. These datasets preserve maximum information:\n")
    f.write(f"   - No data loss from listwise deletion on informative features\n")
    f.write(f"   - Total preserved: {df_filtered.shape[0]:,} rows (vs {df_raw.dropna().shape[0]:,} with listwise deletion)\n")

print(f"   ✓ Summary report: {summary_path}")

print("\n" + "="*80)
print("✅ DATA SEPARATION COMPLETE!")
print("="*80)
print(f"\n💾 Output files saved to: {output_dir}/")
print(f"\n📊 Summary:")
print(f"   • Supervised (with target): {df_with_target.shape[0]:,} rows")
print(f"   • Unsupervised (target missing): {df_target_missing.shape[0]:,} rows")
print(f"   • Total preserved: {df_filtered.shape[0]:,} rows")
print(f"   • Original listwise deletion: {df_raw.dropna().shape[0]:,} rows")
print(f"   • Data recovery: +{df_filtered.shape[0] - df_raw.dropna().shape[0]:,} rows preserved!")
print("\n" + "="*80)
