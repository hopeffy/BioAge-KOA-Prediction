"""
Quick check of raw data missing values
"""
import pandas as pd

df = pd.read_excel(r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx")

print("="*80)
print("RAW DATA QUICK CHECK")
print("="*80)
print(f"\nTotal rows: {len(df):,}")
print(f"Total columns: {len(df.columns)}")

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing': missing.values,
    'Percent': missing_pct.values
})
missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("No missing values!")

complete_cases = (~df.isnull().any(axis=1)).sum()
complete_pct = complete_cases / len(df) * 100

print(f"\n{'='*80}")
print(f"Complete cases (no missing): {complete_cases:,} ({complete_pct:.2f}%)")
print(f"Incomplete cases: {len(df) - complete_cases:,} ({100-complete_pct:.2f}%)")
print("="*80)
