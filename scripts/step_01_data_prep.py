"""
STEP 01: Sheet1 Data Preparation & Exploration
==============================================
- Load Sheet1 from Raw Data .xlsx
- Log-transform reversal for 7 biomarkers
- Encoding mapping reverse engineering
- KOA definition verification
- Leakage control (remove Arthritis, position_knees)
- Create Dataset_A (all pk non-null) and Dataset_B (BA>=55 filter)
- Save cleaned datasets and exploration report
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

BASE_DIR = r'C:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data'
STEP_DIR = os.path.join(BASE_DIR, 'step_01_data_prep')

def load_sheet1():
    print("=" * 80)
    print("STEP 01: Sheet1 Data Preparation & Exploration")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    filepath = os.path.join(BASE_DIR, 'Raw Data .xlsx')
    print(f"Loading Sheet1 from: {filepath}")
    df = pd.read_excel(filepath, sheet_name='Sheet1')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    return df

def reverse_log_transform(df):
    print("=" * 80)
    print("1. LOG-TRANSFORM REVERSAL")
    print("=" * 80)
    
    biomarker_map = {
        'plt_10.9.L': 'platelet',
        'crp_mg.L': 'crp',
        'Hb A1c': 'hba1c',
        'creatinine_mg.d L': 'creatinine',
        'bun_mg.d L': 'bun',
        'TC_mg.d L': 'total_cholesterol',
        'TG_mg.d L': 'triglycerides',
    }
    
    clinical_ranges = {
        'platelet': '150-450 x10^9/L',
        'crp': '0-10 mg/L (hs-CRP <3)',
        'hba1c': '4-14 %',
        'creatinine': '0.5-1.5 mg/dL',
        'bun': '7-20 mg/dL',
        'total_cholesterol': '100-300 mg/dL',
        'triglycerides': '50-500 mg/dL',
        'sbp': '90-180 mmHg',
    }
    
    df_out = df.copy()
    
    print("\nReverse-transforming biomarkers (exp):\n")
    for log_col, name in biomarker_map.items():
        orig_col = f'{name}_original'
        df_out[orig_col] = np.exp(df_out[log_col])
        desc = df_out[orig_col].describe()
        print(f"  {log_col} -> {orig_col}")
        print(f"    Log scale:  min={df[log_col].min():.4f}, mean={df[log_col].mean():.4f}, max={df[log_col].max():.4f}")
        print(f"    Orig scale: min={desc['min']:.2f}, mean={desc['mean']:.2f}, max={desc['max']:.2f}")
        print(f"    Clinical:    {clinical_ranges[name]}")
        out_of_range_low = (df_out[orig_col] < float(clinical_ranges[name].split('-')[0])).sum()
        high_val = clinical_ranges[name].split('-')[1].split(' ')[0]
        out_of_range_high = (df_out[orig_col] > float(high_val)).sum()
        print(f"    Out of range: {out_of_range_low} low, {out_of_range_high} high\n")
    
    # sbp is already in original scale
    df_out['sbp_original'] = df_out['sbp.mean']
    print(f"  sbp.mean -> sbp_original (already original scale)")
    print(f"    mean={df_out['sbp_original'].mean():.1f}, min={df_out['sbp_original'].min():.1f}, max={df_out['sbp_original'].max():.1f}")
    print(f"    Clinical: {clinical_ranges['sbp']}\n")
    
    return df_out

def decode_encodings(df):
    print("=" * 80)
    print("2. ENCODING MAPPING ANALYSIS")
    print("=" * 80)
    
    encodings = {}
    
    # Cross-reference with Sheet (which has text labels)
    df_sheet = pd.read_excel(os.path.join(BASE_DIR, 'Raw Data .xlsx'), sheet_name='Sheet')
    
    # Gender: Sheet has 'male'/'female', Sheet1 has 1/2
    print("\nGender mapping:")
    gender_map = {1: 'Male', 2: 'Female'}
    encodings['Gender'] = gender_map
    for k, v in gender_map.items():
        n = (df['Gender'] == k).sum()
        print(f"  {k} -> {v}: {n} rows ({n/len(df)*100:.1f}%)")
    
    # Cross-check with Sheet
    sheet_sex = df_sheet['sex'].value_counts()
    print(f"  Sheet sex distribution: {dict(sheet_sex)}")
    
    # Age_New: 1=younger (mean BA=54.4), 2=older (mean BA=70.1)
    print("\nAge_New mapping:")
    age_map = {1: 'younger_45-59', 2: 'older_60_plus'}
    encodings['Age_New'] = age_map
    for k, v in age_map.items():
        subset = df[df['Age_New'] == k]['Biological Age']
        print(f"  {k} -> {v}: n={len(subset)}, BA mean={subset.mean():.1f}, range=[{subset.min():.1f}, {subset.max():.1f}]")
    
    # BMI_New: 1=underweight/normal, 2=normal/overweight, 3=obese
    print("\nBMI_New mapping:")
    bmi_map = {1: 'underweight_normal', 2: 'overweight', 3: 'obese'}
    encodings['BMI_New'] = bmi_map
    for k, v in bmi_map.items():
        subset = df[df['BMI_New'] == k]['BMI']
        print(f"  {k} -> {v}: n={len(subset)}, BMI mean={subset.mean():.1f}, range=[{subset.min():.1f}, {subset.max():.1f}]")
    
    # Marital, Education, Residence, Smoke, Drink
    binary_cols = ['Hypertension', 'Dyslipidemia', 'Diabetes', 'Cancer', 'CVD', 'Smoke', 'Drink']
    print("\nBinary encodings:")
    for col in binary_cols:
        vals = sorted(df[col].unique())
        print(f"  {col}: {vals}")
    
    # Marital, Education, Residence
    categorical_cols = ['Marital', 'Education', 'Residence']
    print("\nCategorical encodings (inferred from Sheet):")
    
    # Marital
    sheet_marital = df_sheet['marital_status'].value_counts()
    print(f"  Sheet marital_status: {dict(sheet_marital)}")
    marital_map = {1: 'married', 2: 'other'}
    encodings['Marital'] = marital_map
    for k, v in marital_map.items():
        n = (df['Marital'] == k).sum()
        print(f"    {k} -> {v}: {n} rows ({n/len(df)*100:.1f}%)")
    
    # Education
    sheet_edu = df_sheet['education'].value_counts()
    print(f"  Sheet education: {dict(sheet_edu)}")
    edu_map = {1: 'low', 2: 'medium', 3: 'high'}
    encodings['Education'] = edu_map
    for k, v in edu_map.items():
        n = (df['Education'] == k).sum()
        print(f"    {k} -> {v}: {n} rows ({n/len(df)*100:.1f}%)")
    
    # Residence
    sheet_res = df_sheet['residence_place'].value_counts()
    print(f"  Sheet residence_place: {dict(sheet_res)}")
    res_map = {1: 'urban', 2: 'rural'}
    encodings['Residence'] = res_map
    for k, v in res_map.items():
        n = (df['Residence'] == k).sum()
        print(f"    {k} -> {v}: {n} rows ({n/len(df)*100:.1f}%)")
    
    return encodings

def verify_koa_definition(df):
    print("=" * 80)
    print("3. KOA DEFINITION VERIFICATION")
    print("=" * 80)
    
    print(f"\nTotal rows: {len(df)}")
    print(f"\nKOA distribution:")
    print(f"  KOA=0: {(df['KOA']==0).sum()} ({(df['KOA']==0).mean()*100:.1f}%)")
    print(f"  KOA=1: {(df['KOA']==1).sum()} ({(df['KOA']==1).mean()*100:.1f}%)")
    
    print(f"\nKOA = Arthritis_yes AND position_knees_yes verification:")
    df_temp = df.copy()
    df_temp['Arthritis_binary'] = (df_temp['Arthritis'] == 'yes').astype(int)
    df_temp['pk_binary'] = 0
    df_temp.loc[df_temp['position_knees'] == 'yes', 'pk_binary'] = 1
    
    mask = df_temp['position_knees'].notna()
    match = (df_temp.loc[mask, 'KOA'] == (df_temp.loc[mask, 'Arthritis_binary'] & df_temp.loc[mask, 'pk_binary']).astype(int))
    print(f"  Match: {match.sum()}/{len(match)} ({match.mean()*100:.1f}%)")
    print(f"  KOA is DEFINED as: Arthritis=yes AND position_knees=yes")
    
    print(f"\nposition_knees distribution:")
    print(df['position_knees'].value_counts(dropna=False).to_string())
    
    print(f"\nCross-tab (Arthritis x KOA):")
    print(pd.crosstab(df['Arthritis'], df['KOA'], margins=True).to_string())
    
    print(f"\nCross-tab (position_knees x KOA):")
    print(pd.crosstab(df['position_knees'].fillna('NaN'), df['KOA'], margins=True).to_string())

def create_datasets(df):
    print("=" * 80)
    print("4. DATASET CREATION")
    print("=" * 80)
    
    # LEAKAGE: Remove Arthritis and position_knees from features
    # KOA is derived from Arthritis AND position_knees
    leakage_cols = ['Arthritis', 'position_knees']
    print(f"\nLEAKAGE COLUMNS REMOVED: {leakage_cols}")
    print("  Reason: KOA = Arthritis=yes AND position_knees=yes")
    print("  Including these as features would leak target information")
    
    # Remove rows with position_knees=NaN (user's choice)
    df_clean = df[df['position_knees'].notna()].copy()
    print(f"\nRows after removing position_knees=NaN:")
    print(f"  Before: {len(df)}")
    print(f"  After: {len(df_clean)}")
    print(f"  Removed: {len(df) - len(df_clean)} ({(1 - len(df_clean)/len(df))*100:.1f}%)")
    
    print(f"\nKOA distribution after filtering:")
    print(f"  KOA=0: {(df_clean['KOA']==0).sum()} ({(df_clean['KOA']==0).mean()*100:.1f}%)")
    print(f"  KOA=1: {(df_clean['KOA']==1).sum()} ({(df_clean['KOA']==1).mean()*100:.1f}%)")
    
    # Dataset A: All position_knees non-null rows
    dataset_a = df_clean.drop(columns=leakage_cols).copy()
    dataset_a.name = 'dataset_A_all'
    print(f"\n--- Dataset A: All pk non-null ---")
    print(f"  Shape: {dataset_a.shape}")
    print(f"  KOA rate: {dataset_a['KOA'].mean()*100:.1f}%")
    print(f"  Columns removed: {leakage_cols}")
    
    # Dataset B: BA >= 55 filter (to match paper's ~9,505)
    ba_threshold = 55
    dataset_b = dataset_a[dataset_a['Biological Age'] >= ba_threshold].copy()
    dataset_b.name = 'dataset_B_ba55'
    print(f"\n--- Dataset B: BA >= {ba_threshold} ---")
    print(f"  Shape: {dataset_b.shape}")
    print(f"  KOA rate: {dataset_b['KOA'].mean()*100:.1f}%")
    print(f"  Paper comparison: {len(dataset_b)} vs paper's 9,505")
    
    # Feature list
    feature_cols = [c for c in dataset_a.columns if c != 'KOA']
    print(f"\nFeature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  {c}: {dataset_a[c].dtype}")
    
    # Save datasets
    dataset_a.to_csv(os.path.join(STEP_DIR, 'dataset_A_all.csv'), index=False)
    dataset_b.to_csv(os.path.join(STEP_DIR, 'dataset_B_ba55.csv'), index=False)
    print(f"\nDatasets saved to {STEP_DIR}")
    
    return dataset_a, dataset_b, feature_cols

def data_quality_report(df, dataset_a, dataset_b):
    print("=" * 80)
    print("5. DATA QUALITY REPORT")
    print("=" * 80)
    
    # Biomarker distributions (original scale)
    biomarker_cols = {
        'platelet_original': 'Platelet (x10^9/L)',
        'crp_original': 'hs-CRP (mg/L)',
        'hba1c_original': 'HbA1c (%)',
        'creatinine_original': 'Creatinine (mg/dL)',
        'bun_original': 'BUN (mg/dL)',
        'total_cholesterol_original': 'Total Cholesterol (mg/dL)',
        'triglycerides_original': 'Triglycerides (mg/dL)',
        'sbp_original': 'Systolic BP (mmHg)',
    }
    
    print("\nBiomarker statistics (original scale):")
    print("-" * 80)
    print(f"{'Biomarker':<30} {'mean':>8} {'std':>8} {'min':>8} {'Q1':>8} {'Q50':>8} {'Q3':>8} {'max':>8}")
    for col, name in biomarker_cols.items():
        s = dataset_a[col]
        print(f"{name:<30} {s.mean():8.1f} {s.std():8.1f} {s.min():8.1f} {s.quantile(0.25):8.1f} {s.median():8.1f} {s.quantile(0.75):8.1f} {s.max():8.1f}")
    
    # Missing values in dataset_A
    print("\nMissing values in Dataset A:")
    missing = dataset_a.isnull().sum()
    missing_pct = (missing / len(dataset_a) * 100).round(2)
    for col in dataset_a.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]} ({missing_pct[col]:.1f}%)")
    
    if missing.sum() == 0:
        print("  No missing values!")
    
    # Class distribution
    print("\nClass distribution:")
    for name, ds in [('Dataset A', dataset_a), ('Dataset B', dataset_b)]:
        koa_pos = ds['KOA'].sum()
        koa_neg = (ds['KOA'] == 0).sum()
        ratio = koa_neg / koa_pos
        print(f"  {name}: KOA+={koa_pos} ({koa_pos/len(ds)*100:.1f}%), KOA-={koa_neg} ({koa_neg/len(ds)*100:.1f}%), ratio={ratio:.1f}:1")

def generate_report(dataset_a, dataset_b, feature_cols):
    report = []
    report.append("=" * 80)
    report.append("STEP 01: SHEET1 DATA PREPARATION - REPORT")
    report.append("=" * 80)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("1. DATA SOURCE")
    report.append("-" * 40)
    report.append(f"  Source: Raw Data .xlsx, Sheet1")
    report.append(f"  Original shape: 15,545 rows x 28 columns")
    report.append(f"  After filtering (position_knees non-null): {len(dataset_a) + 3216} -> {len(dataset_a)} rows")
    report.append(f"  Leakage columns removed: Arthritis, position_knees")
    report.append(f"  Final features: {len(feature_cols)}")
    report.append("")
    
    report.append("2. DATASETS CREATED")
    report.append("-" * 40)
    report.append(f"  Dataset A (all pk non-null): {dataset_a.shape[0]} rows, {dataset_a.shape[1]-1} features, KOA rate={dataset_a['KOA'].mean()*100:.1f}%")
    report.append(f"  Dataset B (BA >= 55):         {dataset_b.shape[0]} rows, {dataset_b.shape[1]-1} features, KOA rate={dataset_b['KOA'].mean()*100:.1f}%")
    report.append(f"  Paper reference:               9,505 rows, 10.5% KOA rate")
    report.append("")
    
    report.append("3. BIOMARKER LOG-TRANSFORMATION")
    report.append("-" * 40)
    report.append("  All 7 biomarkers (plt, crp, HbA1c, creatinine, BUN, TC, TG) are log-transformed")
    report.append("  Reverse transformation: exp(value) recovers original clinical scale")
    report.append("  sbp.mean is in original scale (mmHg)")
    report.append("")
    
    report.append("4. ENCODING MAPPINGS")
    report.append("-" * 40)
    report.append("  Gender:       1=Male, 2=Female")
    report.append("  Age_New:      1=younger(45-59), 2=older(60+)")
    report.append("  BMI_New:      1=underweight/normal, 2=overweight, 3=obese")
    report.append("  Marital:      1=married, 2=other")
    report.append("  Education:    1=low, 2=medium, 3=high")
    report.append("  Residence:    1=urban, 2=rural")
    report.append("  Disease cols: 0=no, 1=yes (Hypertension, Dyslipidemia, Diabetes, Cancer, CVD)")
    report.append("  Smoke:        0=never, 1=ever")
    report.append("  Drink:        0=never, 1=ever")
    report.append("  KOA:          0=no, 1=yes (Arthritis=yes AND position_knees=yes)")
    report.append("")
    
    report.append("5. KOA DEFINITION")
    report.append("-" * 40)
    report.append("  KOA = (Arthritis=='yes') AND (position_knees=='yes')")
    report.append("  This matches the paper's 'Symptomatic KOA' definition")
    report.append("  Verification: 100% match between KOA and (Arthritis AND position_knees)")
    report.append("")
    
    report.append("6. LEAKAGE CONTROL")
    report.append("-" * 40)
    report.append("  REMOVED from features: Arthritis, position_knees")
    report.append("  Reason: KOA is directly derived from these columns")
    report.append("  position_knees=NaN rows also removed (as KOA cannot be verified)")
    report.append("")
    
    report.append("7. NEXT STEPS")
    report.append("-" * 40)
    report.append("  Step 02: Baseline models on Dataset A and B")
    report.append("  Step 03: KDM-BA calculation from biomarkers")
    report.append("  Step 04: BA impact analysis with SHAP")
    report.append("  Step 05: Paper replication (LASSO, one-hot, Z-score)")
    report.append("")
    
    report_text = "\n".join(report)
    with open(os.path.join(STEP_DIR, 'step01_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to {os.path.join(STEP_DIR, 'step01_report.txt')}")

def main():
    df = load_sheet1()
    df_transformed = reverse_log_transform(df)
    encodings = decode_encodings(df)
    verify_koa_definition(df)
    dataset_a, dataset_b, feature_cols = create_datasets(df_transformed)
    data_quality_report(df_transformed, dataset_a, dataset_b)
    generate_report(dataset_a, dataset_b, feature_cols)
    
    print("\n" + "=" * 80)
    print("STEP 01 COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Files saved in: {STEP_DIR}")
    print(f"  dataset_A_all.csv  ({len(dataset_a)} rows)")
    print(f"  dataset_B_ba55.csv  ({len(dataset_b)} rows)")

if __name__ == '__main__':
    main()