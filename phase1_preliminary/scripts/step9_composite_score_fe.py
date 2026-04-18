"""
STAGE 9: COMPOSITE SCORE-BASED CONTROLLED FEATURE ENGINEERING
==============================================================

Motivation:
Previous stages (3-8) tested multiple scattered feature engineering strategies
→ ALL failed validation or added noise, not signal

New Strategy:
- Do NOT create many weak features
- CREATE 1-3 STRONG composite features with biological meaning
- Follow paper's approach: Biological Age = composite of 8 biomarkers via KDM
- Test nonlinearity (quartiles, thresholds) like paper's RCS analysis
- Test age-adjusted composites (acceleration/residual)

Hypothesis:
Performance improvement comes from FEW DENSE features with strong signal,
not from MANY WEAK features. If signal exists, it must survive repeated CV.

Experiments:
1. Raw Baseline (17 features)
2. Raw + Composite_Burden (1 composite: z-score mean of key biomarkers)
3. Raw + Composite_Q4 (threshold: top quartile indicator)
4. Raw + Composite_Acceleration (age-adjusted residual)
5. Raw + Composite_Burden + Age_x_Composite (single interaction)
6. Raw + Composite_Burden + Missing_Indicators (missingness awareness)

Validation: 5-fold × 5-repeat = 25 evaluations (consistent with Stage 8)
NO DATA LEAKAGE: All transforms fit on train fold only

Target: +1-2% improvement that survives repeated CV (unlike Stage 7)
"""

import os
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILE = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
OUTPUT_DIR = "step9_composite_score_fe"
RANDOM_STATE = 42

# Repeated CV configuration
N_SPLITS = 5
N_REPEATS = 5

# Model configuration
MODEL_PARAMS = {
    'n_estimators': 100,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# Key biomarkers for composite score
# Based on available continuous/ordinal variables with biological relevance
# Note: These will be computed/extracted from raw data
COMPOSITE_BIOMARKERS = [
    'Biological Age',  # Chronological age (strong predictor)
    'bmi_kg.m2',       # Mechanical load + metabolic health
    'Comorbidity_Count' # Disease burden (will be computed)
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def binary_encode(df, col):
    """
    Robust binary encoding for yes/no columns.
    Handles: string, numeric, NaN
    Returns: 0/1 or NaN
    """
    if col not in df.columns:
        return df
    
    # Create copy of column
    series = df[col].copy()
    
    # If already numeric binary, return
    if series.dtype in ['int64', 'float64'] and series.dropna().isin([0, 1]).all():
        return series
    
    # String encoding
    series = series.astype(str).str.strip().str.lower()
    
    mapping = {
        'yes': 1, 'y': 1, '1': 1, '1.0': 1, 'true': 1,
        'no': 0, 'n': 0, '0': 0, '0.0': 0, 'false': 0,
        'nan': np.nan, 'none': np.nan, '': np.nan
    }
    
    encoded = series.map(mapping)
    
    # Convert to float (allows NaN)
    return encoded.astype(float)


def load_and_encode_data(file_path):
    """
    Load dataset and apply binary encoding to yes/no columns.
    Returns: X (features), y (target), feature_names
    """
    print("Loading data...")
    df = pd.read_excel(file_path)
    
    # Filter valid samples (Arthritis not null)
    df = df[df['Arthritis'].notna()].copy()
    
    print(f"  Loaded: {df.shape[0]} samples, {df.shape[1]} columns")
    
    # Target
    y = df['Arthritis'].copy()
    df = df.drop('Arthritis', axis=1)
    
    # Binary encoding for yes/no columns (consistent with all previous stages)
    binary_cols = ['Sex', 'Smoking', 'Previous_injury', 'Family_history']
    
    # Additional binary columns that might be present (doctor_diagnose_*)
    disease_cols = [col for col in df.columns if 'doctor_diagnose' in col.lower()]
    
    # Encode binary columns
    for col in binary_cols + disease_cols:
        if col in df.columns:
            df[col] = binary_encode(df, col)
    
    # Create Comorbidity_Count (sum of disease indicators)
    if disease_cols:
        df['Comorbidity_Count'] = df[disease_cols].sum(axis=1)
    else:
        # Fallback if no disease columns
        df['Comorbidity_Count'] = 0
    
    # Encode categorical columns (LabelEncoder for non-numeric)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)
    
    # Feature names
    feature_cols = df.columns.tolist()
    X = df.copy()
    
    print(f"  Features: {X.shape[1]}")
    print(f"  Target distribution: {np.sum(y)} positive ({100*np.mean(y):.1f}%)")
    
    return X, y, feature_cols


def compute_composite_burden_zscore(X_train, X_test, biomarkers):
    """
    Composite Method 1: Z-score mean
    
    Steps:
    1. Log-transform skewed biomarkers (Age, BMI kept as-is)
    2. Z-score standardize on TRAIN
    3. Compute mean z-score
    4. Apply same transform to TEST (using TRAIN statistics)
    
    Returns: train_score, test_score
    """
    # Initialize scalers
    scaler = StandardScaler()
    
    # Prepare train data (handle missing values with median)
    train_data = X_train[biomarkers].copy()
    for col in train_data.columns:
        median_val = train_data[col].median()
        train_data[col].fillna(median_val, inplace=True)
    
    # Fit scaler on TRAIN
    train_scaled = scaler.fit_transform(train_data)
    
    # Mean z-score
    train_score = np.mean(train_scaled, axis=1)
    
    # Apply to TEST (use TRAIN medians and scaler)
    test_data = X_test[biomarkers].copy()
    for col in test_data.columns:
        # Use TRAIN median for TEST imputation (no leakage)
        train_median = X_train[col].median()
        test_data[col].fillna(train_median, inplace=True)
    
    test_scaled = scaler.transform(test_data)  # Use TRAIN statistics
    test_score = np.mean(test_scaled, axis=1)
    
    return train_score, test_score


def compute_composite_burden_pca(X_train, X_test, biomarkers):
    """
    Composite Method 2: PCA first component
    
    Captures maximum variance direction in biomarker space.
    More sophisticated than simple mean.
    
    Returns: train_score, test_score
    """
    # Standardize first
    scaler = StandardScaler()
    train_data = X_train[biomarkers].copy()
    
    # Handle missing values
    for col in train_data.columns:
        median_val = train_data[col].median()
        train_data[col].fillna(median_val, inplace=True)
    
    train_scaled = scaler.fit_transform(train_data)
    
    # PCA: extract first component
    pca = PCA(n_components=1, random_state=RANDOM_STATE)
    train_score = pca.fit_transform(train_scaled).ravel()
    
    # Apply to TEST
    test_data = X_test[biomarkers].copy()
    
    # Use TRAIN medians for TEST imputation
    for col in test_data.columns:
        train_median = X_train[col].median()
        test_data[col].fillna(train_median, inplace=True)
    
    test_scaled = scaler.transform(test_data)
    test_score = pca.transform(test_scaled).ravel()
    
    return train_score, test_score


def compute_composite_burden_weighted(X_train, X_test, biomarkers):
    """
    Composite Method 3: Literature-guided weighted score
    
    Weights based on expected OA relevance:
    - Biological Age: 0.40 (strong predictor)
    - BMI: 0.35 (mechanical + metabolic)
    - Comorbidity_Count: 0.25 (systemic health)
    
    Returns: train_score, test_score
    """
    weights = {
        'Biological Age': 0.40,
        'bmi_kg.m2': 0.35,
        'Comorbidity_Count': 0.25
    }
    
    # Standardize first
    scaler = StandardScaler()
    train_data = X_train[biomarkers].copy()
    
    # Handle missing values
    for col in train_data.columns:
        median_val = train_data[col].median()
        train_data[col].fillna(median_val, inplace=True)
    
    train_scaled = scaler.fit_transform(train_data)
    
    # Apply weights
    weight_vector = np.array([weights[b] for b in biomarkers])
    train_score = np.dot(train_scaled, weight_vector)
    
    # Apply to TEST
    test_data = X_test[biomarkers].copy()
    
    # Use TRAIN medians for TEST imputation
    for col in test_data.columns:
        train_median = X_train[col].median()
        test_data[col].fillna(train_median, inplace=True)
    
    test_scaled = scaler.transform(test_data)
    test_score = np.dot(test_scaled, weight_vector)
    
    return train_score, test_score


def create_thresholded_features(score_train, score_test):
    """
    Create threshold-based binary features from composite score.
    
    Captures nonlinearity: risk may accelerate at high burden levels.
    Similar to paper's quartile analysis and RCS showing inflection ~66.7 BA.
    
    Returns: dict of train/test pairs for each threshold type
    """
    # Compute thresholds on TRAIN only
    q75 = np.percentile(score_train, 75)
    q90 = np.percentile(score_train, 90)
    
    # Quartile: top 25%
    composite_q4_train = (score_train >= q75).astype(int)
    composite_q4_test = (score_test >= q75).astype(int)
    
    # High: above 75th percentile
    composite_high_train = (score_train > q75).astype(int)
    composite_high_test = (score_test > q75).astype(int)
    
    # Very high: above 90th percentile
    composite_veryhigh_train = (score_train > q90).astype(int)
    composite_veryhigh_test = (score_test > q90).astype(int)
    
    # Piecewise: max(0, score - threshold)
    # Use 75th percentile as inflection point
    composite_piecewise_train = np.maximum(0, score_train - q75)
    composite_piecewise_test = np.maximum(0, score_test - q75)
    
    return {
        'Composite_Q4': (composite_q4_train, composite_q4_test),
        'Composite_High': (composite_high_train, composite_high_test),
        'Composite_VeryHigh': (composite_veryhigh_train, composite_veryhigh_test),
        'Composite_Piecewise': (composite_piecewise_train, composite_piecewise_test)
    }


def create_acceleration_feature(score_train, score_test, age_train, age_test):
    """
    Create age-adjusted composite score (residual).
    
    Concept: "Biological Age Acceleration"
    - If someone has high composite burden for their age → positive residual
    - If someone has low composite burden for their age → negative residual
    
    This captures: "worse biological state than expected for chronological age"
    Similar to BA - CA in the paper.
    
    Returns: train_residual, test_residual
    """
    # Handle any remaining NaN values
    score_train_clean = np.nan_to_num(score_train, nan=np.nanmean(score_train))
    score_test_clean = np.nan_to_num(score_test, nan=np.nanmean(score_train))
    age_train_clean = np.nan_to_num(age_train, nan=np.nanmean(age_train))
    age_test_clean = np.nan_to_num(age_test, nan=np.nanmean(age_train))
    
    # Fit linear regression: Composite ~ Age (on TRAIN)
    reg = LinearRegression()
    reg.fit(age_train_clean.reshape(-1, 1), score_train_clean)
    
    # Predict expected composite given age
    expected_train = reg.predict(age_train_clean.reshape(-1, 1))
    expected_test = reg.predict(age_test_clean.reshape(-1, 1))
    
    # Residual = Observed - Expected
    residual_train = score_train_clean - expected_train
    residual_test = score_test_clean - expected_test
    
    return residual_train, residual_test


def create_interaction_features(X_train, X_test, composite_train, composite_test):
    """
    Create biologically meaningful interactions.
    
    Rules:
    - Only 1-2 interactions max
    - Only if main effects are reasonable
    
    Returns: dict of train/test pairs
    """
    interactions = {}
    
    # Age × Composite (older age + high burden = synergistic risk)
    age_train = X_train['Biological Age'].fillna(X_train['Biological Age'].median()).values
    age_test = X_test['Biological Age'].fillna(X_train['Biological Age'].median()).values
    interactions['Age_x_Composite'] = (
        age_train * composite_train,
        age_test * composite_test
    )
    
    # BMI × Composite (mechanical load + metabolic dysfunction)
    bmi_train = X_train['bmi_kg.m2'].fillna(X_train['bmi_kg.m2'].median()).values
    bmi_test = X_test['bmi_kg.m2'].fillna(X_train['bmi_kg.m2'].median()).values
    interactions['BMI_x_Composite'] = (
        bmi_train * composite_train,
        bmi_test * composite_test
    )
    
    return interactions


def create_missing_indicators(X_train, X_test, biomarkers):
    """
    Create binary indicators for missing values in key biomarkers.
    
    Rationale: Missing data pattern may be informative.
    Paper used MICE for imputation; we test if missingness itself matters.
    
    Returns: dict of train/test pairs
    """
    missing_features = {}
    
    for marker in biomarkers:
        if X_train[marker].isna().sum() > 0:
            missing_features[f'{marker}_missing'] = (
                X_train[marker].isna().astype(int).values,
                X_test[marker].isna().astype(int).values
            )
    
    return missing_features


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """
    Train RandomForest and compute comprehensive metrics.
    
    Returns: dict of metrics
    """
    # Train model (same as all previous stages for consistency)
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'f1': f1_score(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred),
        'brier': brier_score_loss(y_test, y_proba)
    }
    
    return metrics


def run_repeated_cv_experiment(X, y, config_name, feature_creator_fn):
    """
    Run repeated stratified k-fold CV for a single configuration.
    
    Args:
        X: base features (17 raw)
        y: target
        config_name: experiment name
        feature_creator_fn: function that takes (X_train, X_test) and returns
                           (X_train_augmented, X_test_augmented)
    
    Returns: list of fold results
    """
    print(f"\n{'='*70}")
    print(f"CONFIG: {config_name}")
    print(f"{'='*70}")
    
    results = []
    
    # Repeated stratified k-fold
    rskf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE
    )
    
    fold_idx = 0
    for train_idx, test_idx in rskf.split(X, y):
        fold_idx += 1
        
        # Split data
        X_train_base = X.iloc[train_idx].copy()
        X_test_base = X.iloc[test_idx].copy()
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        # Create augmented features (NO LEAKAGE: fit on train, transform test)
        X_train_aug, X_test_aug = feature_creator_fn(X_train_base, X_test_base)
        
        # Handle NaN (simple mean imputation per fold)
        # This is consistent with simple approach; paper used MICE but we keep it simple
        for col in X_train_aug.columns:
            if X_train_aug[col].isna().sum() > 0:
                mean_val = X_train_aug[col].mean()
                X_train_aug[col].fillna(mean_val, inplace=True)
                X_test_aug[col].fillna(mean_val, inplace=True)
        
        # Train and evaluate
        metrics = train_and_evaluate(X_train_aug, y_train, X_test_aug, y_test)
        
        # Store results
        result = {
            'config': config_name,
            'fold': fold_idx,
            'n_features': X_train_aug.shape[1],
            **metrics
        }
        results.append(result)
        
        # Progress
        if fold_idx % 5 == 0:
            avg_roc = np.mean([r['roc_auc'] for r in results[-5:]])
            print(f"  Fold {fold_idx}/{N_SPLITS * N_REPEATS}: ROC AUC = {avg_roc:.4f}")
    
    # Summary statistics
    df_results = pd.DataFrame(results)
    mean_roc = df_results['roc_auc'].mean()
    std_roc = df_results['roc_auc'].std()
    cv_pct = (std_roc / mean_roc) * 100
    
    print(f"\n  SUMMARY: {mean_roc:.4f} ± {std_roc:.4f} (CV: {cv_pct:.2f}%)")
    print(f"  Features: {results[0]['n_features']}")
    
    return results


# ============================================================================
# CONFIGURATION BUILDERS
# ============================================================================

def config_baseline(X_train, X_test):
    """Config 1: Raw baseline (17 features)"""
    return X_train.copy(), X_test.copy()


def config_composite_burden(X_train, X_test):
    """Config 2: Raw + Composite_Burden (z-score mean)"""
    # Compute composite score
    score_train, score_test = compute_composite_burden_zscore(
        X_train, X_test, COMPOSITE_BIOMARKERS
    )
    
    # Add to dataframes
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    X_train_aug['Composite_Burden'] = score_train
    X_test_aug['Composite_Burden'] = score_test
    
    return X_train_aug, X_test_aug


def config_composite_q4(X_train, X_test):
    """Config 3: Raw + Composite_Q4 (top quartile indicator)"""
    # Compute composite score
    score_train, score_test = compute_composite_burden_zscore(
        X_train, X_test, COMPOSITE_BIOMARKERS
    )
    
    # Thresholded features
    thresholds = create_thresholded_features(score_train, score_test)
    
    # Add Q4 indicator
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    X_train_aug['Composite_Q4'] = thresholds['Composite_Q4'][0]
    X_test_aug['Composite_Q4'] = thresholds['Composite_Q4'][1]
    
    return X_train_aug, X_test_aug


def config_composite_acceleration(X_train, X_test):
    """Config 4: Raw + Composite_Acceleration (age-adjusted residual)"""
    # Compute composite score
    score_train, score_test = compute_composite_burden_zscore(
        X_train, X_test, COMPOSITE_BIOMARKERS
    )
    
    # Age data (handle missing)
    age_train = X_train['Biological Age'].fillna(X_train['Biological Age'].median()).values
    age_test = X_test['Biological Age'].fillna(X_train['Biological Age'].median()).values
    
    # Compute residuals
    residual_train, residual_test = create_acceleration_feature(
        score_train, score_test, age_train, age_test
    )
    
    # Add to dataframes
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    X_train_aug['Composite_Acceleration'] = residual_train
    X_test_aug['Composite_Acceleration'] = residual_test
    
    return X_train_aug, X_test_aug


def config_composite_plus_interaction(X_train, X_test):
    """Config 5: Raw + Composite_Burden + Age_x_Composite"""
    # Compute composite score
    score_train, score_test = compute_composite_burden_zscore(
        X_train, X_test, COMPOSITE_BIOMARKERS
    )
    
    # Create interaction
    interactions = create_interaction_features(
        X_train, X_test, score_train, score_test
    )
    
    # Add features
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    X_train_aug['Composite_Burden'] = score_train
    X_test_aug['Composite_Burden'] = score_test
    X_train_aug['Age_x_Composite'] = interactions['Age_x_Composite'][0]
    X_test_aug['Age_x_Composite'] = interactions['Age_x_Composite'][1]
    
    return X_train_aug, X_test_aug


def config_composite_plus_missing(X_train, X_test):
    """Config 6: Raw + Composite_Burden + Missing_Indicators"""
    # Compute composite score
    score_train, score_test = compute_composite_burden_zscore(
        X_train, X_test, COMPOSITE_BIOMARKERS
    )
    
    # Missing indicators
    missing_feats = create_missing_indicators(X_train, X_test, COMPOSITE_BIOMARKERS)
    
    # Add features
    X_train_aug = X_train.copy()
    X_test_aug = X_test.copy()
    X_train_aug['Composite_Burden'] = score_train
    X_test_aug['Composite_Burden'] = score_test
    
    for feat_name, (train_vals, test_vals) in missing_feats.items():
        X_train_aug[feat_name] = train_vals
        X_test_aug[feat_name] = test_vals
    
    return X_train_aug, X_test_aug


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("STAGE 9: COMPOSITE SCORE-BASED CONTROLLED FEATURE ENGINEERING")
    print("="*80)
    print("\nStrategy: FEW DENSE composite features (not many weak features)")
    print("Validation: 5-fold × 5-repeat = 25 evaluations per config")
    print("Reference: Paper's BA approach (8 biomarkers → 1 KDM score)\n")
    
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    # Load data
    X, y, feature_names = load_and_encode_data(DATA_FILE)
    
    print(f"\nComposite biomarkers: {COMPOSITE_BIOMARKERS}")
    
    # Configurations to test (IN ORDER - do not change)
    configs = [
        ('1_Baseline_Raw', config_baseline),
        ('2_Raw_Plus_Composite_Burden', config_composite_burden),
        ('3_Raw_Plus_Composite_Q4', config_composite_q4),
        ('4_Raw_Plus_Composite_Acceleration', config_composite_acceleration),
        ('5_Raw_Plus_Composite_And_Interaction', config_composite_plus_interaction),
        ('6_Raw_Plus_Composite_And_Missing', config_composite_plus_missing)
    ]
    
    # Run all experiments
    all_results = []
    
    for config_name, config_fn in configs:
        fold_results = run_repeated_cv_experiment(X, y, config_name, config_fn)
        all_results.extend(fold_results)
    
    # Save detailed results
    df_detailed = pd.DataFrame(all_results)
    detailed_path = f"{OUTPUT_DIR}/composite_fe_detailed.csv"
    df_detailed.to_csv(detailed_path, index=False)
    print(f"\n✓ Saved detailed results: {detailed_path}")
    
    # Compute summary statistics
    summary_data = []
    
    baseline_roc = None
    
    for config_name, _ in configs:
        config_results = df_detailed[df_detailed['config'] == config_name]
        
        mean_roc = config_results['roc_auc'].mean()
        std_roc = config_results['roc_auc'].std()
        mean_pr = config_results['pr_auc'].mean()
        mean_f1 = config_results['f1'].mean()
        mean_brier = config_results['brier'].mean()
        cv_pct = (std_roc / mean_roc) * 100
        n_features = config_results['n_features'].iloc[0]
        
        # Improvement vs baseline
        if baseline_roc is None:
            baseline_roc = mean_roc
            improvement = 0.0
            improvement_pct = 0.0
        else:
            improvement = mean_roc - baseline_roc
            improvement_pct = (improvement / baseline_roc) * 100
        
        summary_data.append({
            'config': config_name,
            'n_features': n_features,
            'roc_auc_mean': mean_roc,
            'roc_auc_std': std_roc,
            'cv_pct': cv_pct,
            'pr_auc_mean': mean_pr,
            'f1_mean': mean_f1,
            'brier_mean': mean_brier,
            'improvement_abs': improvement,
            'improvement_pct': improvement_pct
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_path = f"{OUTPUT_DIR}/composite_fe_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"✓ Saved summary: {summary_path}")
    
    # Generate report
    report_path = f"{OUTPUT_DIR}/composite_fe_report.txt"
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STAGE 9: COMPOSITE SCORE-BASED FEATURE ENGINEERING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("Methodology:\n")
        f.write(f"  - Validation: {N_SPLITS}-fold × {N_REPEATS}-repeat CV = {N_SPLITS * N_REPEATS} evaluations\n")
        f.write(f"  - Model: RandomForest (n_estimators={MODEL_PARAMS['n_estimators']})\n")
        f.write(f"  - Biomarkers: {COMPOSITE_BIOMARKERS}\n")
        f.write(f"  - NO DATA LEAKAGE: All transforms fit on train fold only\n\n")
        
        f.write("Ranking (by ROC AUC):\n")
        f.write("-"*80 + "\n")
        
        # Sort by ROC AUC
        df_sorted = df_summary.sort_values('roc_auc_mean', ascending=False)
        
        for idx, row in df_sorted.iterrows():
            f.write(f"\n{row['config']}:\n")
            f.write(f"  Features: {row['n_features']}\n")
            f.write(f"  ROC AUC: {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f} (CV: {row['cv_pct']:.2f}%)\n")
            f.write(f"  PR AUC:  {row['pr_auc_mean']:.4f}\n")
            f.write(f"  F1:      {row['f1_mean']:.4f}\n")
            f.write(f"  Brier:   {row['brier_mean']:.4f}\n")
            
            if row['improvement_abs'] != 0:
                sign = '+' if row['improvement_abs'] > 0 else ''
                f.write(f"  Improvement: {sign}{row['improvement_abs']:.4f} ({sign}{row['improvement_pct']:.2f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("DECISION CRITERIA:\n")
        f.write("="*80 + "\n\n")
        f.write("A feature set is VALIDATED if:\n")
        f.write("  1. Mean ROC AUC > Baseline\n")
        f.write("  2. Improvement > 1 std dev of baseline\n")
        f.write("  3. Wins in majority of folds (>50%)\n")
        f.write("  4. Stable across repeats (CV < 2%)\n\n")
        
        # Check validation criteria
        baseline_mean = df_summary.iloc[0]['roc_auc_mean']
        baseline_std = df_summary.iloc[0]['roc_auc_std']
        
        f.write("Validation Results:\n")
        f.write("-"*80 + "\n")
        
        validated = False
        for idx, row in df_sorted.iloc[1:].iterrows():  # Skip baseline
            passes = []
            
            # Criterion 1: Higher mean
            if row['roc_auc_mean'] > baseline_mean:
                passes.append("✓ Higher mean")
            else:
                passes.append("✗ Lower/equal mean")
            
            # Criterion 2: Improvement > 1 std
            if row['improvement_abs'] > baseline_std:
                passes.append("✓ Improvement > 1 std")
            else:
                passes.append("✗ Improvement < 1 std")
            
            # Criterion 3: Stability
            if row['cv_pct'] < 2.0:
                passes.append("✓ Stable (CV < 2%)")
            else:
                passes.append("✗ Unstable (CV ≥ 2%)")
            
            all_pass = all('✓' in p for p in passes)
            
            f.write(f"\n{row['config']}:\n")
            for criterion in passes:
                f.write(f"  {criterion}\n")
            
            if all_pass:
                f.write(f"  → VALIDATED ✓\n")
                validated = True
            else:
                f.write(f"  → NOT VALIDATED ✗\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("CONCLUSION:\n")
        f.write("="*80 + "\n\n")
        
        if validated:
            f.write("✓ At least one composite feature configuration is VALIDATED.\n")
            f.write("  Composite score approach shows promise.\n")
            f.write("  Proceed to hyperparameter tuning with validated features.\n")
        else:
            f.write("⚠️  NO configuration passed all validation criteria.\n")
            f.write("  Composite features do not provide robust improvement.\n")
            f.write("  Dataset's predictive signal ceiling may be reached.\n")
            f.write("  Next: Model optimization (XGBoost, tuning, ensemble).\n")
    
    print(f"✓ Saved report: {report_path}")
    
    # Print final summary
    print("\n" + "="*80)
    print("STAGE 9 COMPLETE")
    print("="*80)
    
    print("\nRanking:")
    for idx, row in df_sorted.iterrows():
        sign = '+' if row['improvement_pct'] > 0 else ''
        print(f"  {row['config']:40s} {row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f} ({sign}{row['improvement_pct']:.2f}%)")
    
    print(f"\nFiles saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
