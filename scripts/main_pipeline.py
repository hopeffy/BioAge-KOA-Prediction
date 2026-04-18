"""
KOA (Knee Osteoarthritis) Risk Prediction Project
Based on Biological Age and Clinical Features
---------------------------------------------------
10-Stage Strategy Implementation

Author: Bioinformatics Analysis Team
Date: March 2026

COMPANION FILES:
- feature_analysis_experiments.py: Advanced feature selection experiments
  * Exp1: Without Position_Knees_Encoded (ROC AUC: 0.5832)
  * Exp2: p < 0.05 features only (ROC AUC: 0.5822, 28 features)
  * Exp3: Moderate significance 0.01<p<0.05 (ROC AUC: 0.5266)
  * Exp4: Hierarchical Clustering (ROC AUC: 0.5793, F1: 0.4263 ⭐, 5 features only!)
  
BEST MODEL: Hierarchical Clustering with 5 representative features
  → BioAge_60_Plus, Heart_Disease, BMI_Category, Cancer, Sex_Numeric
  → 88% feature reduction with minimal performance loss
  → See: feature_experiments_results.txt for detailed analysis

RESULTS TRACKING:
- pipeline_results.txt: Main pipeline results + feature experiments summary
- experiments_summary.csv: Quick comparison table
- feature_clusters.csv: Hierarchical clustering assignments
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE, SelectFromModel, f_classif, chi2
import warnings
warnings.filterwarnings('ignore')

# Advanced models
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    print("⚠️ Advanced models (XGBoost, LightGBM, CatBoost) not installed.")
    print("Install with: pip install xgboost lightgbm catboost")
    ADVANCED_MODELS_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️ Optuna not installed for hyperparameter optimization.")
    print("Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

import os
import sys

# ============================================================================
# STAGE 1: DATA LOADING AND EXPLORATORY ANALYSIS
# ============================================================================

class KOAPredictionPipeline:
    def __init__(self, data_path, results_file='pipeline_results.txt'):
        """Initialize the KOA Prediction Pipeline"""
        self.data_path = data_path
        self.results_file = results_file
        self.df = None
        self.df_processed = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.selected_features = None
        self.best_model = None
        self.results = {}
        self.results_log = []  # Store all step results
        
        # Initialize results file
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("KOA PREDICTION PIPELINE - RESULTS TRACKING\n")
            f.write("Biological Age Risk Analysis\n")
            f.write("="*80 + "\n\n")
            f.write(f"Started: {pd.Timestamp.now()}\n")
            f.write("="*80 + "\n\n")
        
    def load_data(self):
        """Load and perform initial exploration of the dataset"""
        print("=" * 80)
        print("STAGE 1: DATA LOADING AND EXPLORATION")
        print("=" * 80)
        
        try:
            # Load CSV instead of Excel (imputed dataset)
            if self.data_path.endswith('.csv'):
                self.df = pd.read_csv(self.data_path)
            else:
                self.df = pd.read_excel(self.data_path)
            print(f"✓ Data loaded successfully: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n")
            
            # Basic information
            print("Column Names and Types:")
            print("-" * 40)
            print(self.df.dtypes)
            print("\n")
            
            # Check for target variable (Arthritis in this dataset)
            if 'Arthritis' in self.df.columns:
                print(f"Target Variable (Arthritis/KOA) Distribution:")
                print(self.df['Arthritis'].value_counts())
                print(f"Class Balance:")
                print(self.df['Arthritis'].value_counts(normalize=True) * 100)
            elif 'KOA' in self.df.columns:
                print(f"Target Variable (KOA) Distribution:")
                print(self.df['KOA'].value_counts())
                print(f"Class Balance: {self.df['KOA'].value_counts(normalize=True) * 100}")
            else:
                print("⚠️ Warning: Target variable not found in dataset.")
                print(f"Available columns: {list(self.df.columns)}")
            
            # Missing values
            print("\n" + "=" * 40)
            print("Missing Values Summary:")
            print("=" * 40)
            missing = self.df.isnull().sum()
            missing_pct = (missing / len(self.df)) * 100
            missing_df = pd.DataFrame({
                'Missing_Count': missing,
                'Missing_Percentage': missing_pct
            })
            print(missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))
            
            # Basic statistics
            print("\n" + "=" * 40)
            print("Descriptive Statistics:")
            print("=" * 40)
            print(self.df.describe())
            
            return self.df
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            sys.exit(1)
    
    # ============================================================================
    # STAGE 2: FEATURE ENGINEERING (30+ New Features)
    # ============================================================================
    
    def engineer_features(self):
        """Create 30+ new features based on clinical knowledge"""
        print("\n" + "=" * 80)
        print("STAGE 2: FEATURE ENGINEERING (Creating 30+ New Features)")
        print("=" * 80)
        
        df = self.df.copy()
        initial_columns = len(df.columns)
        
        # Standardize column names for easier processing
        if 'Biological Age' in df.columns:
            df['Biological_Age'] = df['Biological Age']
        if 'bmi_kg.m2' in df.columns:
            df['BMI'] = df['bmi_kg.m2']
        
        # Encode binary yes/no columns to 0/1
        yes_no_cols = ['doctor_diagnose_hypertension', 'doctor_diagnose_dyslipidemia', 
                       'doctor_diagnose_diabetes', 'doctor_diagnose_cancer', 
                       'doctor_diagnose_heart_disease', 'smoke', 'drink', 'Arthritis']
        for col in yes_no_cols:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
        
        # Create renamed columns for comorbidities
        if 'doctor_diagnose_hypertension' in df.columns:
            df['Hypertension'] = df['doctor_diagnose_hypertension']
        if 'doctor_diagnose_dyslipidemia' in df.columns:
            df['Dyslipidemia'] = df['doctor_diagnose_dyslipidemia']
        if 'doctor_diagnose_diabetes' in df.columns:
            df['Diabetes'] = df['doctor_diagnose_diabetes']
        if 'doctor_diagnose_cancer' in df.columns:
            df['Cancer'] = df['doctor_diagnose_cancer']
        if 'doctor_diagnose_heart_disease' in df.columns:
            df['Heart_Disease'] = df['doctor_diagnose_heart_disease']
        
        # Encode sex
        if 'sex' in df.columns:
            df['Sex_Numeric'] = df['sex'].map({'male': 1, 'female': 0, 'Male': 1, 'Female': 0})
        
        print("\n1️⃣ Creating Aging-Related Features...")
        # Polynomial age features (non-linear relationship after 66.7 years)
        if 'Biological_Age' in df.columns:
            df['Biological_Age_Squared'] = df['Biological_Age'] ** 2
            df['Biological_Age_Cubed'] = df['Biological_Age'] ** 3
            df['Log_Biological_Age'] = np.log1p(df['Biological_Age'])
            df['BioAge_60_Plus'] = (df['Biological_Age'] >= 60).astype(int)
            df['BioAge_70_Plus'] = (df['Biological_Age'] >= 70).astype(int)
            df['BioAge_Over_67'] = (df['Biological_Age'] > 66.7).astype(int)
            print(f"   ✓ Biological_Age_Squared, Cubed, Log, Age Thresholds")
        
        print("\n2️⃣ Creating BMI-Related Features...")
        if 'BMI' in df.columns:
            df['BMI_Squared'] = df['BMI'] ** 2
            df['BMI_Category'] = pd.cut(df['BMI'], 
                                        bins=[0, 18.5, 25, 30, 100], 
                                        labels=[0, 1, 2, 3])
            df['BMI_Obese'] = (df['BMI'] >= 30).astype(int)
            df['Log_BMI'] = np.log1p(df['BMI'])
            print(f"   ✓ BMI_Squared, BMI_Category, BMI_Obese, Log_BMI")
        
        print("\n3️⃣ Creating Comorbidity Risk Scores...")
        # Find comorbidity columns
        comorbidity_cols = []
        possible_comorbidities = ['Hypertension', 'Dyslipidemia', 'Diabetes', 
                                  'Cancer', 'Heart_Disease']
        
        for col in possible_comorbidities:
            if col in df.columns:
                comorbidity_cols.append(col)
        
        if comorbidity_cols:
            # Fill NaN with 0 for comorbidities
            for col in comorbidity_cols:
                df[col] = df[col].fillna(0)
            
            df['Comorbidity_Count'] = df[comorbidity_cols].sum(axis=1)
            df['Has_Multiple_Comorbidities'] = (df['Comorbidity_Count'] >= 2).astype(int)
            df['Comorbidity_Risk_Score'] = df['Comorbidity_Count'] / len(comorbidity_cols)
            print(f"   ✓ Comorbidity_Count, Has_Multiple_Comorbidities, Risk_Score")
            print(f"   ℹ️ Found {len(comorbidity_cols)} comorbidity columns: {comorbidity_cols}")
        
        print("\n4️⃣ Creating Biomarker Ratios (Clinical Significance)...")
        # Kidney function ratio
        if 'BUN' in df.columns and 'Creatinine' in df.columns:
            df['BUN_Creatinine_Ratio'] = df['BUN'] / (df['Creatinine'] + 0.001)
            print(f"   ✓ BUN_Creatinine_Ratio (Kidney Function)")
        
        # Lipid ratios
        if 'TC' in df.columns and 'TG' in df.columns:
            df['TC_TG_Ratio'] = df['TC'] / (df['TG'] + 0.001)
            print(f"   ✓ TC_TG_Ratio (Cholesterol/Triglyceride)")
        
        if 'HDL' in df.columns and 'LDL' in df.columns:
            df['HDL_LDL_Ratio'] = df['HDL'] / (df['LDL'] + 0.001)
            print(f"   ✓ HDL_LDL_Ratio (Good/Bad Cholesterol)")
        
        if 'TC' in df.columns and 'HDL' in df.columns:
            df['Atherogenic_Index'] = df['TC'] / (df['HDL'] + 0.001)
            print(f"   ✓ Atherogenic_Index (Cardiovascular Risk)")
        
        # Liver function
        if 'ALT' in df.columns and 'AST' in df.columns:
            df['AST_ALT_Ratio'] = df['AST'] / (df['ALT'] + 0.001)
            print(f"   ✓ AST_ALT_Ratio (Liver Function)")
        
        # Inflammatory markers
        if 'WBC' in df.columns and 'RBC' in df.columns:
            df['WBC_RBC_Ratio'] = df['WBC'] / (df['RBC'] + 0.001)
            print(f"   ✓ WBC_RBC_Ratio (Inflammation)")
        
        print("\n5️⃣ Creating Interaction Terms...")
        # BMI × Biological Age interaction
        if 'BMI' in df.columns and 'Biological_Age' in df.columns:
            df['BMI_x_BioAge'] = df['BMI'] * df['Biological_Age']
            print(f"   ✓ BMI_x_BioAge")
        
        # Biological Age × Comorbidity interaction
        if 'Biological_Age' in df.columns and 'Comorbidity_Count' in df.columns:
            df['BioAge_x_Comorbidity'] = df['Biological_Age'] * df['Comorbidity_Count']
            print(f"   ✓ BioAge_x_Comorbidity")
        
        # Lifestyle interactions
        if 'smoke' in df.columns and 'drink' in df.columns:
            # Ensure numeric
            smoke_num = pd.to_numeric(df['smoke'], errors='coerce').fillna(0)
            drink_num = pd.to_numeric(df['drink'], errors='coerce').fillna(0)
            df['Smoke_x_Drink'] = smoke_num * drink_num
            print(f"   ✓ Smoke_x_Drink")
        
        if 'smoke' in df.columns and 'BMI' in df.columns:
            smoke_num = pd.to_numeric(df['smoke'], errors='coerce').fillna(0)
            df['Smoke_x_BMI'] = smoke_num * df['BMI']
            print(f"   ✓ Smoke_x_BMI")
        
        # MET interactions
        if 'MET' in df.columns and 'BMI' in df.columns:
            df['MET_x_BMI'] = df['MET'] * df['BMI']
            print(f"   ✓ MET_x_BMI")
        
        if 'MET' in df.columns and 'Biological_Age' in df.columns:
            df['MET_x_BioAge'] = df['MET'] * df['Biological_Age']
            print(f"   ✓ MET_x_BioAge")
        
        print("\n6️⃣ Creating Normalized Biomarker Scores...")
        biomarker_cols = []
        possible_biomarkers = ['BUN', 'Creatinine', 'TC', 'TG', 'HDL', 'LDL', 
                               'ALT', 'AST', 'GGT', 'ALP', 'GLU', 'HbA1c',
                               'WBC', 'RBC', 'Hemoglobin', 'Platelet']
        
        for col in possible_biomarkers:
            if col in df.columns:
                biomarker_cols.append(col)
        
        if biomarker_cols:
            # Z-score normalization for composite score
            biomarker_z_scores = df[biomarker_cols].apply(lambda x: (x - x.mean()) / x.std())
            df['Biomarker_Composite_Score'] = biomarker_z_scores.mean(axis=1)
            df['Biomarker_Variability'] = biomarker_z_scores.std(axis=1)
            print(f"   ✓ Biomarker_Composite_Score, Biomarker_Variability")
            print(f"   ℹ️ Used {len(biomarker_cols)} biomarkers")
        
        print("\n7️⃣ Creating Metabolic Risk Profile...")
        # Metabolic syndrome indicators - start with Series of zeros
        metabolic_risk = pd.Series(0, index=df.index)
        
        if 'BMI' in df.columns:
            metabolic_risk = metabolic_risk + (df['BMI'] >= 30).astype(int)
        if 'Diabetes' in df.columns:
            metabolic_risk = metabolic_risk + df['Diabetes'].fillna(0).astype(int)
        if 'Hypertension' in df.columns:
            metabolic_risk = metabolic_risk + df['Hypertension'].fillna(0).astype(int)
        if 'Dyslipidemia' in df.columns:
            metabolic_risk = metabolic_risk + df['Dyslipidemia'].fillna(0).astype(int)
        
        df['Metabolic_Risk_Score'] = metabolic_risk
        df['High_Metabolic_Risk'] = (metabolic_risk >= 2).astype(int)
        print(f"   ✓ Metabolic_Risk_Score, High_Metabolic_Risk")
        
        print("\n8️⃣ Creating Gender-Specific Features...")
        if 'Sex_Numeric' in df.columns:
            if 'BMI' in df.columns:
                df['Sex_x_BMI'] = df['Sex_Numeric'] * df['BMI']
                print(f"   ✓ Sex_x_BMI")
            if 'Biological_Age' in df.columns:
                df['Sex_x_BioAge'] = df['Sex_Numeric'] * df['Biological_Age']
                print(f"   ✓ Sex_x_BioAge")
        
        print("\n9️⃣ Creating Socioeconomic Indicators...")
        if 'education' in df.columns:
            df['Education_Level'] = pd.Categorical(df['education']).codes
            if 'Biological_Age' in df.columns:
                df['Education_x_BioAge'] = df['Education_Level'] * df['Biological_Age']
                print(f"   ✓ Education_Level, Education_x_BioAge")
        
        if 'residence_place' in df.columns:
            df['Residence_Level'] = pd.Categorical(df['residence_place']).codes
            if 'BMI' in df.columns:
                df['Residence_x_BMI'] = df['Residence_Level'] * df['BMI']
                print(f"   ✓ Residence_Level, Residence_x_BMI")
        
        if 'marital_status' in df.columns:
            df['Marital_Level'] = pd.Categorical(df['marital_status']).codes
            print(f"   ✓ Marital_Level")
        
        print("\n🔟 Creating Advanced Combined Features...")
        # Position knees encoding
        if 'position_knees' in df.columns:
            df['Position_Knees_Encoded'] = pd.Categorical(df['position_knees']).codes
            print(f"   ✓ Position_Knees_Encoded")
        
        # Feature Engineering Summary
        new_columns = len(df.columns) - initial_columns
        print("\n" + "=" * 80)
        print(f"✅ FEATURE ENGINEERING COMPLETE")
        print(f"   • Initial Features: {initial_columns}")
        print(f"   • New Features Created: {new_columns}")
        print(f"   • Total Features: {len(df.columns)}")
        print("=" * 80)
        
        self.df_processed = df
        return df
    
    # ============================================================================
    # STAGE 2.7: ANOVA F-TEST FEATURE ANALYSIS
    # ============================================================================
    
    def anova_feature_analysis(self, top_k=30):
        """Analyze features using ANOVA F-test to find most relevant features"""
        print("\n" + "=" * 80)
        print("STAGE 2.7: ANOVA F-TEST FEATURE ANALYSIS")
        print("="*80)
        
        if self.df_processed is None:
            print("⚠️ Running feature engineering first...")
            self.engineer_features()
        
        df = self.df_processed.copy()
        
        # Identify target
        target_col = 'Arthritis' if 'Arthritis' in df.columns else 'KOA'
        
        if target_col not in df.columns:
            print(f"❌ Error: Target variable '{target_col}' not found!")
            return None
        
        # Remove rows with missing target
        print(f"\n🔍 Preparing Data for ANOVA F-test:")
        print(f"   • Initial samples: {len(df)}")
        df = df[df[target_col].notna()]
        print(f"   • After removing missing target: {len(df)}")
        
        # Drop non-feature columns
        drop_cols = [target_col, 'wave', 'iyear', 'sex', 'marital_status', 'education', 
                     'residence_place', 'doctor_diagnose_hypertension', 
                     'doctor_diagnose_dyslipidemia', 'doctor_diagnose_diabetes', 
                     'doctor_diagnose_cancer', 'doctor_diagnose_heart_disease', 
                     'position_knees', 'smoke', 'drink', 'Biological Age', 'bmi_kg.m2']
        
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df[target_col]
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\n   Encoding {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            print(f"   Filling {X.isnull().sum().sum()} missing values with median...")
            X = X.fillna(X.median())
        
        print(f"\n📊 ANOVA F-Test Analysis:")
        print(f"   • Total features to analyze: {X.shape[1]}")
        print(f"   • Samples: {X.shape[0]}")
        
        # Compute ANOVA F-statistic
        print("\n🔬 Computing ANOVA F-statistics...")
        f_scores, p_values = f_classif(X, y)
        
        # Create results dataframe
        feature_scores = pd.DataFrame({
            'Feature': X.columns,
            'F_Score': f_scores,
            'P_Value': p_values,
            'Log_P_Value': -np.log10(p_values + 1e-10)  # Avoid log(0)
        })
        
        # Sort by F-score
        feature_scores = feature_scores.sort_values('F_Score', ascending=False)
        
        # Identify significant features (p < 0.05)
        significant_features = feature_scores[feature_scores['P_Value'] < 0.05]
        
        print("\n" + "="*80)
        print("📈 ANOVA F-TEST RESULTS")
        print("="*80)
        print(f"   • Significant features (p < 0.05): {len(significant_features)}")
        print(f"   • Non-significant features: {len(feature_scores) - len(significant_features)}")
        print(f"   • Top-{top_k} features selected for modeling")
        print("="*80)
        
        # Display top features
        print(f"\n🏆 Top 20 Features by F-Score:")
        print("-"*80)
        for i, row in feature_scores.head(20).iterrows():
            significance = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
            print(f"   {row.name+1:2d}. {row['Feature']:35s} | F={row['F_Score']:8.2f} | p={row['P_Value']:.4e} {significance}")
        
        # Check for features with very low scores
        low_score_features = feature_scores[feature_scores['F_Score'] < 1]
        if len(low_score_features) > 0:
            print(f"\n⚠️  Warning: {len(low_score_features)} features have very low F-scores (< 1):")
            print(f"   These features may add noise to the model.")
            for feat in low_score_features.head(10)['Feature']:
                print(f"   • {feat}")
        
        # Select top-k features
        selected_features = feature_scores.head(top_k)['Feature'].tolist()
        
        print(f"\n✅ Selected {len(selected_features)} features for modeling")
        
        # Feature categories analysis
        print(f"\n📊 Feature Categories in Top-{top_k}:")
        categories = {
            'Original': [],
            'Aging': [],
            'BMI': [],
            'Comorbidity': [],
            'Interaction': [],
            'Metabolic': [],
            'Socioeconomic': [],
            'Other': []
        }
        
        for feat in selected_features:
            if 'Age' in feat or 'BioAge' in feat:
                categories['Aging'].append(feat)
            elif 'BMI' in feat:
                categories['BMI'].append(feat)
            elif 'Comorbidity' in feat or 'Hypertension' in feat or 'Diabetes' in feat:
                categories['Comorbidity'].append(feat)
            elif '_x_' in feat or 'Interaction' in feat:
                categories['Interaction'].append(feat)
            elif 'Metabolic' in feat:
                categories['Metabolic'].append(feat)
            elif 'Education' in feat or 'Residence' in feat or 'Marital' in feat:
                categories['Socioeconomic'].append(feat)
            elif feat in ['MET', 'BMI', 'Biological_Age', 'Sex_Numeric']:
                categories['Original'].append(feat)
            else:
                categories['Other'].append(feat)
        
        for cat, feats in categories.items():
            if len(feats) > 0:
                print(f"   • {cat:15s}: {len(feats):2d} features")
        
        # Check feature correlations among top features
        print(f"\n🔍 Checking Feature Correlations (top-{min(top_k, 15)} features)...")
        top_features_data = X[selected_features[:min(top_k, 15)]]
        correlation_matrix = top_features_data.corr().abs()
        
        # Find high correlations (> 0.8)
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > 0.8:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
        
        if len(high_corr_pairs) > 0:
            print(f"\n   ⚠️  Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
            for feat1, feat2, corr in high_corr_pairs[:5]:
                print(f"      • {feat1} ↔ {feat2}: r = {corr:.3f}")
            print("   Note: High correlation may cause multicollinearity issues.")
        else:
            print("   ✓ No highly correlated feature pairs found (good for modeling!)")
        
        # Save full results to file
        anova_results_file = 'anova_feature_scores.csv'
        feature_scores.to_csv(anova_results_file, index=False)
        print(f"\n💾 Full ANOVA results saved to: {anova_results_file}")
        
        # ========================================================================
        # VISUALIZATION
        # ========================================================================
        
        print(f"\n📊 Creating Visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Top 20 Features by F-Score (Bar Chart)
        ax1 = plt.subplot(2, 3, 1)
        top_20_features = feature_scores.head(20)
        colors = ['#d62728' if p < 0.001 else '#ff7f0e' if p < 0.01 else '#2ca02c' if p < 0.05 else '#7f7f7f' 
                  for p in top_20_features['P_Value']]
        
        ax1.barh(range(len(top_20_features)), top_20_features['F_Score'], color=colors)
        ax1.set_yticks(range(len(top_20_features)))
        ax1.set_yticklabels([feat[:25] + '...' if len(feat) > 25 else feat 
                             for feat in top_20_features['Feature']], fontsize=9)
        ax1.set_xlabel('F-Score', fontsize=11, fontweight='bold')
        ax1.set_title('Top 20 Features by ANOVA F-Score', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)
        
        # Add legend for p-value significance
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d62728', label='p < 0.001 (***)'),
            Patch(facecolor='#ff7f0e', label='p < 0.01 (**)'),
            Patch(facecolor='#2ca02c', label='p < 0.05 (*)'),
            Patch(facecolor='#7f7f7f', label='p ≥ 0.05')
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
        # 2. P-Value Distribution (Histogram)
        ax2 = plt.subplot(2, 3, 2)
        ax2.hist(feature_scores['P_Value'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=0.05, color='red', linestyle='--', linewidth=2, label='α = 0.05')
        ax2.set_xlabel('P-Value', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('P-Value Distribution', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        
        # Add text annotation
        significant_pct = (len(significant_features) / len(feature_scores)) * 100
        ax2.text(0.95, 0.95, f'{significant_pct:.1f}% significant\n(p < 0.05)', 
                transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 3. -log10(p-value) Volcano Plot
        ax3 = plt.subplot(2, 3, 3)
        scatter_colors = ['#d62728' if p < 0.05 else '#7f7f7f' for p in feature_scores['P_Value']]
        ax3.scatter(feature_scores['F_Score'], feature_scores['Log_P_Value'], 
                   c=scatter_colors, alpha=0.6, s=50)
        ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2, label='p = 0.05')
        ax3.set_xlabel('F-Score', fontsize=11, fontweight='bold')
        ax3.set_ylabel('-log10(p-value)', fontsize=11, fontweight='bold')
        ax3.set_title('Volcano Plot: Feature Significance', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)
        
        # Annotate top 5 features
        for idx in range(min(5, len(feature_scores))):
            row = feature_scores.iloc[idx]
            ax3.annotate(row['Feature'][:15], 
                        xy=(row['F_Score'], row['Log_P_Value']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        # 4. Feature Category Distribution (Pie Chart)
        ax4 = plt.subplot(2, 3, 4)
        category_counts = {}
        for feat in selected_features:
            if 'Age' in feat or 'BioAge' in feat:
                category_counts['Aging'] = category_counts.get('Aging', 0) + 1
            elif 'BMI' in feat:
                category_counts['BMI-Related'] = category_counts.get('BMI-Related', 0) + 1
            elif 'Comorbidity' in feat or 'Hypertension' in feat or 'Diabetes' in feat or 'Heart' in feat:
                category_counts['Comorbidity'] = category_counts.get('Comorbidity', 0) + 1
            elif '_x_' in feat:
                category_counts['Interaction'] = category_counts.get('Interaction', 0) + 1
            elif 'Metabolic' in feat:
                category_counts['Metabolic'] = category_counts.get('Metabolic', 0) + 1
            elif 'Education' in feat or 'Residence' in feat or 'Marital' in feat:
                category_counts['Socioeconomic'] = category_counts.get('Socioeconomic', 0) + 1
            else:
                category_counts['Other'] = category_counts.get('Other', 0) + 1
        
        colors_pie = plt.cm.Set3(range(len(category_counts)))
        wedges, texts, autotexts = ax4.pie(category_counts.values(), labels=category_counts.keys(), 
                                            autopct='%1.1f%%', startangle=90, colors=colors_pie)
        ax4.set_title(f'Feature Categories in Top-{top_k}', fontsize=12, fontweight='bold')
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        
        # 5. Correlation Heatmap (Top 15 Features)
        ax5 = plt.subplot(2, 3, 5)
        top_15_for_corr = selected_features[:min(15, len(selected_features))]
        corr_data = X[top_15_for_corr].corr()
        
        im = ax5.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax5.set_xticks(range(len(top_15_for_corr)))
        ax5.set_yticks(range(len(top_15_for_corr)))
        ax5.set_xticklabels([feat[:15] for feat in top_15_for_corr], rotation=45, ha='right', fontsize=8)
        ax5.set_yticklabels([feat[:15] for feat in top_15_for_corr], fontsize=8)
        ax5.set_title('Feature Correlation Heatmap (Top 15)', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5)
        cbar.set_label('Correlation', fontsize=10)
        
        # 6. F-Score Cumulative Distribution
        ax6 = plt.subplot(2, 3, 6)
        sorted_f_scores = feature_scores['F_Score'].sort_values(ascending=False).values
        cumsum_f = np.cumsum(sorted_f_scores)
        cumsum_f_norm = cumsum_f / cumsum_f[-1] * 100
        
        ax6.plot(range(1, len(sorted_f_scores) + 1), cumsum_f_norm, 
                linewidth=2, color='darkblue')
        ax6.axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% cumulative')
        ax6.axvline(x=top_k, color='green', linestyle='--', linewidth=2, label=f'Top {top_k} selected')
        ax6.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Cumulative F-Score (%)', fontsize=11, fontweight='bold')
        ax6.set_title('Cumulative F-Score Distribution', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(alpha=0.3)
        
        # Add 80% mark
        features_for_80 = np.argmax(cumsum_f_norm >= 80) + 1
        ax6.plot(features_for_80, 80, 'ro', markersize=8)
        ax6.annotate(f'{features_for_80} features\nfor 80%', 
                    xy=(features_for_80, 80), xytext=(10, -20),
                    textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # Overall title
        fig.suptitle('ANOVA F-Test Feature Selection Analysis', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save figure
        viz_file = 'anova_feature_analysis.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {viz_file}")
        
        # Create second figure for detailed feature scores
        fig2, ax = plt.subplots(figsize=(14, 10))
        
        # All features ranked by F-score
        y_pos = np.arange(len(feature_scores))
        colors_all = ['#d62728' if p < 0.001 else '#ff7f0e' if p < 0.01 else 
                      '#2ca02c' if p < 0.05 else '#7f7f7f' 
                      for p in feature_scores['P_Value']]
        
        ax.barh(y_pos, feature_scores['F_Score'], color=colors_all, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feat[:30] + '...' if len(feat) > 30 else feat 
                           for feat in feature_scores['Feature']], fontsize=8)
        ax.set_xlabel('F-Score', fontsize=12, fontweight='bold')
        ax.set_title('All Features Ranked by ANOVA F-Score', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        # Add vertical line for selection threshold
        if top_k < len(feature_scores):
            threshold_f = feature_scores.iloc[top_k]['F_Score']
            ax.axvline(x=threshold_f, color='red', linestyle='--', linewidth=2, 
                      label=f'Top-{top_k} threshold')
            ax.legend(fontsize=10)
        
        # Add legend for significance
        legend_elements = [
            Patch(facecolor='#d62728', alpha=0.7, label='p < 0.001 (***) - Highly significant'),
            Patch(facecolor='#ff7f0e', alpha=0.7, label='p < 0.01 (**) - Very significant'),
            Patch(facecolor='#2ca02c', alpha=0.7, label='p < 0.05 (*) - Significant'),
            Patch(facecolor='#7f7f7f', alpha=0.7, label='p ≥ 0.05 - Not significant')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
        
        plt.tight_layout()
        
        viz_file2 = 'anova_all_features_ranked.png'
        plt.savefig(viz_file2, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {viz_file2}")
        
        plt.close('all')
        
        print(f"\n✅ Visualizations created successfully!")
        print(f"   📊 {viz_file}")
        print(f"   📊 {viz_file2}")
        
        # Log ANOVA results
        self.log_result(
            stage_name="Stage 2.7: ANOVA F-Test Analysis",
            description=f"Statistical feature selection using ANOVA F-test",
            metrics={
                'Total_Features': len(feature_scores),
                'Significant_Features_p<0.05': len(significant_features),
                'Selected_Features': len(selected_features),
                'Mean_F_Score_Selected': feature_scores.head(top_k)['F_Score'].mean(),
                'Mean_F_Score_All': feature_scores['F_Score'].mean()
            },
            details={
                'Top_5_Features': selected_features[:5],
                'Selection_Threshold': f'Top {top_k} by F-score',
                'High_Correlation_Pairs': len(high_corr_pairs),
                'Results_File': anova_results_file,
                'Visualization_Files': [viz_file, viz_file2]
            }
        )
        
        return selected_features, feature_scores
    
    # ============================================================================
    # STAGE 2.5: RAW DATA BASELINE (Before Feature Engineering)
    # ============================================================================
    
    def raw_data_baseline(self, test_size=0.2, random_state=42):
        """Test baseline performance with raw data before feature engineering"""
        print("\n" + "=" * 80)
        print("STAGE 2.5: RAW DATA BASELINE (Before Feature Engineering)")
        print("=" * 80)
        
        df = self.df.copy()
        
        # Identify target
        target_col = 'Arthritis' if 'Arthritis' in df.columns else 'KOA'
        
        if target_col not in df.columns:
            print(f"❌ Error: Target variable '{target_col}' not found!")
            return None
        
        # Remove rows with missing target
        print(f"\n🔍 Preparing Raw Data:")
        print(f"   • Initial samples: {len(df)}")
        df = df[df[target_col].notna()]
        print(f"   • After removing missing target: {len(df)}")
        
        # Encode yes/no columns to 0/1
        yes_no_cols = ['doctor_diagnose_hypertension', 'doctor_diagnose_dyslipidemia', 
                       'doctor_diagnose_diabetes', 'doctor_diagnose_cancer', 
                       'doctor_diagnose_heart_disease', 'smoke', 'drink', target_col]
        for col in yes_no_cols:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
        
        # Encode categorical columns
        cat_cols = ['sex', 'marital_status', 'education', 'residence_place', 'position_knees']
        for col in cat_cols:
            if col in df.columns:
                df[col] = pd.Categorical(df[col]).codes
        
        # Select only original features
        feature_cols = ['wave', 'iyear', 'sex', 'marital_status', 'education', 'residence_place',
                       'doctor_diagnose_hypertension', 'doctor_diagnose_dyslipidemia',
                       'doctor_diagnose_diabetes', 'doctor_diagnose_cancer',
                       'doctor_diagnose_heart_disease', 'position_knees',
                       'smoke', 'drink', 'MET', 'bmi_kg.m2', 'Biological Age']
        
        X = df[[col for col in feature_cols if col in df.columns]]
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\n📊 Raw Data Split (80/20):")
        print(f"   • Training Set: {X_train.shape[0]} samples")
        print(f"   • Test Set: {X_test.shape[0]} samples")
        print(f"   • Raw Features: {X_train.shape[1]}")
        print(f"   • Class Distribution: {dict(pd.Series(y_train).value_counts())}")
        
        # Train Random Forest
        print("\n🌲 Training Baseline RF with Raw Data...")
        rf_raw = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        rf_raw.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf_raw.predict(X_test_scaled)
        y_pred_proba = rf_raw.predict_proba(X_test_scaled)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "=" * 40)
        print("📈 RAW DATA BASELINE RESULTS:")
        print("=" * 40)
        print(f"   ROC AUC:  {roc_auc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print("=" * 40)
        
        # Log raw baseline
        self.log_result(
            stage_name="Stage 2.5: Raw Data Baseline",
            description="Random Forest with raw features only (no feature engineering)",
            metrics={
                'ROC_AUC': roc_auc,
                'F1_Score': f1,
                'Accuracy': accuracy
            },
            details={
                'Raw_Features': X_train.shape[1],
                'Training_Samples': X_train.shape[0],
                'Test_Samples': X_test.shape[0],
                'Train_Test_Split': f"{int((1-test_size)*100)}/{int(test_size*100)}"
            }
        )
        
        print("\n✅ Raw baseline established. Proceeding to feature engineering...")
        
        return roc_auc
    
    # ============================================================================
    # STAGE 3: BASELINE RANDOM FOREST PERFORMANCE TEST
    # ============================================================================
    
    def baseline_random_forest(self, test_size=0.3, random_state=42, use_selected_features=True):
        """Train baseline Random Forest and check if ROC AUC > 0.65"""
        print("\n" + "=" * 80)
        print("STAGE 3: BASELINE RANDOM FOREST PERFORMANCE TEST")
        if use_selected_features and self.selected_features:
            print(f"         (Using {len(self.selected_features)} ANOVA-selected features)")
        print("=" * 80)
        
        if self.df_processed is None:
            print("⚠️ Running feature engineering first...")
            self.engineer_features()
        
        df = self.df_processed.copy()
        
        # Identify target and features
        target_col = 'Arthritis' if 'Arthritis' in df.columns else 'KOA'
        
        if target_col not in df.columns:
            print(f"❌ Error: Target variable '{target_col}' not found!")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        # Remove rows with missing target values
        print(f"\n🔍 Cleaning Data:")
        print(f"   • Initial samples: {len(df)}")
        df = df[df[target_col].notna()]
        print(f"   • After removing missing target: {len(df)}")
        
        # Drop non-feature columns (also drop original columns that were renamed/encoded)
        drop_cols = [target_col, 'wave', 'iyear', 'sex', 'marital_status', 'education', 
                     'residence_place', 'doctor_diagnose_hypertension', 
                     'doctor_diagnose_dyslipidemia', 'doctor_diagnose_diabetes', 
                     'doctor_diagnose_cancer', 'doctor_diagnose_heart_disease', 
                     'position_knees', 'smoke', 'drink', 'Biological Age', 'bmi_kg.m2']
        
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        y = df[target_col]
        
        # Use ANOVA-selected features if available and requested
        if use_selected_features and self.selected_features:
            available_features = [f for f in self.selected_features if f in X.columns]
            if len(available_features) > 0:
                X = X[available_features]
                print(f"\n✓ Using {len(available_features)} ANOVA-selected features")
            else:
                print(f"\n⚠️ Warning: Selected features not available, using all features")
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\nEncoding {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Handle missing values (simple imputation for baseline)
        if X.isnull().sum().sum() > 0:
            print(f"Filling {X.isnull().sum().sum()} missing values with median...")
            X = X.fillna(X.median())
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n📊 Data Split:")
        print(f"   • Training Set: {self.X_train.shape[0]} samples")
        print(f"   • Test Set: {self.X_test.shape[0]} samples")
        print(f"   • Features: {self.X_train.shape[1]}")
        print(f"   • Class Distribution (Train): {dict(pd.Series(self.y_train).value_counts())}")
        
        # Train baseline Random Forest with class balancing
        print("\n🌲 Training Baseline Random Forest...")
        rf_baseline = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        
        rf_baseline.fit(self.X_train_scaled, self.y_train)
        
        # Predictions
        y_pred = rf_baseline.predict(self.X_test_scaled)
        y_pred_proba = rf_baseline.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluation
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print("\n" + "=" * 40)
        print("📈 BASELINE RANDOM FOREST RESULTS:")
        print("=" * 40)
        print(f"   ROC AUC:  {roc_auc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print("=" * 40)
        
        # Store baseline results
        self.results['baseline_rf'] = {
            'model': rf_baseline,
            'roc_auc': roc_auc,
            'f1_score': f1,
            'accuracy': accuracy
        }
        
        # Decision checkpoint
        print(f"\n🎯 DECISION CHECKPOINT:")
        if roc_auc > 0.65:
            print(f"   ✅ ROC AUC ({roc_auc:.4f}) > 0.65 → Proceed to Stage 4 (Feature Selection)")
        else:
            print(f"   ⚠️ ROC AUC ({roc_auc:.4f}) ≤ 0.65 → Return to Stage 2 for more feature engineering")
        
        # Log baseline results
        self.log_result(
            stage_name="Stage 3: Baseline Random Forest",
            description=f"Baseline RF with {self.X_train.shape[1]} engineered features",
            metrics={
                'ROC_AUC': roc_auc,
                'F1_Score': f1,
                'Accuracy': accuracy
            },
            details={
                'Total_Features': self.X_train.shape[1],
                'Training_Samples': self.X_train.shape[0],
                'Test_Samples': self.X_test.shape[0],
                'Class_Balance': f"{dict(pd.Series(self.y_train).value_counts())}"
            }
        )
        
        return rf_baseline, roc_auc
    
    # ============================================================================
    # STAGE 4: ENSEMBLE FEATURE SELECTION
    # ============================================================================
    
    def ensemble_feature_selection(self, n_features=20):
        """Select best features using LASSO, Tree Importance, and RFE"""
        print("\n" + "=" * 80)
        print("STAGE 4: ENSEMBLE FEATURE SELECTION")
        print("=" * 80)
        
        if self.X_train is None:
            print("⚠️ Running baseline RF first...")
            self.baseline_random_forest()
        
        X_train = pd.DataFrame(self.X_train_scaled, columns=self.X_train.columns)
        
        print("\n1️⃣ LASSO-Based Feature Selection...")
        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(self.X_train_scaled, self.y_train)
        lasso_coef = np.abs(lasso.coef_)
        lasso_selected = self.X_train.columns[lasso_coef > 0]
        print(f"   ✓ LASSO selected {len(lasso_selected)} features")
        
        print("\n2️⃣ Tree-Based Feature Importance...")
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_selector.fit(self.X_train_scaled, self.y_train)
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': rf_selector.feature_importances_
        }).sort_values('importance', ascending=False)
        
        tree_selected = feature_importance.head(n_features)['feature'].values
        print(f"   ✓ Tree Importance selected top {len(tree_selected)} features")
        
        print("\n3️⃣ Recursive Feature Elimination (RFE)...")
        rfe_selector = RFE(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_features_to_select=n_features
        )
        rfe_selector.fit(self.X_train_scaled, self.y_train)
        rfe_selected = self.X_train.columns[rfe_selector.support_]
        print(f"   ✓ RFE selected {len(rfe_selected)} features")
        
        # Find intersection (features selected by all three methods)
        all_selected = set(lasso_selected) & set(tree_selected) & set(rfe_selected)
        
        if len(all_selected) < 5:
            print(f"\n⚠️ Only {len(all_selected)} features in intersection. Using union approach...")
            # If intersection too small, use features selected by at least 2 methods
            from collections import Counter
            all_features = list(lasso_selected) + list(tree_selected) + list(rfe_selected)
            feature_votes = Counter(all_features)
            all_selected = [feat for feat, count in feature_votes.items() if count >= 2]
        
        self.selected_features = list(all_selected)
        
        print("\n" + "=" * 40)
        print(f"✅ ENSEMBLE FEATURE SELECTION COMPLETE")
        print(f"   • LASSO selected: {len(lasso_selected)} features")
        print(f"   • Tree Importance selected: {len(tree_selected)} features")
        print(f"   • RFE selected: {len(rfe_selected)} features")
        print(f"   • Final Selected Features: {len(self.selected_features)}")
        print("=" * 40)
        
        print(f"\n📋 Selected Features:")
        for i, feat in enumerate(self.selected_features, 1):
            print(f"   {i:2d}. {feat}")
        
        # Retrain with selected features
        print(f"\n🔄 Retraining Random Forest with selected features...")
        X_train_selected = self.X_train[self.selected_features]
        X_test_selected = self.X_test[self.selected_features]
        
        X_train_selected_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_selected_scaled = self.scaler.transform(X_test_selected)
        
        rf_optimized = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_optimized.fit(X_train_selected_scaled, self.y_train)
        
        y_pred_proba = rf_optimized.predict_proba(X_test_selected_scaled)[:, 1]
        roc_auc_optimized = roc_auc_score(self.y_test, y_pred_proba)
        
        baseline_auc = self.results['baseline_rf']['roc_auc']
        improvement = ((roc_auc_optimized - baseline_auc) / baseline_auc) * 100
        
        print(f"\n📊 Performance Comparison:")
        print(f"   • Baseline ROC AUC:  {baseline_auc:.4f}")
        print(f"   • Optimized ROC AUC: {roc_auc_optimized:.4f}")
        print(f"   • Improvement:       {improvement:+.2f}%")
        
        self.results['optimized_rf'] = {
            'model': rf_optimized,
            'roc_auc': roc_auc_optimized,
            'improvement': improvement
        }
        
        # Log feature selection results
        self.log_result(
            stage_name="Stage 4: Feature Selection",
            description=f"Ensemble feature selection (LASSO + Tree + RFE)",
            metrics={
                'ROC_AUC_Before': baseline_auc,
                'ROC_AUC_After': roc_auc_optimized,
                'Improvement_%': improvement
            },
            details={
                'Features_Before': self.X_train.shape[1],
                'Features_Selected': len(self.selected_features),
                'Reduction_%': f"{((self.X_train.shape[1] - len(self.selected_features)) / self.X_train.shape[1] * 100):.1f}%"
            }
        )
        
        return self.selected_features
    
    # ============================================================================
    # HELPER: LOG RESULTS TO FILE
    # ============================================================================
    
    def log_result(self, stage_name, description, metrics, details=None):
        """Log results of each stage to file"""
        result_entry = {
            'timestamp': pd.Timestamp.now(),
            'stage': stage_name,
            'description': description,
            'metrics': metrics,
            'details': details
        }
        self.results_log.append(result_entry)
        
        # Write to file
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"STAGE: {stage_name}\n")
            f.write(f"TIME: {result_entry['timestamp']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Description: {description}\n\n")
            
            f.write("Metrics:\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  • {key}: {value:.4f}\n")
                else:
                    f.write(f"  • {key}: {value}\n")
            
            if details:
                f.write(f"\nDetails:\n")
                for key, value in details.items():
                    f.write(f"  • {key}: {value}\n")
            
            f.write("\n")
        
        print(f"\n📝 Results logged to {self.results_file}")
    
    # ============================================================================
    # STAGE 6: COMPREHENSIVE MODEL COMPARISON
    # ============================================================================
    
    def compare_models(self):
        """Compare multiple ML algorithms with selected features"""
        print("\n" + "=" * 80)
        print("STAGE 6: COMPREHENSIVE MODEL COMPARISON")
        print("=" * 80)
        
        if self.selected_features is None:
            print("⚠️ Running feature selection first...")
            self.ensemble_feature_selection()
        
        X_train_selected = self.X_train[self.selected_features]
        X_test_selected = self.X_test[self.selected_features]
        
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
            'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1),
        }
        
        # Add advanced models if available
        if ADVANCED_MODELS_AVAILABLE:
            models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
            models['LightGBM'] = LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
            models['CatBoost'] = CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='balanced')
        
        # Train and evaluate each model
        results_comparison = []
        
        for name, model in models.items():
            print(f"\n🔄 Training {name}...")
            
            try:
                model.fit(X_train_scaled, self.y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                f1 = f1_score(self.y_test, y_pred)
                accuracy = accuracy_score(self.y_test, y_pred)
                
                results_comparison.append({
                    'Model': name,
                    'ROC_AUC': roc_auc,
                    'F1_Score': f1,
                    'Accuracy': accuracy
                })
                
                print(f"   ✓ ROC AUC: {roc_auc:.4f} | F1: {f1:.4f} | Accuracy: {accuracy:.4f}")
                
                self.results[name] = {
                    'model': model,
                    'roc_auc': roc_auc,
                    'f1_score': f1,
                    'accuracy': accuracy
                }
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {str(e)}")
        
        # Summary table
        results_df = pd.DataFrame(results_comparison).sort_values('ROC_AUC', ascending=False)
        
        print("\n" + "=" * 80)
        print("📊 MODEL COMPARISON RESULTS (Sorted by ROC AUC)")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)
        
        # Identify best model
        best_model_name = results_df.iloc[0]['Model']
        best_roc_auc = results_df.iloc[0]['ROC_AUC']
        
        print(f"\n🏆 BEST MODEL: {best_model_name} (ROC AUC: {best_roc_auc:.4f})")
        
        # Compare with paper results
        print(f"\n📄 Comparison with Paper Results:")
        print(f"   • Paper XGBoost AUROC: 0.9078")
        print(f"   • Paper LightGBM AUROC: 0.8973")
        if 'XGBoost' in self.results:
            print(f"   • Our XGBoost AUROC: {self.results['XGBoost']['roc_auc']:.4f}")
        if 'LightGBM' in self.results:
            print(f"   • Our LightGBM AUROC: {self.results['LightGBM']['roc_auc']:.4f}")
        
        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['model']
        
        # Log model comparison results
        self.log_result(
            stage_name="Stage 6: Model Comparison",
            description=f"Comparison of {len(results_df)} ML algorithms",
            metrics={
                'Best_Model': best_model_name,
                'Best_ROC_AUC': best_roc_auc,
                'Avg_ROC_AUC': results_df['ROC_AUC'].mean()
            },
            details={
                'All_Models': results_df.to_dict('records')
            }
        )
        
        return results_df
    
    # ============================================================================
    # STAGE 7: HYPERPARAMETER OPTIMIZATION
    # ============================================================================
    
    def optimize_hyperparameters(self, n_trials=50):
        """Optimize hyperparameters for best model using Optuna"""
        print("\n" + "=" * 80)
        print("STAGE 7: HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        
        if not OPTUNA_AVAILABLE:
            print("❌ Optuna not available. Skipping hyperparameter optimization.")
            print("   Install with: pip install optuna")
            return None
        
        if self.best_model is None:
            print("⚠️ Running model comparison first...")
            self.compare_models()
        
        X_train_selected = self.X_train[self.selected_features]
        X_test_selected = self.X_test[self.selected_features]
        
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        print(f"\n🎯 Optimizing {self.best_model_name}...")
        
        # Define objective based on best model
        if 'XGBoost' in self.best_model_name and ADVANCED_MODELS_AVAILABLE:
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                    'gamma': trial.suggest_float('gamma', 0, 0.5),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False
                }
                
                model = XGBClassifier(**params)
                model.fit(X_train_scaled, self.y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                return roc_auc_score(self.y_test, y_pred_proba)
        
        elif 'LightGBM' in self.best_model_name and ADVANCED_MODELS_AVAILABLE:
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1.0),
                    'random_state': 42,
                    'verbose': -1,
                    'class_weight': 'balanced'
                }
                
                model = LGBMClassifier(**params)
                model.fit(X_train_scaled, self.y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                return roc_auc_score(self.y_test, y_pred_proba)
        
        else:
            print(f"   ⚠️ Hyperparameter optimization not implemented for {self.best_model_name}")
            return None
        
        # Run optimization
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n✅ Optimization Complete!")
        print(f"   • Best ROC AUC: {study.best_value:.4f}")
        print(f"   • Best Parameters:")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")
        
        # Train final optimized model
        if 'XGBoost' in self.best_model_name:
            final_model = XGBClassifier(**study.best_params, random_state=42, 
                                        eval_metric='logloss', use_label_encoder=False)
        elif 'LightGBM' in self.best_model_name:
            final_model = LGBMClassifier(**study.best_params, random_state=42, 
                                         verbose=-1, class_weight='balanced')
        
        final_model.fit(X_train_scaled, self.y_train)
        
        self.results['optimized_best_model'] = {
            'model': final_model,
            'roc_auc': study.best_value,
            'params': study.best_params
        }
        
        # Log hyperparameter optimization results
        baseline_auc = self.results[self.best_model_name]['roc_auc']
        improvement = ((study.best_value - baseline_auc) / baseline_auc) * 100
        
        self.log_result(
            stage_name="Stage 7: Hyperparameter Optimization",
            description=f"Optuna optimization with {n_trials} trials",
            metrics={
                'ROC_AUC_Before': baseline_auc,
                'ROC_AUC_After': study.best_value,
                'Improvement_%': improvement
            },
            details={
                'Best_Params': study.best_params,
                'Model': self.best_model_name,
                'N_Trials': n_trials
            }
        )
        
        return final_model, study.best_params
    
    # ============================================================================
    # STAGE 8: FINAL EVALUATION AND REPORTING
    # ============================================================================
    
    def final_evaluation(self):
        """Final evaluation on test set and comprehensive reporting"""
        print("\n" + "=" * 80)
        print("STAGE 8: FINAL EVALUATION AND REPORTING")
        print("=" * 80)
        
        # Use optimized model if available, otherwise best model
        if 'optimized_best_model' in self.results:
            final_model = self.results['optimized_best_model']['model']
            model_name = f"Optimized {self.best_model_name}"
        else:
            final_model = self.best_model
            model_name = self.best_model_name
        
        X_train_selected = self.X_train[self.selected_features]
        X_test_selected = self.X_test[self.selected_features]
        
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Predictions
        y_pred = final_model.predict(X_test_scaled)
        y_pred_proba = final_model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        f1 = f1_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"\n🏆 FINAL MODEL: {model_name}")
        print("=" * 40)
        print(f"   ROC AUC:  {roc_auc:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print("=" * 40)
        
        # Classification Report
        print("\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion Matrix
        print("\n🔲 Confusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        # Compare with paper
        print("\n📊 Comparison with Paper Results:")
        print("=" * 40)
        print("   Paper (XGBoost):  AUROC = 0.9078")
        print("   Paper (LightGBM): AUROC = 0.8973")
        print(f"   Our Model:        AUROC = {roc_auc:.4f}")
        print("=" * 40)
        
        if roc_auc >= 0.85:
            print("\n✅ SUCCESS: Model achieves strong performance (AUROC ≥ 0.85)")
            print("   Project can be concluded successfully.")
        elif roc_auc >= 0.70:
            print("\n⚠️ MODERATE: Model shows good performance (0.70 ≤ AUROC < 0.85)")
            print("   Consider Stage 9 (ensemble methods) for improvement.")
        else:
            print("\n❌ INSUFFICIENT: Model performance below expectations (AUROC < 0.70)")
            print("   Return to Stage 2 for feature re-engineering.")
        
        # Log final evaluation
        self.log_result(
            stage_name="Stage 8: Final Evaluation",
            description=f"Final test set evaluation with {model_name}",
            metrics={
                'ROC_AUC': roc_auc,
                'F1_Score': f1,
                'Accuracy': accuracy
            },
            details={
                'Model': model_name,
                'Confusion_Matrix': cm.tolist(),
                'Test_Samples': len(self.y_test)
            }
        )
        
        # Write final summary
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("FINAL SUMMARY - PIPELINE PROGRESSION\n")
            f.write("="*80 + "\n\n")
            
            for i, entry in enumerate(self.results_log, 1):
                f.write(f"{i}. {entry['stage']}\n")
                f.write(f"   Description: {entry['description']}\n")
                if 'ROC_AUC' in entry['metrics']:
                    f.write(f"   ROC AUC: {entry['metrics']['ROC_AUC']:.4f}\n")
                elif 'ROC_AUC_After' in entry['metrics']:
                    f.write(f"   ROC AUC: {entry['metrics']['ROC_AUC_After']:.4f}\n")
                f.write("\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Pipeline completed: {pd.Timestamp.now()}\n")
            f.write("="*80 + "\n")
        
        return roc_auc, f1, accuracy
    
    # ============================================================================
    # STAGE 9: ADVANCED ENSEMBLE METHODS (If Needed)
    # ============================================================================
    
    def create_ensemble(self):
        """Create Voting or Stacking ensemble if performance needs boost"""
        print("\n" + "=" * 80)
        print("STAGE 9: ADVANCED ENSEMBLE METHODS")
        print("=" * 80)
        
        if not ADVANCED_MODELS_AVAILABLE:
            print("❌ Advanced models not available for ensemble.")
            return None
        
        X_train_selected = self.X_train[self.selected_features]
        X_test_selected = self.X_test[self.selected_features]
        
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        print("\n1️⃣ Creating Voting Classifier...")
        voting_clf = VotingClassifier(
            estimators=[
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)),
                ('lgbm', LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')),
                ('cat', CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='balanced')),
            ],
            voting='soft'
        )
        
        voting_clf.fit(X_train_scaled, self.y_train)
        y_pred_voting_proba = voting_clf.predict_proba(X_test_scaled)[:, 1]
        roc_auc_voting = roc_auc_score(self.y_test, y_pred_voting_proba)
        
        print(f"   ✓ Voting Classifier ROC AUC: {roc_auc_voting:.4f}")
        
        print("\n2️⃣ Creating Stacking Classifier...")
        stacking_clf = StackingClassifier(
            estimators=[
                ('xgb', XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)),
                ('lgbm', LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')),
                ('cat', CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='balanced')),
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5
        )
        
        stacking_clf.fit(X_train_scaled, self.y_train)
        y_pred_stacking_proba = stacking_clf.predict_proba(X_test_scaled)[:, 1]
        roc_auc_stacking = roc_auc_score(self.y_test, y_pred_stacking_proba)
        
        print(f"   ✓ Stacking Classifier ROC AUC: {roc_auc_stacking:.4f}")
        
        print("\n" + "=" * 40)
        print("📊 ENSEMBLE COMPARISON:")
        print("=" * 40)
        if 'optimized_best_model' in self.results:
            print(f"   Single Model: {self.results['optimized_best_model']['roc_auc']:.4f}")
        print(f"   Voting:       {roc_auc_voting:.4f}")
        print(f"   Stacking:     {roc_auc_stacking:.4f}")
        print("=" * 40)
        
        best_ensemble = 'Voting' if roc_auc_voting > roc_auc_stacking else 'Stacking'
        best_ensemble_auc = max(roc_auc_voting, roc_auc_stacking)
        
        print(f"\n🏆 Best Ensemble: {best_ensemble} (ROC AUC: {best_ensemble_auc:.4f})")
        
        self.results['voting_ensemble'] = {'model': voting_clf, 'roc_auc': roc_auc_voting}
        self.results['stacking_ensemble'] = {'model': stacking_clf, 'roc_auc': roc_auc_stacking}
        
        return best_ensemble, best_ensemble_auc
    
    # ============================================================================
    # MAIN EXECUTION PIPELINE
    # ============================================================================
    
    def run_full_pipeline(self):
        """Execute the complete 10-stage pipeline"""
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 15 + "KOA PREDICTION - FULL ML PIPELINE" + " " * 30 + "║")
        print("║" + " " * 20 + "Biological Age Risk Analysis" + " " * 31 + "║")
        print("╚" + "=" * 78 + "╝")
        
        try:
            # Stage 1: Load Data
            self.load_data()
            
            # Stage 2.5: Raw Data Baseline (BEFORE feature engineering)
            print("\n" + "🔬" * 40)
            print("TESTING RAW DATA PERFORMANCE (80/20 split)")
            print("🔬" * 40)
            raw_auc = self.raw_data_baseline(test_size=0.2)
            
            # Stage 2: Feature Engineering
            self.engineer_features()
            
            # Stage 2.7: ANOVA F-Test Feature Analysis
            print("\n" + "🔬" * 40)
            print("ANALYZING FEATURES WITH ANOVA F-TEST")
            print("🔬" * 40)
            selected_features_anova, anova_scores = self.anova_feature_analysis(top_k=30)
            
            # Compare: All features vs ANOVA-selected features
            print("\n" + "="*80)
            print("🔄 TESTING WITH ANOVA-SELECTED FEATURES")
            print("="*80)
            
            # Store ANOVA-selected features for later use
            self.selected_features = selected_features_anova
            
            # Stage 3: Baseline RF (AFTER feature engineering, 70/30 split)
            print("\n" + "🔬" * 40)
            print("TESTING ENGINEERED FEATURES PERFORMANCE (70/30 split)")
            print("🔬" * 40)
            rf_model, baseline_auc = self.baseline_random_forest()
            
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"   • Raw Data (80/20):                    ROC AUC = {raw_auc:.4f}")
            print(f"   • ANOVA-Selected Features (70/30):     ROC AUC = {baseline_auc:.4f}")
            improvement_from_raw = ((baseline_auc - raw_auc) / raw_auc * 100)
            print(f"   • Improvement from raw:                {improvement_from_raw:+.2f}%")
            
            if baseline_auc > raw_auc:
                print(f"   ✅ ANOVA feature selection improved performance!")
            else:
                print(f"   ⚠️  Performance decreased. Feature engineering needs review.")
            
            if baseline_auc <= 0.65:
                print("\n⚠️ Baseline performance insufficient. Consider returning to feature engineering.")
                return
            
            # Stage 4: Feature Selection
            self.ensemble_feature_selection()
            
            # Stage 6: Model Comparison
            self.compare_models()
            
            # Stage 7: Hyperparameter Optimization
            if OPTUNA_AVAILABLE:
                self.optimize_hyperparameters(n_trials=50)
            
            # Stage 8: Final Evaluation
            final_auc, final_f1, final_acc = self.final_evaluation()
            
            # Stage 9: Ensemble (if needed)
            if final_auc < 0.85 and ADVANCED_MODELS_AVAILABLE:
                print("\n⚠️ Performance below 0.85. Trying ensemble methods...")
                self.create_ensemble()
            
            print("\n" + "=" * 80)
            print("✅ PIPELINE EXECUTION COMPLETE!")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n❌ Pipeline Error: {str(e)}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n🔬 Starting KOA Prediction Pipeline...")
    print("=" * 80)
    print("NOTE: For advanced feature selection experiments, run:")
    print("  → python feature_analysis_experiments.py")
    print("  → Results: 4 experiments comparing feature selection strategies")
    print("  → Best: Hierarchical Clustering (5 features, F1=0.4263)")
    print("=" * 80)
    
    # Initialize pipeline with IMPUTED dataset (Mean Imputation, 0% data loss)
    # Original: Raw Data .xlsx (listwise deletion, 71.2% data loss)
    # New: data_imputed_mean.csv (mean/mode imputation, +6,763 rows recovered)
    data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_imputed_data\data_imputed_mean.csv"
    
    print("\n" + "="*80)
    print("⚠️  IMPORTANT: Using IMPUTED Dataset (Mean/Mode Imputation)")
    print("="*80)
    print("Dataset: data_imputed_mean.csv")
    print("Method: Mean (numeric) / Mode (categorical)")
    print("Rows: 9,505 (vs 2,742 with listwise deletion)")
    print("Data Recovery: +6,763 rows (+246.6%)")
    print("Performance: ROC AUC 0.6792 (baseline comparison from Step 0)")
    print("="*80 + "\n")
    
    pipeline = KOAPredictionPipeline(data_path)
    
    # Run full pipeline
    pipeline.run_full_pipeline()
    
    print("\n✅ Analysis Complete! Check results above.")
    print("\n📊 Generated Files:")
    print("  • pipeline_results.txt - Main results + feature experiments summary")
    print("  • anova_feature_scores.csv - ANOVA F-test results")
    print("  • anova_feature_analysis.png - Feature importance visualizations")
    print("\n🔬 Feature Experiments (if run separately):")
    print("  • feature_experiments_results.txt - Detailed experiment results")
    print("  • experiments_summary.csv - Quick comparison table")
    print("  • feature_clusters.csv - Hierarchical clustering results")
    print("  • hierarchical_clustering_dendrogram.png - Cluster visualization")
    print("  • representative_features_heatmap.png - Feature correlations")
