"""
Step 0: Imputation Methods Comparison
======================================
Farklı missing value imputation yöntemlerini test edip karşılaştırır.
Sonuçları step 2, 3, 4 için baseline oluşturur.

Imputation Methods:
1. Listwise Deletion (Mevcut yöntem) - Missing rows silinir
2. Mean Imputation - Numeric: mean, Categorical: mode
3. Median Imputation - Numeric: median, Categorical: mode
4. KNN Imputation - K-Nearest Neighbors ile tahmin
5. Iterative Imputation - MICE benzeri iterative imputation
6. Simple Constant - Numeric: 0, Categorical: "missing"

Author: Bioinformatics Analysis Team
Date: March 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import warnings
warnings.filterwarnings('ignore')

import os
import sys

# ============================================================================
# IMPUTATION COMPARISON PIPELINE
# ============================================================================

class ImputationComparison:
    def __init__(self, data_path, output_dir='step0_imputation_comparison'):
        """Initialize Imputation Comparison"""
        self.data_path = data_path
        self.output_dir = output_dir
        self.df_raw = None
        self.imputation_results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results file
        self.results_file = os.path.join(output_dir, 'imputation_comparison_results.txt')
        
    def load_raw_data(self):
        """Load raw data without any cleaning"""
        print("=" * 80)
        print("STEP 0: LOADING RAW DATA (NO CLEANING)")
        print("=" * 80)
        
        self.df_raw = pd.read_excel(self.data_path)
        print(f"✓ Raw data loaded: {self.df_raw.shape[0]} rows × {self.df_raw.shape[1]} columns\n")
        
        # Missing value analysis
        print("📊 Missing Values Analysis:")
        print("-" * 80)
        missing = self.df_raw.isnull().sum()
        missing_pct = (missing / len(self.df_raw)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Missing_Percentage': missing_pct.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df.to_string(index=False))
            print(f"\n⚠️ Total missing values: {missing['Missing_Count'].sum():,}")
            print(f"⚠️ Percentage of complete cases: {(~self.df_raw.isnull().any(axis=1)).sum() / len(self.df_raw) * 100:.2f}%")
        else:
            print("✓ No missing values found!")
        
        # Age distribution
        if 'Biological Age' in self.df_raw.columns:
            print(f"\n📊 Age Distribution:")
            print(f"   • Min age: {self.df_raw['Biological Age'].min():.0f}")
            print(f"   • Max age: {self.df_raw['Biological Age'].max():.0f}")
            print(f"   • Mean age: {self.df_raw['Biological Age'].mean():.1f}")
            print(f"   • Median age: {self.df_raw['Biological Age'].median():.1f}")
            
            age_under_18 = (self.df_raw['Biological Age'] < 18).sum()
            if age_under_18 > 0:
                print(f"   • ⚠️ Ages < 18: {age_under_18} ({age_under_18/len(self.df_raw)*100:.2f}%)")
        
        # Target distribution
        if 'Arthritis' in self.df_raw.columns:
            print(f"\n📊 Target Variable Distribution:")
            target_counts = self.df_raw['Arthritis'].value_counts()
            print(target_counts)
            print(f"\nClass Balance:")
            print(self.df_raw['Arthritis'].value_counts(normalize=True) * 100)
        
        return self.df_raw
    
    def apply_age_filter(self, df):
        """Apply Age >= 18 filter (common preprocessing)"""
        if 'Biological Age' in df.columns:
            before = len(df)
            df = df[df['Biological Age'] >= 18].copy()
            after = len(df)
            removed = before - after
            if removed > 0:
                print(f"   • Age filter (≥18): Removed {removed} rows ({removed/before*100:.2f}%)")
        return df
    
    def method_1_listwise_deletion(self, df):
        """Method 1: Listwise Deletion (Current Method)"""
        print("\n" + "=" * 80)
        print("METHOD 1: LISTWISE DELETION (CURRENT METHOD)")
        print("=" * 80)
        
        df_clean = df.copy()
        before = len(df_clean)
        
        # Remove missing values
        df_clean = df_clean.dropna()
        after = len(df_clean)
        removed = before - after
        
        print(f"\n📊 Listwise Deletion Results:")
        print(f"   • Original rows: {before:,}")
        print(f"   • Rows with missing values removed: {removed:,} ({removed/before*100:.2f}%)")
        print(f"   • Final rows: {after:,} ({after/before*100:.2f}% retained)")
        
        return df_clean, f"Listwise Deletion (n={after:,}, {removed/before*100:.1f}% removed)"
    
    def method_2_mean_imputation(self, df):
        """Method 2: Mean/Mode Imputation"""
        print("\n" + "=" * 80)
        print("METHOD 2: MEAN/MODE IMPUTATION")
        print("=" * 80)
        
        df_imputed = df.copy()
        imputed_count = 0
        
        # Separate numeric and categorical columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from imputation
        if 'Arthritis' in numeric_cols:
            numeric_cols.remove('Arthritis')
        if 'Arthritis' in categorical_cols:
            categorical_cols.remove('Arthritis')
        
        print(f"\n📊 Imputing Missing Values:")
        
        # Numeric: Mean imputation
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                missing_count = df_imputed[col].isnull().sum()
                if missing_count > 0:
                    mean_val = df_imputed[col].mean()
                    df_imputed[col].fillna(mean_val, inplace=True)
                    imputed_count += missing_count
                    print(f"   • {col}: {missing_count} values → Mean={mean_val:.2f}")
        
        # Categorical: Mode imputation
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                missing_count = df_imputed[col].isnull().sum()
                if missing_count > 0:
                    mode_val = df_imputed[col].mode()[0] if len(df_imputed[col].mode()) > 0 else 'missing'
                    df_imputed[col].fillna(mode_val, inplace=True)
                    imputed_count += missing_count
                    print(f"   • {col}: {missing_count} values → Mode={mode_val}")
        
        # Remove rows still missing target
        if 'Arthritis' in df_imputed.columns:
            before = len(df_imputed)
            df_imputed = df_imputed[df_imputed['Arthritis'].notna()]
            after = len(df_imputed)
            if before != after:
                print(f"\n   ⚠️ Removed {before-after} rows with missing target")
        
        print(f"\n✓ Total values imputed: {imputed_count:,}")
        print(f"✓ Final dataset: {len(df_imputed):,} rows")
        
        return df_imputed, f"Mean/Mode Imputation (n={len(df_imputed):,}, {imputed_count:,} imputed)"
    
    def method_3_median_imputation(self, df):
        """Method 3: Median/Mode Imputation"""
        print("\n" + "=" * 80)
        print("METHOD 3: MEDIAN/MODE IMPUTATION")
        print("=" * 80)
        
        df_imputed = df.copy()
        imputed_count = 0
        
        # Separate numeric and categorical columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from imputation
        if 'Arthritis' in numeric_cols:
            numeric_cols.remove('Arthritis')
        if 'Arthritis' in categorical_cols:
            categorical_cols.remove('Arthritis')
        
        print(f"\n📊 Imputing Missing Values:")
        
        # Numeric: Median imputation
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                missing_count = df_imputed[col].isnull().sum()
                if missing_count > 0:
                    median_val = df_imputed[col].median()
                    df_imputed[col].fillna(median_val, inplace=True)
                    imputed_count += missing_count
                    print(f"   • {col}: {missing_count} values → Median={median_val:.2f}")
        
        # Categorical: Mode imputation
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                missing_count = df_imputed[col].isnull().sum()
                if missing_count > 0:
                    mode_val = df_imputed[col].mode()[0] if len(df_imputed[col].mode()) > 0 else 'missing'
                    df_imputed[col].fillna(mode_val, inplace=True)
                    imputed_count += missing_count
                    print(f"   • {col}: {missing_count} values → Mode={mode_val}")
        
        # Remove rows still missing target
        if 'Arthritis' in df_imputed.columns:
            before = len(df_imputed)
            df_imputed = df_imputed[df_imputed['Arthritis'].notna()]
            after = len(df_imputed)
            if before != after:
                print(f"\n   ⚠️ Removed {before-after} rows with missing target")
        
        print(f"\n✓ Total values imputed: {imputed_count:,}")
        print(f"✓ Final dataset: {len(df_imputed):,} rows")
        
        return df_imputed, f"Median/Mode Imputation (n={len(df_imputed):,}, {imputed_count:,} imputed)"
    
    def method_4_knn_imputation(self, df, n_neighbors=5):
        """Method 4: KNN Imputation"""
        print("\n" + "=" * 80)
        print(f"METHOD 4: KNN IMPUTATION (k={n_neighbors})")
        print("=" * 80)
        
        df_imputed = df.copy()
        
        # Encode categorical variables first
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()
        label_encoders = {}
        
        print(f"\n📊 Preprocessing:")
        if len(categorical_cols) > 0:
            print(f"   • Encoding {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                if col != 'Arthritis':
                    le = LabelEncoder()
                    # Handle NaN in categorical
                    mask = df_imputed[col].notna()
                    df_imputed.loc[mask, col] = le.fit_transform(df_imputed.loc[mask, col].astype(str))
                    label_encoders[col] = le
        
        # Prepare data for KNN imputer
        feature_cols = [col for col in df_imputed.columns if col != 'Arthritis']
        
        # Remove rows with missing target
        if 'Arthritis' in df_imputed.columns:
            before = len(df_imputed)
            df_imputed = df_imputed[df_imputed['Arthritis'].notna()]
            after = len(df_imputed)
            if before != after:
                print(f"   • Removed {before-after} rows with missing target")
        
        # Count missing values before imputation
        missing_before = df_imputed[feature_cols].isnull().sum().sum()
        
        print(f"\n📊 KNN Imputation:")
        print(f"   • Missing values to impute: {missing_before:,}")
        print(f"   • Using k={n_neighbors} nearest neighbors")
        print(f"   • This may take a few minutes for large datasets...")
        
        # Apply KNN imputation
        if missing_before > 0:
            imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform')
            df_imputed[feature_cols] = imputer.fit_transform(df_imputed[feature_cols])
            print(f"\n✓ KNN imputation completed!")
            print(f"✓ Final dataset: {len(df_imputed):,} rows")
        else:
            print(f"\n✓ No missing values to impute!")
        
        return df_imputed, f"KNN Imputation k={n_neighbors} (n={len(df_imputed):,}, {missing_before:,} imputed)"
    
    def method_5_iterative_imputation(self, df, max_iter=10):
        """Method 5: Iterative Imputation (MICE-like)"""
        print("\n" + "=" * 80)
        print(f"METHOD 5: ITERATIVE IMPUTATION (max_iter={max_iter})")
        print("=" * 80)
        
        df_imputed = df.copy()
        
        # Encode categorical variables first
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()
        label_encoders = {}
        
        print(f"\n📊 Preprocessing:")
        if len(categorical_cols) > 0:
            print(f"   • Encoding {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                if col != 'Arthritis':
                    le = LabelEncoder()
                    # Handle NaN in categorical
                    mask = df_imputed[col].notna()
                    df_imputed.loc[mask, col] = le.fit_transform(df_imputed.loc[mask, col].astype(str))
                    label_encoders[col] = le
        
        # Prepare data for iterative imputer
        feature_cols = [col for col in df_imputed.columns if col != 'Arthritis']
        
        # Remove rows with missing target
        if 'Arthritis' in df_imputed.columns:
            before = len(df_imputed)
            df_imputed = df_imputed[df_imputed['Arthritis'].notna()]
            after = len(df_imputed)
            if before != after:
                print(f"   • Removed {before-after} rows with missing target")
        
        # Count missing values before imputation
        missing_before = df_imputed[feature_cols].isnull().sum().sum()
        
        print(f"\n📊 Iterative Imputation (MICE-like):")
        print(f"   • Missing values to impute: {missing_before:,}")
        print(f"   • Max iterations: {max_iter}")
        print(f"   • Using BayesianRidge estimator")
        print(f"   • This may take several minutes...")
        
        # Apply iterative imputation
        if missing_before > 0:
            imputer = IterativeImputer(max_iter=max_iter, random_state=42, verbose=0)
            df_imputed[feature_cols] = imputer.fit_transform(df_imputed[feature_cols])
            print(f"\n✓ Iterative imputation completed!")
            print(f"✓ Final dataset: {len(df_imputed):,} rows")
        else:
            print(f"\n✓ No missing values to impute!")
        
        return df_imputed, f"Iterative Imputation iter={max_iter} (n={len(df_imputed):,}, {missing_before:,} imputed)"
    
    def method_6_constant_imputation(self, df):
        """Method 6: Constant Imputation"""
        print("\n" + "=" * 80)
        print("METHOD 6: CONSTANT IMPUTATION")
        print("=" * 80)
        
        df_imputed = df.copy()
        imputed_count = 0
        
        # Separate numeric and categorical columns
        numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from imputation
        if 'Arthritis' in numeric_cols:
            numeric_cols.remove('Arthritis')
        if 'Arthritis' in categorical_cols:
            categorical_cols.remove('Arthritis')
        
        print(f"\n📊 Imputing Missing Values:")
        
        # Numeric: Fill with 0
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                missing_count = df_imputed[col].isnull().sum()
                if missing_count > 0:
                    df_imputed[col].fillna(0, inplace=True)
                    imputed_count += missing_count
                    print(f"   • {col}: {missing_count} values → 0")
        
        # Categorical: Fill with "missing"
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                missing_count = df_imputed[col].isnull().sum()
                if missing_count > 0:
                    df_imputed[col].fillna('missing', inplace=True)
                    imputed_count += missing_count
                    print(f"   • {col}: {missing_count} values → 'missing'")
        
        # Remove rows still missing target
        if 'Arthritis' in df_imputed.columns:
            before = len(df_imputed)
            df_imputed = df_imputed[df_imputed['Arthritis'].notna()]
            after = len(df_imputed)
            if before != after:
                print(f"\n   ⚠️ Removed {before-after} rows with missing target")
        
        print(f"\n✓ Total values imputed: {imputed_count:,}")
        print(f"✓ Final dataset: {len(df_imputed):,} rows")
        
        return df_imputed, f"Constant Imputation (n={len(df_imputed):,}, {imputed_count:,} imputed)"
    
    def evaluate_imputation_method(self, df, method_name):
        """Evaluate imputation method with baseline model"""
        print(f"\n" + "=" * 80)
        print(f"EVALUATING: {method_name}")
        print("=" * 80)
        
        # Basic encoding for target
        target_col = 'Arthritis'
        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' not found!")
            return None
        
        # Encode yes/no to 0/1
        df_eval = df.copy()
        if df_eval[target_col].dtype == 'object':
            df_eval[target_col] = df_eval[target_col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
        
        # Prepare features and target
        drop_cols = [target_col]
        X = df_eval.drop(columns=[col for col in drop_cols if col in df_eval.columns])
        y = df_eval[target_col]
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train baseline model (Logistic Regression)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Baseline Model Performance (Logistic Regression):")
        print(f"   • ROC AUC: {roc_auc:.4f}")
        print(f"   • F1 Score: {f1:.4f}")
        print(f"   • Accuracy: {accuracy:.4f}")
        print(f"   • Training samples: {len(X_train):,}")
        print(f"   • Test samples: {len(X_test):,}")
        print(f"   • Features: {X.shape[1]}")
        
        return {
            'method': method_name,
            'n_samples': len(df),
            'n_features': X.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'roc_auc': roc_auc,
            'f1_score': f1,
            'accuracy': accuracy
        }
    
    def run_comparison(self):
        """Run all imputation methods and compare"""
        print("\n" + "=" * 80)
        print("IMPUTATION METHODS COMPARISON")
        print("=" * 80)
        
        # Load raw data
        self.load_raw_data()
        
        # Apply age filter (common preprocessing)
        df_filtered = self.apply_age_filter(self.df_raw)
        print(f"\n✓ After age filter (≥18): {len(df_filtered):,} rows")
        
        results_list = []
        
        # Method 1: Listwise Deletion
        try:
            df1, desc1 = self.method_1_listwise_deletion(df_filtered)
            result1 = self.evaluate_imputation_method(df1, desc1)
            if result1:
                results_list.append(result1)
        except Exception as e:
            print(f"❌ Method 1 failed: {str(e)}")
        
        # Method 2: Mean Imputation
        try:
            df2, desc2 = self.method_2_mean_imputation(df_filtered)
            result2 = self.evaluate_imputation_method(df2, desc2)
            if result2:
                results_list.append(result2)
        except Exception as e:
            print(f"❌ Method 2 failed: {str(e)}")
        
        # Method 3: Median Imputation
        try:
            df3, desc3 = self.method_3_median_imputation(df_filtered)
            result3 = self.evaluate_imputation_method(df3, desc3)
            if result3:
                results_list.append(result3)
        except Exception as e:
            print(f"❌ Method 3 failed: {str(e)}")
        
        # Method 4: KNN Imputation
        try:
            df4, desc4 = self.method_4_knn_imputation(df_filtered, n_neighbors=5)
            result4 = self.evaluate_imputation_method(df4, desc4)
            if result4:
                results_list.append(result4)
        except Exception as e:
            print(f"❌ Method 4 failed: {str(e)}")
        
        # Method 5: Iterative Imputation
        try:
            df5, desc5 = self.method_5_iterative_imputation(df_filtered, max_iter=10)
            result5 = self.evaluate_imputation_method(df5, desc5)
            if result5:
                results_list.append(result5)
        except Exception as e:
            print(f"❌ Method 5 failed: {str(e)}")
        
        # Method 6: Constant Imputation
        try:
            df6, desc6 = self.method_6_constant_imputation(df_filtered)
            result6 = self.evaluate_imputation_method(df6, desc6)
            if result6:
                results_list.append(result6)
        except Exception as e:
            print(f"❌ Method 6 failed: {str(e)}")
        
        # Create comparison dataframe
        if len(results_list) > 0:
            results_df = pd.DataFrame(results_list)
            results_df = results_df.sort_values('roc_auc', ascending=False)
            
            # Save results
            results_df.to_csv(os.path.join(self.output_dir, 'imputation_comparison_summary.csv'), index=False)
            
            # Print comparison
            print("\n" + "=" * 80)
            print("📊 IMPUTATION METHODS COMPARISON SUMMARY")
            print("=" * 80)
            print(results_df.to_string(index=False))
            
            # Best method
            best_method = results_df.iloc[0]
            print(f"\n🏆 BEST METHOD: {best_method['method']}")
            print(f"   • ROC AUC: {best_method['roc_auc']:.4f}")
            print(f"   • F1 Score: {best_method['f1_score']:.4f}")
            print(f"   • Samples retained: {best_method['n_samples']:,}")
            
            # Save detailed results to file
            with open(self.results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("IMPUTATION METHODS COMPARISON - DETAILED RESULTS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Date: {pd.Timestamp.now()}\n")
                f.write(f"Original dataset: {len(self.df_raw):,} rows\n")
                f.write(f"After age filter (≥18): {len(df_filtered):,} rows\n\n")
                f.write("=" * 80 + "\n")
                f.write("COMPARISON SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write(results_df.to_string(index=False))
                f.write("\n\n")
                f.write("=" * 80 + "\n")
                f.write(f"BEST METHOD: {best_method['method']}\n")
                f.write("=" * 80 + "\n")
                f.write(f"ROC AUC: {best_method['roc_auc']:.4f}\n")
                f.write(f"F1 Score: {best_method['f1_score']:.4f}\n")
                f.write(f"Accuracy: {best_method['accuracy']:.4f}\n")
                f.write(f"Samples: {best_method['n_samples']:,}\n")
            
            print(f"\n✓ Results saved to: {self.results_file}")
            
            # Create visualization
            self.plot_comparison(results_df)
        
        return results_df if len(results_list) > 0 else None
    
    def plot_comparison(self, results_df):
        """Plot comparison of imputation methods"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. ROC AUC comparison
        ax1 = axes[0, 0]
        methods = [m.split('(')[0].strip() for m in results_df['method']]
        ax1.barh(methods, results_df['roc_auc'], color='steelblue')
        ax1.set_xlabel('ROC AUC')
        ax1.set_title('ROC AUC by Imputation Method')
        ax1.set_xlim([0, 1])
        for i, v in enumerate(results_df['roc_auc']):
            ax1.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # 2. F1 Score comparison
        ax2 = axes[0, 1]
        ax2.barh(methods, results_df['f1_score'], color='coral')
        ax2.set_xlabel('F1 Score')
        ax2.set_title('F1 Score by Imputation Method')
        ax2.set_xlim([0, 1])
        for i, v in enumerate(results_df['f1_score']):
            ax2.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        # 3. Sample size comparison
        ax3 = axes[1, 0]
        ax3.barh(methods, results_df['n_samples'], color='lightgreen')
        ax3.set_xlabel('Number of Samples')
        ax3.set_title('Sample Size by Imputation Method')
        for i, v in enumerate(results_df['n_samples']):
            ax3.text(v + 50, i, f'{v:,}', va='center')
        
        # 4. Combined metric (ROC AUC vs Sample Size)
        ax4 = axes[1, 1]
        scatter = ax4.scatter(results_df['n_samples'], results_df['roc_auc'], 
                            s=results_df['f1_score']*1000, alpha=0.6, c=range(len(results_df)), cmap='viridis')
        ax4.set_xlabel('Number of Samples')
        ax4.set_ylabel('ROC AUC')
        ax4.set_title('ROC AUC vs Sample Size (bubble size = F1 Score)')
        for i, method in enumerate(methods):
            ax4.annotate(method, (results_df.iloc[i]['n_samples'], results_df.iloc[i]['roc_auc']),
                        fontsize=8, ha='right')
        
        plt.tight_layout()
        plot_file = os.path.join(self.output_dir, 'imputation_comparison_plot.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {plot_file}")
        plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("IMPUTATION METHODS COMPARISON PIPELINE")
    print("Testing 6 different imputation strategies")
    print("="*80 + "\n")
    
    # File path
    data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found at {data_path}")
        sys.exit(1)
    
    # Run comparison
    pipeline = ImputationComparison(data_path)
    results = pipeline.run_comparison()
    
    if results is not None:
        print("\n" + "="*80)
        print("✅ IMPUTATION COMPARISON COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"\nResults saved to: step0_imputation_comparison/")
        print("  • imputation_comparison_summary.csv")
        print("  • imputation_comparison_results.txt")
        print("  • imputation_comparison_plot.png")
    else:
        print("\n❌ Comparison failed - no results generated")
