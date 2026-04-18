"""
Feature Selection Methods Comparison
6 farklı feature selection yönteminin karşılaştırmalı değerlendirmesi

Yöntemler:
1. ANOVA F-test (Univariate Statistical Test)
2. Mutual Information (Information Theory)
3. L1 Logistic Regression (LASSO - L1 Regularization)
4. Random Forest / XGBoost Feature Importance (Tree-based)
5. RFECV - Recursive Feature Elimination with CV (Wrapper Method)
6. Correlation Clustering / Hierarchical Clustering (Redundancy Removal)

Her yöntem için:
- Feature seçimi yapılır
- Random Forest model eğitilir
- ROC AUC, F1 Score, Accuracy karşılaştırılır
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    f_classif, mutual_info_classif, SelectKBest, 
    SelectFromModel, RFECV, RFE
)
from sklearn.linear_model import LogisticRegression
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# XGBoost için
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available. Using RF only for tree-based importance.")
    XGBOOST_AVAILABLE = False


class FeatureSelectionComparison:
    """6 farklı feature selection yöntemini karşılaştırır"""
    
    def __init__(self, data_path, n_features_to_select=15, results_file='feature_selection_comparison_results.txt'):
        self.data_path = data_path
        self.n_features = n_features_to_select
        self.results_file = results_file
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.results = {}
        self.selected_features = {}
        
        # Sonuç dosyasını başlat
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("FEATURE SELECTION METHODS COMPARISON\n")
            f.write("KOA Prediction - 6 Different Approaches\n")
            f.write("=" * 100 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target number of features: {n_features_to_select}\n")
            f.write("=" * 100 + "\n\n")
    
    def log_result(self, method_name, metrics, selected_features, additional_info=""):
        """Sonuçları kaydet"""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*100}\n")
            f.write(f"METHOD: {method_name}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*100}\n")
            
            if additional_info:
                f.write(f"\n{additional_info}\n")
                f.write(f"{'-'*100}\n")
            
            f.write(f"\nSelected Features ({len(selected_features)}):\n")
            for i, feat in enumerate(selected_features[:20], 1):  # Top 20
                f.write(f"  {i:2d}. {feat}\n")
            if len(selected_features) > 20:
                f.write(f"  ... and {len(selected_features) - 20} more\n")
            
            f.write(f"\n{'-'*100}\n")
            f.write("Performance Metrics:\n")
            for key, value in metrics.items():
                f.write(f"  • {key}: {value}\n")
            f.write("\n")
    
    def load_and_prepare_data(self):
        """Veriyi yükle ve feature engineering yap"""
        print("\n" + "="*100)
        print("DATA LOADING AND PREPARATION")
        print("="*100)
        
        print("Loading data...")
        self.df = pd.read_excel(self.data_path, sheet_name=0)
        print(f"Loaded: {self.df.shape}")
        
        # Feature engineering (simplified version)
        print("Feature engineering...")
        self.engineer_features()
        
        # Target ve features'ları ayır
        target_col = 'Arthritis'
        valid_data = self.df[self.df[target_col].notna()].copy()
        
        # Target encode
        valid_data[target_col] = valid_data[target_col].map({'yes': 1, 'no': 0, 1: 1, 0: 0})
        
        # Sadece numeric features
        exclude_cols = [target_col, 'ID', 'id']
        feature_cols = [col for col in valid_data.columns if col not in exclude_cols]
        X_all = valid_data[feature_cols]
        numeric_cols = X_all.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        self.X = X_all[numeric_cols].copy()
        self.y = valid_data[target_col].copy()
        self.feature_names = self.X.columns.tolist()
        
        # NaN handling
        self.X = self.X.fillna(self.X.median())
        self.X = self.X.fillna(0)
        self.X = self.X.replace([np.inf, -np.inf], 0)
        
        print(f"Final data: X={self.X.shape}, y={self.y.shape}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Class distribution: {self.y.value_counts().to_dict()}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        print(f"Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        
        return self.X, self.y
    
    def engineer_features(self):
        """Feature engineering - simplified"""
        df = self.df.copy()
        
        # Standardize column names
        if 'Biological Age' in df.columns:
            df['Biological_Age'] = df['Biological Age']
        if 'bmi_kg.m2' in df.columns:
            df['BMI'] = df['bmi_kg.m2']
        
        # Encode binary columns
        yes_no_cols = ['doctor_diagnose_hypertension', 'doctor_diagnose_dyslipidemia', 
                       'doctor_diagnose_diabetes', 'doctor_diagnose_cancer', 
                       'doctor_diagnose_heart_disease', 'smoke', 'drink', 'Arthritis']
        for col in yes_no_cols:
            if col in df.columns:
                df[col] = df[col].map({'yes': 1, 'no': 0, 'Yes': 1, 'No': 0})
        
        # Comorbidities
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
        
        # Sex encoding
        if 'sex' in df.columns:
            df['Sex_Numeric'] = df['sex'].map({'male': 1, 'female': 0, 'Male': 1, 'Female': 0})
        
        # Aging features
        if 'Biological_Age' in df.columns:
            df['Biological_Age_Squared'] = df['Biological_Age'] ** 2
            df['Biological_Age_Cubed'] = df['Biological_Age'] ** 3
            df['Log_Biological_Age'] = np.log1p(df['Biological_Age'])
            df['BioAge_60_Plus'] = (df['Biological_Age'] >= 60).astype(int)
            df['BioAge_70_Plus'] = (df['Biological_Age'] >= 70).astype(int)
            df['BioAge_Over_67'] = (df['Biological_Age'] > 66.7).astype(int)
        
        # BMI features
        if 'BMI' in df.columns:
            df['BMI_Squared'] = df['BMI'] ** 2
            df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
            df['BMI_Category'] = df['BMI_Category'].astype(float)
            df['BMI_Obese'] = (df['BMI'] >= 30).astype(int)
            df['Log_BMI'] = np.log1p(df['BMI'])
        
        # Comorbidity features
        comorbidity_cols = []
        for col in ['Hypertension', 'Dyslipidemia', 'Diabetes', 'Cancer', 'Heart_Disease']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                comorbidity_cols.append(col)
        
        if comorbidity_cols:
            df['Comorbidity_Count'] = df[comorbidity_cols].sum(axis=1)
            df['Has_Multiple_Comorbidities'] = (df['Comorbidity_Count'] >= 2).astype(int)
            df['Comorbidity_Risk_Score'] = df['Comorbidity_Count'] / len(comorbidity_cols)
        
        # Interaction features
        if 'BMI' in df.columns and 'Biological_Age' in df.columns:
            df['BMI_x_BioAge'] = df['BMI'] * df['Biological_Age']
        
        if 'Biological_Age' in df.columns and 'Comorbidity_Count' in df.columns:
            df['Age_x_Comorbidity'] = df['Biological_Age'] * df['Comorbidity_Count']
        
        if 'BMI' in df.columns and 'Comorbidity_Count' in df.columns:
            df['BMI_x_Comorbidity'] = df['BMI'] * df['Comorbidity_Count']
        
        # Position_Knees encoding
        if 'position_knees' in df.columns:
            df['Position_Knees_Encoded'] = pd.Categorical(df['position_knees']).codes
        
        self.df = df
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, method_name):
        """Model eğit ve değerlendir"""
        print(f"  Training model with {X_train.shape[1]} features...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf.predict(X_test)
        y_pred_proba = rf.predict_proba(X_test)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        
        metrics = {
            'Method': method_name,
            'Feature_Count': X_train.shape[1],
            'ROC_AUC': f"{roc_auc:.4f}",
            'F1_Score': f"{f1:.4f}",
            'Accuracy': f"{accuracy:.4f}"
        }
        
        print(f"  ROC AUC: {roc_auc:.4f} | F1: {f1:.4f} | Acc: {accuracy:.4f}")
        
        return metrics, rf, roc_auc, f1, accuracy
    
    # ========================================================================
    # METHOD 1: ANOVA F-test
    # ========================================================================
    
    def method_1_anova_ftest(self):
        """ANOVA F-test ile feature selection"""
        print("\n" + "="*100)
        print("METHOD 1: ANOVA F-test (Univariate Statistical Test)")
        print("="*100)
        
        # F-test
        F_scores, p_values = f_classif(self.X_train, self.y_train)
        
        # Top K features seç
        selector = SelectKBest(score_func=f_classif, k=self.n_features)
        selector.fit(self.X_train, self.y_train)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        # Skorları göster
        feature_scores = pd.DataFrame({
            'Feature': self.feature_names,
            'F_Score': F_scores,
            'p_value': p_values
        }).sort_values('F_Score', ascending=False)
        
        print(f"\nTop 10 F-scores:")
        print(feature_scores.head(10).to_string(index=False))
        
        # Transform data
        X_train_selected = selector.transform(self.X_train)
        X_test_selected = selector.transform(self.X_test)
        
        # Train model
        metrics, rf, auc, f1, acc = self.train_and_evaluate(
            X_train_selected, X_test_selected, self.y_train, self.y_test,
            "ANOVA F-test"
        )
        
        # Save results
        additional_info = f"Top {self.n_features} features by F-score\nMean F-score: {F_scores[selected_indices].mean():.2f}"
        self.log_result("1. ANOVA F-test", metrics, selected_features, additional_info)
        
        self.results['ANOVA F-test'] = metrics
        self.selected_features['ANOVA F-test'] = selected_features
        
        return selected_features, auc, f1, acc
    
    # ========================================================================
    # METHOD 2: Mutual Information
    # ========================================================================
    
    def method_2_mutual_information(self):
        """Mutual Information ile feature selection"""
        print("\n" + "="*100)
        print("METHOD 2: Mutual Information (Information Theory)")
        print("="*100)
        
        # Mutual information
        mi_scores = mutual_info_classif(self.X_train, self.y_train, random_state=42)
        
        # Top K features seç
        selector = SelectKBest(score_func=mutual_info_classif, k=self.n_features)
        selector.fit(self.X_train, self.y_train)
        
        selected_indices = selector.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        # Skorları göster
        feature_scores = pd.DataFrame({
            'Feature': self.feature_names,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        print(f"\nTop 10 MI scores:")
        print(feature_scores.head(10).to_string(index=False))
        
        # Transform data
        X_train_selected = selector.transform(self.X_train)
        X_test_selected = selector.transform(self.X_test)
        
        # Train model
        metrics, rf, auc, f1, acc = self.train_and_evaluate(
            X_train_selected, X_test_selected, self.y_train, self.y_test,
            "Mutual Information"
        )
        
        # Save results
        additional_info = f"Top {self.n_features} features by Mutual Information\nMean MI score: {mi_scores[selected_indices].mean():.4f}"
        self.log_result("2. Mutual Information", metrics, selected_features, additional_info)
        
        self.results['Mutual Information'] = metrics
        self.selected_features['Mutual Information'] = selected_features
        
        return selected_features, auc, f1, acc
    
    # ========================================================================
    # METHOD 3: L1 Logistic Regression (LASSO)
    # ========================================================================
    
    def method_3_l1_logistic_regression(self):
        """L1 Logistic Regression (LASSO) ile feature selection"""
        print("\n" + "="*100)
        print("METHOD 3: L1 Logistic Regression (LASSO)")
        print("="*100)
        
        # Scale data (important for L1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        # L1 Logistic Regression
        lasso = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,  # Regularization strength
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        lasso.fit(X_train_scaled, self.y_train)
        
        # Get feature importance (absolute coefficients)
        importance = np.abs(lasso.coef_[0])
        
        # Select top K features
        top_indices = np.argsort(importance)[::-1][:self.n_features]
        selected_features = [self.feature_names[i] for i in top_indices]
        
        # Show coefficients
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Abs_Coefficient': importance
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print(f"\nTop 10 L1 coefficients:")
        print(feature_importance.head(10).to_string(index=False))
        print(f"Non-zero coefficients: {np.sum(importance > 0)}/{len(importance)}")
        
        # Select features
        X_train_selected = X_train_scaled[:, top_indices]
        X_test_selected = X_test_scaled[:, top_indices]
        
        # Train RF model
        metrics, rf, auc, f1, acc = self.train_and_evaluate(
            X_train_selected, X_test_selected, self.y_train, self.y_test,
            "L1 Logistic Regression"
        )
        
        # Save results
        additional_info = (f"L1 regularization (C=0.1)\n"
                          f"Non-zero coefficients: {np.sum(importance > 0)}\n"
                          f"Selected top {self.n_features} by absolute coefficient")
        self.log_result("3. L1 Logistic Regression (LASSO)", metrics, selected_features, additional_info)
        
        self.results['L1 LASSO'] = metrics
        self.selected_features['L1 LASSO'] = selected_features
        
        return selected_features, auc, f1, acc
    
    # ========================================================================
    # METHOD 4: Random Forest / XGBoost Feature Importance
    # ========================================================================
    
    def method_4_tree_based_importance(self):
        """Random Forest / XGBoost feature importance ile selection"""
        print("\n" + "="*100)
        print("METHOD 4: Tree-Based Feature Importance (RF + XGBoost)")
        print("="*100)
        
        importance_scores = {}
        
        # Random Forest importance
        print("\n  4a. Random Forest importance...")
        rf_temp = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf_temp.fit(self.X_train, self.y_train)
        importance_scores['RF'] = rf_temp.feature_importances_
        
        # XGBoost importance (if available)
        if XGBOOST_AVAILABLE:
            print("  4b. XGBoost importance...")
            xgb_temp = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )
            xgb_temp.fit(self.X_train, self.y_train)
            importance_scores['XGB'] = xgb_temp.feature_importances_
            
            # Average importance
            combined_importance = (importance_scores['RF'] + importance_scores['XGB']) / 2
            print("  Using averaged RF + XGBoost importance")
        else:
            combined_importance = importance_scores['RF']
            print("  Using RF importance only (XGBoost not available)")
        
        # Select top K features
        top_indices = np.argsort(combined_importance)[::-1][:self.n_features]
        selected_features = [self.feature_names[i] for i in top_indices]
        
        # Show importance
        feature_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': combined_importance
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 feature importances:")
        print(feature_importance.head(10).to_string(index=False))
        
        # Select features
        selected_cols = [self.feature_names.index(f) for f in selected_features]
        X_train_selected = self.X_train.values[:, selected_cols]
        X_test_selected = self.X_test.values[:, selected_cols]
        
        # Train model
        metrics, rf, auc, f1, acc = self.train_and_evaluate(
            X_train_selected, X_test_selected, self.y_train, self.y_test,
            "Tree-Based Importance"
        )
        
        # Save results
        method_desc = "RF + XGBoost average" if XGBOOST_AVAILABLE else "RF only"
        additional_info = f"Tree-based feature importance ({method_desc})\nTop {self.n_features} features by importance"
        self.log_result("4. Tree-Based Feature Importance", metrics, selected_features, additional_info)
        
        self.results['Tree-Based'] = metrics
        self.selected_features['Tree-Based'] = selected_features
        
        return selected_features, auc, f1, acc
    
    # ========================================================================
    # METHOD 5: RFECV - Recursive Feature Elimination with CV
    # ========================================================================
    
    def method_5_rfecv(self):
        """RFECV ile feature selection"""
        print("\n" + "="*100)
        print("METHOD 5: RFECV - Recursive Feature Elimination with Cross-Validation")
        print("="*100)
        
        # Base estimator
        base_estimator = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # RFECV
        print(f"  Running RFECV (target: {self.n_features} features, may take time)...")
        rfecv = RFECV(
            estimator=base_estimator,
            step=1,
            cv=StratifiedKFold(3),  # 3-fold CV
            scoring='roc_auc',
            min_features_to_select=self.n_features,
            n_jobs=-1
        )
        
        rfecv.fit(self.X_train, self.y_train)
        
        # Get selected features
        selected_indices = rfecv.get_support(indices=True)
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        print(f"\n  Optimal number of features: {rfecv.n_features_}")
        print(f"  Best CV score: {rfecv.cv_results_['mean_test_score'].max():.4f}")
        
        # Show ranking
        feature_ranking = pd.DataFrame({
            'Feature': self.feature_names,
            'Ranking': rfecv.ranking_,
            'Selected': rfecv.support_
        }).sort_values('Ranking')
        
        print(f"\nTop 10 selected features:")
        print(feature_ranking[feature_ranking['Selected']].head(10).to_string(index=False))
        
        # Transform data
        X_train_selected = rfecv.transform(self.X_train)
        X_test_selected = rfecv.transform(self.X_test)
        
        # Train model
        metrics, rf, auc, f1, acc = self.train_and_evaluate(
            X_train_selected, X_test_selected, self.y_train, self.y_test,
            "RFECV"
        )
        
        # Save results
        additional_info = (f"Recursive Feature Elimination with 3-fold CV\n"
                          f"Optimal features: {rfecv.n_features_}\n"
                          f"Best CV score: {rfecv.cv_results_['mean_test_score'].max():.4f}")
        self.log_result("5. RFECV", metrics, selected_features, additional_info)
        
        self.results['RFECV'] = metrics
        self.selected_features['RFECV'] = selected_features
        
        return selected_features, auc, f1, acc
    
    # ========================================================================
    # METHOD 6: Hierarchical Clustering (Redundancy Removal)
    # ========================================================================
    
    def method_6_hierarchical_clustering(self):
        """Hierarchical clustering ile redundancy removal"""
        print("\n" + "="*100)
        print("METHOD 6: Hierarchical Clustering (Correlation-Based Redundancy Removal)")
        print("="*100)
        
        # Correlation matrix
        print("  Computing correlation matrix...")
        corr_matrix = self.X_train.corr().abs()
        
        # Distance matrix
        distance_matrix = (1 - corr_matrix).values.copy()
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=1.0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # Hierarchical clustering
        print("  Running hierarchical clustering...")
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Create clusters
        n_clusters = self.n_features
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Get ANOVA F-scores to select best from each cluster
        F_scores, _ = f_classif(self.X_train, self.y_train)
        
        # Select one representative from each cluster (highest F-score)
        print(f"  Selecting best feature from each of {n_clusters} clusters...")
        selected_features = []
        cluster_info = []
        
        for cluster_id in range(1, n_clusters + 1):
            cluster_mask = clusters == cluster_id
            cluster_features = [self.feature_names[i] for i, mask in enumerate(cluster_mask) if mask]
            cluster_f_scores = F_scores[cluster_mask]
            
            # Select best feature in cluster
            best_idx = np.argmax(cluster_f_scores)
            best_feature = cluster_features[best_idx]
            selected_features.append(best_feature)
            
            cluster_info.append({
                'Cluster': cluster_id,
                'Size': len(cluster_features),
                'Representative': best_feature,
                'F_Score': cluster_f_scores[best_idx]
            })
        
        cluster_df = pd.DataFrame(cluster_info)
        print(f"\nCluster summary:")
        print(cluster_df.to_string(index=False))
        
        # Select columns
        selected_cols = [self.feature_names.index(f) for f in selected_features]
        X_train_selected = self.X_train.values[:, selected_cols]
        X_test_selected = self.X_test.values[:, selected_cols]
        
        # Train model
        metrics, rf, auc, f1, acc = self.train_and_evaluate(
            X_train_selected, X_test_selected, self.y_train, self.y_test,
            "Hierarchical Clustering"
        )
        
        # Save results
        additional_info = (f"Hierarchical clustering (ward linkage)\n"
                          f"Clusters: {n_clusters}\n"
                          f"One representative per cluster (highest F-score)")
        self.log_result("6. Hierarchical Clustering", metrics, selected_features, additional_info)
        
        self.results['Hierarchical Clustering'] = metrics
        self.selected_features['Hierarchical Clustering'] = selected_features
        
        return selected_features, auc, f1, acc
    
    # ========================================================================
    # COMPARISON AND VISUALIZATION
    # ========================================================================
    
    def compare_and_visualize(self):
        """Tüm yöntemleri karşılaştır ve görselleştir"""
        print("\n" + "="*100)
        print("COMPARISON AND VISUALIZATION")
        print("="*100)
        
        # Create comparison dataframe
        comparison_data = []
        for method_name, metrics in self.results.items():
            comparison_data.append({
                'Method': method_name,
                'ROC_AUC': float(metrics['ROC_AUC']),
                'F1_Score': float(metrics['F1_Score']),
                'Accuracy': float(metrics['Accuracy']),
                'Features': int(metrics['Feature_Count'])
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ROC_AUC', ascending=False)
        
        print("\n" + "="*100)
        print("FINAL COMPARISON TABLE")
        print("="*100)
        print(comparison_df.to_string(index=False))
        
        # Save to CSV
        comparison_df.to_csv('feature_selection_methods_comparison.csv', index=False)
        print("\n✓ Saved: feature_selection_methods_comparison.csv")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. ROC AUC comparison
        ax1 = axes[0, 0]
        bars1 = ax1.barh(comparison_df['Method'], comparison_df['ROC_AUC'], color='skyblue')
        ax1.set_xlabel('ROC AUC', fontsize=12, fontweight='bold')
        ax1.set_title('ROC AUC Comparison', fontsize=14, fontweight='bold')
        ax1.set_xlim(0.5, 0.7)
        for i, (bar, val) in enumerate(zip(bars1, comparison_df['ROC_AUC'])):
            ax1.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10)
        ax1.grid(axis='x', alpha=0.3)
        
        # 2. F1 Score comparison
        ax2 = axes[0, 1]
        bars2 = ax2.barh(comparison_df['Method'], comparison_df['F1_Score'], color='lightcoral')
        ax2.set_xlabel('F1 Score', fontsize=12, fontweight='bold')
        ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlim(0.3, 0.5)
        for i, (bar, val) in enumerate(zip(bars2, comparison_df['F1_Score'])):
            ax2.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. Accuracy comparison
        ax3 = axes[1, 0]
        bars3 = ax3.barh(comparison_df['Method'], comparison_df['Accuracy'], color='lightgreen')
        ax3.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax3.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax3.set_xlim(0.5, 0.7)
        for i, (bar, val) in enumerate(zip(bars3, comparison_df['Accuracy'])):
            ax3.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=10)
        ax3.grid(axis='x', alpha=0.3)
        
        # 4. Multi-metric radar chart
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary text
        best_auc = comparison_df.iloc[0]
        best_f1 = comparison_df.loc[comparison_df['F1_Score'].idxmax()]
        best_acc = comparison_df.loc[comparison_df['Accuracy'].idxmax()]
        
        summary_text = f"""
BEST PERFORMERS

🥇 Best ROC AUC:
   {best_auc['Method']} = {best_auc['ROC_AUC']:.4f}

🥇 Best F1 Score:
   {best_f1['Method']} = {best_f1['F1_Score']:.4f}

🥇 Best Accuracy:
   {best_acc['Method']} = {best_acc['Accuracy']:.4f}

📊 Average Performance:
   ROC AUC: {comparison_df['ROC_AUC'].mean():.4f}
   F1 Score: {comparison_df['F1_Score'].mean():.4f}
   Accuracy: {comparison_df['Accuracy'].mean():.4f}

📈 Performance Range:
   ROC AUC: {comparison_df['ROC_AUC'].min():.4f} - {comparison_df['ROC_AUC'].max():.4f}
   Δ = {comparison_df['ROC_AUC'].max() - comparison_df['ROC_AUC'].min():.4f}
"""
        
        ax4.text(0.1, 0.5, summary_text, 
                fontsize=11, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('feature_selection_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: feature_selection_comparison.png")
        plt.close()
        
        # Feature overlap analysis
        self.analyze_feature_overlap()
        
        return comparison_df
    
    def analyze_feature_overlap(self):
        """Yöntemler arasında feature overlap analizi"""
        print("\n" + "="*100)
        print("FEATURE OVERLAP ANALYSIS")
        print("="*100)
        
        # Create overlap matrix
        methods = list(self.selected_features.keys())
        n_methods = len(methods)
        overlap_matrix = np.zeros((n_methods, n_methods))
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                set1 = set(self.selected_features[method1])
                set2 = set(self.selected_features[method2])
                overlap = len(set1.intersection(set2))
                overlap_matrix[i, j] = overlap
        
        # Visualize overlap
        plt.figure(figsize=(12, 10))
        sns.heatmap(overlap_matrix, annot=True, fmt='.0f', 
                   xticklabels=methods, yticklabels=methods,
                   cmap='YlOrRd', cbar_kws={'label': 'Number of Overlapping Features'})
        plt.title('Feature Selection Methods - Overlap Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_overlap_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: feature_overlap_heatmap.png")
        plt.close()
        
        # Find consensus features (selected by multiple methods)
        all_features = []
        for features in self.selected_features.values():
            all_features.extend(features)
        
        feature_counts = pd.Series(all_features).value_counts()
        consensus_features = feature_counts[feature_counts >= 3]  # Selected by at least 3 methods
        
        if len(consensus_features) > 0:
            print(f"\nConsensus Features (selected by ≥3 methods):")
            for feat, count in consensus_features.items():
                print(f"  • {feat}: selected by {count}/{n_methods} methods")
        else:
            print("\nNo strong consensus features (none selected by ≥3 methods)")
        
        # Log to file
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*100 + "\n")
            f.write("FEATURE OVERLAP ANALYSIS\n")
            f.write("="*100 + "\n")
            f.write(f"\nOverlap Matrix:\n")
            overlap_df = pd.DataFrame(overlap_matrix, index=methods, columns=methods)
            f.write(overlap_df.to_string())
            f.write("\n\n")
            if len(consensus_features) > 0:
                f.write(f"Consensus Features (≥3 methods):\n")
                for feat, count in consensus_features.items():
                    f.write(f"  • {feat}: {count}/{n_methods} methods\n")
    
    # ========================================================================
    # MAIN EXECUTION
    # ========================================================================
    
    def run_all_comparisons(self):
        """Tüm feature selection yöntemlerini çalıştır"""
        print("\n" + "="*100)
        print("STARTING FEATURE SELECTION METHODS COMPARISON")
        print("="*100)
        
        # Load data
        self.load_and_prepare_data()
        
        # Run all methods
        print("\n🔬 Running 6 feature selection methods...\n")
        
        self.method_1_anova_ftest()
        self.method_2_mutual_information()
        self.method_3_l1_logistic_regression()
        self.method_4_tree_based_importance()
        self.method_5_rfecv()
        self.method_6_hierarchical_clustering()
        
        # Compare and visualize
        comparison_df = self.compare_and_visualize()
        
        # Final summary
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*100 + "\n")
            f.write("FINAL SUMMARY\n")
            f.write("="*100 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n")
        
        print("\n" + "="*100)
        print("✅ ALL COMPARISONS COMPLETE!")
        print("="*100)
        print("\n📁 Generated Files:")
        print(f"  • {self.results_file}")
        print("  • feature_selection_methods_comparison.csv")
        print("  • feature_selection_comparison.png")
        print("  • feature_overlap_heatmap.png")
        
        return comparison_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\Raw Data .xlsx"
    n_features = 15  # Her yöntemle 15 feature seç
    
    comparator = FeatureSelectionComparison(data_path, n_features_to_select=n_features)
    comparison_df = comparator.run_all_comparisons()
    
    print("\n✅ Feature selection comparison complete!")


if __name__ == "__main__":
    main()
