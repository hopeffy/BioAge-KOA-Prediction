"""
Feature Analysis Experiments
Farklı feature seçim stratejilerinin KOA prediction üzerindeki etkisini test eder.

Deneyler:
1. Position_Knees_Encoded çıkarılarak test
2. p < 0.05 olan features ile test
3. 0.01 < p < 0.05 aralığındaki features ile test
4. Hierarchical clustering of features + görselleştirme
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalysisExperiments:
    """ANOVA ve hierarchical clustering dayalı feature analiz deneyleri"""
    
    def __init__(self, data_path, results_file='feature_experiments_results.txt'):
        self.data_path = data_path
        self.results_file = results_file
        self.df = None
        self.X = None
        self.y = None
        self.feature_scores = None
        
        # Sonuç dosyasını başlat
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FEATURE ANALYSIS EXPERIMENTS - KOA PREDICTION\n")
            f.write("=" * 80 + "\n")
            f.write(f"Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
    
    def log_result(self, experiment_name, metrics, additional_info=""):
        """Deney sonuçlarını kaydet"""
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"DENEY: {experiment_name}\n")
            f.write(f"Zaman: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n")
            
            if additional_info:
                f.write(f"{additional_info}\n")
                f.write(f"{'-'*80}\n")
            
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
    
    def load_and_prepare_data(self):
        """Veriyi yükle ve feature engineering yap"""
        print("Veri yükleniyor...")
        # Load CSV or Excel
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        else:
            self.df = pd.read_excel(self.data_path, sheet_name=0)
        print(f"Yüklenen veri: {self.df.shape}")
        
        # Feature engineering
        print("Feature engineering yapılıyor...")
        self.engineer_features()
        
        # Target ve features'ları ayır
        target_col = 'Arthritis'
        valid_data = self.df[self.df[target_col].notna()].copy()
        
        # Target'ı encode et
        valid_data[target_col] = valid_data[target_col].map({'yes': 1, 'no': 0, 1: 1, 0: 0})
        
        # Features ve target'ı ayır (sadece numeric kolonlar)
        exclude_cols = [target_col, 'ID']
        feature_cols = [col for col in valid_data.columns if col not in exclude_cols]
        
        # Select only numeric columns
        X_all = valid_data[feature_cols]
        numeric_cols = X_all.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        self.X = X_all[numeric_cols].copy()
        self.y = valid_data[target_col].copy()
        
        # Eksik değerleri median ile doldur
        self.X = self.X.fillna(self.X.median())
        
        # Hala NaN varsa 0 ile doldur
        self.X = self.X.fillna(0)
        
        # Inf check
        self.X = self.X.replace([np.inf, -np.inf], 0)
        
        print(f"Hazır veri: X={self.X.shape}, y={self.y.shape}")
        print(f"Feature sayısı: {self.X.shape[1]}")
        
        return self.X, self.y
    
    def engineer_features(self):
        """Ana pipeline'daki aynı feature engineering - simplified"""
        df = self.df.copy()
        
        # Standardize column names
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
        
        print("  - Aging-related features...")
        if 'Biological_Age' in df.columns:
            df['Biological_Age_Squared'] = df['Biological_Age'] ** 2
            df['Biological_Age_Cubed'] = df['Biological_Age'] ** 3
            df['Log_Biological_Age'] = np.log1p(df['Biological_Age'])
            df['BioAge_60_Plus'] = (df['Biological_Age'] >= 60).astype(int)
            df['BioAge_70_Plus'] = (df['Biological_Age'] >= 70).astype(int)
            df['BioAge_Over_67'] = (df['Biological_Age'] > 66.7).astype(int)
        
        print("  - BMI-related features...")
        if 'BMI' in df.columns:
            df['BMI_Squared'] = df['BMI'] ** 2
            df['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3])
            df['BMI_Category'] = df['BMI_Category'].astype(float)
            df['BMI_Obese'] = (df['BMI'] >= 30).astype(int)
            df['Log_BMI'] = np.log1p(df['BMI'])
        
        print("  - Comorbidity features...")
        comorbidity_cols = []
        possible_comorbidities = ['Hypertension', 'Dyslipidemia', 'Diabetes', 'Cancer', 'Heart_Disease']
        
        for col in possible_comorbidities:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                comorbidity_cols.append(col)
        
        if comorbidity_cols:
            df['Comorbidity_Count'] = df[comorbidity_cols].sum(axis=1)
            df['Has_Multiple_Comorbidities'] = (df['Comorbidity_Count'] >= 2).astype(int)
            df['Comorbidity_Risk_Score'] = df['Comorbidity_Count'] / len(comorbidity_cols)
        
        print("  - Interaction features...")
        if 'BMI' in df.columns and 'Biological_Age' in df.columns:
            df['BMI_x_BioAge'] = df['BMI'] * df['Biological_Age']
        
        if 'Biological_Age' in df.columns and 'Comorbidity_Count' in df.columns:
            df['Age_x_Comorbidity'] = df['Biological_Age'] * df['Comorbidity_Count']
        
        if 'BMI' in df.columns and 'Comorbidity_Count' in df.columns:
            df['BMI_x_Comorbidity'] = df['BMI'] * df['Comorbidity_Count']
        
        # Smoking ve Drinking
        if 'smoke' in df.columns:
            df['Smoking_Encoded'] = df['smoke']
        if 'drink' in df.columns:
            df['Drinking_Encoded'] = df['drink']
        
        if 'Smoking_Encoded' in df.columns and 'Drinking_Encoded' in df.columns and 'BMI_Obese' in df.columns:
            df['Smoke_x_Drink'] = df['Smoking_Encoded'] * df['Drinking_Encoded']
            df['Lifestyle_Risk'] = df['Smoking_Encoded'] + df['Drinking_Encoded'] + df['BMI_Obese']
            df['High_Risk_Lifestyle'] = (df['Lifestyle_Risk'] >= 2).astype(int)
        
        print("  - Metabolic features...")
        if 'BMI_Obese' in df.columns and 'Diabetes' in df.columns and 'Hypertension' in df.columns:
            metabolic_risk = pd.Series(0, index=df.index)
            metabolic_risk += df['BMI_Obese'] * 2
            metabolic_risk += df['Diabetes'] * 3
            metabolic_risk += df['Hypertension'] * 2
            df['Metabolic_Risk_Score'] = metabolic_risk
            df['High_Metabolic_Risk'] = (df['Metabolic_Risk_Score'] >= 4).astype(int)
        
        # Position_Knees encoding
        if 'position_knees' in df.columns:
            df['Position_Knees_Encoded'] = pd.Categorical(df['position_knees']).codes
        
        print(f"Feature engineering tamamlandı. Toplam kolon: {df.shape[1]}")
        self.df = df
    
    def compute_anova_scores(self):
        """ANOVA F-test skorlarını hesapla"""
        print("\nANOVA F-test hesaplanıyor...")
        F_scores, p_values = f_classif(self.X, self.y)
        
        self.feature_scores = pd.DataFrame({
            'Feature': self.X.columns,
            'F_Score': F_scores,
            'p_value': p_values
        }).sort_values('F_Score', ascending=False)
        
        print(f"Top 10 features:")
        print(self.feature_scores.head(10))
        
        # CSV olarak kaydet
        self.feature_scores.to_csv('feature_scores_detailed.csv', index=False)
        
        return self.feature_scores
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test, experiment_name):
        """Model eğit ve değerlendir"""
        print(f"\n{experiment_name} - Model eğitiliyor...")
        print(f"Feature sayısı: {X_train.shape[1]}")
        
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
            'Feature Count': X_train.shape[1],
            'ROC AUC': f"{roc_auc:.4f}",
            'F1 Score': f"{f1:.4f}",
            'Accuracy': f"{accuracy:.4f}"
        }
        
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return metrics, rf
    
    def experiment_1_without_position_knees(self):
        """Deney 1: Position_Knees_Encoded olmadan"""
        print("\n" + "="*80)
        print("DENEY 1: Position_Knees_Encoded Çıkarılarak Test")
        print("="*80)
        
        # Position_Knees_Encoded'ı çıkar
        X_filtered = self.X.drop(columns=['Position_Knees_Encoded'], errors='ignore')
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Model eğit
        metrics, rf = self.train_and_evaluate(
            X_train, X_test, y_train, y_test,
            "Position_Knees_Encoded Olmadan"
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_filtered.columns,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop 10 Important Features:")
        print(feature_importance.head(10))
        
        # Sonuçları kaydet
        additional_info = f"Çıkarılan feature: Position_Knees_Encoded\nKalan feature sayısı: {X_filtered.shape[1]}"
        self.log_result("Deney 1: Position_Knees_Encoded Çıkarılarak", metrics, additional_info)
        
        return metrics
    
    def experiment_2_p_less_than_005(self):
        """Deney 2: p < 0.05 olan features"""
        print("\n" + "="*80)
        print("DENEY 2: p < 0.05 olan Features ile Test")
        print("="*80)
        
        # p < 0.05 olanları seç
        significant_features = self.feature_scores[self.feature_scores['p_value'] < 0.05]['Feature'].tolist()
        print(f"p < 0.05 olan feature sayısı: {len(significant_features)}")
        
        X_filtered = self.X[significant_features]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Model eğit
        metrics, rf = self.train_and_evaluate(
            X_train, X_test, y_train, y_test,
            "p < 0.05 Features"
        )
        
        # Sonuçları kaydet
        additional_info = f"Seçilen features: p < 0.05\nFeature sayısı: {len(significant_features)}"
        self.log_result("Deney 2: p < 0.05 Features", metrics, additional_info)
        
        return metrics
    
    def experiment_3_p_between_001_and_005(self):
        """Deney 3: 0.01 < p < 0.05 aralığındaki features"""
        print("\n" + "="*80)
        print("DENEY 3: 0.01 < p < 0.05 Aralığındaki Features ile Test")
        print("="*80)
        
        # 0.01 < p < 0.05 olanları seç
        moderate_features = self.feature_scores[
            (self.feature_scores['p_value'] > 0.01) & 
            (self.feature_scores['p_value'] < 0.05)
        ]['Feature'].tolist()
        
        print(f"0.01 < p < 0.05 aralığındaki feature sayısı: {len(moderate_features)}")
        
        if len(moderate_features) == 0:
            print("Bu aralıkta feature bulunamadı!")
            return None
        
        X_filtered = self.X[moderate_features]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Model eğit
        metrics, rf = self.train_and_evaluate(
            X_train, X_test, y_train, y_test,
            "0.01 < p < 0.05 Features"
        )
        
        # Sonuçları kaydet
        additional_info = f"Seçilen features: 0.01 < p < 0.05\nFeature sayısı: {len(moderate_features)}\nFeatures: {', '.join(moderate_features)}"
        self.log_result("Deney 3: 0.01 < p < 0.05 Features", metrics, additional_info)
        
        return metrics
    
    def experiment_4_hierarchical_clustering(self):
        """Deney 4: Hierarchical clustering of features"""
        print("\n" + "="*80)
        print("DENEY 4: Hierarchical Clustering of Features")
        print("="*80)
        
        # Correlation matrix hesapla
        print("Korelasyon matrisi hesaplanıyor...")
        corr_matrix = self.X.corr().abs()
        
        # Distance matrix (1 - correlation) - symmetric olarak
        distance_matrix = (1 - corr_matrix).values.copy()
        
        # Replace any NaN or Inf with 1 (max distance)
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=1.0)
        
        # Ensure symmetry
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # Convert to condensed form
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Hierarchical clustering
        print("Hierarchical clustering yapılıyor...")
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Dendrogram çizimi
        plt.figure(figsize=(20, 10))
        
        # Ana dendrogram
        plt.subplot(2, 1, 1)
        dendrogram(
            linkage_matrix,
            labels=self.X.columns,
            leaf_rotation=90,
            leaf_font_size=8
        )
        plt.title('Hierarchical Clustering Dendrogram - All Features', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Distance (1 - |correlation|)', fontsize=12)
        plt.tight_layout()
        
        # Cluster sayılarına göre renklendirme
        plt.subplot(2, 1, 2)
        dendrogram(
            linkage_matrix,
            labels=self.X.columns,
            leaf_rotation=90,
            leaf_font_size=8,
            color_threshold=0.7 * max(linkage_matrix[:, 2])
        )
        plt.title('Hierarchical Clustering with Color Threshold', fontsize=14, fontweight='bold')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.axhline(y=0.7 * max(linkage_matrix[:, 2]), c='red', linestyle='--', label='Threshold')
        plt.legend()
        plt.tight_layout()
        
        plt.savefig('hierarchical_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
        print("Dendrogram kaydedildi: hierarchical_clustering_dendrogram.png")
        plt.close()
        
        # Cluster'lara ayır (örneğin 5 cluster)
        n_clusters = 5
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        cluster_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Cluster': clusters
        }).sort_values('Cluster')
        
        cluster_df.to_csv('feature_clusters.csv', index=False)
        print(f"\nFeature cluster'ları kaydedildi: feature_clusters.csv")
        
        # Her cluster'dan bir representative feature seç
        print(f"\nCluster analizi ({n_clusters} cluster):")
        representative_features = []
        
        for cluster_id in range(1, n_clusters + 1):
            cluster_features = cluster_df[cluster_df['Cluster'] == cluster_id]['Feature'].tolist()
            print(f"\nCluster {cluster_id} ({len(cluster_features)} features):")
            print(f"  {', '.join(cluster_features[:5])}{'...' if len(cluster_features) > 5 else ''}")
            
            # En yüksek F-score'a sahip feature'ı seç
            cluster_scores = self.feature_scores[self.feature_scores['Feature'].isin(cluster_features)]
            if not cluster_scores.empty:
                best_feature = cluster_scores.iloc[0]['Feature']
                representative_features.append(best_feature)
                print(f"  Representative: {best_feature}")
        
        # Representative features ile model eğit
        print(f"\n\nRepresentative features ile model eğitiliyor ({len(representative_features)} features)...")
        X_clustered = self.X[representative_features]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_clustered, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        metrics, rf = self.train_and_evaluate(
            X_train, X_test, y_train, y_test,
            "Hierarchical Clustering Representatives"
        )
        
        # Heatmap çiz (representative features)
        plt.figure(figsize=(12, 10))
        corr_subset = self.X[representative_features].corr()
        sns.heatmap(corr_subset, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Heatmap - Representative Features (n={len(representative_features)})', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('representative_features_heatmap.png', dpi=300, bbox_inches='tight')
        print("Representative features heatmap kaydedildi: representative_features_heatmap.png")
        plt.close()
        
        # Sonuçları kaydet
        additional_info = (f"Hierarchical Clustering (ward method)\n"
                          f"Cluster sayısı: {n_clusters}\n"
                          f"Representative features: {len(representative_features)}\n"
                          f"Features: {', '.join(representative_features)}")
        self.log_result("Deney 4: Hierarchical Clustering", metrics, additional_info)
        
        return metrics, cluster_df, representative_features
    
    def run_all_experiments(self):
        """Tüm deneyleri çalıştır"""
        print("\n" + "="*80)
        print("TÜM DENEYLER BAŞLATILIYOR")
        print("="*80)
        
        # Veriyi hazırla
        self.load_and_prepare_data()
        
        # ANOVA skorlarını hesapla
        self.compute_anova_scores()
        
        # Deneyleri çalıştır
        results = {}
        
        results['exp1'] = self.experiment_1_without_position_knees()
        results['exp2'] = self.experiment_2_p_less_than_005()
        results['exp3'] = self.experiment_3_p_between_001_and_005()
        results['exp4'], cluster_df, rep_features = self.experiment_4_hierarchical_clustering()
        
        # Özet rapor
        print("\n\n" + "="*80)
        print("DENEYLER TAMAMLANDI - ÖZET RAPOR")
        print("="*80)
        
        summary = pd.DataFrame({
            'Deney': [
                'Exp1: Position_Knees_Encoded Yok',
                'Exp2: p < 0.05',
                'Exp3: 0.01 < p < 0.05',
                'Exp4: Hierarchical Clustering'
            ],
            'ROC_AUC': [
                results['exp1']['ROC AUC'],
                results['exp2']['ROC AUC'],
                results['exp3']['ROC AUC'] if results['exp3'] else 'N/A',
                results['exp4']['ROC AUC']
            ],
            'F1_Score': [
                results['exp1']['F1 Score'],
                results['exp2']['F1 Score'],
                results['exp3']['F1 Score'] if results['exp3'] else 'N/A',
                results['exp4']['F1 Score']
            ],
            'Feature_Count': [
                results['exp1']['Feature Count'],
                results['exp2']['Feature Count'],
                results['exp3']['Feature Count'] if results['exp3'] else 0,
                results['exp4']['Feature Count']
            ]
        })
        
        print(summary.to_string(index=False))
        
        summary.to_csv('experiments_summary.csv', index=False)
        
        # Final log
        with open(self.results_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write("ÖZET RAPOR\n")
            f.write("="*80 + "\n")
            f.write(summary.to_string(index=False))
            f.write("\n\n")
            f.write(f"Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
        
        print(f"\n\nTüm sonuçlar kaydedildi:")
        print(f"  - {self.results_file}")
        print(f"  - feature_scores_detailed.csv")
        print(f"  - feature_clusters.csv")
        print(f"  - experiments_summary.csv")
        print(f"  - hierarchical_clustering_dendrogram.png")
        print(f"  - representative_features_heatmap.png")
        
        return results, summary


def main():
    """Ana çalıştırma fonksiyonu"""
    # Data path - UPDATED to use imputed dataset (Mean/Mode Imputation, 0% data loss)
    data_path = r"c:\Users\eftel\OneDrive\Masaüstü\bioinformatics-data\step0_imputed_data\data_imputed_mean.csv"
    
    print("="*80)
    print("⚠️  IMPORTANT: Using IMPUTED Dataset (Mean/Mode Imputation)")
    print("="*80)
    print(f"Dataset: data_imputed_mean.csv")
    print(f"Rows: 9,505 (vs 2,742 with listwise deletion)")
    print(f"Data Recovery: +6,763 rows (+246.6%)")
    print("="*80 + "\n")
    
    # Experiment pipeline oluştur
    pipeline = FeatureAnalysisExperiments(data_path)
    
    # Tüm deneyleri çalıştır
    results, summary = pipeline.run_all_experiments()
    
    print("\n✅ Tüm deneyler başarıyla tamamlandı!")


if __name__ == "__main__":
    main()
