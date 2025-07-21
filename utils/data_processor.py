"""
Data Processing Utilities
Functions for data cleaning, transformation, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Class for data processing and transformation"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
        self.feature_names = []
        
    def clean_data(self, df):
        """Clean and preprocess the dataset"""
        df_clean = df.copy()
        
        # Remove duplicates
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_count - len(df_clean)
        
        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        
        # For this dataset, we'll fill missing values if any
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        missing_after = df_clean.isnull().sum().sum()
        
        # Handle outliers using IQR method for salary
        if 'salary_in_usd' in df_clean.columns:
            Q1 = df_clean['salary_in_usd'].quantile(0.25)
            Q3 = df_clean['salary_in_usd'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            df_clean['salary_in_usd'] = df_clean['salary_in_usd'].clip(lower=lower_bound, upper=upper_bound)
        
        # Create cleaning report
        cleaning_report = {
            'duplicates_removed': duplicates_removed,
            'missing_values_before': missing_before,
            'missing_values_after': missing_after,
            'outliers_capped': len(df[(df['salary_in_usd'] < lower_bound) | (df['salary_in_usd'] > upper_bound)]) if 'salary_in_usd' in df.columns else 0,
            'final_records': len(df_clean)
        }
        
        return df_clean, cleaning_report
    
    def encode_features(self, df, target_column='salary_in_usd'):
        """Encode categorical features for machine learning"""
        df_encoded = df.copy()
        
        # Separate features and target
        if target_column in df_encoded.columns:
            y = df_encoded[target_column]
            X = df_encoded.drop(columns=[target_column])
        else:
            y = None
            X = df_encoded
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in ['experience_level', 'company_size']:
                # Ordinal encoding for these features
                if col == 'experience_level':
                    mapping = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
                elif col == 'company_size':
                    mapping = {'S': 0, 'M': 1, 'L': 2}
                
                df_encoded[f'{col}_encoded'] = X[col].map(mapping)
                self.feature_names.append(f'{col}_encoded')
                
            else:
                # Label encoding for other categorical features
                le = LabelEncoder()
                df_encoded[f'{col}_encoded'] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                self.feature_names.append(f'{col}_encoded')
        
        # Keep numerical features
        for col in numerical_cols:
            if col != target_column:
                self.feature_names.append(col)
        
        # Create feature matrix
        X_processed = df_encoded[self.feature_names]
        
        return X_processed, y, self.feature_names
    
    def create_additional_features(self, df):
        """Create additional features for analysis"""
        df_features = df.copy()
        
        # Salary categories
        df_features['salary_category'] = pd.cut(
            df_features['salary_in_usd'],
            bins=[0, 75000, 125000, 200000, float('inf')],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        # Experience level full names
        exp_mapping = {
            'EN': 'Entry Level',
            'MI': 'Mid Level', 
            'SE': 'Senior Level',
            'EX': 'Executive Level'
        }
        df_features['experience_full'] = df_features['experience_level'].map(exp_mapping)
        
        # Company size full names
        size_mapping = {
            'S': 'Small',
            'M': 'Medium',
            'L': 'Large'
        }
        df_features['company_size_full'] = df_features['company_size'].map(size_mapping)
        
        # Employment type full names
        emp_mapping = {
            'FT': 'Full Time',
            'PT': 'Part Time',
            'CT': 'Contract',
            'FL': 'Freelance'
        }
        df_features['employment_full'] = df_features['employment_type'].map(emp_mapping)
        
        # Remote work categories
        df_features['remote_category'] = df_features['remote_ratio'].apply(
            lambda x: 'Fully Remote' if x == 100 
                     else 'Hybrid' if x == 50 
                     else 'On-site'
        )
        
        # Salary per experience year (approximation)
        exp_years_mapping = {'EN': 1, 'MI': 3, 'SE': 7, 'EX': 12}
        df_features['approx_exp_years'] = df_features['experience_level'].map(exp_years_mapping)
        df_features['salary_per_exp_year'] = df_features['salary_in_usd'] / df_features['approx_exp_years']
        
        return df_features
    
    def prepare_for_modeling(self, df, test_size=0.2, random_state=42):
        """Prepare data for machine learning modeling"""
        # Clean data
        df_clean, cleaning_report = self.clean_data(df)
        
        # Create additional features
        df_features = self.create_additional_features(df_clean)
        
        # Encode features
        X, y, feature_names = self.encode_features(df_features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'cleaning_report': cleaning_report,
            'scaler': self.scaler,
            'encoders': self.label_encoders
        }
    
    def transform_new_data(self, df):
        """Transform new data using fitted encoders and scaler"""
        df_transformed = df.copy()
        
        # Apply encodings
        for col, encoder in self.label_encoders.items():
            if col in df_transformed.columns:
                df_transformed[f'{col}_encoded'] = encoder.transform(df_transformed[col].astype(str))
        
        # Apply manual encodings
        if 'experience_level' in df_transformed.columns:
            mapping = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
            df_transformed['experience_level_encoded'] = df_transformed['experience_level'].map(mapping)
        
        if 'company_size' in df_transformed.columns:
            mapping = {'S': 0, 'M': 1, 'L': 2}
            df_transformed['company_size_encoded'] = df_transformed['company_size'].map(mapping)
        
        # Select features and scale
        X_new = df_transformed[self.feature_names]
        X_new_scaled = self.scaler.transform(X_new)
        
        return pd.DataFrame(X_new_scaled, columns=self.feature_names)

# Convenience functions
def clean_data(df):
    """Quick function to clean data"""
    processor = DataProcessor()
    return processor.clean_data(df)

def encode_features(df, target_column='salary_in_usd'):
    """Quick function to encode features"""
    processor = DataProcessor()
    return processor.encode_features(df, target_column)

def create_salary_insights(df):
    """Create salary-related insights"""
    insights = {}
    
    # Basic statistics
    insights['basic_stats'] = {
        'mean': df['salary_in_usd'].mean(),
        'median': df['salary_in_usd'].median(),
        'std': df['salary_in_usd'].std(),
        'min': df['salary_in_usd'].min(),
        'max': df['salary_in_usd'].max()
    }
    
    # Experience level analysis
    insights['experience_analysis'] = df.groupby('experience_level')['salary_in_usd'].agg([
        'mean', 'median', 'count'
    ]).to_dict()
    
    # Company size analysis
    insights['company_size_analysis'] = df.groupby('company_size')['salary_in_usd'].agg([
        'mean', 'median', 'count'
    ]).to_dict()
    
    # Remote work analysis
    insights['remote_analysis'] = df.groupby('remote_ratio')['salary_in_usd'].agg([
        'mean', 'median', 'count'
    ]).to_dict()
    
    # Geographic analysis
    insights['geographic_analysis'] = df.groupby('company_location')['salary_in_usd'].agg([
        'mean', 'median', 'count'
    ]).sort_values('mean', ascending=False).head(10).to_dict()
    
    # Job title analysis
    insights['job_title_analysis'] = df.groupby('job_title')['salary_in_usd'].agg([
        'mean', 'median', 'count'
    ]).sort_values('mean', ascending=False).head(10).to_dict()
    
    return insights

def detect_outliers(df, column='salary_in_usd', method='iqr'):
    """Detect outliers in specified column"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        return {
            'outliers': outliers,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > 3]
        
        return {
            'outliers': outliers,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(df) * 100,
            'threshold': 3
        }

def create_correlation_analysis(df):
    """Create correlation analysis for numeric columns"""
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlation matrix
    correlation_matrix = df[numeric_cols].corr()
    
    # Find strong correlations (> 0.5 or < -0.5)
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > 0.5:
                strong_correlations.append({
                    'feature_1': correlation_matrix.columns[i],
                    'feature_2': correlation_matrix.columns[j],
                    'correlation': corr_value
                })
    
    return {
        'correlation_matrix': correlation_matrix,
        'strong_correlations': strong_correlations
    }