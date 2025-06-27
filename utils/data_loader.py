"""
Data Loader Utilities
Functions to load dataset and machine learning model
"""

import pandas as pd
import joblib
import os
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Class untuk loading dan managing data"""
    
    def __init__(self, data_path: str = 'data/', model_path: str = 'data/model.pkl'):
        self.data_path = data_path
        self.model_path = model_path
        self._df = None
        self._model = None
        self._original_df = None
        self._processed_df = None
    
    def load_dataset(self, filename: str = 'ds_salaries.csv') -> pd.DataFrame:
        """Load dataset dari file CSV"""
        try:
            file_path = os.path.join(self.data_path, filename)
            self._df = pd.read_csv(file_path)
            self._original_df = self._df.copy()
            
            print(f"✅ Dataset loaded successfully: {len(self._df)} records")
            print(f"📊 Columns: {list(self._df.columns)}")
            return self._df
            
        except FileNotFoundError:
            print(f"❌ File {filename} not found in {self.data_path}")
            return self._create_sample_data()
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return self._create_sample_data()
    
    def load_model(self) -> Optional[object]:
        """Load trained machine learning model"""
        try:
            self._model = joblib.load(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            return self._model
        except FileNotFoundError:
            print(f"❌ Model file not found: {self.model_path}")
            return None
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return None
    
    def get_dataset_info(self) -> dict:
        """Get basic information about the dataset"""
        if self._df is None:
            self.load_dataset()
        
        return {
            'total_records': len(self._df),
            'total_columns': len(self._df.columns),
            'columns': list(self._df.columns),
            'data_types': self._df.dtypes.to_dict(),
            'missing_values': self._df.isnull().sum().to_dict(),
            'memory_usage': self._df.memory_usage(deep=True).sum(),
            'shape': self._df.shape
        }
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics"""
        if self._df is None:
            self.load_dataset()
        
        numeric_cols = self._df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self._df.select_dtypes(include=['object']).columns
        
        return {
            'numeric_summary': self._df[numeric_cols].describe().to_dict(),
            'categorical_summary': {col: self._df[col].value_counts().head().to_dict() 
                                  for col in categorical_cols},
            'numeric_columns': list(numeric_cols),
            'categorical_columns': list(categorical_cols)
        }
    
    def get_data_quality_report(self) -> dict:
        """Generate data quality report"""
        if self._df is None:
            self.load_dataset()
        
        total_records = len(self._df)
        
        quality_report = {
            'completeness': {
                'total_records': total_records,
                'complete_records': total_records - self._df.isnull().any(axis=1).sum(),
                'missing_by_column': self._df.isnull().sum().to_dict(),
                'missing_percentage': (self._df.isnull().sum() / total_records * 100).to_dict()
            },
            'duplicates': {
                'total_duplicates': self._df.duplicated().sum(),
                'unique_records': len(self._df.drop_duplicates())
            },
            'data_types': self._df.dtypes.to_dict()
        }
        
        return quality_report
    
    def filter_data(self, filters: dict) -> pd.DataFrame:
        """Filter dataset based on given criteria"""
        if self._df is None:
            self.load_dataset()
        
        filtered_df = self._df.copy()
        
        for column, criteria in filters.items():
            if column in filtered_df.columns:
                if isinstance(criteria, dict):
                    if 'min' in criteria and 'max' in criteria:
                        # Range filter
                        filtered_df = filtered_df[
                            (filtered_df[column] >= criteria['min']) & 
                            (filtered_df[column] <= criteria['max'])
                        ]
                    elif 'values' in criteria:
                        # Categorical filter
                        filtered_df = filtered_df[filtered_df[column].isin(criteria['values'])]
                elif isinstance(criteria, list):
                    # List of values
                    filtered_df = filtered_df[filtered_df[column].isin(criteria)]
        
        return filtered_df
    
    def get_unique_values(self, column: str) -> list:
        """Get unique values for a specific column"""
        if self._df is None:
            self.load_dataset()
        
        if column in self._df.columns:
            return sorted(self._df[column].unique().tolist())
        return []
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample data if original file not found"""
        print("📝 Creating sample data...")
        
        sample_data = {
            'work_year': [2023] * 100,
            'experience_level': ['SE', 'MI', 'EN', 'EX'] * 25,
            'employment_type': ['FT', 'PT', 'CT', 'FL'] * 25,
            'job_title': ['Data Scientist', 'Data Engineer', 'Data Analyst', 'ML Engineer'] * 25,
            'salary': [100000, 80000, 60000, 120000] * 25,
            'salary_currency': ['USD'] * 100,
            'salary_in_usd': [100000, 80000, 60000, 120000] * 25,
            'employee_residence': ['US', 'UK', 'CA', 'DE'] * 25,
            'remote_ratio': [0, 50, 100] * 33 + [0],
            'company_location': ['US', 'UK', 'CA', 'DE'] * 25,
            'company_size': ['M', 'L', 'S'] * 33 + ['M']
        }
        
        return pd.DataFrame(sample_data)
    
    @property
    def df(self) -> pd.DataFrame:
        """Get current dataframe"""
        if self._df is None:
            self.load_dataset()
        return self._df
    
    @property
    def model(self):
        """Get loaded model"""
        if self._model is None:
            self.load_model()
        return self._model
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = 'processed_data.pkl'):
        """Save processed dataframe"""
        try:
            file_path = os.path.join(self.data_path, filename)
            df.to_pickle(file_path)
            print(f"✅ Processed data saved: {file_path}")
        except Exception as e:
            print(f"❌ Error saving processed data: {str(e)}")

# Global instance
data_loader = DataLoader()

# Convenience functions
def load_data() -> pd.DataFrame:
    """Quick function to load data"""
    return data_loader.load_dataset()

def load_model():
    """Quick function to load model"""
    return data_loader.load_model()

def get_data_info() -> dict:
    """Quick function to get data info"""
    return data_loader.get_dataset_info()

if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()
    df = loader.load_dataset()
    model = loader.load_model()
    info = loader.get_dataset_info()
    
    print("\n📋 Dataset Info:")
    print(f"Records: {info['total_records']}")
    print(f"Columns: {info['total_columns']}")
    print(f"Shape: {info['shape']}")
    
    quality = loader.get_data_quality_report()
    print(f"\n📊 Data Quality:")
    print(f"Complete records: {quality['completeness']['complete_records']}")
    print(f"Duplicates: {quality['duplicates']['total_duplicates']}")