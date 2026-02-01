"""
Data Analysis Module
데이터 분석 관련 Python 코드

This module contains data analysis examples including:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Feature engineering
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_and_explore_data(data):
    """
    Basic data exploration
    
    Args:
        data: pandas DataFrame
    
    Returns:
        dict: Basic information about the dataset
    """
    return {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.to_dict(),
        'missing_values': data.isnull().sum().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum()
    }


def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in dataset
    
    Args:
        data: pandas DataFrame
        strategy: 'mean', 'median', 'mode', or 'drop'
    
    Returns:
        pandas DataFrame: Data with missing values handled
    """
    data_copy = data.copy()
    
    if strategy == 'drop':
        return data_copy.dropna()
    
    for col in data_copy.columns:
        if data_copy[col].isnull().sum() > 0:
            if data_copy[col].dtype in ['float64', 'int64']:
                if strategy == 'mean':
                    data_copy[col] = data_copy[col].fillna(data_copy[col].mean())
                elif strategy == 'median':
                    data_copy[col] = data_copy[col].fillna(data_copy[col].median())
            else:
                data_copy[col] = data_copy[col].fillna(data_copy[col].mode()[0])
    
    return data_copy


def normalize_data(data, method='standard'):
    """
    Normalize numerical data
    
    Args:
        data: pandas DataFrame or numpy array
        method: 'standard' or 'minmax'
    
    Returns:
        Normalized data
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Method must be 'standard' or 'minmax'")
    
    return scaler.fit_transform(data)


def detect_outliers_iqr(data, column):
    """
    Detect outliers using IQR method
    
    Args:
        data: pandas DataFrame
        column: Column name to check for outliers
    
    Returns:
        tuple: (outlier indices, lower bound, upper bound)
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)].index
    
    return outliers, lower_bound, upper_bound


def correlation_matrix_analysis(data):
    """
    Calculate and visualize correlation matrix
    
    Args:
        data: pandas DataFrame with numerical columns
    
    Returns:
        pandas DataFrame: Correlation matrix
    """
    correlation_matrix = data.corr()
    return correlation_matrix


def create_feature_interactions(data, col1, col2):
    """
    Create interaction features between two columns
    
    Args:
        data: pandas DataFrame
        col1: First column name
        col2: Second column name
    
    Returns:
        pandas DataFrame: Original data with new interaction features
    """
    data_copy = data.copy()
    
    # Multiplication interaction
    data_copy[f'{col1}_x_{col2}'] = data_copy[col1] * data_copy[col2]
    
    # Addition interaction
    data_copy[f'{col1}_plus_{col2}'] = data_copy[col1] + data_copy[col2]
    
    # Ratio interaction (avoid division by zero)
    if (data_copy[col2] != 0).all():
        data_copy[f'{col1}_div_{col2}'] = data_copy[col1] / data_copy[col2]
    
    return data_copy


if __name__ == "__main__":
    # Example usage
    print("=== Data Analysis Example ===")
    
    # Create sample dataset
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, 1000),
        'feature2': np.random.normal(50, 10, 1000),
        'feature3': np.random.exponential(2, 1000),
        'category': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Add some missing values
    sample_data.loc[np.random.choice(sample_data.index, 50), 'feature1'] = np.nan
    
    print("\n=== Data Exploration ===")
    exploration = load_and_explore_data(sample_data)
    print(f"Shape: {exploration['shape']}")
    print(f"Columns: {exploration['columns']}")
    print(f"Missing values: {exploration['missing_values']}")
    
    print("\n=== Handle Missing Values ===")
    cleaned_data = handle_missing_values(sample_data, strategy='mean')
    print(f"Missing values after cleaning: {cleaned_data.isnull().sum().sum()}")
    
    print("\n=== Outlier Detection ===")
    outliers, lower, upper = detect_outliers_iqr(cleaned_data, 'feature1')
    print(f"Number of outliers: {len(outliers)}")
    print(f"Bounds: [{lower:.2f}, {upper:.2f}]")
    
    print("\n=== Correlation Analysis ===")
    numerical_data = cleaned_data[['feature1', 'feature2', 'feature3']]
    corr_matrix = correlation_matrix_analysis(numerical_data)
    print("Correlation Matrix:")
    print(corr_matrix)
    
    print("\n=== Feature Engineering ===")
    enhanced_data = create_feature_interactions(cleaned_data, 'feature1', 'feature2')
    print(f"New features created: {[col for col in enhanced_data.columns if col not in sample_data.columns]}")
