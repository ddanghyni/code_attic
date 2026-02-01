# Data Analysis Module

데이터 분석 관련 Python 코드 모음

## Contents

- `data_preprocessing.py`: 데이터 전처리 및 분석 함수

## Features

### Data Exploration
- Basic data information
- Missing value analysis
- Data type inspection
- Memory usage analysis

### Data Preprocessing
- Missing value handling (mean, median, mode, drop)
- Data normalization (StandardScaler, MinMaxScaler)
- Outlier detection (IQR method)
- Outlier removal

### Feature Engineering
- Feature interactions
- Polynomial features
- Feature transformations
- Feature selection

### Exploratory Data Analysis
- Correlation analysis
- Distribution analysis
- Statistical summaries
- Data visualization

## Usage

```python
from data_analysis.data_preprocessing import handle_missing_values, detect_outliers_iqr
import pandas as pd

# Handle missing values
df = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, None, 7, 8]})
cleaned_df = handle_missing_values(df, strategy='mean')

# Detect outliers
outliers, lower, upper = detect_outliers_iqr(df, 'A')
print(f"Outliers: {outliers}")
```

## Requirements

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
