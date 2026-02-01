# Statistics Module

통계 분석 관련 Python 코드 모음

## Contents

- `statistical_analysis.py`: 통계 분석 주요 함수 구현

## Features

### Descriptive Statistics
- Mean, median, standard deviation
- Quartiles and percentiles
- Variance analysis

### Hypothesis Testing
- Independent t-test
- Paired t-test
- Statistical significance testing

### Correlation Analysis
- Pearson correlation
- Spearman correlation
- Correlation significance testing

### Distribution Testing
- Normality testing (Shapiro-Wilk test)
- Distribution fitting
- Q-Q plots

## Usage

```python
from stat_analysis.statistical_analysis import descriptive_statistics, t_test

# Descriptive statistics
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
stats = descriptive_statistics(data)
print(stats)

# T-test
group1 = [100, 102, 98, 105, 103]
group2 = [110, 108, 112, 115, 109]
result = t_test(group1, group2)
print(result)
```

## Requirements

- numpy
- scipy
- pandas
