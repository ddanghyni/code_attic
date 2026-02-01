# Anomaly Detection Module

이상 탐지(Anomaly Detection) 관련 Python 코드 모음

## Contents

- `anomaly_methods.py`: 다양한 이상 탐지 알고리즘 구현

## Features

### Statistical Methods
- **Z-Score Method**: 통계적 표준 점수 기반
- **IQR Method**: 사분위수 범위 기반

### Machine Learning Methods
- **Isolation Forest**: 앙상블 기반 이상 탐지
- **Local Outlier Factor (LOF)**: 지역 밀도 기반
- **One-Class SVM**: 서포트 벡터 머신 기반

### Deep Learning Methods
- **Autoencoder**: 재구성 오차 기반 이상 탐지
- Custom neural network architectures

### Evaluation
- Precision, Recall, F1-score
- Confusion matrix
- ROC curves
- Anomaly scoring

## Usage

```python
from anomaly_detection.anomaly_methods import (
    zscore_anomaly_detection,
    isolation_forest_detection,
    AnomalyAutoencoder
)
import numpy as np

# Statistical method
data = np.random.normal(0, 1, 1000)
anomalies, z_scores = zscore_anomaly_detection(data, threshold=3)

# Machine learning method
X = np.random.randn(1000, 10)
predictions = isolation_forest_detection(X, contamination=0.1)

# Deep learning method
import torch
model = AnomalyAutoencoder(input_dim=10, encoding_dim=3)
# Train model and detect anomalies
```

## Methods Comparison

| Method | Type | Use Case | Pros | Cons |
|--------|------|----------|------|------|
| Z-Score | Statistical | Univariate | Simple, fast | Assumes normal distribution |
| IQR | Statistical | Univariate | Robust to outliers | Less sensitive |
| Isolation Forest | ML | Multivariate | Fast, scalable | May miss local anomalies |
| LOF | ML | Multivariate | Good for local anomalies | Computationally expensive |
| One-Class SVM | ML | Multivariate | Powerful, flexible | Sensitive to parameters |
| Autoencoder | DL | Multivariate | Learns complex patterns | Requires training data |

## Requirements

- numpy
- pandas
- scikit-learn
- torch
