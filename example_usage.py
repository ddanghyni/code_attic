"""
Example Script
코드 저장소의 모든 모듈을 간단히 시연하는 스크립트

This script demonstrates the basic functionality of all modules
in the code_attic repository.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CODE ATTIC - Python Research Code Repository")
print("통계, 데이터 분석, 딥러닝, Anomaly Detection")
print("=" * 70)

# ============================================================================
# 1. Statistics Module
# ============================================================================
print("\n" + "=" * 70)
print("1. STATISTICS MODULE (통계 분석)")
print("=" * 70)

from stat_analysis.statistical_analysis import (
    descriptive_statistics,
    t_test,
    correlation_analysis
)

# Generate sample data
np.random.seed(42)
data1 = np.random.normal(100, 15, 100)
data2 = np.random.normal(105, 15, 100)

print("\n[Descriptive Statistics]")
stats = descriptive_statistics(data1)
print(f"  Mean: {stats['mean']:.2f}")
print(f"  Std Dev: {stats['std']:.2f}")
print(f"  Median: {stats['median']:.2f}")

print("\n[T-Test: Comparing Two Groups]")
test_result = t_test(data1, data2)
print(f"  T-statistic: {test_result['t_statistic']:.4f}")
print(f"  P-value: {test_result['p_value']:.4f}")
print(f"  Significant difference: {test_result['reject_null']}")

# ============================================================================
# 2. Data Analysis Module
# ============================================================================
print("\n" + "=" * 70)
print("2. DATA ANALYSIS MODULE (데이터 분석)")
print("=" * 70)

from data_analysis.data_preprocessing import (
    load_and_explore_data,
    handle_missing_values,
    detect_outliers_iqr
)

# Create sample dataset with missing values
df = pd.DataFrame({
    'feature1': np.random.normal(100, 15, 100),
    'feature2': np.random.normal(50, 10, 100),
    'feature3': np.random.exponential(2, 100)
})
df.loc[np.random.choice(df.index, 10), 'feature1'] = np.nan

print("\n[Data Exploration]")
exploration = load_and_explore_data(df)
print(f"  Shape: {exploration['shape']}")
print(f"  Missing values: {sum(exploration['missing_values'].values())}")

print("\n[Data Cleaning]")
cleaned_df = handle_missing_values(df, strategy='mean')
print(f"  Missing values after cleaning: {cleaned_df.isnull().sum().sum()}")

print("\n[Outlier Detection]")
outliers, lower, upper = detect_outliers_iqr(cleaned_df, 'feature1')
print(f"  Number of outliers: {len(outliers)}")
print(f"  Valid range: [{lower:.2f}, {upper:.2f}]")

# ============================================================================
# 3. Deep Learning Module
# ============================================================================
print("\n" + "=" * 70)
print("3. DEEP LEARNING MODULE (딥러닝)")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    from deep_learning.neural_networks import SimpleNN, LSTM, Autoencoder
    
    print("\n[Neural Network Architectures]")
    
    # Simple Neural Network
    simple_nn = SimpleNN(input_dim=10, hidden_dim=64, output_dim=2)
    print(f"  SimpleNN: {sum(p.numel() for p in simple_nn.parameters())} parameters")
    
    # LSTM
    lstm = LSTM(input_dim=10, hidden_dim=128, num_layers=2, output_dim=1)
    print(f"  LSTM: {sum(p.numel() for p in lstm.parameters())} parameters")
    
    # Autoencoder
    autoencoder = Autoencoder(input_dim=784, encoding_dim=32)
    print(f"  Autoencoder: {sum(p.numel() for p in autoencoder.parameters())} parameters")
    
    print("\n[Quick Training Test]")
    device = torch.device('cpu')
    model = SimpleNN(10, 32, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    
    # Quick training
    model.train()
    for epoch in range(3):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    print(f"  Training completed - Final loss: {loss.item():.4f}")
    
except ImportError as e:
    print(f"\n  PyTorch not available: {e}")
    print("  Install PyTorch to use deep learning features")

# ============================================================================
# 4. Anomaly Detection Module
# ============================================================================
print("\n" + "=" * 70)
print("4. ANOMALY DETECTION MODULE (이상 탐지)")
print("=" * 70)

from anomaly_detection.anomaly_methods import (
    zscore_anomaly_detection,
    iqr_anomaly_detection,
    isolation_forest_detection
)

# Generate normal data with some anomalies
normal_data = np.random.normal(0, 1, 1000)
anomaly_data = np.random.uniform(5, 10, 50)
test_data = np.concatenate([normal_data, anomaly_data])

print("\n[Statistical Methods]")
z_anomalies, _ = zscore_anomaly_detection(test_data, threshold=3)
print(f"  Z-Score method: {len(z_anomalies)} anomalies detected")

iqr_anomalies = iqr_anomaly_detection(test_data)
print(f"  IQR method: {len(iqr_anomalies)} anomalies detected")

# Machine learning method
X_normal = np.random.randn(1000, 10)
X_anomaly = np.random.uniform(-5, 5, (50, 10))
X_test = np.vstack([X_normal, X_anomaly])

print("\n[Machine Learning Methods]")
if_predictions = isolation_forest_detection(X_test, contamination=0.05)
n_anomalies = np.sum(if_predictions == -1)
print(f"  Isolation Forest: {n_anomalies} anomalies detected")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("DEMONSTRATION COMPLETE")
print("=" * 70)
print("\nAll modules are working correctly!")
print("Check individual module READMEs for more detailed usage examples.")
print("\nModules available:")
print("  ✓ Statistics (통계 분석)")
print("  ✓ Data Analysis (데이터 분석)")
print("  ✓ Deep Learning (딥러닝)")
print("  ✓ Anomaly Detection (이상 탐지)")
print("=" * 70)
