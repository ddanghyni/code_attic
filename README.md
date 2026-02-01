# Code Attic

통계, 데이터 분석, 딥러닝, Anomaly Detection 등 연구 분야와 관련된 Python 코드 저장소

A repository for Python codes related to statistics, data analysis, deep learning, and anomaly detection research.

## 📚 Repository Structure

```
code_attic/
├── stat_analysis/          # 통계 분석 (Statistical Analysis)
│   ├── statistical_analysis.py
│   └── README.md
├── data_analysis/          # 데이터 분석 (Data Analysis)
│   ├── data_preprocessing.py
│   └── README.md
├── deep_learning/          # 딥러닝 (Deep Learning)
│   ├── neural_networks.py
│   └── README.md
├── anomaly_detection/      # 이상 탐지 (Anomaly Detection)
│   ├── anomaly_methods.py
│   └── README.md
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ddanghyni/code_attic.git
cd code_attic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Quick Start

Each module can be run independently:

```bash
# Statistics examples
python stat_analysis/statistical_analysis.py

# Data analysis examples
python data_analysis/data_preprocessing.py

# Deep learning examples
python deep_learning/neural_networks.py

# Anomaly detection examples
python anomaly_detection/anomaly_methods.py
```

## 📖 Modules

### 1. Statistics (통계)
Statistical analysis tools and methods including:
- Descriptive statistics
- Hypothesis testing (t-test, chi-square, etc.)
- Correlation analysis
- Distribution fitting and testing

[→ View Statistics Module](./stat_analysis/README.md)

### 2. Data Analysis (데이터 분석)
Data preprocessing and exploratory data analysis:
- Data cleaning and handling missing values
- Outlier detection and removal
- Feature engineering
- Data visualization
- Correlation analysis

[→ View Data Analysis Module](./data_analysis/README.md)

### 3. Deep Learning (딥러닝)
Neural network architectures and training utilities:
- Feedforward Neural Networks
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (LSTM)
- Autoencoders
- Training and evaluation utilities

[→ View Deep Learning Module](./deep_learning/README.md)

### 4. Anomaly Detection (이상 탐지)
Various anomaly detection methods:
- Statistical methods (Z-score, IQR)
- Machine learning methods (Isolation Forest, LOF, One-Class SVM)
- Deep learning methods (Autoencoders)
- Evaluation metrics

[→ View Anomaly Detection Module](./anomaly_detection/README.md)

## 🛠️ Dependencies

Main libraries used in this repository:

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **PyTorch**: Deep learning framework
- **SciPy**: Scientific computing
- **Matplotlib & Seaborn**: Data visualization
- **PyOD**: Outlier detection library

See [requirements.txt](./requirements.txt) for complete list.

## 📝 Usage Examples

### Statistics Example
```python
from stat_analysis.statistical_analysis import t_test, correlation_analysis

# Perform t-test
group1 = [100, 102, 98, 105, 103]
group2 = [110, 108, 112, 115, 109]
result = t_test(group1, group2)
print(f"P-value: {result['p_value']}")
```

### Data Analysis Example
```python
from data_analysis.data_preprocessing import handle_missing_values
import pandas as pd

# Handle missing values
df = pd.DataFrame({'A': [1, 2, None, 4], 'B': [5, None, 7, 8]})
cleaned_df = handle_missing_values(df, strategy='mean')
```

### Deep Learning Example
```python
from deep_learning.neural_networks import SimpleNN
import torch

# Create and use a neural network
model = SimpleNN(input_dim=10, hidden_dim=64, output_dim=2)
x = torch.randn(32, 10)
output = model(x)
```

### Anomaly Detection Example
```python
from anomaly_detection.anomaly_methods import isolation_forest_detection
import numpy as np

# Detect anomalies
X = np.random.randn(1000, 10)
predictions = isolation_forest_detection(X, contamination=0.1)
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new algorithms or methods
- Improve existing code
- Add documentation
- Report bugs or issues

## 📄 License

This project is open source and available for research and educational purposes.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This repository is actively maintained and updated with new research code and methods.