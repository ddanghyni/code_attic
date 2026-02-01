# Deep Learning Module

딥러닝 관련 Python 코드 모음

## Contents

- `neural_networks.py`: 다양한 신경망 구조 구현

## Features

### Neural Network Architectures
- **SimpleNN**: 기본 feedforward 신경망
- **ConvNet**: 합성곱 신경망 (CNN)
- **LSTM**: 순환 신경망 (RNN)
- **Autoencoder**: 오토인코더

### Training Utilities
- Training loop implementation
- Model evaluation
- Loss tracking
- Optimization utilities

### Model Components
- Activation functions
- Dropout layers
- Batch normalization
- Various optimizers

## Usage

```python
from deep_learning.neural_networks import SimpleNN, train_model
import torch
import torch.nn as nn
import torch.optim as optim

# Create model
model = SimpleNN(input_dim=10, hidden_dim=64, output_dim=2)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
losses = train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
```

## Model Architectures

### SimpleNN
- Multi-layer perceptron
- Fully connected layers
- Dropout for regularization

### ConvNet
- Convolutional layers
- Max pooling
- Image classification

### LSTM
- Sequence modeling
- Time series prediction
- Text processing

### Autoencoder
- Dimensionality reduction
- Feature learning
- Anomaly detection

## Requirements

- torch
- numpy
