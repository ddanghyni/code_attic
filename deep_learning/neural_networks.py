"""
Deep Learning Module
딥러닝 관련 Python 코드

This module contains deep learning examples including:
- Neural network architectures
- Training utilities
- Model evaluation
- PyTorch implementations
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SimpleNN(nn.Module):
    """
    Simple feedforward neural network for classification
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class ConvNet(nn.Module):
    """
    Convolutional Neural Network for image classification
    """
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


class LSTM(nn.Module):
    """
    LSTM network for sequence prediction
    """
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class Autoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction and anomaly detection
    """
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    Training function for neural networks
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
        num_epochs: Number of training epochs
    
    Returns:
        list: Training losses per epoch
    """
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    return losses


def evaluate_model(model, test_loader, device):
    """
    Evaluation function for classification models
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        float: Accuracy score
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":
    print("=== Deep Learning Models Example ===")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example 1: Simple Neural Network
    print("\n=== Simple Neural Network ===")
    model = SimpleNN(input_dim=10, hidden_dim=64, output_dim=2)
    print(f"Model architecture:\n{model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Example 2: CNN
    print("\n=== Convolutional Neural Network ===")
    cnn_model = ConvNet(num_classes=10)
    print(f"CNN parameters: {sum(p.numel() for p in cnn_model.parameters())}")
    
    # Example 3: LSTM
    print("\n=== LSTM Network ===")
    lstm_model = LSTM(input_dim=10, hidden_dim=128, num_layers=2, output_dim=1)
    print(f"LSTM parameters: {sum(p.numel() for p in lstm_model.parameters())}")
    
    # Example 4: Autoencoder
    print("\n=== Autoencoder ===")
    ae_model = Autoencoder(input_dim=784, encoding_dim=32)
    print(f"Autoencoder parameters: {sum(p.numel() for p in ae_model.parameters())}")
    
    # Example training with random data
    print("\n=== Training Example ===")
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    simple_model = SimpleNN(10, 64, 2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
    
    print("Training for 5 epochs...")
    losses = train_model(simple_model, train_loader, criterion, optimizer, device, num_epochs=5)
    print(f"Final loss: {losses[-1]:.4f}")
