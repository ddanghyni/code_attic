"""
Anomaly Detection Module
이상 탐지(Anomaly Detection) 관련 Python 코드

This module contains anomaly detection examples including:
- Statistical methods
- Machine learning-based methods
- Deep learning-based methods
- Evaluation metrics
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch
import torch.nn as nn


def zscore_anomaly_detection(data, threshold=3):
    """
    Detect anomalies using Z-score method
    
    Args:
        data: numpy array or pandas Series
        threshold: Number of standard deviations (default: 3)
    
    Returns:
        tuple: (anomaly indices, z-scores)
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = np.abs((data - mean) / std)
    
    anomalies = np.where(z_scores > threshold)[0]
    
    return anomalies, z_scores


def iqr_anomaly_detection(data):
    """
    Detect anomalies using Interquartile Range (IQR) method
    
    Args:
        data: numpy array or pandas Series
    
    Returns:
        numpy array: Indices of anomalies
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    anomalies = np.where((data < lower_bound) | (data > upper_bound))[0]
    
    return anomalies


def isolation_forest_detection(X, contamination=0.1, random_state=42):
    """
    Detect anomalies using Isolation Forest
    
    Args:
        X: Feature matrix (n_samples, n_features)
        contamination: Expected proportion of outliers
        random_state: Random seed
    
    Returns:
        numpy array: Predictions (-1 for anomaly, 1 for normal)
    """
    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(X)
    
    return predictions


def local_outlier_factor_detection(X, n_neighbors=20, contamination=0.1):
    """
    Detect anomalies using Local Outlier Factor (LOF)
    
    Args:
        X: Feature matrix (n_samples, n_features)
        n_neighbors: Number of neighbors to consider
        contamination: Expected proportion of outliers
    
    Returns:
        numpy array: Predictions (-1 for anomaly, 1 for normal)
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = model.fit_predict(X)
    
    return predictions


def one_class_svm_detection(X, nu=0.1, kernel='rbf', gamma='scale'):
    """
    Detect anomalies using One-Class SVM
    
    Args:
        X: Feature matrix (n_samples, n_features)
        nu: Upper bound on the fraction of training errors
        kernel: Kernel type
        gamma: Kernel coefficient
    
    Returns:
        numpy array: Predictions (-1 for anomaly, 1 for normal)
    """
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    predictions = model.fit_predict(X)
    
    return predictions


class AnomalyAutoencoder(nn.Module):
    """
    Autoencoder-based anomaly detection
    """
    def __init__(self, input_dim, encoding_dim=32):
        super(AnomalyAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection"""
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error


def detect_anomalies_with_autoencoder(model, data, threshold_percentile=95):
    """
    Detect anomalies using trained autoencoder
    
    Args:
        model: Trained autoencoder model
        data: Test data tensor
        threshold_percentile: Percentile for threshold (default: 95)
    
    Returns:
        tuple: (anomaly predictions, reconstruction errors)
    """
    errors = model.get_reconstruction_error(data)
    threshold = np.percentile(errors.cpu().numpy(), threshold_percentile)
    predictions = (errors > threshold).cpu().numpy()
    
    return predictions, errors.cpu().numpy()


def evaluate_anomaly_detection(y_true, y_pred):
    """
    Evaluate anomaly detection performance
    
    Args:
        y_true: True labels (0 for normal, 1 for anomaly)
        y_pred: Predicted labels (0 for normal, 1 for anomaly)
    
    Returns:
        dict: Evaluation metrics
    """
    # Convert predictions if they're in -1/1 format
    y_pred_binary = np.where(y_pred == -1, 1, 0)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred_binary, average='binary', pos_label=1
    )
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    }


if __name__ == "__main__":
    print("=== Anomaly Detection Examples ===")
    
    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, 1000)
    anomalies = np.random.uniform(5, 10, 50)
    data = np.concatenate([normal_data, anomalies])
    
    print("\n=== Z-Score Method ===")
    anomaly_indices, z_scores = zscore_anomaly_detection(data, threshold=3)
    print(f"Detected {len(anomaly_indices)} anomalies")
    
    print("\n=== IQR Method ===")
    iqr_anomalies = iqr_anomaly_detection(data)
    print(f"Detected {len(iqr_anomalies)} anomalies")
    
    # Generate multivariate data
    X_normal = np.random.randn(1000, 10)
    X_anomaly = np.random.uniform(-5, 5, (50, 10))
    X = np.vstack([X_normal, X_anomaly])
    
    print("\n=== Isolation Forest ===")
    if_predictions = isolation_forest_detection(X, contamination=0.05)
    n_anomalies_if = np.sum(if_predictions == -1)
    print(f"Detected {n_anomalies_if} anomalies")
    
    print("\n=== Local Outlier Factor ===")
    lof_predictions = local_outlier_factor_detection(X, contamination=0.05)
    n_anomalies_lof = np.sum(lof_predictions == -1)
    print(f"Detected {n_anomalies_lof} anomalies")
    
    print("\n=== One-Class SVM ===")
    svm_predictions = one_class_svm_detection(X, nu=0.05)
    n_anomalies_svm = np.sum(svm_predictions == -1)
    print(f"Detected {n_anomalies_svm} anomalies")
    
    print("\n=== Autoencoder-based Detection ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ae_model = AnomalyAutoencoder(input_dim=10, encoding_dim=3).to(device)
    
    # Train on normal data
    X_train_tensor = torch.FloatTensor(X_normal[:800]).to(device)
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    print("Training autoencoder...")
    ae_model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        reconstructed = ae_model(X_train_tensor)
        loss = criterion(reconstructed, X_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
    
    # Detect anomalies
    X_test_tensor = torch.FloatTensor(X[800:]).to(device)
    ae_predictions, errors = detect_anomalies_with_autoencoder(ae_model, X_test_tensor)
    n_anomalies_ae = np.sum(ae_predictions)
    print(f"Detected {n_anomalies_ae} anomalies")
    
    # Evaluate if true labels are known
    y_true = np.array([0] * 200 + [1] * 50)
    print("\n=== Evaluation ===")
    metrics = evaluate_anomaly_detection(y_true, if_predictions[800:])
    print(f"Isolation Forest - Precision: {metrics['precision']:.3f}, Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}")
