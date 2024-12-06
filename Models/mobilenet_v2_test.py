# -*- coding: utf-8 -*-
"""
MobileNet V2 Training and Testing on MNIST
==========================================
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
import psutil
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

def load_and_preprocess_data():
    """Load and preprocess MNIST data."""
    # Load MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize and reshape to [batch, channels, height, width]
    X_train = np.expand_dims(X_train / 255.0, axis=1)  # Shape: [batch, 1, 28, 28]
    X_test = np.expand_dims(X_test / 255.0, axis=1)

    # Convert to PyTorch tensors
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=64):
    """Create DataLoaders for training and testing."""
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def modify_mobilenet_for_mnist():
    """Load and modify MobileNetV2 for MNIST (grayscale)."""
    mobilenet = models.mobilenet_v2(pretrained=True)
    mobilenet.features[0][0] = torch.nn.Conv2d(
        in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False
    )
    mobilenet.classifier[1] = torch.nn.Linear(mobilenet.last_channel, 10)
    return mobilenet

def train_model(model, train_loader, criterion, optimizer, epochs=5):
    """Train the model."""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

def evaluate_model(model, test_loader):
    """Evaluate the model and return predictions and labels."""
    model.eval()
    all_outputs = []
    all_labels = []
    start_time = time.time()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            all_outputs.append(outputs)
            all_labels.append(batch_y)
    end_time = time.time()

    # Concatenate results
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Calculate inference time and predictions
    inference_time = end_time - start_time
    predictions = torch.argmax(all_outputs, dim=1).numpy()
    accuracy = np.mean(predictions == all_labels.numpy())

    return predictions, all_labels, inference_time, accuracy

def visualize_results(all_labels, predictions, inference_time, accuracy):
    """Visualize results and display metrics."""
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels.numpy(), predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

    # Metrics summary
    fps = len(all_labels) / inference_time
    cpu_utilization = psutil.cpu_percent()
    memory_utilization = psutil.virtual_memory().percent
    print(f"Inference Time: {inference_time:.2f} seconds")
    print(f"Inference Speed: {fps:.2f} FPS")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"CPU Utilization: {cpu_utilization}%")
    print(f"Memory Utilization: {memory_utilization}%")

def main():
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    # Create DataLoaders
    train_loader, test_loader = create_dataloaders(X_train, y_train, X_test, y_test)

    # Initialize model, criterion, and optimizer
    model = modify_mobilenet_for_mnist()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer)

    # Evaluate the model
    predictions, all_labels, inference_time, accuracy = evaluate_model(model, test_loader)

    # Visualize results
    visualize_results(all_labels, predictions, inference_time, accuracy)

if __name__ == "__main__":
    main()