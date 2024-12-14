# -*- coding: utf-8 -*-
"""
使用 EfficientNet-B0 识别 MNIST 数据集 (输入 32x32x3)
========================================
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import psutil

# 检测是否有 GPU 可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

""""@@@ 数据块 """
# 加载 MNIST 数据集
from torchvision.datasets import MNIST

transform = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整为 32x32 尺寸
    transforms.Grayscale(num_output_channels=3),  # 转换为三通道
    transforms.ToTensor(),  # 转为 Tensor 格式
    transforms.Normalize((0.5,), (0.5,))  # 归一化
])

train_dataset = MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = MNIST(root="./data", train=False, transform=transform, download=True)

# 创建数据加载器
batch_size = 16
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

""""@@@ 网络块 EfficientNet-B0 """
# 加载 EfficientNet-B0 模型
model = efficientnet_b0(weights=None, num_classes=10)  # 使用随机初始化权重
model = model.to(device)  # 将模型加载到设备

# 定义损失函数与优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

""""@@@ 训练模型 """
epochs = 5
train_history = {"accuracy": [], "loss": []}
start_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        correct += (outputs.argmax(dim=1) == y_batch).sum().item()

    accuracy = correct / len(train_loader.dataset)
    train_history["accuracy"].append(accuracy)
    train_history["loss"].append(epoch_loss / len(train_loader))

    print(f"Epoch {epoch + 1}/{epochs}: Loss={epoch_loss:.4f}, Accuracy={accuracy:.4%}")

end_time = time.time()
training_time = end_time - start_time
print(f"\nTraining Time: {training_time:.2f} seconds")

""""@@@ 评估模型 """
model.eval()
start_time = time.time()

correct = 0
y_pred = []
y_true = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        preds = outputs.argmax(dim=1)
        y_pred.extend(preds.cpu().numpy())
        y_true.extend(y_batch.cpu().numpy())
        correct += (preds == y_batch).sum().item()

end_time = time.time()
inference_time = end_time - start_time
fps = len(test_loader.dataset) / inference_time
accuracy = correct / len(test_loader.dataset)

# 获取系统 CPU 和内存利用率
cpu_utilization = psutil.cpu_percent()
memory_info = psutil.virtual_memory()
memory_utilization = memory_info.percent

# 输出评估指标
print("\nModel Evaluation Metrics:")
print(f"Inference Time: {inference_time:.2f} seconds")
print(f"Inference Speed (FPS): {fps:.2f} frames/second")
print(f"Accuracy: {accuracy:.2%}")
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

""""@@@ 混淆矩阵 """
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

""""@@@ 输出模型评估表格 """
metrics_data = {
    "Metric": ["Inference Time", "Inference Speed (FPS)", "Accuracy", "CPU Utilization", "Memory Utilization"],
    "Value": [
        f"{inference_time:.2f} seconds",
        f"{fps:.2f} frames/second",
        f"{accuracy:.2%}",
        f"{cpu_utilization}%",
        f"{memory_utilization}%"
    ]
}

metrics_df = pd.DataFrame(metrics_data)
print("\nModel Evaluation Metrics Table:")
print(metrics_df)

""""@@@ 可视化训练历史 """
def show_train_history(metric_name):
    plt.plot(train_history[metric_name])
    plt.title(f"Train History ({metric_name.capitalize()})")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.capitalize())
    plt.show()

show_train_history("accuracy")
show_train_history("loss")