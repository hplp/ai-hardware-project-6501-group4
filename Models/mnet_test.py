# -*- coding: utf-8 -*-
"""
优化后的 MobileNet V2 识别 MNIST 数据集代码
保持数据量不变
"""
import time
import numpy as np
import torch
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import psutil

""""@@@ 数据块 """
# 加载 MNIST 数据集
def load_and_preprocess_data():
    # 使用 TensorFlow 加载 MNIST 数据集
    mnist = tf.keras.datasets.mnist
    (X_img_train, y_label_train), (X_img_test, y_label_test) = mnist.load_data()

    # 数据预处理：归一化和维度调整
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # 数据归一化
    ])

    # 转换为 PyTorch 格式
    X_img_test = [preprocess(img) for img in X_img_test]  # 转换为每张图片的张量
    X_img_test = torch.stack(X_img_test)  # 合并为 (batch, channels, height, width) 格式
    y_label_test = torch.tensor(y_label_test, dtype=torch.int64)  # 标签转为张量
    return X_img_test, y_label_test

# 加载全量测试集
X_img_test, y_label_test = load_and_preprocess_data()

""""@@@ 网络块 MobileNet V2 """
# 加载预训练 MobileNet V2 模型
model = models.mobilenet_v2(weights=True)
model.classifier[1] = torch.nn.Linear(model.last_channel, 10)  # 修改分类器输出层为 10 类
model.eval()

""""@@@ 推理与评估 """
# 分批次推理
def evaluate_model_in_batches(model, data, labels, batch_size=128):
    num_samples = data.size(0)
    predictions = []
    start_time = time.time()

    with torch.no_grad():  # 禁用梯度计算以节省内存
        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i + batch_size]
            outputs = model(batch_data)
            batch_predictions = torch.argmax(outputs, dim=1)
            predictions.append(batch_predictions)

    end_time = time.time()
    predictions = torch.cat(predictions)  # 合并所有批次预测结果
    accuracy = torch.mean((predictions == labels).float()).item()  # 计算准确率
    inference_time = end_time - start_time
    fps = num_samples / inference_time
    return accuracy, predictions, inference_time, fps

# 运行评估
accuracy, predictions, inference_time, fps = evaluate_model_in_batches(model, X_img_test, y_label_test)
cpu_utilization = psutil.cpu_percent()
memory_info = psutil.virtual_memory()
memory_utilization = memory_info.percent
# 混淆矩阵
conf_matrix = confusion_matrix(y_label_test.numpy(), predictions.numpy())
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 打印评估指标
print(f"\nModel Evaluation Metrics:")
print(f"Inference Time: {inference_time:.2f} seconds")
print(f"Inference Speed (FPS): {fps:.2f} frames/second")
print(f"Accuracy: {accuracy:.2%}")
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")