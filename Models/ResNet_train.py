# -*- coding: utf-8 -*-

"""
使用 ResNet50 识别 MNIST 数据集 (输入 28x28x3)
========================================
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import psutil

""""@@@ 数据块 """
# 导入 MNIST 数据集
mnist = tf.keras.datasets.mnist
(X_img_train, y_label_train), (X_img_test, y_label_test) = mnist.load_data()

# 数据归一化并增加通道维度
X_img_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_img_train[..., tf.newaxis] / 255.0, dtype=tf.float32))
X_img_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_img_test[..., tf.newaxis] / 255.0, dtype=tf.float32))

# 将图像调整到32x32
X_img_train = tf.image.resize(X_img_train, [32, 32])
X_img_test = tf.image.resize(X_img_test, [32, 32])

# 标签 OneHot 化
y_label_train_OneHot = tf.keras.utils.to_categorical(y_label_train, 10)
y_label_test_OneHot = tf.keras.utils.to_categorical(y_label_test, 10)


""""@@@ 网络块 ResNet50 """
# 使用 ResNet50 预训练模型
base_model = tf.keras.applications.ResNet50(
    input_shape=(32, 32, 3),
    include_top=False,
    weights=None  # 不使用ImageNet权重，重新训练
)

# 构建模型
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 输出模型结构
print(model.summary())


""""@@@ 训练模型 """
epochs = 20
# Set a custom learning rate
learning_rate = 0.0001  # Example learning rate

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model with the custom optimizer
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# 保存最佳模型
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)

# 训练模型
t1 = time.time()
train_history = model.fit(
    X_img_train,
    y_label_train_OneHot,
    validation_split=0.2,
    epochs=epochs,
    batch_size=128,
    verbose=1,
    callbacks=[checkpoint]  # 保存最佳模型
)
t2 = time.time()
CNNResNet50 = float(t2 - t1)
print("Time taken: {} seconds".format(CNNResNet50))


""""@@@ 评估模型的准确率 """
# 模型评估
t_start = time.time()
loss, accuracy = model.evaluate(X_img_test, y_label_test_OneHot, verbose=0)
t_end = time.time()
inference_time = t_end - t_start
fps = len(X_img_test) / inference_time

# 获取系统 CPU 和内存利用率
cpu_utilization = psutil.cpu_percent()
memory_info = psutil.virtual_memory()
memory_utilization = memory_info.percent

# 输出模型评估指标
print("\nModel Evaluation Metrics:")
print(f"Inference Time: {inference_time:.2f} seconds")
print(f"Inference Speed (FPS): {fps:.2f} frames/second")
print(f"Accuracy: {accuracy:.2%}")
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

# 绘制训练过程中的准确率和损失
def show_train_history(train_acc, test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title("Train History")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

show_train_history("accuracy", "val_accuracy")
show_train_history("loss", "val_loss")


""""@@@ 输出模型评估表格 """
import pandas as pd

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