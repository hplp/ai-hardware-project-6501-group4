import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import psutil  # 用于获取 CPU 和内存利用率
import pandas as pd  # 用于生成结果表格

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(X_img_train, y_label_train), (X_img_test, y_label_test) = mnist.load_data()

# 数据归一化并增加通道维度
X_img_train = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_img_train[..., tf.newaxis] / 255.0, dtype=tf.float32))
X_img_test = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_img_test[..., tf.newaxis] / 255.0, dtype=tf.float32))

# 将图像调整到 ResNet50 期望的输入尺寸 (224x224)
X_img_train = tf.image.resize(X_img_train, [224, 224])
X_img_test = tf.image.resize(X_img_test, [224, 224])

# 标签 OneHot 化
y_label_train_OneHot = tf.keras.utils.to_categorical(y_label_train, 10)
y_label_test_OneHot = tf.keras.utils.to_categorical(y_label_test, 10)

# 定义 ResNet50 模型
base_model = tf.keras.applications.ResNet50(
    weights=None,  # 可切换为 'imagenet' 加载预训练权重
    input_shape=(224, 224, 3),
    include_top=False
)

# 添加分类头
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
x = tf.keras.layers.Dense(256, activation='relu')(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

# 构建模型
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
)

# 训练模型（如有需要）
# 如果已有保存模型可跳过训练部分直接加载模型
# model.fit(X_img_train, y_label_train_OneHot, epochs=10, batch_size=32, validation_split=0.1)

# 加载保存的模型
# model = tf.keras.models.load_model('resnet50_model.h5')

# 测试集推理时间
start_time = time.time()
test_loss, test_acc = model.evaluate(X_img_test, y_label_test_OneHot, verbose=0)
end_time = time.time()

# 计算推理时间和推理速度
inference_time = end_time - start_time
fps = len(X_img_test) / inference_time

# 获取系统资源利用率
cpu_utilization = psutil.cpu_percent()
memory_info = psutil.virtual_memory()
memory_utilization = memory_info.percent

# 输出结果
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
print(f"Inference Time: {inference_time:.2f} seconds")
print(f"Inference Speed (FPS): {fps:.2f} frames/second")
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

# 在测试集上进行预测
predictions = model.predict(X_img_test)
predictions_class = np.argmax(predictions, axis=1)

# 打印分类报告
print("\nClassification Report:")
print(classification_report(y_label_test, predictions_class))

# 混淆矩阵
conf_matrix = confusion_matrix(y_label_test, predictions_class)
label_dict = {i: str(i) for i in range(10)}

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict.values(), yticklabels=label_dict.values())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 创建模型评估指标表格
metrics_data = {
    "Metric": ["Inference Time", "Inference Speed (FPS)", "Accuracy", "CPU Utilization", "Memory Utilization"],
    "Value": [
        f"{inference_time:.2f} seconds",
        f"{fps:.2f} frames/second",
        f"{test_acc:.2%}",
        f"{cpu_utilization}%",
        f"{memory_utilization}%"
    ]
}

metrics_df = pd.DataFrame(metrics_data)
print("\nModel Evaluation Metrics Table:")
print(metrics_df)