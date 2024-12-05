import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import psutil  # 用于获取 CPU 和内存利用率

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(X_img_train, y_label_train), (X_img_test, y_label_test) = mnist.load_data()

# 数据归一化并扩展通道维度
X_img_train = np.expand_dims(X_img_train / 255.0, axis=-1)
X_img_test = np.expand_dims(X_img_test / 255.0, axis=-1)

# 将灰度图像扩展为伪 RGB
X_img_train = np.repeat(X_img_train, 3, axis=-1)
X_img_test = np.repeat(X_img_test, 3, axis=-1)

# 调整图像大小到 32x32
X_img_train = tf.image.resize(X_img_train, [32, 32])
X_img_test = tf.image.resize(X_img_test, [32, 32])

# 使用小型 ResNet（ResNet18 模拟）
def build_resnet18():
    def residual_block(x, filters, downsample=False):
        shortcut = x
        strides = (2, 2) if downsample else (1, 1)
        
        # 主路径
        x = tf.keras.layers.Conv2D(filters, (3, 3), strides=strides, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
        
        # 如果需要降采样或调整通道数，对 shortcut 进行变换
        if downsample or shortcut.shape[-1] != filters:
            shortcut = tf.keras.layers.Conv2D(filters, (1, 1), strides=strides, padding="same")(shortcut)
        
        # 残差连接
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation("relu")(x)
        return x
    
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    # 堆叠残差块
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128, downsample=True)
    x = residual_block(x, 128)
    x = residual_block(x, 256, downsample=True)
    x = residual_block(x, 256)
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# 构建模型
model = build_resnet18()

# 编译模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 训练模型
start_train_time = time.time()
history = model.fit(
    X_img_train, y_label_train,
    epochs=5, batch_size=16, validation_split=0.1  # 减小批量大小
)
end_train_time = time.time()

# 测试模型
start_inference_time = time.time()
test_loss, test_acc = model.evaluate(X_img_test, y_label_test, batch_size=16, verbose=0)  # 减小测试批量
end_inference_time = time.time()

# 性能指标
train_time = end_train_time - start_train_time
inference_time = end_inference_time - start_inference_time
fps = len(X_img_test) / inference_time

# 获取系统资源利用率
cpu_utilization = psutil.cpu_percent()
memory_info = psutil.virtual_memory()
memory_utilization = memory_info.percent

# 输出性能结果
print(f"\nTraining Time: {train_time:.2f} seconds")
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
print(f"Inference Time: {inference_time:.2f} seconds")
print(f"Inference Speed (FPS): {fps:.2f} frames/second")
print(f"CPU Utilization: {cpu_utilization}%")
print(f"Memory Utilization: {memory_utilization}%")

# 分类报告和混淆矩阵
predictions = model.predict(X_img_test, batch_size=16)  # 减小推理批量
predictions_class = np.argmax(predictions, axis=1)
print("\nClassification Report:")
print(classification_report(y_label_test, predictions_class))

conf_matrix = confusion_matrix(y_label_test, predictions_class)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()