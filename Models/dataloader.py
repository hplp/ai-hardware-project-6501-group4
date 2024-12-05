import tensorflow as tf
import numpy as np
from PIL import Image
import os

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 创建保存图像的文件夹
os.makedirs('mnist_images/train', exist_ok=True)
os.makedirs('mnist_images/test', exist_ok=True)

# 定义保存图像的函数
def save_images(images, labels, folder):
    for i, (image, label) in enumerate(zip(images, labels)):
        img = Image.fromarray(image)  # 将 NumPy 数组转换为 PIL 图像
        label_folder = os.path.join(folder, str(label))
        os.makedirs(label_folder, exist_ok=True)
        img.save(os.path.join(label_folder, f'{i}.png'))  # 保存为 PNG 格式
        if i % 1000 == 0:  # 每1000张打印一次
            print(f'Saved {i} images in {folder}.')

# 保存训练集和测试集的图像
save_images(train_images, train_labels, 'mnist_images/train')
save_images(test_images, test_labels, 'mnist_images/test')

print("All images saved.")

