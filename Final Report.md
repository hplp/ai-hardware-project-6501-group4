# Testing of Image Classification Task Based on Raspberry Pi 4

## 1. Team Name: 
- ECE6501-Group4
---

## 2. Team Members:
- **Jiandi Wang(ctf2we)**: Hardware setup, environment debugging, model configuration and adjustments, testing, and data analysis  
- **Yunwei Cai(ftf8kf)**: Model research and selection  
- **Zilin Wang(akw4py)**: Environment debugging, testing, data analysis, and slides preparation  
- **Henan Zhao(cnw7cq)**: Model research and selection


---

## 3. Objective and Motivation

The primary objective of this project is to evaluate the performance of three different CNN-based models—**ResNet_18**, **MobileNet_V2**, and **EfficientNet-B0**—trained on the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) and deployed on the Raspberry Pi 4. Key performance metrics include **inference time**, **frames per second (FPS)**, **accuracy**, and **CPU/memory utilization**.

### MNIST Dataset
The MNIST dataset consists of 70,000 images of handwritten digits (0–9), each 28x28 pixels in size, serving as a standard benchmark for evaluating image classification models.

![MNIST Visualization](https://datasets.activeloop.ai/wp-content/uploads/2019/12/MNIST-handwritten-digits-dataset-visualized-by-Activeloop.webp)

Our initial plan was to leverage the **Hailo-8 accelerator** to further improve inference speed. However, due to driver incompatibilities with the Raspberry Pi 4, we resorted to pure CPU-based inference. Although this choice limited the performance boost we originally anticipated, it allowed us to deeply explore the Pi’s native capabilities for on-device inference.


---

## 4. Purpose of Using Raspberry Pi 4

The Raspberry Pi 4 was selected due to its affordability, portability, and flexibility:

- **Cost-Effectiveness**: Its low price point enables widespread accessibility for research and prototyping.
- **Edge Computing Potential**: Running inference locally reduces latency and reliance on cloud services, enhancing security and responsiveness.
- **Portability and Flexibility**: Its small form factor and low power consumption facilitate integration into IoT systems, robotics, and remote field deployments.
- **Reproducibility and Accessibility**: A strong community and open-source ecosystem ensure experiments can be easily replicated and shared.

---

### Real-World Applications

By demonstrating effective on-device image classification, our research can be applied to various practical scenarios:

1. **Smart Surveillance Systems**: Real-time detection of anomalies without cloud dependency.
2. **Automated Quality Control**: Efficient classification of products on factory lines, reducing inspection time.
3. **IoT and Wearables**: Low-latency AI services on portable or battery-powered devices for healthcare, agriculture, or personalized user experiences.

---


## 5. Introduction

Edge computing requires devices that are compact, energy-efficient, and capable of performing AI tasks locally. The Raspberry Pi 4, a low-cost single-board computer, represents a promising candidate for such applications. By training and deploying lightweight models on this platform, we explore its feasibility for image classification tasks, focusing on performance and resource constraints. This work has significant implications for real-time applications, including:

- **Smart IoT Systems**: Enabling object detection and classification for applications such as home automation, inventory management, and agriculture monitoring.
- **On-Device Handwriting Recognition**: Facilitating tools in education or financial systems where offline handwriting analysis is required.
- **AI-Enhanced Accessibility Solutions**: Powering portable devices to assist individuals with disabilities through real-time visual recognition.

### 5.1 Hardware Preparation

To ensure seamless execution on the Raspberry Pi 4, the following hardware setup was implemented:
![](https://assets.raspberrypi.com/static/raspberry-pi-4-labelled@2x-1c8c2d74ade597b9c9c7e9e2fff16dd4.png)
- **Raspberry Pi 4**: A quad-core ARM Cortex-A72 processor and 4GB RAM for sufficient computational capacity.
- **Power Supply**: A stable 5V/3A power adapter to support high workloads during inference tasks.
- **MicroSD Card**: A high-speed 32GB card for storing the Raspberry Pi OS, datasets, and models.
- **Cooling System**: Passive heat sinks and a cooling fan to manage thermal dissipation during extended model testing.
- **Display, Keyboard, and Mouse**: Peripherals for interactive control and debugging.
- **Camera Module (Optional)**: For testing live inference on real-time data inputs.

### 5.2 Software Preparation

The software setup was tailored to optimize the Raspberry Pi 4 for edge AI tasks. The following steps were undertaken:
5.2.1. **Operating System Installation**:
   - Installed **Raspberry Pi OS (64-bit)** to leverage the full capabilities of the 64-bit ARM architecture.
   - Flashed the OS image onto the microSD card using tools like [Raspberry Pi Imager](https://www.raspberrypi.com/software/).
     ![](https://assets.raspberrypi.com/static/4d26bd8bf3fa72e6c0c424f9aa7c32ea/d1b7c/imager.webp)
5.2.2. **Python Environment Setup**:
   - Updated all system packages and installed Python 3.9 or later.
Run the following commands to update system packages and set up a Python virtual environment:

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.9 and pip
sudo apt install python3.9 python3-pip -y

# Create and activate a virtual environment
python3 -m venv my_project_env
source my_project_env/bin/activate

# Verify Python version
python --version
```
   - Created a virtual environment using `venv` to manage dependencies cleanly.
5.2.3. **Library Installation**:
   - Installed essential libraries, including `TensorFlow Lite`, `ONNX Runtime`, `numpy`, and `scipy` for model inference.
   - Added `matplotlib` and `seaborn` for visualizing performance metrics and `psutil` for monitoring resource usage.
```bash
# Install TensorFlow Lite
pip install tflite-runtime

# Install ONNX Runtime
pip install onnxruntime

# Install other essential libraries
pip install numpy scipy matplotlib seaborn psutil
```
5.2.4. **Dataset Preparation**:
   - Downloaded the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist), a standard benchmark for handwritten digit classification tasks.
   - Preprocessed the data to align with the input format of the selected models.
```python
import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Expand dimensions for model compatibility
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Save the preprocessed data for later use
import numpy as np
np.savez_compressed('mnist_preprocessed.npz', 
                    x_train=x_train, y_train=y_train, 
                    x_test=x_test, y_test=y_test)
```
5.2.5. **Model Conversion**:
   - Converted ResNet_18, MobileNet_V2, and EfficientNet_M models into TensorFlow Lite and ONNX formats for compatibility with the Raspberry Pi.
5.2.6. **TensorFlow Lite Conversion:**
```python
import tensorflow as tf

# Load a trained Keras model
model = tf.keras.models.load_model('resnet18_model.h5')

# Convert to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('resnet18_model.tflite', 'wb') as f:
    f.write(tflite_model)
```
5.2.7. **ONNX Conversion (using tf2onnx):**
```bash
# Install tf2onnx if not already installed
pip install tf2onnx

# Convert a Keras model to ONNX format
python -m tf2onnx.convert --saved-model resnet18_model --output resnet18_model.onnx
```
   - Applied quantization techniques where applicable to reduce model size and improve inference speed.
5.2.8. **Testing Scripts**:
   - Developed Python scripts to automate inference, collect performance metrics, and log results for comparative analysis.
5.2.9. **Inference Script:**
```python
import tflite_runtime.interpreter as tflite
import numpy as np
import time
import psutil

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="resnet18_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load preprocessed MNIST test data
data = np.load('mnist_preprocessed.npz')
x_test, y_test = data['x_test'], data['y_test']

# Run inference and collect metrics
start_time = time.time()
cpu_usage = []
memory_usage = []
accuracy = 0

for i in range(len(x_test)):
    input_data = np.expand_dims(x_test[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Measure CPU and memory usage
    cpu_usage.append(psutil.cpu_percent())
    memory_usage.append(psutil.virtual_memory().percent)
    
    # Perform inference
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Calculate accuracy
    if np.argmax(output_data) == y_test[i]:
        accuracy += 1

# Compute overall metrics
inference_time = time.time() - start_time
fps = len(x_test) / inference_time
accuracy_percentage = (accuracy / len(x_test)) * 100

print(f"Inference Time: {inference_time:.2f} seconds")
print(f"Frames Per Second: {fps:.2f}")
print(f"Accuracy: {accuracy_percentage:.2f}%")
print(f"Average CPU Usage: {np.mean(cpu_usage):.2f}%")
print(f"Average Memory Usage: {np.mean(memory_usage):.2f}%")
```
This robust preparation ensured that the experiments were conducted in a controlled and repeatable manner, enabling precise evaluation of the Raspberry Pi 4’s capabilities for edge AI tasks.

---

## 6. Models
### [ResNet_18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
![](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*rrlou8xyh7DeWdtHZk_m4Q.png)
### [MobileNet_V2](https://arxiv.org/abs/1801.04381)
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2023/12/image-150.png)
### [EfficientNet](https://arxiv.org/abs/1905.11946)
![](https://wisdomml.in/wp-content/uploads/2023/03/eff_banner.png)
- [ResNet_18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html): A compact residual network optimized for image recognition, balancing accuracy and computational cost.
- [MobileNet_V2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/): Utilizes depthwise separable convolutions and inverted residuals to reduce resource consumption, making it ideal for embedded systems.
- [EfficientNet_M](https://pytorch.org/vision/main/models/efficientnet.html): Employs Neural Architecture Search (NAS) to scale model dimensions for optimal accuracy and efficiency.

---
## 7. Project Outline
1.	**Train ResNet_18, MobileNet_V2, and EfficientNet_M models on the MNIST dataset.**
2.	**Optimize models for deployment on Raspberry Pi 4 using TensorFlow Lite.**
3.	**Benchmark models under identical conditions to measure:**
-  Inference time
-  Frames per second (FPS)
-  Accuracy
-  CPU and memory utilization
4.	**Compare results to determine the most efficient model for edge deployment.**


---

## 8. Experiment Flow

1. Preprocess MNIST and ensure compatibility with the models.
2. Train and quantize models on a desktop machine.
3. Deploy the converted models to Raspberry Pi 4.
4. Connect the Raspberry Pi to a local network and configure SSH for remote control.
5. Execute inference on test data and collect performance metrics.
6. Evaluate results and draw conclusions.

---
## 9. Methodology

### 9.1 System Setup
#### Step 1: Install Raspberry Pi OS

1. **Download the OS**:
   - Download the [Raspberry Pi OS (64-bit)](https://www.raspberrypi.com/software/operating-systems/) image from the official Raspberry Pi website.
   
2. **Flash the OS to SD Card**:
   - Use [Raspberry Pi Imager](https://www.raspberrypi.com/software/) to write the OS image to a high-speed microSD card.

3. **Enable SSH and Configure Wi-Fi**:
   - After flashing, insert the SD card into your computer.
   - Create a file named `ssh` (no extension) in the `/boot` partition to enable SSH.
   - Create a `wpa_supplicant.conf` file in the `/boot` partition with the following content (replace `SSID` and `PASSWORD` with your Wi-Fi details):

     ```plaintext
     country=US
     ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
     update_config=1

     network={
         ssid="YOUR_SSID"
         psk="YOUR_PASSWORD"
         key_mgmt=WPA-PSK
     }
     ```

4. **Boot Up the Raspberry Pi**:
   - Insert the SD card into the Raspberry Pi, connect it to a power source, and boot up the system.

#### Step 2: Find Raspberry Pi's IP Address

To connect to the Raspberry Pi:
- Use your router's admin page to find the Raspberry Pi's IP address.
- Alternatively, use a network scanner tool like `nmap`:

  ```bash
  nmap -sn 192.168.1.0/24
  ```
#### Step 3: Connect to Raspberry Pi via SSH

Connect to the Raspberry Pi from your computer using the IP address:
```bash
ssh pi@<Raspberry_Pi_IP>
```
- The default username is pi and the default password is raspberry.

#### Step 4: Update and Upgrade the Raspberry Pi

After logging in via SSH, update the system and install necessary tools:
```bash
sudo apt update && sudo apt upgrade -y
```

#### Step 5: Enable VNC

For GUI-based access to the Raspberry Pi:

- Run the Raspberry Pi configuration tool:
  ```bash
  sudo raspi-config
  ```
- Navigate to Interfacing Options > VNC and enable it.

### 9.2 Workflow for Experiments

#### Step 1: Transfer Models and Data to Raspberry Pi
Use scp to copy trained .tflite models and preprocessed dataset files to the Raspberry Pi:
```bash
scp model.tflite pi@<Raspberry_Pi_IP>:/home/pi/
scp mnist_preprocessed.npz pi@<Raspberry_Pi_IP>:/home/pi/
```
#### Step 2: Run Inference Script
Navigate to the directory containing the model and dataset, then execute the inference script:
```bash
python3 inference_script.py
```

#### Step 3: Monitor Resource Usage
Install and use htop to monitor CPU and memory usage during inference:
```bash
sudo apt install htop
htop
```

#### Step 4: Visualize Results
After inference, transfer result logs back to your computer for further analysis:
```bash
scp pi@<Raspberry_Pi_IP>:/home/pi/results.csv ./local_results/
```

### 9.3 Experiment Execution Summary
1.	**Set Up SSH:** Enabled SSH for remote control, allowing model transfer and command execution from a laptop.
2.	**Deploy Models:** Transferred .tflite models and datasets to the Raspberry Pi for local inference.
3.	**Run Inference:** Executed the Python inference scripts on the Raspberry Pi.
4.	**Monitor Performance:** Used tools like psutil and htop to measure CPU and memory usage during the tests.
5.	**Analyze Results:** Retrieved logs and visualized performance metrics such as FPS, accuracy, and resource utilization.



---
## 10. Results and Discussion

### 10.1 Preliminary Results (Before Final Pre)

| Metrics                 | ResNet_18 | MobileNet_V2 | EfficientNet_M |
| ----------------------- | --------- | ------------ | -------------- |
| Inference Speed (img/s) | 70.37     | 287.92       | 104.58         |
| Accuracy (%)            | 98.59     | 6.48         | 11.6           |
| CPU Utilization (%)     | 95.3      | 72.5         | 85.6           |
| Memory Utilization (%)  | 24.7      | 15.7         | 33.0           |

### 10.2 Updated Results (After Final Pre and Corrections)

| Metrics                 | ResNet_18  | MobileNet_V2  | EfficientNet-B0 |
| ----------------------- | ---------- | ------------- | --------------- |
| FPS (Frames/s)          | 70.37      | 287.92        | 104.58          |
| Accuracy (%)            | 98.59      | 92.30         | 94.50           |
| CPU Utilization (%)     | 95.3       | 72.5          | 85.6            |
| Memory Utilization (%)  | 24.7       | 15.7          | 33.0            |

### 10.3 Observations

- **ResNet_18** achieved the highest accuracy (98.59%) and reasonable FPS, making it suitable for applications where accuracy is the primary concern.
- **MobileNet_V2** demonstrated the highest FPS (287.92) and efficient resource usage, ideal for latency-sensitive and resource-constrained scenarios.
- **EfficientNet-B0** provided a good balance of accuracy and performance but required more resources compared to MobileNet_V2, making it less advantageous for MNIST on the Raspberry Pi 4.

### 10.4 Reasons for Limited Performance  

#### 10.4.1 Model Complexity and Dataset Simplicity  
- **EfficientNet_M** is optimized for complex, large-scale datasets (e.g., ImageNet) using advanced scaling techniques. However, its architectural advantages are underutilized on the simple MNIST dataset, leading to suboptimal results.  
- **MobileNet_V2**, designed for mobile applications, might not fully leverage its optimizations due to the grayscale, low-resolution nature of MNIST.  

#### 10.4.2 Optimization for Edge Devices  
- Both models rely on techniques like **quantization** and **pruning** to optimize performance on resource-constrained devices. These processes can degrade accuracy, particularly for complex architectures like EfficientNet_M.  
- Lack of hardware accelerators on the Raspberry Pi 4 to support these optimizations further limits their efficiency.  

#### 10.4.3 Resource Constraints on Raspberry Pi 4  
- **EfficientNet_M** is more resource-intensive compared to ResNet_18 and MobileNet_V2. The Raspberry Pi 4’s limited CPU and memory may have throttled its performance, especially during inference tasks.  

#### 10.4.4 Architectural Trade-offs  
- **MobileNet_V2** is designed to prioritize speed and lightweight deployment. While effective in mobile scenarios, its accuracy can suffer when handling noisy or challenging inputs.  
- **EfficientNet_M** focuses on achieving high accuracy for complex datasets. Its computational demands make it less suited for constrained environments like the Raspberry Pi 4.  

By understanding these limitations, this project highlights the importance of selecting models appropriate to the dataset and hardware environment. 

### 10.5 Practical Implications
- The results highlight the viability of deploying lightweight machine learning models on edge devices for applications such as:
- Automated document scanning and handwriting recognition.
- Portable AI systems for offline image processing.
- Real-time decision-making in embedded IoT devices.

---
## 11. Conclusion and Future Work
This project demonstrates the Raspberry Pi 4's capability to support lightweight AI models for edge computing tasks, showcasing its potential for cost-effective and portable AI solutions. Among the tested models, **ResNet_18** stood out for its superior accuracy and balanced trade-off between speed and resource usage, making it a strong candidate for scenarios where accuracy is critical, such as quality inspection or medical imaging.

However, for real-world deployments that prioritize speed and resource efficiency, **MobileNet_V2** emerged as the most practical choice. Its lightweight architecture allowed for faster inference times and lower CPU/memory utilization, making it suitable for applications like IoT devices, where responsiveness and energy efficiency are key factors.  

The relatively lower performance of **EfficientNet_M** on the MNIST dataset highlights the importance of aligning model complexity with the dataset and deployment environment. Designed for larger, more complex datasets, EfficientNet_M struggled to leverage its architectural strengths in this simple, grayscale classification task. Additionally, its higher computational requirements made it less suitable for the resource-constrained Raspberry Pi 4 platform.  

In conclusion, the selection of the appropriate model should consider the specific application requirements:
- Use **ResNet_18** for tasks demanding high accuracy with moderate computational resources.
- Choose **MobileNet_V2** for scenarios that require real-time processing and energy efficiency.
- Avoid overly complex architectures like EfficientNet_M unless optimized for the target environment or dataset.

This study underscores the versatility of the Raspberry Pi 4 in handling diverse AI workloads and provides a framework for selecting models tailored to edge computing constraints.
**Future work could include:**
- Expanding the dataset to include more complex tasks (e.g., CIFAR-10).
- Investigating other accelerators compatible with Raspberry Pi 4.
- Exploring advanced optimization techniques like pruning and quantization-aware training.

---
## 12. References

1. Andrew G. Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, 2017.  
2. Raspberry Pi Official Documentation: [https://www.raspberrypi.org/documentation/](https://www.raspberrypi.org/documentation/)  
3. TensorFlow Lite Official Guide: [https://www.tensorflow.org/lite](https://www.tensorflow.org/lite)  
4. PyTorch Vision Models: [https://pytorch.org/vision/main/models](https://pytorch.org/vision/main/models)  
5. Mingxing Tan and Quoc V. Le, *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019, arXiv:1905.11946
