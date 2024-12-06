# Accelerated Testing of Image Classification Task Based on Raspberry Pi 4

## Team Name: 
- ECE6501-Group4
## Team Members:
- **Jiandi Wang(ctf2we)**: Hardware, Environment Debugging, Models adjusting, Test and Data analysis
- **Yunwei Cai(ftf8kf)**: Model Finding
- **Zilin Wang(akw4py)**: Environment Debugging, Test, Data analysis, and Slides
- **Henan Zhao(cnw7cq)**: Model Finding

## Objective and Motivation

The primary objective of this project is to evaluate the performance of **ResNet_18**, **MobileNet_V2**, and **EfficientNet_M** models trained on the [MNIST dataset](https://www.tensorflow.org/datasets/catalog/mnist) and deployed on the Raspberry Pi 4. Metrics such as **inference time**, **frames per second (FPS)**, **accuracy**, and **CPU/memory utilization** were analyzed.  

Although we initially aimed to utilize the **Hailo-8 accelerator** to enhance model inference speed and efficiency, we had to pivot to standalone execution on the Raspberry Pi 4 due to the accelerator's incompatibility with this platform. This change allowed us to explore the potential of deploying lightweight machine learning models directly on the Raspberry Pi 4 without reliance on external accelerators.  

## Purpose of Using Raspberry Pi 4  
The Raspberry Pi 4 was chosen for its **affordability, portability, and flexibility**, making it a compelling platform for edge computing research. Below are the key reasons for its selection:  

- **Cost-Effectiveness**: As a low-cost device, the Raspberry Pi 4 is accessible to developers and researchers with limited resources.  
- **Edge Computing Potential**: By performing AI inference tasks locally, the Raspberry Pi 4 reduces dependency on cloud resources, enabling faster and more secure processing.  
- **Portability and Deployment Flexibility**: Its small size and energy efficiency make it ideal for real-world applications such as IoT systems and portable AI solutions.  
- **Reproducibility and Accessibility**: The open-source ecosystem of the Raspberry Pi ensures that experiments can be easily reproduced or extended.  

### Real-World Applications  
This study demonstrates the feasibility of deploying lightweight AI models on edge devices, offering insights into various real-world use cases:  

1. **Smart Surveillance Systems**: Perform real-time anomaly detection on live camera feeds without relying on cloud processing.  
2. **Automated Quality Control**: Enable efficient inspection of manufactured products using local image classification.  
3. **IoT and Wearable Devices**: Provide on-device AI capabilities for healthcare, environmental monitoring, or personalized user experiences.  

---
## Models
### ResNet_18
![](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*rrlou8xyh7DeWdtHZk_m4Q.png)
### MobileNet_V2
![](https://cdn.analyticsvidhya.com/wp-content/uploads/2023/12/image-150.png)
### EfficientNet
![](https://wisdomml.in/wp-content/uploads/2023/03/eff_banner.png)
## Introduction
- Edge computing requires devices that are compact, energy-efficient, and capable of performing AI tasks locally. The Raspberry Pi 4, a low-cost single-board computer, represents a promising candidate for such applications.
By training and deploying lightweight models on this platform, we explore its feasibility for image classification tasks, focusing on performance and resource constraints. This work has significant implications for real-time applications, including:
-	Smart IoT systems for object detection and classification.
- On-device handwriting recognition (e.g., in education or financial systems).
- I-enhanced portable devices for accessibility solutions.
![](https://assets.raspberrypi.com/static/raspberry-pi-4-labelled@2x-1c8c2d74ade597b9c9c7e9e2fff16dd4.png)
## Project Outline
- 1.	Train ResNet_18, MobileNet_V2, and EfficientNet_M models on the MNIST dataset.
- 2.	Optimize models for deployment on Raspberry Pi 4 using TensorFlow Lite.
- 3.	Benchmark models under identical conditions to measure:
- **Inference time**
- **Frames per second (FPS)**
- **Accuracy**
- **CPU and memory utilization**
- 4.	Compare results to determine the most efficient model for edge deployment.

## Model Comparisons
- [ResNet_18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html): A compact residual network optimized for image recognition, balancing accuracy and computational cost.
- [MobileNet_V2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/): Utilizes depthwise separable convolutions and inverted residuals to reduce resource consumption, making it ideal for embedded systems.
- [EfficientNet_M](https://pytorch.org/vision/main/models/efficientnet.html): Employs Neural Architecture Search (NAS) to scale model dimensions for optimal accuracy and efficiency.

## Experiment Flow
- 1.	Preprocess the MNIST dataset into a format suitable for TensorFlow training.
- 2.	Train and quantize models on a local machine before transferring them to Raspberry Pi 4.
- 3.	Perform inference on a test set and collect performance metrics, including:
- **Average inference time**
- **FPS**
- **Accuracy**
- **Resource usage (CPU, memory)**
- 4.	Discuss trade-offs between performance and accuracy among the models.

## Methodology
- 1.	Prepare the Raspberry Pi 4 environment, ensuring necessary libraries (TensorFlow Lite, psutil, etc.) are installed.
- 2.	Train models on the MNIST dataset using Python scripts.
- 3.	Deploy quantized TFLite versions of the models on the Raspberry Pi 4.
- 4.	Execute inference tasks and log metrics using monitoring tools.
### Example inference command:
```python
python3 classify_image.py --model_file <model.tflite> --image <test_image>
```
## Results and Discussion
### Metrics Table
| Metrics  | ResNet_18 | MobileNet_V2 | EfficientNet_M |
| ------------- | ------------- | ------------- | ------------- | 
| Inference Speed(/s)  | 70.37  | 287.92 | 104.58 |
| Accuracy(%)  | 98.59  | 6.48 | 11.6 |
| CPU Utiization(%) | 95.3 | 72.5 | 85.6 |
| Memory Utilization(%) | 24.7 | 15.7 | 33.0 |

### Observations 
Among the tested models, **ResNet_18** demonstrated superior accuracy and a balanced trade-off between speed and resource usage. In contrast, **MobileNet_V2** and **EfficientNet_M** showed relatively lower performance, either in terms of inference speed or accuracy.  

### Reasons for Limited Performance  

#### 1. Model Complexity and Dataset Simplicity  
- **EfficientNet_M** is optimized for complex, large-scale datasets (e.g., ImageNet) using advanced scaling techniques. However, its architectural advantages are underutilized on the simple MNIST dataset, leading to suboptimal results.  
- **MobileNet_V2**, designed for mobile applications, might not fully leverage its optimizations due to the grayscale, low-resolution nature of MNIST.  

#### 2. Optimization for Edge Devices  
- Both models rely on techniques like **quantization** and **pruning** to optimize performance on resource-constrained devices. These processes can degrade accuracy, particularly for complex architectures like EfficientNet_M.  
- Lack of hardware accelerators on the Raspberry Pi 4 to support these optimizations further limits their efficiency.  

#### 3. Resource Constraints on Raspberry Pi 4  
- **EfficientNet_M** is more resource-intensive compared to ResNet_18 and MobileNet_V2. The Raspberry Pi 4’s limited CPU and memory may have throttled its performance, especially during inference tasks.  

#### 4. Architectural Trade-offs  
- **MobileNet_V2** is designed to prioritize speed and lightweight deployment. While effective in mobile scenarios, its accuracy can suffer when handling noisy or challenging inputs.  
- **EfficientNet_M** focuses on achieving high accuracy for complex datasets. Its computational demands make it less suited for constrained environments like the Raspberry Pi 4.  

---
By understanding these limitations, this project highlights the importance of selecting models appropriate to the dataset and hardware environment. Future work could include:  
- **Dataset Augmentation**: Enhancing the MNIST dataset with noise or complex patterns to challenge model performance.  
- **Advanced Optimization Techniques**: Exploring alternative quantization strategies or fine-tuning models specifically for the Raspberry Pi architecture.  
- **Alternative Hardware**: Testing on devices with integrated accelerators or exploring newer Raspberry Pi models with improved hardware support.  
### Practical Implications
- The results highlight the viability of deploying lightweight machine learning models on edge devices for applications such as:
- Automated document scanning and handwriting recognition.
- Portable AI systems for offline image processing.
- Real-time decision-making in embedded IoT devices.

## Conclusion and Future Work
This project demonstrates the Raspberry Pi 4's capability to support lightweight AI models for edge computing tasks, showcasing its potential for cost-effective and portable AI solutions. Among the tested models, **ResNet_18** stood out for its superior accuracy and balanced trade-off between speed and resource usage, making it a strong candidate for scenarios where accuracy is critical, such as quality inspection or medical imaging.

However, for real-world deployments that prioritize speed and resource efficiency, **MobileNet_V2** emerged as the most practical choice. Its lightweight architecture allowed for faster inference times and lower CPU/memory utilization, making it suitable for applications like IoT devices, where responsiveness and energy efficiency are key factors.  

The relatively lower performance of **EfficientNet_M** on the MNIST dataset highlights the importance of aligning model complexity with the dataset and deployment environment. Designed for larger, more complex datasets, EfficientNet_M struggled to leverage its architectural strengths in this simple, grayscale classification task. Additionally, its higher computational requirements made it less suitable for the resource-constrained Raspberry Pi 4 platform.  

In conclusion, the selection of the appropriate model should consider the specific application requirements:
- Use **ResNet_18** for tasks demanding high accuracy with moderate computational resources.
- Choose **MobileNet_V2** for scenarios that require real-time processing and energy efficiency.
- Avoid overly complex architectures like EfficientNet_M unless optimized for the target environment or dataset.

This study underscores the versatility of the Raspberry Pi 4 in handling diverse AI workloads and provides a framework for selecting models tailored to edge computing constraints.
- Future work could include:
- **Expanding the dataset to include more complex tasks (e.g., CIFAR-10).**
- **Investigating other accelerators compatible with Raspberry Pi 4.**
- **Exploring advanced optimization techniques like pruning and quantization-aware training.**

## References
-	1.	A. G. Howard, et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, 2017.
-	2.	Raspberry Pi Official Documentation: https://www.raspberrypi.org/documentation/
-	3.	TensorFlow Lite Official Guide: https://www.tensorflow.org/lite
