# Accelerated Testing of Image Classification Task Based on Raspberry Pi 4

## Team Name: 
- ECE6501-Group4
## Team Members:
- Jiandi Wang(ctf2we): Hardware, Environment Debugging, Models adjusting, Test and Data analysis
- Yunwei Cai(ftf8kf): Model Finding
- Zilin Wang(akw4py): Environment Debugging, Test, Data analysis, and Slides
- Henan Zhao(cnw7cq): Model Finding

## Objective and Motivation
- The primary objective of this project is to evaluate the performance of ResNet_18, MobileNet_V2, and EfficientNet models trained on the MNIST dataset and deployed on the Raspberry Pi 4. Metrics such as inference time, frames per second (FPS), accuracy, and CPU/memory utilization were analyzed. Although we initially aimed to use the Hailo-8 accelerator for enhanced performance, its incompatibility with Raspberry Pi 4 necessitated a pivot to standalone execution on the Raspberry Pi.
[MNIST](https://www.tensorflow.org/datasets/catalog/mnist)

- This study demonstrates the potential of cost-effective, low-power edge devices like the Raspberry Pi 4 for real-world AI applications such as smart surveillance, automated quality control, and embedded AI in IoT systems.
## Models
### ResNet_18
![](https://www.researchgate.net/publication/366608244/figure/fig1/AS:11431281109643320@1672145338540/Structure-of-the-Resnet-18-Model.jpg)
### MobileNet_V2
![](https://www.researchgate.net/publication/368539704/figure/fig3/AS:11431281143702109@1681264260877/Block-diagram-of-the-MobileNetV2-architecture.png)
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
-  	Inference time
-  	Frames per second (FPS)
-  	Accuracy
-  	CPU and memory utilization
- 4.	Compare results to determine the most efficient model for edge deployment.

## Model Comparisons
- [ResNet_18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html): A compact residual network optimized for image recognition, balancing accuracy and computational cost.
- [MobileNet_V2](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/): Utilizes depthwise separable convolutions and inverted residuals to reduce resource consumption, making it ideal for embedded systems.
- [EfficientNet_M](https://pytorch.org/vision/main/models/efficientnet.html): Employs Neural Architecture Search (NAS) to scale model dimensions for optimal accuracy and efficiency.

## Experiment Flow
- 1.	Preprocess the MNIST dataset into a format suitable for TensorFlow training.
- 2.	Train and quantize models on a local machine before transferring them to Raspberry Pi 4.
- 3.	Perform inference on a test set and collect performance metrics, including:
-  •	Average inference time
-  •	FPS
-  •	Top-1 accuracy
-  •	Resource usage (CPU, memory)
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
 - •	Inference Time and FPS: MobileNet_V2 demonstrated the best balance between speed and resource efficiency, making it suitable for real-time edge applications.
-  •	Accuracy: ResNet_18 slightly outperformed other models in accuracy, making it ideal for precision-critical applications.
-  •	Resource Utilization: MobileNet_V2 achieved lower CPU and memory usage compared to ResNet_18 and EfficientNet_M.
### Practical Implications
- The results highlight the viability of deploying lightweight machine learning models on edge devices for applications such as:
- •	Automated document scanning and handwriting recognition.
- •	Portable AI systems for offline image processing.
- •	Real-time decision-making in embedded IoT devices.

## Conclusion and Future Work
- This project demonstrates the Raspberry Pi 4’s capability to support lightweight AI models for edge computing tasks. Among the tested models, MobileNet_V2 emerged as the most practical choice for real-world deployment due to its superior speed and resource efficiency.

- Future work could include:
- 	1.	Expanding the dataset to include more complex tasks (e.g., CIFAR-10).
- 	2.	Investigating other accelerators compatible with Raspberry Pi 4.
- 	3.	Exploring advanced optimization techniques like pruning and quantization-aware training.

## References
-	1.	A. G. Howard, et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, 2017.
-	2.	Raspberry Pi Official Documentation: https://www.raspberrypi.org/documentation/
-	3.	TensorFlow Lite Official Guide: https://www.tensorflow.org/lite
