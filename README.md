# Comparative Deep Learning Benchmarking for Brain Tumor Classification on MRI Scans

## Project Overview
This repository contains an end-to-end medical image classification pipeline designed to automate the detection and classification of brain tumors from Magnetic Resonance Imaging (MRI) scans. Using a curated dataset of MRI scans, this project conducts a rigorous comparative analysis across three distinct, state-of-the-art Deep Convolutional Neural Network (CNN) architectures to evaluate performance, generalization, and computational efficiency in a simulated clinical workflow:

1. **ResNet-50:** Utilizing residual learning frameworks to alleviate vanishing gradient constraints in deep features.
2. **DenseNet121:** Maximizing feature reuse and gradient flow through dense connectivity patterns, optimizing performance on smaller medical imaging batches.
3. **EfficientNetV2-S:** Leveraging fused-MBConv layers and progressive learning configurations to optimize parameter efficiency and training speed without compromising accuracy.

## Key Engineering Highlights
* **End-to-End Data Pipeline:** Features a comprehensive data lake pipeline transforming raw MRI data through automated validation, preprocessing, and structured visualization scripts.
* **Robustness Evaluation:** Includes strict Out-of-Distribution (OOD) dataset testing to evaluate model generalization and guard against overfitting on baseline data characteristics.
* **Functional Application Interface:** Features an interactive interface integrating the top-performing optimized model configuration, moving the project from an abstract script to a deployment-ready diagnostic asset.
