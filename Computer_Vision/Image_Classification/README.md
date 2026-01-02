# Handwritten Digit Classification using CNN (MNIST)

## Overview
This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits (0–9) using the **MNIST dataset**.  
The goal is to demonstrate a complete **Computer Vision pipeline**: data loading, preprocessing, model building, training, evaluation, and result visualization.

MNIST is a standard benchmark dataset in computer vision and deep learning, making this project both **academically valid** and **industry-relevant**.

---

## Problem Statement
To build a deep learning model that can accurately recognize handwritten digits from grayscale images.

---

## Dataset
- **Name:** MNIST Handwritten Digits
- **Source:** Built-in Keras dataset
- **Images:** 28 × 28 grayscale
- **Classes:** 10 (digits 0–9)
- **Training Samples:** 60,000
- **Test Samples:** 10,000

---

## Approach

### 1. Data Preprocessing
- Normalized pixel values to range [0, 1]
- Reshaped images to `(28, 28, 1)` for CNN compatibility
- Labels retained as integers for sparse categorical loss

### 2. Model Architecture
A sequential CNN model consisting of:
- Input layer
- Convolutional layers for feature extraction
- MaxPooling layers for spatial reduction
- Fully connected (Dense) layers for classification
- Softmax output layer for multi-class prediction

### 3. Training
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy
- Trained on training set and validated on test set

### 4. Evaluation & Visualization
- Test accuracy evaluation
- Training vs validation accuracy plot
- Sample predictions visualization

---

## Model Summary
- **Architecture:** Convolutional Neural Network (CNN)
- **Total Parameters:** ~225K
- **Activation Functions:** ReLU, Softmax
- **Framework:** TensorFlow / Keras

---

## Results
- The model achieves **high accuracy** on the MNIST test set.
- Correctly classifies most handwritten digits.
- Demonstrates effective feature learning using convolutional layers.

---

## Outputs
The following outputs are generated and saved:
- Training and validation accuracy plot
- Sample prediction results
- Model evaluation metrics

All output files are stored inside the `outputs/` directory.

---

## Project Structure
Computer_Vision/
│
├── MNIST_Image_Classification.ipynb
├── outputs/
│ ├── accuracy_plot.png
│ ├── sample_predictions.png
│ └── metrics.txt
└── README.md


---

## Key Learnings
- Fundamentals of Convolutional Neural Networks
- Image preprocessing for deep learning
- Model evaluation and visualization
- Practical implementation of computer vision workflows

---

## Conclusion
This project demonstrates a complete and effective CNN-based solution for handwritten digit recognition.  
It serves as a strong foundational computer vision project and can be extended to more complex image classification tasks.

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

