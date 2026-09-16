# CodeAlpha Task 3: Handwritten Character Recognition

## Overview
This project implements an end-to-end Deep Learning system to identify and classify handwritten characters/digits using **Convolutional Neural Networks (CNN)** trained on the standard **MNIST** benchmark dataset.

## Key Features
- **Data Augmentation & Preprocessing**: Random rotations, affine transformations, and channel-wise normalization.
- **Deep CNN Architecture**: Multi-stage Convolutional, Batch Normalization, MaxPooling, and Dropout layers for robust feature extraction.
- **Evaluation & Metrics**: Comprehensive evaluation with Loss Curves, Test Accuracy (~99%), Classification Report (Precision, Recall, F1-Score), and Confusion Matrix.
- **Interactive Visualizer**: Sample prediction visualizer with ground truth vs. predicted labels.

## Project Structure
- `handwritten_character_recognition.ipynb`: Complete step-by-step Jupyter Notebook containing all data loading, preprocessing, CNN model training, evaluation, and visual inference cells.

## Getting Started
Open and run all cells in `handwritten_character_recognition.ipynb`:
```bash
jupyter notebook handwritten_character_recognition.ipynb
```
