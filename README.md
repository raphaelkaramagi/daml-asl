# DAML-ASL: American Sign Language Recognition

Deep learning project for recognizing American Sign Language (ASL) alphabet gestures using TensorFlow and Keras.

## Overview

This project implements a computer vision model to classify ASL alphabet signs (A-Z, plus special characters) from images.

**Dataset:** 87,000+ images across 29 classes from the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

## Quick Start

See [SETUP.md](SETUP.md) for detailed installation and setup instructions.

```bash
# 1. Create environment
conda create -n daml-asl python=3.11 -y
conda activate daml-asl

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks
jupyter notebook
```

## Notebooks

### Data Preparation
- `01-data-preprocessing.ipynb` - Data loading, augmentation, and preprocessing pipeline
- `03-mediapipe-feature-extraction.ipynb` - Hand landmark extraction using MediaPipe

### Model Training
- `05-training-approach-1.ipynb` - ResNet50 transfer learning on raw images
- `06-training-approach-2-landmarks.ipynb` - Neural network on hand landmarks

### Evaluation
- `07-model-comparison-evaluation.ipynb` - Model comparison and test set evaluation

## Approaches

### Approach 1: ResNet50 Transfer Learning
- Fine-tuned ResNet50 pre-trained on ImageNet
- Two-phase training: frozen base → fine-tuning
- Input: 96×96 RGB images with data augmentation
- Expected accuracy: 95-99% on validation set

### Approach 2: Landmark-Based Neural Network
- MediaPipe hand landmark extraction (21 landmarks × 3 coordinates = 63 features)
- Lightweight neural network (128→64→29 architecture)
- Translation-invariant (normalized to wrist position)
- Expected accuracy: 85-95% on test set
- Fast inference, suitable for mobile/edge devices

## Quick Start Guide

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed step-by-step instructions.

```bash
# 1. Extract landmarks (required for Approach 2)
# Run: notebooks/03-mediapipe-feature-extraction.ipynb (~30 min)

# 2. Train Approach 1 (ResNet50)
# Run: notebooks/05-training-approach-1.ipynb (~2-4 hours with GPU)

# 3. Train Approach 2 (Landmarks)
# Run: notebooks/06-training-approach-2-landmarks.ipynb (~5-10 min)

# 4. Compare models
# Run: notebooks/07-model-comparison-evaluation.ipynb (~2 min)
```

## Tech Stack

- **TensorFlow/Keras** - Deep learning framework
- **MediaPipe** - Hand landmark detection
- **NumPy/Pandas** - Data manipulation
- **Matplotlib/Seaborn** - Visualization
- **OpenCV** - Image processing
- **scikit-learn** - ML utilities and evaluation
- **Jupyter** - Interactive development

## Project Structure

```
daml-asl/
├── data/                              # ASL alphabet dataset
│   ├── asl_alphabet_train/           # Training images (87,000)
│   ├── asl_alphabet_test/            # Test images (29)
│   └── asl_landmarks_train.csv       # Extracted landmarks (generated)
├── notebooks/                         # Jupyter notebooks
│   ├── 01-data-preprocessing.ipynb
│   ├── 03-mediapipe-feature-extraction.ipynb
│   ├── 05-training-approach-1.ipynb
│   ├── 06-training-approach-2-landmarks.ipynb
│   ├── 07-model-comparison-evaluation.ipynb
│   └── landmark_processor.py          # Helper for parallel processing
├── requirements.txt                   # Python dependencies
├── SETUP.md                          # Setup instructions
├── TRAINING_GUIDE.md                 # Step-by-step training guide
└── README.md                         # This file
```

## Requirements

- Python 3.11+
- TensorFlow 2.20+


## Acknowledgments

Dataset provided by [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
