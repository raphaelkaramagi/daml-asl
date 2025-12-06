# ASL Alphabet Recognition

Deep learning models for classifying American Sign Language alphabet gestures.

## Overview

This project implements two approaches for ASL alphabet recognition (A-Z + special characters):
1. **ResNet50 Transfer Learning** - Fine-tuned CNN on raw images
2. **Landmark Neural Network** - Lightweight NN on MediaPipe hand landmarks

**Dataset**: 87,000+ images, 29 classes ([Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))

## Quick Start

```bash
# Setup
conda create -n daml-asl python=3.11 -y
conda activate daml-asl
pip install -r requirements.txt

# Download data (required for training)
# Get from: https://www.kaggle.com/datasets/grassknoted/asl-alphabet
# Extract to: data/asl_alphabet_train/ and data/asl_alphabet_test/

# Run demo
python demo.py

# Or run notebooks
jupyter notebook
```

## Demo

```bash
# Test on all test images
python demo.py

# Test on specific image
python demo.py path/to/hand_image.jpg
```

## Project Structure

```
├── models/                    # Trained models
│   ├── best_asl_resnet50_phase2.h5   # ResNet50 (71.43% test acc)
│   └── asl_resnet50_final.keras
├── data/                      # Data files
│   ├── nn_landmark_model.keras       # Landmark NN (98.88% val acc)
│   ├── label_encoder.joblib
│   └── scaler.joblib
├── notebooks/                 # Training notebooks
│   ├── 01-data-preprocessing.ipynb
│   ├── 03-mediapipe-feature-extraction.ipynb
│   ├── 05-training-approach-1.ipynb
│   ├── 06-training-approach-2-landmarks.ipynb
│   └── 07-model-comparison-evaluation.ipynb
└── demo.py                    # Quick demo script
```

## Results

### Validation/Test Performance

| Approach | Val Accuracy | Test Accuracy | Model Size | Training Time |
|----------|--------------|---------------|------------|---------------|
| **ResNet50** | 47.24% | **71.43%** | 208 MB | ~4 hours |
| **Landmark NN** | 98.88% | 53.57% | ~1 MB | ~10 min |

**Winner on Test Set**: ResNet50 (+17.86% higher accuracy)

*Note: Test set contains only 28 images (missing "del" class). Landmark NN achieved 98.88% on its validation set.*

## Tech Stack

TensorFlow • Keras • MediaPipe • OpenCV • scikit-learn

## Usage

See [SETUP.md](SETUP.md) for detailed instructions.

---

**Dataset**: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
