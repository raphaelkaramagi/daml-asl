# ASL Alphabet Recognition

Deep learning models for classifying American Sign Language alphabet gestures.

## Overview

This project implements two approaches for ASL alphabet recognition (A-Z + special characters):
1. **ResNet50 Transfer Learning** - Fine-tuned CNN on raw images
2. **Landmark Neural Network** - Lightweight NN on MediaPipe hand landmarks

**Dataset**: 87,000+ images, 29 classes ([Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))

## Quick Start

```bash
# 1. Setup environment
conda create -n daml-asl python=3.11 -y
conda activate daml-asl
pip install -r requirements.txt

# 2. Download dataset from Kaggle
# https://www.kaggle.com/datasets/grassknoted/asl-alphabet
# Extract to: data/asl_alphabet_train/ and data/asl_alphabet_test/

# 3. Train models (see Training section below)

# 4. Run demo
python demo.py
```

## Training

Run notebooks in order to train both models:

```bash
jupyter notebook
```

| Step | Notebook | Output | Time |
|------|----------|--------|------|
| 1 | `01-data-preprocessing.ipynb` | Data exploration | 2 min |
| 2 | `03-mediapipe-feature-extraction.ipynb` | `data/asl_landmarks_train.csv` | 30 min |
| 3 | `05-training-approach-1.ipynb` | `models/best_asl_resnet50_phase2.h5` | 4 hours |
| 4 | `06-training-approach-2-landmarks.ipynb` | `data/nn_landmark_model.keras` | 10 min |
| 5 | `07-model-comparison-evaluation.ipynb` | Evaluation results | 2 min |

**Note:** ResNet50 training benefits significantly from GPU acceleration.

## Demo

```bash
python demo.py           # Interactive menu
python demo.py --test    # Run full test set
python demo.py image.jpg # Test specific image
```

## Project Structure

```
├── models/                    # Trained models (generated)
│   └── best_asl_resnet50_phase2.h5
├── data/                      # Data files
│   ├── asl_alphabet_train/    # Training images (download)
│   ├── asl_alphabet_test/     # Test images (download)
│   ├── asl_landmarks_train.csv # Generated landmarks
│   ├── nn_landmark_model.keras # Landmark NN (generated)
│   ├── label_encoder.joblib
│   └── scaler.joblib
├── notebooks/                 # Training notebooks
│   ├── 01-data-preprocessing.ipynb
│   ├── 03-mediapipe-feature-extraction.ipynb
│   ├── 05-training-approach-1.ipynb
│   ├── 06-training-approach-2-landmarks.ipynb
│   └── 07-model-comparison-evaluation.ipynb
└── demo.py                    # Demo script
```

## Results

### Validation/Test Performance

| Approach | Val Accuracy | Test Accuracy | Model Size | Training Time |
|----------|--------------|---------------|------------|---------------|
| **ResNet50** | 47.24% | **71.43%** | 208 MB | ~4 hours |
| **Landmark NN** | **98.88%** | **71.43%** | ~1 MB | ~10 min |

**🔑 Key Finding:** Landmark NN achieves **100% accuracy when hands are detected!** The bottleneck is MediaPipe hand detection, not the model. With proper confidence tuning (0.1), both models achieve equal 71.4% test accuracy.

*Note: Test set contains only 28 images (missing "del" class).*

## Tech Stack

TensorFlow • Keras • MediaPipe • OpenCV • scikit-learn

## Usage

See [SETUP.md](SETUP.md) for detailed instructions.

---

**Dataset**: [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
