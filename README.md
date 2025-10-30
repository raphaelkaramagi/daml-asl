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

- `01-data-preprocessing.ipynb` - Data loading, augmentation, and preprocessing pipeline

## Tech Stack

- **TensorFlow/Keras** - Deep learning framework
- **NumPy/Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **OpenCV** - Image processing
- **Jupyter** - Interactive development

## Project Structure

```
daml-asl/
├── data/               # ASL alphabet dataset (not in repo)
├── notebooks/          # Jupyter notebooks
├── requirements.txt    # Python dependencies
├── SETUP.md           # Setup instructions
└── README.md          # This file
```

## Requirements

- Python 3.11+
- TensorFlow 2.20+


## Acknowledgments

Dataset provided by [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
