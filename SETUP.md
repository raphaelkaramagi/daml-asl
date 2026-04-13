# Setup Guide

Complete environment setup for the ASL Alphabet Recognition project.

---

## Prerequisites

### Python 3.11+

**macOS (Homebrew):**
```bash
brew install python@3.11
```

**Windows:**
```powershell
winget install Python.Python.3.11
# OR download from https://www.python.org/downloads/
# ⚠️ Check "Add Python to PATH" during installation
```

### Conda (Recommended)

**macOS:**
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh  # Apple Silicon
# OR
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh  # Intel
bash Miniconda3-latest-*.sh
```

**Windows:**
Download and run the installer from [docs.conda.io/miniconda](https://docs.conda.io/en/latest/miniconda.html).

---

## Setup Steps

### 1. Clone the repo

```bash
git clone git@github.com:raphaelkaramagi/daml-asl.git
cd daml-asl
```

### 2. Create the environment

**Option A: Conda (Recommended)**
```bash
conda create -n daml-asl python=3.11 -y
conda activate daml-asl
pip install -r requirements.txt
```

**Option B: venv (macOS/Linux)**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Option B: venv (Windows)**
```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Apple Silicon — replace TensorFlow

On M1/M2/M3 Macs, replace the standard TensorFlow with the Apple Silicon builds for GPU acceleration:

```bash
pip uninstall tensorflow -y
pip install tensorflow-macos tensorflow-metal
```

### 4. Verify installation

```bash
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import mediapipe as mp; print('MediaPipe OK')"
python -c "import numpy as np; print('NumPy:', np.__version__)"
```

---

## Dataset

Download the ASL Alphabet dataset from Kaggle and extract it into `data/`:

**[Kaggle: ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)**

The expected directory structure after extraction:

```
data/
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/         # ~3,000 images per class
│       ├── B/
│       └── ...        # 29 classes total
└── asl_alphabet_test/
    └── asl_alphabet_test/
        ├── A_test.jpg
        ├── B_test.jpg
        └── ...        # 28 test images (one per class, missing "del")
```

> The dataset is ~1.3 GB uncompressed.

---

## Running Notebooks

### Option 1: VS Code (Recommended)

1. Install [VS Code](https://code.visualstudio.com/)
2. Install the **Python** and **Jupyter** extensions
3. Open the project folder in VS Code
4. Open any notebook in `notebooks/`
5. Click **Select Kernel** → choose the `daml-asl` environment
6. Run cells with `Shift+Enter` or **Run All**

### Option 2: Jupyter Notebook

```bash
conda activate daml-asl
jupyter notebook
# Navigate to notebooks/ in the browser
```

### Option 3: JupyterLab

```bash
conda activate daml-asl
jupyter lab
```

### Notebook order

| Notebook | Purpose |
|---|---|
| `01-data-preprocessing.ipynb` | Data loading, exploration, class distribution |
| `03-mediapipe-feature-extraction.ipynb` | Extract 63-feature landmark vectors → `data/asl_landmarks_train.csv` |
| `05-training-approach-1.ipynb` | ResNet50 two-phase transfer learning → `models/best_asl_resnet50_phase2.h5` |
| `06-training-approach-2-landmarks.ipynb` | Landmark NN training → `data/nn_landmark_model.keras` |
| `07-model-comparison-evaluation.ipynb` | Side-by-side evaluation on test set |

---

## CLI Demo

```bash
conda activate daml-asl
python demo.py              # Interactive menu
python demo.py --test       # Evaluate on full test set
python demo.py image.jpg    # Predict a single image
```

---

## Web Demo (Local)

```bash
cd web
npm install
npm run dev   # http://localhost:3000
```

See [`web/README.md`](web/README.md) for the web demo setup, model conversion, and Vercel deployment.

---

## Session Commands

```bash
# Start of session
conda activate daml-asl
jupyter notebook

# End of session
conda deactivate
```

---

## Troubleshooting

**`ModuleNotFoundError`**
```bash
# Make sure your environment is activated, then:
pip install -r requirements.txt
```

**`No kernel available` in VS Code**
```bash
pip install ipykernel
python -m ipykernel install --user --name=daml-asl
```

**Out of memory during ResNet training**
- Reduce `BATCH_SIZE` in the notebook: 64 → 32 → 16
- Remove `.cache()` from the data pipeline
- ResNet50 at 96×96 requires ~4–6 GB VRAM comfortably

**PowerShell execution policy (Windows)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**MediaPipe errors on Apple Silicon**
```bash
pip install --upgrade mediapipe
```
