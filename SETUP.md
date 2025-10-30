Quick setup instructions 


## Prerequisites

### Install Python 3.11+

**macOS:**
```bash
# Using Homebrew
brew install python@3.11
```

**Windows:**
```powershell
# Using winget
winget install Python.Python.3.11
# OR download from: https://www.python.org/downloads/
# ⚠️ Check "Add Python to PATH" during installation
```

### Install Conda (Optional but Recommended)

**macOS:**
```bash
# Download Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh
```

**Windows:**
```powershell
# Download and run installer from:
# https://docs.conda.io/en/latest/miniconda.html
```

---

## Setup Steps

### 1. Clone/Download Project
```bash
cd /path/to/your/projects
git clone git@github.com:raphaelkaramagi/daml-asl.git
cd daml-asl
```

### 2. Create Environment

**Option A: Using Conda (Recommended)**
```bash
# Create environment
conda create -n daml-asl python=3.11 -y
conda activate daml-asl

# Install dependencies
pip install -r requirements.txt
```

**Option B: Using venv**

*macOS:*
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*Windows:*
```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install TensorFlow for Apple Silicon 
```bash
pip uninstall tensorflow
pip install tensorflow-macos tensorflow-metal
```

### 4. Verify Installation
```bash
python -c "import tensorflow as tf; print('✅ TensorFlow:', tf.__version__)"
python -c "import numpy as np; print('✅ NumPy:', np.__version__)"
```

---

## Running Notebooks

### Option 1: VS Code (Recommended)
1. Install [VS Code](https://code.visualstudio.com/)
2. Install Python + Jupyter extensions
3. Open project folder in VS Code
4. Open notebook: `notebooks/01-data-preprocessing.ipynb`
5. Select kernel: Click "Select Kernel" → Choose your environment
6. Run cells with `Shift+Enter` or "Run All"

### Option 2: Jupyter Notebook
```bash
# Activate environment first
conda activate daml-asl  # OR: source venv/bin/activate

# Start Jupyter
jupyter notebook

# Navigate to notebooks/ folder and open desired notebook
```

### Option 3: JupyterLab
```bash
jupyter lab
```

---

## Dataset

The project expects ASL Alphabet dataset in this structure:
```
data/
├── asl_alphabet_train/
│   └── asl_alphabet_train/
│       ├── A/
│       ├── B/
│       └── ...
└── asl_alphabet_test/
    └── asl_alphabet_test/
        ├── A_test.jpg
        └── ...
```

**Download:** [Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

---

## Troubleshooting

**"ModuleNotFoundError"**
```bash
# Ensure environment is activated
pip install -r requirements.txt
```

**"No kernel available" in VS Code**
```bash
pip install ipykernel
python -m ipykernel install --user --name=daml-asl
```

**Memory errors**
- Reduce `BATCH_SIZE` in notebooks (32 → 16 → 8)
- Remove `.cache()` from data pipelines

**PowerShell execution policy (Windows)**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Quick Start Commands

**Every session:**
```bash
# Activate environment
conda activate daml-asl  # Conda
# OR
source venv/bin/activate  # macOS venv
# OR
venv\Scripts\activate  # Windows venv

# Start Jupyter
jupyter notebook
```

**Deactivate when done:**
```bash
conda deactivate  # Conda
# OR
deactivate  # venv
```

---


