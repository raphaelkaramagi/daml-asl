# ASL Alphabet Recognition

Deep learning models for classifying American Sign Language (ASL) alphabet gestures, with a live interactive web demo.

Two fundamentally different approaches are compared side by side:

1. **ResNet50 Transfer Learning** — fine-tuned CNN on raw 96×96 images
2. **Landmark Neural Network** — lightweight dense NN on 63 MediaPipe hand landmark features

**Dataset:** 87,000+ images, 29 classes — A–Z + `del`, `nothing`, `space` ([Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))

**Live Demo:** [daml-asl.vercel.app](https://daml-asl.vercel.app) · **Source:** [github.com/raphaelkaramagi/daml-asl](https://github.com/raphaelkaramagi/daml-asl)

---

## Results

| Model | Val Accuracy | Test Accuracy | Model Size | Training Time |
|---|---|---|---|---|
| **ResNet50** (Phase 2 fine-tuning) | 48.06% | 67.9% (19/28) | 208 MB (23 MB quantized) | ~4 hours |
| **Landmark NN** | 98.97% | 98.88%* | ~244 KB | ~10 min |

> \* On the 12,724-sample landmark feature test split. On the 28-photo test set, Landmark NN drops to ~54% due to MediaPipe detection failures — not classification errors.

**Key finding:** The Landmark NN classifier is near-perfect when MediaPipe detects a hand. The bottleneck is detection, not classification. ResNet50's lower accuracy is primarily due to limited training (20 epochs) and small input resolution (96×96).

---

## Quick Start

```bash
# 1. Set up environment
conda create -n daml-asl python=3.11 -y
conda activate daml-asl
pip install -r requirements.txt

# 2. Download dataset from Kaggle and extract to data/
# https://www.kaggle.com/datasets/grassknoted/asl-alphabet

# 3. Run the CLI demo
python demo.py
```

See [SETUP.md](SETUP.md) for full setup instructions including Apple Silicon, Windows, and troubleshooting.

---

## Training Pipeline

Run notebooks in order:

| Step | Notebook | Output | Time |
|---|---|---|---|
| 1 | `01-data-preprocessing.ipynb` | Data exploration & visualisation | ~2 min |
| 2 | `03-mediapipe-feature-extraction.ipynb` | `data/asl_landmarks_train.csv` | ~30 min |
| 3 | `05-training-approach-1.ipynb` | `models/best_asl_resnet50_phase2.h5` | ~4 hours |
| 4 | `06-training-approach-2-landmarks.ipynb` | `data/nn_landmark_model.keras` | ~10 min |
| 5 | `07-model-comparison-evaluation.ipynb` | Evaluation results & comparison | ~2 min |

```bash
conda activate daml-asl
jupyter notebook   # then open notebooks/ in the browser
```

> **Note:** ResNet50 training (notebook 05) benefits significantly from GPU acceleration. On CPU expect 2–3× longer.

---

## CLI Demo

```bash
python demo.py              # Interactive menu
python demo.py --test       # Run full test set evaluation
python demo.py image.jpg    # Predict a single image
```

---

## Web Demo

A full interactive web demo lives in `web/`. It runs entirely in the browser — no Python backend needed.

**Features:**
- Live webcam prediction with MediaPipe hand skeleton overlay
- Image upload prediction
- Side-by-side ResNet50 vs Landmark NN results with confidence bars
- Training replay — animated epoch-by-epoch charts from real training runs
- In-browser micro training (TF.js)
- Model architecture comparison
- Sample gallery — browse all 29 classes, click to predict

See [`web/README.md`](web/README.md) for development and deployment instructions.

---

## Project Structure

```
daml-asl/
├── README.md
├── SETUP.md
├── requirements.txt
├── demo.py                          # CLI demo script
│
├── notebooks/
│   ├── 01-data-preprocessing.ipynb        # Data loading, exploration, visualisation
│   ├── 03-mediapipe-feature-extraction.ipynb  # Extract hand landmarks → CSV
│   ├── 05-training-approach-1.ipynb       # ResNet50 transfer learning (2-phase)
│   ├── 06-training-approach-2-landmarks.ipynb # Landmark NN training
│   └── 07-model-comparison-evaluation.ipynb   # Side-by-side evaluation
│
├── models/                          # Trained model files (generated)
│   ├── best_asl_resnet50_phase1.h5  # ResNet50 after Phase 1 (frozen backbone)
│   ├── best_asl_resnet50_phase2.h5  # ResNet50 after Phase 2 (fine-tuned) ← used for web
│   └── asl_resnet50_final.keras     # Final Keras format export
│
├── data/                            # Datasets and generated artifacts
│   ├── asl_alphabet_train/          # Training images (download from Kaggle)
│   ├── asl_alphabet_test/           # Test images (download from Kaggle)
│   ├── asl_landmarks_train.csv      # MediaPipe landmark features (generated)
│   ├── nn_landmark_model.keras      # Trained Landmark NN (generated)
│   ├── label_encoder.joblib         # LabelEncoder for class names
│   └── scaler.joblib                # StandardScaler for landmark features
│
├── scripts/                         # Web asset preparation (run from repo root)
│   ├── convert_models.py            # Convert Keras → TF.js format
│   ├── prepare_samples.py           # Resize sample images for web gallery
│   └── extract_training_data.py     # Export training metrics → JSON
│
└── web/                             # Next.js web demo
    ├── public/
    │   ├── models/
    │   │   ├── landmark-nn/         # TF.js layers model (~72 KB weights)
    │   │   └── resnet-graph/        # TF.js graph model (~23 MB, uint8 quantized)
    │   ├── samples/                 # Resized sample images + manifest.json
    │   └── training-data.json       # Epoch metrics for training replay
    └── src/
        ├── app/                     # Next.js App Router (layout, page, globals)
        ├── components/              # UI components
        ├── hooks/                   # useModels, usePrediction, useWebcam
        ├── lib/                     # models.ts, landmarks.ts, constants.ts
        └── store/                   # Zustand global state
```

---

## Model Details

### Approach 1: ResNet50 Transfer Learning

- **Input:** 96×96 RGB images, normalised to [0, 1]
- **Architecture:** ResNet50 (ImageNet pretrained) → GlobalAveragePooling2D → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.2) → Dense(29, Softmax)
- **Training:** 2-phase — Phase 1 (10 epochs, frozen backbone, lr=0.001) then Phase 2 (10 epochs, layers 143+ unfrozen, lr=0.0001)
- **Parameters:** ~23.6M
- **Web:** Converted to TF.js graph model with uint8 quantisation (208 MB → 23 MB)

### Approach 2: Landmark Neural Network

- **Input:** 63 features — 21 MediaPipe hand landmarks × (x, y, z), wrist-relative, StandardScaler normalised
- **Architecture:** Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(29, Softmax)
- **Training:** Adam, sparse categorical crossentropy, early stopping (patience=5), trained for 29 epochs
- **Parameters:** ~18K
- **Web:** TF.js layers model, 244 KB

---

## Tech Stack

**ML / Python:** TensorFlow 2.x · Keras · MediaPipe · OpenCV · scikit-learn · NumPy · Matplotlib

**Web demo:** Next.js 16 · TypeScript · Tailwind CSS · TensorFlow.js · MediaPipe Tasks Vision · Recharts · Framer Motion · Zustand
