# ASL Alphabet Recognition

Deep learning models for classifying American Sign Language (ASL) alphabet gestures, with a live interactive web demo.

Two fundamentally different approaches are compared side by side:

1. **ResNet50 Transfer Learning** — fine-tuned CNN on raw 96×96 images
2. **Landmark Neural Network** — lightweight dense NN on 63 MediaPipe hand landmark features

**Dataset:** 87,000+ images, 29 classes — A–Z + `del`, `nothing`, `space` ([Kaggle ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet))

**Live Demo:** [daml-asl.vercel.app](https://daml-asl.vercel.app) · **Source:** [github.com/raphaelkaramagi/daml-asl](https://github.com/raphaelkaramagi/daml-asl)

---

## Results

See **[docs/RESULTS.md](docs/RESULTS.md)** for full training and evaluation metrics.

| Model | Val / split accuracy | 28-photo test (end-to-end) | Parameters | Web deploy size |
|---|---|---|---|---|
| **ResNet50** (hand-cropped retrain) | — | **96.4%** (27/28) | ~23.6M | ~91 MB TF.js graph |
| **Landmark NN** (deployed weights) | **98.97%** val | **67.9%** e2e · **100%** given detection | ~18K | ~72 KB TF.js |

> End-to-end accuracy on the 28-photo test set depends on MediaPipe detection rate (67.9% at conf=0.2). The Landmark NN classifier is perfect when a hand is detected. Run `python scripts/evaluate_models.py` after retraining to refresh metrics.

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

Run notebooks in order, or use the improved scripts (recommended):

```bash
# 1. Re-extract landmarks with shared detection (multi-scale retry)
python scripts/reextract_landmarks.py

# 2. Train Landmark NN (notebook 06) on new CSV

# 3. Generate hand crops for ResNet
python scripts/generate_hand_crops.py

# 4. Train improved ResNet50 (3-phase, no horizontal flip, adapted stem)
python scripts/train_resnet_improved.py

# 5. Convert models for web + evaluate
python scripts/convert_models.py
python scripts/evaluate_models.py
python scripts/extract_training_data.py
```

| Step | Notebook / Script | Output | Time |
|---|---|---|---|
| 1 | `03-mediapipe-feature-extraction.ipynb` or `scripts/reextract_landmarks.py` | `data/asl_landmarks_train.csv` | ~30 min |
| 2 | `06-training-approach-2-landmarks.ipynb` | `data/nn_landmark_model.keras` | ~10 min |
| 3 | `scripts/generate_hand_crops.py` | `data/asl_hand_crops/` | ~1 hr |
| 4 | `05-training-approach-1.ipynb` or `scripts/train_resnet_improved.py` | `models/best_asl_resnet50_phase2.h5` | ~4–8 hrs GPU |
| 5 | `07-model-comparison-evaluation.ipynb` or `scripts/evaluate_models.py` | Evaluation metrics | ~2 min |

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
- Model architecture comparison with evaluation metrics
- Sample gallery — browse all 29 classes, click to predict
- Settings — toggle models, adjust detection confidence

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
├── scripts/                         # Training, detection, and web asset prep
│   ├── mediapipe_detect.py          # Shared hand detection (training + inference parity)
│   ├── reextract_landmarks.py       # Re-extract landmark CSV with improved detection
│   ├── generate_hand_crops.py       # Hand-cropped images for ResNet training
│   ├── train_resnet_improved.py     # Improved 3-phase ResNet50 training
│   ├── evaluate_models.py           # Fair evaluation on 28-photo test set
│   ├── convert_models.py            # Convert Keras → TF.js graph model
│   ├── prepare_samples.py           # Resize sample images for web gallery
│   └── extract_training_data.py     # Export training metrics → JSON
│
└── web/                             # Next.js web demo
    ├── public/
    │   ├── models/
│   │   ├── landmark-nn/         # TF.js layers model (~72 KB weights)
│   │   └── resnet-graph/        # TF.js float32 graph model (~91 MB)
│   ├── samples/                 # Resized sample images + manifest.json
    └── src/
        ├── app/                     # Next.js App Router (layout, page, globals)
        ├── components/              # UI components
        ├── hooks/                   # useModels, usePrediction, useWebcam
        ├── lib/                     # models.ts, landmarks.ts, image-utils.ts, constants.ts
        └── store/                   # Zustand global state
```

---

## Model Details

### Approach 1: ResNet50 Transfer Learning

- **Input:** 96×96 RGB images, normalised to [0, 1]
- **Architecture:** ResNet50 (ImageNet pretrained) → GlobalAveragePooling2D → Dropout(0.2) → Dense(128, ReLU) → Dropout(0.2) → Dense(29, Softmax)
- **Training:** 2-phase — Phase 1 (10 epochs, frozen backbone, lr=0.001) then Phase 2 (10 epochs, layers 143+ unfrozen, lr=0.0001)
- **Parameters:** ~23.6M
- **Web:** Converted to TF.js float32 graph model (~91 MB deployed; 208 MB Keras checkpoint)

### Approach 2: Landmark Neural Network

- **Input:** 63 features — 21 MediaPipe hand landmarks × (x, y, z), wrist-relative, StandardScaler normalised
- **Architecture:** Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(29, Softmax)
- **Training:** Adam, sparse categorical crossentropy, early stopping (patience=5), trained for 29 epochs
- **Parameters:** ~18K
- **Web:** TF.js layers model, ~72 KB

---

## Tech Stack

**ML / Python:** TensorFlow 2.x · Keras · MediaPipe · OpenCV · scikit-learn · NumPy · Matplotlib

**Web demo:** Next.js 16 · TypeScript · Tailwind CSS · TensorFlow.js · MediaPipe Tasks Vision · Recharts · Framer Motion · Zustand
