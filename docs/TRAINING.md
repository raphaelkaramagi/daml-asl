# Training Pipeline

Scripts for retraining both models. Run from the repo root with `conda activate daml-asl`.

## Prerequisites

- Kaggle ASL dataset in `data/asl_alphabet_train/` and `data/asl_alphabet_test/`
- Apple Silicon: `pip install tensorflow-metal` for GPU-accelerated training

## Performance notes (Apple M-series)

| Step | GPU? | Speed tip |
|---|---|---|
| `reextract_landmarks.py` | CPU (MediaPipe) | `--workers 8` (~6 min for 87k images) |
| `generate_hand_crops.py` | CPU (MediaPipe) | `--workers 8` (~6 min) |
| `train_landmark_nn.py` | Metal GPU | ~4 min (80 epochs, early stopping) |
| `train_resnet_improved.py` | Metal GPU | ~4–8 hours |

MediaPipe batch scripts use CPU by design — parallel workers are faster and more stable than GPU delegate on macOS.

## Landmark-only retrain (fast)

When only the landmark classifier needs updating:

```bash
conda activate daml-asl
python scripts/run_landmark_pipeline.py
```

This trains, exports TF.js (`--landmark-only`), evaluates, updates docs, and builds the web app.

## Full pipeline

```bash
conda activate daml-asl

# 1. Re-extract landmarks (shared detection module)
python scripts/reextract_landmarks.py --workers 8

# 2. Train landmark classifier
python scripts/train_landmark_nn.py

# 3. Generate hand-cropped ResNet training set
python scripts/generate_hand_crops.py --workers 8

# 4. Train ResNet (3-phase, hand-cropped data)
python scripts/train_resnet_improved.py

# 5. Export for web + evaluate
python scripts/run_post_training.py   # convert, evaluate, update docs, build
```

## What each script changes

| Script | Output |
|---|---|
| `reextract_landmarks.py` | `data/asl_landmarks_train.csv` |
| `train_landmark_nn.py` | `data/nn_landmark_model.keras`, scaler, label encoder, `results/landmark_training.json` |
| `generate_hand_crops.py` | `data/asl_hand_crops/` + manifest |
| `train_resnet_improved.py` | `models/best_asl_resnet50_phase2.h5` |
| `convert_models.py` | `web/public/models/resnet-graph/`, `landmark-nn/`, `preprocessing.json` |
| `convert_models.py --landmark-only` | Landmark NN + preprocessing only (ResNet unchanged) |
| `evaluate_models.py` | `results/evaluation_results.json` + sync to `web/public/evaluation-results.json` |
| `run_landmark_pipeline.py` | Landmark-only train → export → evaluate → docs → build |

See [RESULTS.md](RESULTS.md) for latest metrics.
