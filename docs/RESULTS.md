# Model Results

Metrics from the improved training pipeline (`scripts/mediapipe_detect.py` + hand-cropped ResNet).

Canonical evaluation JSON: [`results/evaluation_results.json`](../results/evaluation_results.json)

## Dataset extraction (MediaPipe)

| Metric | Before | After |
|---|---|---|
| Detection rate (87k train) | 73.2% | **78.7%** |
| Landmark rows used for training | 63,616 | **68,429** |

Per-class detection improved for most letters. Weak classes remain `nothing` (0.7%, expected — no hand) and `N` (55.1%).

## Landmark neural network

### Deployed in web demo (original weights)

| Metric | Value |
|---|---|
| Val accuracy | **98.97%** |
| Feature test split | **98.88%** |
| Parameters | **~18K** |
| TF.js size | **~72 KB** |

### Latest retrain pipeline (`results/landmark_training.json`)

| Metric | Value |
|---|---|
| Training samples | 68,429 |
| Feature noise std | 0.015 |
| Best val accuracy | **95.7%** |
| Held-out test accuracy | **95.7%** |

The live site uses the **original landmark weights** for webcam parity with unmirrored Kaggle training data. End-to-end accuracy on the 28-photo test set is limited by MediaPipe detection rate, not classifier error.

## ResNet50

| Metric | Before retrain | After retrain |
|---|---|---|
| Val accuracy (phase 2) | 48.06% | See training checkpoint |
| End-to-end (28 photos) | 67.9% (19/28) | **96.4%** (27/28) |

Training uses hand-cropped 96×96 images, no horizontal flip, 3-phase schedule with early stopping on `val_accuracy`.

## Fair evaluation (28-photo test set)

Evaluated with `scripts/evaluate_models.py` using shared detection (`scripts/mediapipe_detect.py`, conf=0.2). ResNet uses hand-crop when landmarks are found; otherwise full frame.

| Model | Detection rate | Given detection | End-to-end |
|---|---|---|---|
| ResNet50 | 67.9%* | — | **96.4%** (27/28) |
| Landmark NN | 67.9% | **100.0%** | **67.9%** (19/28) |

> \* Detection rate is shared — both models use the same MediaPipe hand detection for fair comparison.

Classes with no detection on test photos: `A, C, D, E, N, V, X, Z, nothing`

### Previous baseline (pre-retrain ResNet)

| Model | End-to-end | Given detection |
|---|---|---|
| ResNet50 | 67.9% | — |
| Landmark NN | 53.6% | 100% |

## Web inference

Browser deployment:

- **ResNet50** — TF.js **float32 graph-model** at `web/public/models/resnet-graph/` (~91 MB). Keras 3 layers export and uint8 graph quant were unreliable in the browser.
- **Landmark NN** — original TF.js weights at `web/public/models/landmark-nn/` (~72 KB) for live parity with unmirrored training data.
- **Webcam** — MediaPipe on raw unmirrored video pixels; CSS mirrors the preview and landmark x-coordinates are flipped in the overlay only.
- **Gallery/upload** — single-pass IMAGE-mode `detectHandFromImage`; small images (&lt;300px) are upscaled before detection.
- **ResNet webcam path** — full-frame resize to 96×96 (matches pre-accuracy-work behavior).
- **Settings** — detection confidence default **0.3**; toggle models on/off.

**ResNet50** is recommended for live demo (96.4% end-to-end on the 28-photo test set). **Landmark NN** is 100% accurate when MediaPipe detects a hand, but end-to-end accuracy is capped by the 67.9% detection rate.

Results are synced to `web/public/evaluation-results.json` for the demo evaluation banner.
