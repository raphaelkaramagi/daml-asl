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

| Metric | Value |
|---|---|
| Training samples | 68429 |
| Feature noise std | 0.015 |
| Best val accuracy | **95.7%** |
| Held-out test accuracy | **95.7%** |


Training uses wrist-relative 63-d features with Gaussian noise augmentation for live robustness.
End-to-end accuracy is limited by MediaPipe detection rate on the 28-photo test set.

## ResNet50

| Metric | Before | After |
|---|---|---|
| Val accuracy (phase 2) | 48.06% | See training checkpoint |
| Test accuracy (28 photos) | 67.9% | **96.4%** (27/28) |

Training uses hand-cropped 96×96 images, no horizontal flip, 3-phase schedule with early stopping on `val_accuracy`.

## Fair evaluation (28-photo test set)

Evaluated with `scripts/evaluate_models.py` using shared detection (`scripts/mediapipe_detect.py`, conf=0.2).

| Model | Detection rate | Given detection | End-to-end |
|---|---|---|---|
| ResNet50 | 67.9%* | — | **96.4%** |
| Landmark NN | 67.9% | **100.0%** | **67.9%** |

> \* Detection rate is shared — both models use the same MediaPipe hand detection for fair comparison.

Classes with no detection on test photos: `A, C, D, E, N, V, X, Z, nothing`

### Previous baseline

| Model | End-to-end | Given detection |
|---|---|---|
| ResNet50 | 67.9% | — |
| Landmark NN | 53.6% | 100% |

## Web inference

Browser deployment uses training-parity detection:

- **ResNet50** loads as a TF.js **layers-model** (`web/public/models/resnet/`) — the uint8 graph export was unreliable in WebGL.
- **Landmark NN** uses pre-session weights (`adaf4d5`) for live parity with unmirrored Kaggle training data.
- Webcam: MediaPipe runs on **raw unmirrored** video pixels; the preview is CSS-mirrored and the skeleton overlay flips x for display only.
- Gallery/upload: single-pass IMAGE-mode `detectHandFromImage` (no multi-scale upscale).
- Hand-crop before ResNet when landmarks are found.
- Configurable detection confidence (Settings, default **0.5**).

**ResNet50** is recommended for live demo. **Landmark NN** depends on MediaPipe detecting a hand first.

Results are synced to `web/public/evaluation-results.json` for the demo evaluation banner.
