#!/usr/bin/env python3
"""Update docs/RESULTS.md from results/evaluation_results.json and landmark training summary."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
EVAL_PATH = RESULTS_DIR / "evaluation_results.json"
LANDMARK_TRAINING_PATH = RESULTS_DIR / "landmark_training.json"
RESULTS_MD = PROJECT_ROOT / "docs" / "RESULTS.md"

# Deployed web demo uses original landmark weights (pre-retrain)
DEPLOYED_LANDMARK_VAL = 98.97
DEPLOYED_LANDMARK_TEST = 98.88


def main() -> None:
    if not EVAL_PATH.exists():
        print(f"No evaluation results at {EVAL_PATH}")
        return

    with open(EVAL_PATH) as f:
        m = json.load(f)

    retrain_rows = ""
    if LANDMARK_TRAINING_PATH.exists():
        with open(LANDMARK_TRAINING_PATH) as f:
            lt = json.load(f)
        retrain_rows = f"""| Training samples | {lt.get('samples', '—')} |
| Feature noise std | {lt.get('noiseStd', '—')} |
| Best val accuracy | **{lt.get('bestValAccuracy', '—')}%** |
| Held-out test accuracy | **{lt.get('testAccuracy', '—')}%** |"""

    no_det = m.get("noDetectionClasses", [])
    no_det_note = f"`{', '.join(no_det)}`" if no_det else "—"
    conf = m.get("detectionConfidence", 0.2)

    content = f"""# Model Results

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
| Val accuracy | **{DEPLOYED_LANDMARK_VAL:.2f}%** |
| Feature test split | **{DEPLOYED_LANDMARK_TEST:.2f}%** |
| Parameters | **~18K** |
| TF.js size | **~72 KB** |

### Latest retrain pipeline (`results/landmark_training.json`)

| Metric | Value |
|---|---|
{retrain_rows or "| (run train_landmark_nn.py) | — |"}

The live site uses the **original landmark weights** for webcam parity with unmirrored Kaggle training data. End-to-end accuracy on the 28-photo test set is limited by MediaPipe detection rate, not classifier error.

## ResNet50

| Metric | Before retrain | After retrain |
|---|---|---|
| Val accuracy (phase 2) | 48.06% | See training checkpoint |
| End-to-end (28 photos) | 67.9% (19/28) | **{m['resnetEndToEnd']:.1f}%** ({m['resnetCorrect']}/{m['testImages']}) |

Training uses hand-cropped 96×96 images, no horizontal flip, 3-phase schedule with early stopping on `val_accuracy`.

## Fair evaluation (28-photo test set)

Evaluated with `scripts/evaluate_models.py` using shared detection (`scripts/mediapipe_detect.py`, conf={conf}). ResNet uses hand-crop when landmarks are found; otherwise full frame.

| Model | Detection rate | Given detection | End-to-end |
|---|---|---|---|
| ResNet50 | {m['detectionRate']:.1f}%* | — | **{m['resnetEndToEnd']:.1f}%** ({m['resnetCorrect']}/{m['testImages']}) |
| Landmark NN | {m['detectionRate']:.1f}% | **{m['landmarkAccuracyGivenDetection']:.1f}%** | **{m['landmarkEndToEnd']:.1f}%** ({m['landmarkCorrect']}/{m['testImages']}) |

> \\* Detection rate is shared — both models use the same MediaPipe hand detection for fair comparison.

Classes with no detection on test photos: {no_det_note}

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
- **Gallery/upload** — single-pass IMAGE-mode `detectHandFromImage`; small images (<300px) are upscaled before detection.
- **ResNet webcam path** — full-frame resize to 96×96 (matches pre-accuracy-work behavior).
- **Settings** — detection confidence default **0.3**; toggle models on/off.

**ResNet50** is recommended for live demo ({m['resnetEndToEnd']:.1f}% end-to-end on the 28-photo test set). **Landmark NN** is {m['landmarkAccuracyGivenDetection']:.0f}% accurate when MediaPipe detects a hand, but end-to-end accuracy is capped by the {m['detectionRate']:.1f}% detection rate.

Results are synced to `web/public/evaluation-results.json` for the demo evaluation banner.
"""
    RESULTS_MD.write_text(content)
    print(f"Updated {RESULTS_MD}")


if __name__ == "__main__":
    main()
