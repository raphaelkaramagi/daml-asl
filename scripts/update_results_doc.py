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


def main() -> None:
    if not EVAL_PATH.exists():
        print(f"No evaluation results at {EVAL_PATH}")
        return

    with open(EVAL_PATH) as f:
        m = json.load(f)

    landmark_section = "| Metric | Value |\n|---|---|\n"
    if LANDMARK_TRAINING_PATH.exists():
        with open(LANDMARK_TRAINING_PATH) as f:
            lt = json.load(f)
        landmark_section = f"""| Metric | Value |
|---|---|
| Training samples | {lt.get('samples', '—')} |
| Feature noise std | {lt.get('noiseStd', '—')} |
| Best val accuracy | **{lt.get('bestValAccuracy', '—')}%** |
| Held-out test accuracy | **{lt.get('testAccuracy', '—')}%** |
"""
    else:
        landmark_section += "| (run train_landmark_nn.py) | — |"

    no_det = m.get("noDetectionClasses", [])
    no_det_note = f"`{', '.join(no_det)}`" if no_det else "—"

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

{landmark_section}

Training uses wrist-relative 63-d features with Gaussian noise augmentation for live robustness.
End-to-end accuracy is limited by MediaPipe detection rate on the 28-photo test set.

## ResNet50

| Metric | Before | After |
|---|---|---|
| Val accuracy (phase 2) | 48.06% | See training checkpoint |
| Test accuracy (28 photos) | 67.9% | **{m['resnetEndToEnd']:.1f}%** ({m['resnetCorrect']}/{m['testImages']}) |

Training uses hand-cropped 96×96 images, no horizontal flip, 3-phase schedule with early stopping on `val_accuracy`.

## Fair evaluation (28-photo test set)

Evaluated with `scripts/evaluate_models.py` using shared detection (`scripts/mediapipe_detect.py`, conf={m.get('detectionConfidence', 0.2)}).

| Model | Detection rate | Given detection | End-to-end |
|---|---|---|---|
| ResNet50 | {m['detectionRate']:.1f}%* | — | **{m['resnetEndToEnd']:.1f}%** |
| Landmark NN | {m['detectionRate']:.1f}% | **{m['landmarkAccuracyGivenDetection']:.1f}%** | **{m['landmarkEndToEnd']:.1f}%** |

> \\* Detection rate is shared — both models use the same MediaPipe hand detection for fair comparison.

Classes with no detection on test photos: {no_det_note}

### Previous baseline

| Model | End-to-end | Given detection |
|---|---|---|
| ResNet50 | 67.9% | — |
| Landmark NN | 53.6% | 100% |

## Web inference

Browser detection matches Python training/eval:

- IMAGE-mode multi-scale retry (1×, 1.5×, 2×) with upscale to 300px minimum
- Hand-crop before ResNet when landmarks found
- Temporal hold on webcam; pauses when scrolled off-screen
- Configurable detection confidence (Settings, default 0.2)

**ResNet50** is recommended for live demo. **Landmark NN** depends on MediaPipe detecting a hand first.

Results are synced to `web/public/evaluation-results.json` for the demo evaluation banner.
"""
    RESULTS_MD.write_text(content)
    print(f"Updated {RESULTS_MD}")


if __name__ == "__main__":
    main()
