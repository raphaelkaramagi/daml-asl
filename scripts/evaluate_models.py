#!/usr/bin/env python3
"""
Fair model evaluation on the 28-photo test set using shared MediaPipe detection.

Reports:
- Detection rate (Landmark path)
- Classification accuracy given detection
- End-to-end accuracy
- Per-image predictions for both models

Usage:
    python scripts/evaluate_models.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mediapipe_detect import detect_hand, crop_hand, DEFAULT_CONFIDENCE

TEST_DIR = PROJECT_ROOT / "data" / "asl_alphabet_test" / "asl_alphabet_test"
RESNET_MODEL = PROJECT_ROOT / "models" / "best_asl_resnet50_phase2.h5"
LANDMARK_MODEL = PROJECT_ROOT / "data" / "nn_landmark_model.keras"
LABEL_ENCODER = PROJECT_ROOT / "data" / "label_encoder.joblib"
SCALER = PROJECT_ROOT / "data" / "scaler.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "evaluation_results.json"
WEB_RESULTS_JSON = PROJECT_ROOT / "web" / "public" / "evaluation-results.json"

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "del", "nothing", "space",
]


def load_test_images():
    samples = []
    for fname in sorted(os.listdir(TEST_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        label = fname.replace("_test.jpg", "").replace(".jpg", "")
        path = TEST_DIR / fname
        samples.append((label, path))
    return samples


def sync_web_results(results: dict) -> None:
    """Copy evaluation JSON to web public for the demo banner."""
    WEB_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RESULTS_JSON, WEB_RESULTS_JSON)


def evaluate():
    import joblib
    from tensorflow import keras

    resnet = keras.models.load_model(RESNET_MODEL, compile=False)
    landmark = keras.models.load_model(LANDMARK_MODEL, compile=False)
    encoder = joblib.load(LABEL_ENCODER)
    scaler = joblib.load(SCALER)

    samples = load_test_images()
    resnet_true, resnet_pred = [], []
    lm_true, lm_pred_detected = [], []
    lm_end_to_end_true, lm_end_to_end_pred = [], []
    per_image = []
    detected_count = 0

    print(f"Evaluating {len(samples)} test images (detection conf={DEFAULT_CONFIDENCE})...\n")

    for true_label, path in samples:
        image = cv2.imread(str(path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hand = detect_hand(image_rgb, conf=DEFAULT_CONFIDENCE)
        detected = hand is not None
        if detected:
            detected_count += 1

        # ResNet (hand-cropped when possible)
        if hand is not None:
            img_resnet = crop_hand(image_rgb, hand.landmarks)
        else:
            img_resnet = image_rgb
        img_resnet = cv2.resize(img_resnet, (96, 96)) / 255.0
        pred = resnet.predict(np.expand_dims(img_resnet, 0), verbose=0)
        resnet_label = CLASS_NAMES[int(np.argmax(pred[0]))]
        resnet_true.append(true_label)
        resnet_pred.append(resnet_label)

        # Landmark NN
        lm_end_to_end_true.append(true_label)
        if hand is not None:
            features = scaler.transform([hand.features])
            lp = landmark.predict(features, verbose=0)
            lm_label = encoder.inverse_transform([int(np.argmax(lp[0]))])[0]
            lm_true.append(true_label)
            lm_pred_detected.append(lm_label)
            lm_end_to_end_pred.append(lm_label)
        else:
            lm_label = "NO_DETECTION"
            lm_end_to_end_pred.append(lm_label)

        per_image.append({
            "true": true_label,
            "detected": detected,
            "resnetPred": resnet_label,
            "resnetCorrect": resnet_label == true_label,
            "landmarkPred": lm_label,
            "landmarkCorrect": lm_label == true_label if detected else False,
        })

    resnet_correct = sum(t == p for t, p in zip(resnet_true, resnet_pred))
    resnet_e2e = resnet_correct / len(samples)

    print("=== ResNet50 ===")
    print(classification_report(resnet_true, resnet_pred, zero_division=0))
    print(f"\nResNet end-to-end accuracy: {resnet_e2e:.1%} ({resnet_correct}/{len(samples)})")

    print("\n=== Landmark NN (given detection) ===")
    acc_given = 0.0
    if lm_true:
        acc_given = sum(t == p for t, p in zip(lm_true, lm_pred_detected)) / len(lm_true)
        print(f"Detection rate: {detected_count}/{len(samples)} ({100*detected_count/len(samples):.1f}%)")
        print(f"Accuracy given detection: {acc_given:.1%}")
        print(classification_report(lm_true, lm_pred_detected, zero_division=0))
    else:
        print("No hands detected.")

    lm_correct = sum(
        1 for row in per_image if row["landmarkCorrect"]
    )
    e2e_acc = lm_correct / len(samples)
    print(f"\nLandmark end-to-end accuracy: {e2e_acc:.1%} ({lm_correct}/{len(samples)})")

    missed = [r["true"] for r in per_image if not r["detected"]]
    if missed:
        print(f"\nNo detection: {', '.join(missed)}")

    results = {
        "testImages": len(samples),
        "detectionConfidence": DEFAULT_CONFIDENCE,
        "detectionRate": round(100 * detected_count / len(samples), 2),
        "resnetEndToEnd": round(100 * resnet_e2e, 2),
        "resnetCorrect": resnet_correct,
        "landmarkAccuracyGivenDetection": round(100 * acc_given, 2),
        "landmarkEndToEnd": round(100 * e2e_acc, 2),
        "landmarkCorrect": lm_correct,
        "noDetectionClasses": missed,
        "resnetPredictions": [
            {"true": r["true"], "pred": r["resnetPred"], "correct": r["resnetCorrect"]}
            for r in per_image
        ],
        "landmarkPredictions": [
            {
                "true": r["true"],
                "pred": r["landmarkPred"],
                "detected": r["detected"],
                "correct": r["landmarkCorrect"],
            }
            for r in per_image
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    sync_web_results(results)
    print(f"\nResults saved to {RESULTS_JSON}")
    print(f"Synced to {WEB_RESULTS_JSON}")
    return results


if __name__ == "__main__":
    evaluate()
