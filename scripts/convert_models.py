#!/usr/bin/env python3
"""
Convert trained Keras models to TensorFlow.js format for browser inference.
Also exports scaler and label encoder as JSON.

Usage:
    pip install tensorflowjs
    python scripts/convert_models.py
"""

import os
import sys
import json
import subprocess
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_MODELS_DIR = os.path.join(PROJECT_ROOT, "web", "public", "models")

LANDMARK_MODEL = os.path.join(PROJECT_ROOT, "data", "nn_landmark_model.keras")
RESNET_MODEL = os.path.join(PROJECT_ROOT, "models", "best_asl_resnet50_phase2.h5")
LABEL_ENCODER = os.path.join(PROJECT_ROOT, "data", "label_encoder.joblib")
SCALER = os.path.join(PROJECT_ROOT, "data", "scaler.joblib")


def convert_landmark_model():
    """Convert the lightweight landmark NN to TF.js layers format."""
    out_dir = os.path.join(WEB_MODELS_DIR, "landmark-nn")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting landmark model: {LANDMARK_MODEL}")
    subprocess.run([
        sys.executable, "-m", "tensorflowjs.converters.keras_h5_to_tfjs",
        LANDMARK_MODEL, out_dir
    ], check=True)
    print(f"  -> Saved to {out_dir}")


def convert_resnet_model():
    """Convert ResNet50 to TF.js with uint8 quantization."""
    out_dir = os.path.join(WEB_MODELS_DIR, "resnet")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Converting ResNet model: {RESNET_MODEL}")
    subprocess.run([
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
        "--quantize_uint8",
        RESNET_MODEL, out_dir
    ], check=True)
    print(f"  -> Saved to {out_dir}")


def export_preprocessing():
    """Export scaler parameters and label encoder classes as JSON."""
    import joblib

    out_path = os.path.join(WEB_MODELS_DIR, "preprocessing.json")
    os.makedirs(WEB_MODELS_DIR, exist_ok=True)

    scaler = joblib.load(SCALER)
    encoder = joblib.load(LABEL_ENCODER)

    data = {
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "classes": encoder.classes_.tolist(),
    }

    with open(out_path, "w") as f:
        json.dump(data, f)
    print(f"  -> Preprocessing JSON saved to {out_path}")


def main():
    if not os.path.exists(LANDMARK_MODEL):
        print(f"ERROR: Landmark model not found at {LANDMARK_MODEL}")
        print("Train the model first using notebook 06.")
        sys.exit(1)

    export_preprocessing()

    try:
        convert_landmark_model()
    except Exception as e:
        print(f"  Landmark conversion failed: {e}")
        print("  Install tensorflowjs: pip install tensorflowjs")

    if os.path.exists(RESNET_MODEL):
        try:
            convert_resnet_model()
        except Exception as e:
            print(f"  ResNet conversion failed: {e}")
    else:
        print(f"WARNING: ResNet model not found at {RESNET_MODEL}, skipping.")


if __name__ == "__main__":
    main()
