#!/usr/bin/env python3
"""
Convert trained Keras models to TensorFlow.js format for browser inference.
Also exports scaler and label encoder as JSON.

Usage:
    pip install tensorflowjs
    python scripts/convert_models.py
"""

import os
import shutil
import sys
import tempfile
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_MODELS_DIR = os.path.join(PROJECT_ROOT, "web", "public", "models")

LANDMARK_MODEL = os.path.join(PROJECT_ROOT, "data", "nn_landmark_model.keras")
RESNET_MODEL = os.path.join(PROJECT_ROOT, "models", "best_asl_resnet50_phase2.h5")
LABEL_ENCODER = os.path.join(PROJECT_ROOT, "data", "label_encoder.joblib")
SCALER = os.path.join(PROJECT_ROOT, "data", "scaler.joblib")
RESNET_GRAPH_OUT = os.path.join(WEB_MODELS_DIR, "resnet-graph")
RESNET_LAYERS_OUT = os.path.join(WEB_MODELS_DIR, "resnet")


def fix_keras3_layers_model_json(model_json_path: str) -> None:
    """Patch Keras 3 TF.js export for @tensorflow/tfjs layer deserialization."""
    with open(model_json_path, encoding="utf-8") as f:
        topology = json.load(f)

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("class_name") == "InputLayer" and "config" in obj:
                cfg = obj["config"]
                if "batch_shape" in cfg and "batch_input_shape" not in cfg:
                    cfg["batch_input_shape"] = cfg.pop("batch_shape")
            if obj.get("class_name") == "Functional":
                obj["class_name"] = "Model"
            if isinstance(obj.get("dtype"), dict):
                dtype_name = obj["dtype"].get("config", {}).get("name")
                if dtype_name:
                    obj["dtype"] = dtype_name
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(topology)
    with open(model_json_path, "w", encoding="utf-8") as f:
        json.dump(topology, f)
    print(f"  -> Patched Keras 3 layers JSON for TF.js: {model_json_path}")


def convert_resnet_layers_model():
    """Convert ResNet50 to TF.js layers model (browser-compatible, no uint8 graph quant)."""
    import tensorflow as tf

    if not os.path.exists(RESNET_MODEL):
        raise FileNotFoundError(f"ResNet model not found: {RESNET_MODEL}")

    print(f"Converting ResNet layers model: {RESNET_MODEL}")
    model = tf.keras.models.load_model(RESNET_MODEL, compile=False)

    if os.path.exists(RESNET_LAYERS_OUT):
        shutil.rmtree(RESNET_LAYERS_OUT)
    os.makedirs(RESNET_LAYERS_OUT, exist_ok=True)

    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, RESNET_LAYERS_OUT)
        fix_keras3_layers_model_json(os.path.join(RESNET_LAYERS_OUT, "model.json"))
        print(f"  -> Saved layers model to {RESNET_LAYERS_OUT}")
        return
    except Exception as exc:
        print(f"  Layers conversion via Python API failed: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = os.path.join(tmp, "saved_model")
        tf.saved_model.save(model, saved_model_dir)
        subprocess.run([
            sys.executable, "-m", "tensorflowjs.converters.converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_layers_model",
            "--saved_model_tags=serve",
            saved_model_dir, RESNET_LAYERS_OUT,
        ], check=True)
        fix_keras3_layers_model_json(os.path.join(RESNET_LAYERS_OUT, "model.json"))
        print(f"  -> Saved layers model to {RESNET_LAYERS_OUT}")


def convert_landmark_model():
    """Convert the lightweight landmark NN to TF.js layers format."""
    script = os.path.join(PROJECT_ROOT, "scripts", "export_landmark_tfjs.py")
    print(f"Converting landmark model: {LANDMARK_MODEL}")
    subprocess.run([sys.executable, script], check=True)
    print(f"  -> Saved to {os.path.join(WEB_MODELS_DIR, 'landmark-nn')}")


def convert_resnet_graph_model():
    """Convert ResNet50 to TF.js graph model (float32, browser-compatible)."""
    import tensorflow as tf

    if not os.path.exists(RESNET_MODEL):
        raise FileNotFoundError(f"ResNet model not found: {RESNET_MODEL}")

    print(f"Converting ResNet graph model: {RESNET_MODEL}")
    model = tf.keras.models.load_model(RESNET_MODEL, compile=False)

    if os.path.exists(RESNET_GRAPH_OUT):
        shutil.rmtree(RESNET_GRAPH_OUT)
    os.makedirs(RESNET_GRAPH_OUT, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = os.path.join(tmp, "saved_model")
        tf.saved_model.save(model, saved_model_dir)

        try:
            from tensorflowjs.converters import tf_saved_model_conversion_v2 as converter
            converter.convert_tf_saved_model(saved_model_dir, RESNET_GRAPH_OUT)
            print(f"  -> Saved graph model to {RESNET_GRAPH_OUT}")
            return
        except Exception as exc:
            print(f"  Graph conversion via Python API failed: {exc}")
            print("  Falling back to tensorflowjs_converter CLI...")

        subprocess.run([
            sys.executable, "-m", "tensorflowjs.converters.converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            "--saved_model_tags=serve",
            saved_model_dir, RESNET_GRAPH_OUT,
        ], check=True)
        print(f"  -> Saved graph model to {RESNET_GRAPH_OUT}")


def convert_resnet_graph_model_quantized():
    """Convert ResNet50 to TF.js graph model with uint8 quantization (optional)."""
    import tensorflow as tf

    if not os.path.exists(RESNET_MODEL):
        raise FileNotFoundError(f"ResNet model not found: {RESNET_MODEL}")

    print(f"Converting ResNet model: {RESNET_MODEL}")
    model = tf.keras.models.load_model(RESNET_MODEL, compile=False)

    if os.path.exists(RESNET_GRAPH_OUT):
        shutil.rmtree(RESNET_GRAPH_OUT)
    os.makedirs(RESNET_GRAPH_OUT, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        saved_model_dir = os.path.join(tmp, "saved_model")
        tf.saved_model.save(model, saved_model_dir)

        try:
            from tensorflowjs.converters import tf_saved_model_conversion_v2 as converter
            converter.convert_tf_saved_model(
                saved_model_dir,
                RESNET_GRAPH_OUT,
                quantization_dtype="uint8",
            )
            print(f"  -> Saved graph model to {RESNET_GRAPH_OUT}")
            return
        except Exception as exc:
            print(f"  Graph conversion via Python API failed: {exc}")
            print("  Falling back to tensorflowjs_converter CLI...")

        subprocess.run([
            sys.executable, "-m", "tensorflowjs.converters.converter",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            "--quantize_uint8",
            "--saved_model_tags=serve",
            saved_model_dir, RESNET_GRAPH_OUT,
        ], check=True)
        print(f"  -> Saved graph model to {RESNET_GRAPH_OUT}")


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
    import argparse

    parser = argparse.ArgumentParser(description="Convert models to TF.js for web deployment")
    parser.add_argument(
        "--landmark-only",
        action="store_true",
        help="Export landmark NN and preprocessing only (skip ResNet conversion)",
    )
    parser.add_argument(
        "--resnet-only",
        action="store_true",
        help="Export ResNet layers model only (skip landmark conversion)",
    )
    args = parser.parse_args()

    if args.resnet_only:
        if os.path.exists(RESNET_MODEL):
            try:
                convert_resnet_graph_model()
            except Exception as e:
                print(f"  ResNet graph conversion failed: {e}")
                sys.exit(1)
        else:
            print(f"ERROR: ResNet model not found at {RESNET_MODEL}")
            sys.exit(1)
        print("\nResNet-only export complete.")
        return

    if not os.path.exists(LANDMARK_MODEL):
        print(f"ERROR: Landmark model not found at {LANDMARK_MODEL}")
        print("Train the model first: python scripts/train_landmark_nn.py")
        sys.exit(1)

    export_preprocessing()

    try:
        convert_landmark_model()
    except Exception as e:
        print(f"  Landmark conversion failed: {e}")
        print("  Install tensorflowjs: pip install tensorflowjs")
        sys.exit(1)

    if args.landmark_only:
        print("\nLandmark-only export complete.")
        return

    if os.path.exists(RESNET_MODEL):
        try:
            convert_resnet_graph_model()
        except Exception as e:
            print(f"  ResNet graph conversion failed: {e}")
    else:
        print(f"WARNING: ResNet model not found at {RESNET_MODEL}, skipping.")


if __name__ == "__main__":
    main()
