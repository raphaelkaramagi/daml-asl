#!/usr/bin/env python3
"""Export landmark NN to TF.js layers format compatible with @tensorflow/tfjs in browser."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "data" / "nn_landmark_model.keras"
OUT_DIR = PROJECT_ROOT / "web" / "public" / "models" / "landmark-nn"


def export_model() -> None:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    layers: list[dict] = []
    for layer in model.layers:
        cfg: dict = {"name": layer.name, "trainable": True, "dtype": "float32"}
        if isinstance(layer, tf.keras.layers.Dense):
            if not layers:
                cfg["batch_input_shape"] = [None, 63]
            cfg.update(
                {
                    "units": layer.units,
                    "activation": layer.get_config()["activation"],
                    "use_bias": True,
                    "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": None}},
                    "bias_initializer": {"class_name": "Zeros", "config": {}},
                    "kernel_regularizer": None,
                    "bias_regularizer": None,
                    "activity_regularizer": None,
                    "kernel_constraint": None,
                    "bias_constraint": None,
                }
            )
            layers.append({"class_name": "Dense", "config": cfg})
        elif isinstance(layer, tf.keras.layers.Dropout):
            cfg.update({"rate": layer.rate, "seed": None, "noise_shape": None})
            layers.append({"class_name": "Dropout", "config": cfg})

    weight_entries: list[tuple[str, np.ndarray]] = []
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        weight_entries.append((f"{layer.name}/kernel", weights[0].astype(np.float32)))
        if len(weights) > 1:
            weight_entries.append((f"{layer.name}/bias", weights[1].astype(np.float32)))

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    shard_path = "group1-shard1of1.bin"
    manifest_weights = []
    with open(OUT_DIR / shard_path, "wb") as f:
        for name, arr in weight_entries:
            manifest_weights.append({"name": name, "shape": list(arr.shape), "dtype": "float32"})
            f.write(arr.tobytes())

    model_json = {
        "format": "layers-model",
        "generatedBy": "scripts/export_landmark_tfjs.py",
        "convertedBy": "TensorFlow.js Converter v4.22.0",
        "modelTopology": {
            "class_name": "Sequential",
            "config": {"name": "sequential", "layers": layers},
        },
        "weightsManifest": [{"paths": [shard_path], "weights": manifest_weights}],
    }

    with open(OUT_DIR / "model.json", "w") as f:
        json.dump(model_json, f)

    print(f"Exported {len(weight_entries)} tensors to {OUT_DIR}")
    print(f"First weight: {manifest_weights[0]['name']}")


if __name__ == "__main__":
    export_model()
