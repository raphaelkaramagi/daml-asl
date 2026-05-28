#!/usr/bin/env python3
"""
Train the landmark neural network on re-extracted CSV features.

Usage:
    python scripts/reextract_landmarks.py
    python scripts/train_landmark_nn.py
    python scripts/train_landmark_nn.py --noise-std 0.015 --epochs 80
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "asl_landmarks_train.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "nn_landmark_model.keras"
ENCODER_PATH = PROJECT_ROOT / "data" / "label_encoder.joblib"
SCALER_PATH = PROJECT_ROOT / "data" / "scaler.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"
TRAINING_JSON = RESULTS_DIR / "landmark_training.json"


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected — training on CPU.")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"Using GPU: {gpus[0].name}")


def build_model(num_classes: int) -> keras.Model:
    return keras.Sequential([
        keras.layers.Input(shape=(63,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(num_classes, activation="softmax"),
    ])


def train(
    csv_path: Path = CSV_PATH,
    epochs: int = 80,
    noise_std: float = 0.015,
    batch_size: int = 64,
) -> dict:
    df = pd.read_csv(csv_path)
    if "source_path" in df.columns:
        feature_cols = [c for c in df.columns if c.startswith("lm_")]
        df_clean = df.dropna(subset=feature_cols)
    else:
        df_clean = df.dropna()

    X = df_clean[[c for c in df_clean.columns if c.startswith("lm_")]].values.astype(np.float32)
    y = df_clean["label"].values
    print(f"Samples: {len(X)} | Features: {X.shape[1]} | Classes: {len(np.unique(y))}")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_scaled, y_train))
        .shuffle(len(X_train_scaled), seed=42)
        .batch(batch_size)
        .map(
            lambda x, y: (
                x + tf.random.normal(tf.shape(x), stddev=noise_std) if noise_std > 0 else x,
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test)).batch(batch_size)

    num_classes = len(encoder.classes_)
    model = build_model(num_classes)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=4, min_lr=1e-6,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    val_acc = max(history.history["val_accuracy"])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Best val accuracy: {val_acc:.2%}")
    print(f"Test accuracy: {test_acc:.2%}")

    summary = {
        "samples": len(X),
        "trainSamples": len(X_train),
        "testSamples": len(X_test),
        "noiseStd": noise_std,
        "epochsRun": len(history.history["loss"]),
        "bestValAccuracy": round(float(val_acc) * 100, 2),
        "testAccuracy": round(float(test_acc) * 100, 2),
        "finalTrainAccuracy": round(float(history.history["accuracy"][-1]) * 100, 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to {TRAINING_JSON}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Train landmark NN on CSV features")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--noise-std", type=float, default=0.015, help="Feature noise augmentation")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    configure_gpu()
    train(
        csv_path=args.csv,
        epochs=args.epochs,
        noise_std=args.noise_std,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
