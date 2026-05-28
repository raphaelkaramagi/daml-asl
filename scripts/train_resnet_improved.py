#!/usr/bin/env python3
"""
Improved ResNet50 training for ASL alphabet recognition.

Changes vs original notebook 05:
- No horizontal flip (ASL signs are handedness-sensitive)
- RandomBrightness augmentation
- Adapted ResNet50 stem for 96x96 (3x3 conv stride 1, no maxpool)
- 3-phase training schedule with val_accuracy monitoring
- Optional hand-cropped dataset from scripts/generate_hand_crops.py

Usage:
    python scripts/generate_hand_crops.py   # optional but recommended
    python scripts/train_resnet_improved.py
    python scripts/train_resnet_improved.py --data-dir data/asl_alphabet_train/asl_alphabet_train
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_DIR = PROJECT_ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
CROPS_DIR = PROJECT_ROOT / "data" / "asl_hand_crops"
MODELS_DIR = PROJECT_ROOT / "models"

IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 123
NUM_CLASSES = 29


def configure_gpu() -> None:
    """Enable Apple Metal / GPU memory growth when available."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected — training will use CPU (much slower).")
        print("Apple Silicon: pip install tensorflow-metal")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"Using GPU: {gpus[0].name}")


def build_model() -> tuple[keras.Model, keras.Model]:
    """Build ResNet50 classifier. Returns model and frozen backbone reference."""
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3),
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs, name="asl_resnet50_improved")
    return model, base_model


def make_augmentation() -> keras.Sequential:
    return keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(factor=0.1),
    ], name="data_augmentation")


def load_datasets(data_dir: Path):
    train_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
    )
    return train_ds, val_ds


def prepare_datasets(train_ds, val_ds):
    aug = make_augmentation()
    norm = layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return (
        train_ds.cache().prefetch(tf.data.AUTOTUNE),
        val_ds.cache().prefetch(tf.data.AUTOTUNE),
    )


def make_callbacks(phase_name: str, filepath: Path) -> list:
    return [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            str(filepath),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


def train(data_dir: Path):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds = load_datasets(data_dir)
    train_ds, val_ds = prepare_datasets(train_ds, val_ds)

    model, base_model = build_model()

    # Phase 1: frozen backbone
    print("\n=== Phase 1: Head only (frozen backbone) ===")
    model.compile(optimizer=optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=make_callbacks("phase1", MODELS_DIR / "best_asl_resnet50_phase1.h5"),
    )

    # Phase 2: unfreeze top layers
    print("\n=== Phase 2: Fine-tune top blocks ===")
    base_model.trainable = True
    freeze_until = max(0, len(base_model.layers) - 30)
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    model.compile(optimizer=optimizers.Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=make_callbacks("phase2", MODELS_DIR / "best_asl_resnet50_phase2.h5"),
    )

    # Phase 3: full fine-tune
    print("\n=== Phase 3: Full fine-tune (differential LR) ===")
    for layer in base_model.layers:
        layer.trainable = True

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=make_callbacks("phase3", MODELS_DIR / "best_asl_resnet50_phase2.h5"),
    )

    final_path = MODELS_DIR / "asl_resnet50_final.keras"
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")
    print(f"Best phase-2 checkpoint: {MODELS_DIR / 'best_asl_resnet50_phase2.h5'}")


def main():
    parser = argparse.ArgumentParser(description="Train improved ResNet50 for ASL")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Training data directory (default: hand crops if available, else full dataset)",
    )
    args = parser.parse_args()

    configure_gpu()

    if args.data_dir:
        data_dir = args.data_dir
    elif CROPS_DIR.exists() and any(CROPS_DIR.iterdir()):
        data_dir = CROPS_DIR
        print(f"Using hand-cropped dataset: {CROPS_DIR}")
    else:
        data_dir = DEFAULT_TRAIN_DIR
        print(f"Using full dataset: {data_dir}")
        print("Tip: run scripts/generate_hand_crops.py first for better accuracy.")

    train(data_dir)


if __name__ == "__main__":
    main()
