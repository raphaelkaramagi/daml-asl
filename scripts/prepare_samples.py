#!/usr/bin/env python3
"""
Prepare sample images for the web demo.
Copies and resizes test images + a few training images per class.

Usage:
    python scripts/prepare_samples.py
"""

import os
import shutil
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    import cv2
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "asl_alphabet_test" / "asl_alphabet_test"
TRAIN_DIR = PROJECT_ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
OUT_DIR = PROJECT_ROOT / "web" / "public" / "samples"

TARGET_SIZE = (200, 200)
TRAIN_SAMPLES_PER_CLASS = 2


def resize_and_save(src: Path, dst: Path):
    """Resize an image and save it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if Image:
        img = Image.open(src).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        img.save(dst, "JPEG", quality=85)
    else:
        img = cv2.imread(str(src))
        img = cv2.resize(img, TARGET_SIZE)
        cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0

    if TEST_DIR.exists():
        for img_file in sorted(TEST_DIR.glob("*.jpg")):
            label = img_file.stem.replace("_test", "")
            dst = OUT_DIR / f"{label}_test.jpg"
            resize_and_save(img_file, dst)
            copied += 1
        print(f"Copied {copied} test images")

    train_copied = 0
    if TRAIN_DIR.exists():
        for class_dir in sorted(TRAIN_DIR.iterdir()):
            if not class_dir.is_dir():
                continue
            images = sorted(class_dir.glob("*.jpg"))[:TRAIN_SAMPLES_PER_CLASS]
            for i, img_file in enumerate(images):
                dst = OUT_DIR / f"{class_dir.name}_train_{i}.jpg"
                resize_and_save(img_file, dst)
                train_copied += 1
        print(f"Copied {train_copied} training sample images")

    manifest = []
    for f in sorted(OUT_DIR.glob("*.jpg")):
        manifest.append(f.name)

    import json
    with open(OUT_DIR / "manifest.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"Manifest written with {len(manifest)} entries")


if __name__ == "__main__":
    main()
