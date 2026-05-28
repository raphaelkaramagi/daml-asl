#!/usr/bin/env python3
"""
Generate hand-cropped 96x96 training images for ResNet using shared MediaPipe detection.

Usage:
    python scripts/generate_hand_crops.py
    python scripts/generate_hand_crops.py --workers 8
    python scripts/generate_hand_crops.py --limit 1000  # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mediapipe_detect import detect_hand, crop_hand, DEFAULT_CONFIDENCE

DATA_DIR = PROJECT_ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
OUT_DIR = PROJECT_ROOT / "data" / "asl_hand_crops"
MANIFEST_PATH = OUT_DIR / "manifest.json"
IMAGE_SIZE = (96, 96)
NOTHING_CLASS = "nothing"


def _default_workers() -> int:
    n = cpu_count() or 4
    return max(1, min(n - 1, 10))


def _process_crop(item: tuple[int, str, str]) -> tuple[int, dict, dict[str, int]]:
    """Worker: crop one image and write to disk."""
    index, label, path_str = item
    path = Path(path_str)
    class_stats = {"detected": 0, "fallback": 0, "total": 1}

    image = cv2.imread(path_str)
    if image is None:
        return index, {}, class_stats

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out_name = f"{label}_{path.stem}.jpg"
    out_path = OUT_DIR / label / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if label == NOTHING_CLASS:
        resized = cv2.resize(image_rgb, IMAGE_SIZE)
        class_stats["fallback"] = 1
        crop_type = "full_frame"
    else:
        hand = detect_hand(image_rgb, conf=DEFAULT_CONFIDENCE)
        if hand is not None:
            cropped = crop_hand(image_rgb, hand.landmarks)
            resized = cv2.resize(cropped, IMAGE_SIZE)
            class_stats["detected"] = 1
            crop_type = "hand_crop"
        else:
            resized = cv2.resize(image_rgb, IMAGE_SIZE)
            class_stats["fallback"] = 1
            crop_type = "fallback_full"

    cv2.imwrite(str(out_path), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    manifest_entry = {
        "source": str(path.relative_to(PROJECT_ROOT)),
        "output": str(out_path.relative_to(PROJECT_ROOT)),
        "label": label,
        "crop_type": crop_type,
    }
    return index, manifest_entry, class_stats


def process_dataset(limit: int | None = None, workers: int = 1) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"detected": 0, "fallback": 0, "total": 0})
    manifest: list[dict | None] = []

    classes = sorted(d for d in os.listdir(DATA_DIR) if (DATA_DIR / d).is_dir())
    image_paths: list[tuple[str, Path]] = []
    for label in classes:
        class_dir = DATA_DIR / label
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append((label, class_dir / fname))

    if limit:
        image_paths = image_paths[:limit]

    tasks = [(i, label, str(path)) for i, (label, path) in enumerate(image_paths)]
    manifest = [None] * len(tasks)

    if workers <= 1:
        iterator = (_process_crop(t) for t in tasks)
        for index, entry, class_stats in tqdm(iterator, total=len(tasks), desc="Generating hand crops"):
            if entry:
                manifest[index] = entry
            label = image_paths[index][0]
            stats[label]["total"] += class_stats["total"]
            stats[label]["detected"] += class_stats["detected"]
            stats[label]["fallback"] += class_stats["fallback"]
    else:
        chunksize = max(16, len(tasks) // (workers * 8))
        with Pool(processes=workers) as pool:
            for index, entry, class_stats in tqdm(
                pool.imap_unordered(_process_crop, tasks, chunksize=chunksize),
                total=len(tasks),
                desc=f"Generating hand crops ({workers} workers)",
            ):
                if entry:
                    manifest[index] = entry
                label = image_paths[index][0]
                stats[label]["total"] += class_stats["total"]
                stats[label]["detected"] += class_stats["detected"]
                stats[label]["fallback"] += class_stats["fallback"]

    manifest = [m for m in manifest if m is not None]

    summary = {
        "total": sum(s["total"] for s in stats.values()),
        "detected": sum(s["detected"] for s in stats.values()),
        "fallback": sum(s["fallback"] for s in stats.values()),
        "per_class": dict(stats),
    }
    with open(MANIFEST_PATH, "w") as f:
        json.dump({"summary": summary, "images": manifest}, f, indent=2)

    print(f"\nSaved crops to {OUT_DIR}")
    print(f"Detection rate: {summary['detected']}/{summary['total']} "
          f"({100 * summary['detected'] / max(summary['total'], 1):.1f}%)")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate hand-cropped ResNet training set")
    parser.add_argument("--limit", type=int, default=None, help="Max images (for testing)")
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help=f"Parallel CPU workers (default: {_default_workers()})",
    )
    args = parser.parse_args()
    process_dataset(limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
