#!/usr/bin/env python3
"""
Re-extract MediaPipe landmarks for all training images using shared detection module.

Replaces notebook 03 inline legacy Hands API with the same pipeline as the web demo.

Usage:
    python scripts/reextract_landmarks.py
    python scripts/reextract_landmarks.py --workers 8
    python scripts/reextract_landmarks.py --limit 500
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from mediapipe_detect import detect_hand, DEFAULT_CONFIDENCE

DATA_DIR = PROJECT_ROOT / "data" / "asl_alphabet_train" / "asl_alphabet_train"
OUTPUT_FILE = PROJECT_ROOT / "data" / "asl_landmarks_train.csv"


def _default_workers() -> int:
    n = cpu_count() or 4
    return max(1, min(n - 1, 10))


def _process_image(item: tuple[str, str]) -> tuple[int, str, str, list | None, bool]:
    """Worker: returns (index, label, source_path, features, detected)."""
    index, label, path_str = item
    path = Path(path_str)
    rel = str(path.relative_to(PROJECT_ROOT))

    image = cv2.imread(path_str)
    if image is None:
        return index, label, rel, None, False

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hand = detect_hand(image_rgb, conf=DEFAULT_CONFIDENCE)
    if hand is not None:
        return index, label, rel, hand.features, True
    return index, label, rel, None, False


def reextract(limit: int | None = None, workers: int = 1) -> pd.DataFrame:
    image_paths: list[tuple[str, Path]] = []
    for label in sorted(os.listdir(DATA_DIR)):
        class_dir = DATA_DIR / label
        if not class_dir.is_dir():
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                image_paths.append((label, class_dir / fname))

    if limit:
        image_paths = image_paths[:limit]

    tasks = [
        (i, label, str(path))
        for i, (label, path) in enumerate(image_paths)
    ]

    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"detected": 0, "failed": 0, "total": 0})
    rows: list[list | None] = [None] * len(tasks)
    labels: list[str] = [""] * len(tasks)
    sources: list[str] = [""] * len(tasks)

    if workers <= 1:
        iterator = (_process_image(t) for t in tasks)
        progress = tqdm(iterator, total=len(tasks), desc="Extracting landmarks")
        for index, label, rel, features, detected in progress:
            labels[index] = label
            sources[index] = rel
            stats[label]["total"] += 1
            if detected and features is not None:
                rows[index] = features
                stats[label]["detected"] += 1
            else:
                rows[index] = [np.nan] * 63
                stats[label]["failed"] += 1
    else:
        chunksize = max(16, len(tasks) // (workers * 8))
        with Pool(processes=workers) as pool:
            for index, label, rel, features, detected in tqdm(
                pool.imap_unordered(_process_image, tasks, chunksize=chunksize),
                total=len(tasks),
                desc=f"Extracting landmarks ({workers} workers)",
            ):
                labels[index] = label
                sources[index] = rel
                stats[label]["total"] += 1
                if detected and features is not None:
                    rows[index] = features
                    stats[label]["detected"] += 1
                else:
                    rows[index] = [np.nan] * 63
                    stats[label]["failed"] += 1

    landmark_cols = [f"lm_{i}" for i in range(63)]
    df = pd.DataFrame(rows, columns=landmark_cols)
    df["label"] = labels
    df["source_path"] = sources

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    total = len(image_paths)
    detected = sum(s["detected"] for s in stats.values())
    print(f"\nSaved {OUTPUT_FILE}")
    print(f"Detection rate: {detected}/{total} ({100 * detected / max(total, 1):.1f}%)")
    for label in sorted(stats.keys()):
        s = stats[label]
        t = s.get("total", s["detected"] + s["failed"])
        rate = 100 * s["detected"] / max(t, 1)
        print(f"  {label}: {s['detected']}/{t} ({rate:.1f}%)")

    return df


def main():
    parser = argparse.ArgumentParser(description="Re-extract landmarks with shared MediaPipe module")
    parser.add_argument("--limit", type=int, default=None, help="Max images (for testing)")
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_workers(),
        help=f"Parallel CPU workers (default: {_default_workers()})",
    )
    args = parser.parse_args()
    reextract(limit=args.limit, workers=args.workers)


if __name__ == "__main__":
    main()
