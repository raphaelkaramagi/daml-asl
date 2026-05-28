#!/usr/bin/env python3
"""
Shared MediaPipe hand detection for training, evaluation, and demo.py.

Uses MediaPipe Tasks HandLandmarker (same API family as the web app).
Provides multi-scale retry, padded bbox cropping, and wrist-relative features.
"""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "scripts" / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

DEFAULT_CONFIDENCE = 0.2
DEFAULT_TRACKING_CONFIDENCE = 0.5
DEFAULT_SCALES: tuple[float, ...] = (1.0, 1.5, 2.0)
MIN_DETECTION_DIM = 300  # matches web/src/lib/image-utils.ts upscaleCanvas
BBOX_PADDING = 0.2

_detector = None
_detector_confidence: float | None = None


@dataclass
class HandLandmarks:
    """Normalized landmark coordinates (0-1) plus derived features."""

    landmarks: list[tuple[float, float, float]]  # 21 x (x, y, z)

    @property
    def features(self) -> list[float]:
        return extract_features(self.landmarks)


def ensure_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        print(f"Downloading HandLandmarker model to {MODEL_PATH}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def _get_detector(
    min_confidence: float = DEFAULT_CONFIDENCE,
    min_tracking_confidence: float = DEFAULT_TRACKING_CONFIDENCE,
):
    global _detector, _detector_confidence
    if _detector is not None and _detector_confidence == min_confidence:
        return _detector

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    model_path = ensure_model()
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    _detector = vision.HandLandmarker.create_from_options(options)
    _detector_confidence = min_confidence
    return _detector


def extract_features(landmarks: Sequence[tuple[float, float, float]]) -> list[float]:
    """Wrist-relative 63-dim feature vector (matches web/src/lib/landmarks.ts)."""
    wrist = landmarks[0]
    features: list[float] = []
    for x, y, z in landmarks:
        features.extend([x - wrist[0], y - wrist[1], z - wrist[2]])
    return features


def get_padded_bbox(
    landmarks: Sequence[tuple[float, float, float]],
    padding: float = BBOX_PADDING,
) -> tuple[float, float, float, float]:
    """Return normalized bbox (min_x, min_y, max_x, max_y)."""
    xs = [lm[0] for lm in landmarks]
    ys = [lm[1] for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w, h = max_x - min_x, max_y - min_y
    min_x = max(0.0, min_x - w * padding)
    min_y = max(0.0, min_y - h * padding)
    max_x = min(1.0, max_x + w * padding)
    max_y = min(1.0, max_y + h * padding)
    return min_x, min_y, max_x, max_y


def bbox_area(landmarks: Sequence[tuple[float, float, float]]) -> float:
    min_x, min_y, max_x, max_y = get_padded_bbox(landmarks, padding=0)
    return (max_x - min_x) * (max_y - min_y)


def _parse_result(result) -> HandLandmarks | None:
    if not result.hand_landmarks:
        return None
    hand = result.hand_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in hand]
    return HandLandmarks(landmarks=landmarks)


def _upscale_image(image_rgb: np.ndarray, min_dim: int = MIN_DETECTION_DIM) -> np.ndarray:
    """Upscale so smallest dimension is at least min_dim (web parity)."""
    h, w = image_rgb.shape[:2]
    if h >= min_dim and w >= min_dim:
        return image_rgb
    scale = max(min_dim / w, min_dim / h, 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _detect_on_image(
    image_rgb: np.ndarray,
    min_confidence: float,
    min_tracking_confidence: float,
) -> HandLandmarks | None:
    import mediapipe as mp

    detector = _get_detector(min_confidence, min_tracking_confidence)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    return _parse_result(detector.detect(mp_image))


def detect_hand(
    image_rgb: np.ndarray,
    conf: float = DEFAULT_CONFIDENCE,
    scales: Iterable[float] = DEFAULT_SCALES,
    min_tracking_confidence: float = DEFAULT_TRACKING_CONFIDENCE,
) -> HandLandmarks | None:
    """Detect hand with upscale + multi-scale retry (matches browser pipeline)."""
    base = _upscale_image(image_rgb)
    h, w = base.shape[:2]
    for scale in scales:
        if scale != 1.0:
            scaled = cv2.resize(
                base,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            scaled = base
        result = _detect_on_image(scaled, conf, min_tracking_confidence)
        if result is not None:
            return result
    return None


def crop_hand(
    image_rgb: np.ndarray,
    landmarks: Sequence[tuple[float, float, float]],
    padding: float = BBOX_PADDING,
) -> np.ndarray:
    """Crop padded hand region from RGB image."""
    h, w = image_rgb.shape[:2]
    min_x, min_y, max_x, max_y = get_padded_bbox(landmarks, padding)
    x1 = int(min_x * w)
    y1 = int(min_y * h)
    x2 = max(x1 + 1, int(max_x * w))
    y2 = max(y1 + 1, int(max_y * h))
    return image_rgb[y1:y2, x1:x2].copy()


def reset_detector() -> None:
    """Reset cached detector (e.g. after confidence change)."""
    global _detector, _detector_confidence
    if _detector is not None:
        _detector.close()
    _detector = None
    _detector_confidence = None
