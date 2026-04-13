#!/usr/bin/env python3
"""
Extract training metrics from notebook outputs and save as JSON
for the web training replay visualizer.

Usage:
    python scripts/extract_training_data.py
"""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "web" / "public" / "training-data.json"


def main():
    data = {
        "resnet": {
            "phase1": {
                "label": "Phase 1: Frozen Base (Transfer Learning)",
                "description": "ResNet50 backbone frozen, only training custom classification head",
                "optimizer": "Adam (lr=0.001)",
                "epochs": [
                    {"epoch": 1, "loss": 3.2458, "accuracy": 0.0815, "val_loss": 2.9772, "val_accuracy": 0.1563},
                    {"epoch": 2, "loss": 3.0595, "accuracy": 0.1217, "val_loss": 2.7860, "val_accuracy": 0.1995},
                    {"epoch": 3, "loss": 2.9806, "accuracy": 0.1384, "val_loss": 2.6945, "val_accuracy": 0.2164},
                    {"epoch": 4, "loss": 2.9364, "accuracy": 0.1480, "val_loss": 2.6333, "val_accuracy": 0.2356},
                    {"epoch": 5, "loss": 2.9038, "accuracy": 0.1536, "val_loss": 2.5831, "val_accuracy": 0.2532},
                    {"epoch": 6, "loss": 2.8799, "accuracy": 0.1594, "val_loss": 2.5786, "val_accuracy": 0.2564},
                    {"epoch": 7, "loss": 2.8623, "accuracy": 0.1633, "val_loss": 2.5454, "val_accuracy": 0.2554},
                    {"epoch": 8, "loss": 2.8462, "accuracy": 0.1667, "val_loss": 2.5074, "val_accuracy": 0.2550},
                    {"epoch": 9, "loss": 2.8353, "accuracy": 0.1680, "val_loss": 2.4927, "val_accuracy": 0.2593},
                    {"epoch": 10, "loss": 2.8222, "accuracy": 0.1735, "val_loss": 2.4864, "val_accuracy": 0.2642},
                ],
                "annotations": [
                    {"epoch": 10, "text": "Best val_accuracy: 26.42%"}
                ]
            },
            "phase2": {
                "label": "Phase 2: Fine-Tuning (Unfrozen Layers)",
                "description": "Unfreezing ResNet50 layers 143+ for fine-tuning with lower learning rate",
                "optimizer": "Adam (lr=0.0001)",
                "epochs": [
                    {"epoch": 1, "loss": 3.0791, "accuracy": 0.1616, "val_loss": 2.4233, "val_accuracy": 0.2876},
                    {"epoch": 2, "loss": 2.4399, "accuracy": 0.2697, "val_loss": 2.0520, "val_accuracy": 0.3798},
                    {"epoch": 3, "loss": 2.2283, "accuracy": 0.3204, "val_loss": 2.0954, "val_accuracy": 0.3570},
                    {"epoch": 4, "loss": 2.0715, "accuracy": 0.3603, "val_loss": 2.7869, "val_accuracy": 0.3194},
                    {"epoch": 5, "loss": 1.9209, "accuracy": 0.4029, "val_loss": 1.9186, "val_accuracy": 0.3945},
                    {"epoch": 6, "loss": 1.7732, "accuracy": 0.4427, "val_loss": 2.4739, "val_accuracy": 0.4213},
                    {"epoch": 7, "loss": 1.6436, "accuracy": 0.4790, "val_loss": 1.7146, "val_accuracy": 0.4806},
                    {"epoch": 8, "loss": 1.5221, "accuracy": 0.5164, "val_loss": 2.1988, "val_accuracy": 0.3921},
                    {"epoch": 9, "loss": 1.4175, "accuracy": 0.5469, "val_loss": 2.3608, "val_accuracy": 0.4228},
                    {"epoch": 10, "loss": 1.3242, "accuracy": 0.5747, "val_loss": 1.8604, "val_accuracy": 0.4724},
                ],
                "annotations": [
                    {"epoch": 1, "text": "Layers 143+ unfrozen"},
                    {"epoch": 7, "text": "Best val_accuracy: 48.06%"},
                    {"epoch": 10, "text": "LR reduced to 5e-5"}
                ]
            },
            "summary": {
                "architecture": "ResNet50 + GlobalAvgPool + Dense(128) + Dense(29)",
                "input": "96x96 RGB images",
                "parameters": "~23.6M",
                "modelSize": "208 MB",
                "testAccuracy": 71.43,
                "trainingTime": "~4 hours"
            }
        },
        "landmark": {
            "training": {
                "label": "Landmark Neural Network Training",
                "description": "Dense network on 63 MediaPipe hand landmark features (21 landmarks x 3 coords)",
                "optimizer": "Adam",
                "epochs": [
                    {"epoch": 1, "loss": 0.7082, "accuracy": 0.7943, "val_loss": 0.1510, "val_accuracy": 0.9669},
                    {"epoch": 2, "loss": 0.2213, "accuracy": 0.9365, "val_loss": 0.1186, "val_accuracy": 0.9695},
                    {"epoch": 3, "loss": 0.1664, "accuracy": 0.9537, "val_loss": 0.0804, "val_accuracy": 0.9807},
                    {"epoch": 4, "loss": 0.1399, "accuracy": 0.9627, "val_loss": 0.0748, "val_accuracy": 0.9814},
                    {"epoch": 5, "loss": 0.1219, "accuracy": 0.9666, "val_loss": 0.0647, "val_accuracy": 0.9829},
                    {"epoch": 6, "loss": 0.1169, "accuracy": 0.9671, "val_loss": 0.0602, "val_accuracy": 0.9846},
                    {"epoch": 7, "loss": 0.1058, "accuracy": 0.9700, "val_loss": 0.0620, "val_accuracy": 0.9825},
                    {"epoch": 8, "loss": 0.1004, "accuracy": 0.9727, "val_loss": 0.0625, "val_accuracy": 0.9807},
                    {"epoch": 9, "loss": 0.0951, "accuracy": 0.9737, "val_loss": 0.0544, "val_accuracy": 0.9861},
                    {"epoch": 10, "loss": 0.0891, "accuracy": 0.9751, "val_loss": 0.0499, "val_accuracy": 0.9886},
                    {"epoch": 11, "loss": 0.0883, "accuracy": 0.9742, "val_loss": 0.0499, "val_accuracy": 0.9879},
                    {"epoch": 12, "loss": 0.0847, "accuracy": 0.9764, "val_loss": 0.0518, "val_accuracy": 0.9880},
                    {"epoch": 13, "loss": 0.0826, "accuracy": 0.9766, "val_loss": 0.0501, "val_accuracy": 0.9877},
                    {"epoch": 14, "loss": 0.0801, "accuracy": 0.9771, "val_loss": 0.0508, "val_accuracy": 0.9879},
                    {"epoch": 15, "loss": 0.0795, "accuracy": 0.9778, "val_loss": 0.0468, "val_accuracy": 0.9879},
                    {"epoch": 16, "loss": 0.0744, "accuracy": 0.9788, "val_loss": 0.0469, "val_accuracy": 0.9890},
                    {"epoch": 17, "loss": 0.0781, "accuracy": 0.9774, "val_loss": 0.0455, "val_accuracy": 0.9884},
                    {"epoch": 18, "loss": 0.0719, "accuracy": 0.9790, "val_loss": 0.0447, "val_accuracy": 0.9894},
                    {"epoch": 19, "loss": 0.0717, "accuracy": 0.9795, "val_loss": 0.0493, "val_accuracy": 0.9887},
                    {"epoch": 20, "loss": 0.0710, "accuracy": 0.9798, "val_loss": 0.0485, "val_accuracy": 0.9876},
                    {"epoch": 21, "loss": 0.0693, "accuracy": 0.9790, "val_loss": 0.0459, "val_accuracy": 0.9889},
                    {"epoch": 22, "loss": 0.0684, "accuracy": 0.9803, "val_loss": 0.0445, "val_accuracy": 0.9896},
                    {"epoch": 23, "loss": 0.0720, "accuracy": 0.9799, "val_loss": 0.0436, "val_accuracy": 0.9893},
                    {"epoch": 24, "loss": 0.0696, "accuracy": 0.9796, "val_loss": 0.0427, "val_accuracy": 0.9886},
                    {"epoch": 25, "loss": 0.0655, "accuracy": 0.9810, "val_loss": 0.0440, "val_accuracy": 0.9897},
                    {"epoch": 26, "loss": 0.0661, "accuracy": 0.9818, "val_loss": 0.0466, "val_accuracy": 0.9894},
                    {"epoch": 27, "loss": 0.0660, "accuracy": 0.9805, "val_loss": 0.0434, "val_accuracy": 0.9885},
                    {"epoch": 28, "loss": 0.0660, "accuracy": 0.9811, "val_loss": 0.0455, "val_accuracy": 0.9897},
                    {"epoch": 29, "loss": 0.0666, "accuracy": 0.9808, "val_loss": 0.0438, "val_accuracy": 0.9889},
                ],
                "annotations": [
                    {"epoch": 1, "text": "Rapid convergence: 79.4% accuracy"},
                    {"epoch": 6, "text": "98.46% val accuracy"},
                    {"epoch": 25, "text": "Best val_accuracy: 98.97%"},
                    {"epoch": 29, "text": "Early stopping triggered"}
                ]
            },
            "summary": {
                "architecture": "Dense(128, ReLU) → Dropout(0.3) → Dense(64, ReLU) → Dropout(0.3) → Dense(29, Softmax)",
                "input": "63 features (21 landmarks × 3 coordinates)",
                "parameters": "~18K",
                "modelSize": "~244 KB",
                "testAccuracy": 98.88,
                "finalTestAccuracy": 71.43,
                "trainingTime": "~10 min",
                "keyFinding": "100% accuracy when hands are detected. Bottleneck is MediaPipe detection, not classification."
            }
        }
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Training data written to {OUT_PATH}")


if __name__ == "__main__":
    main()
