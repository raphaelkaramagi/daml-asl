#!/usr/bin/env python3
"""
Run post-training export, evaluation, and web asset generation.

Usage:
    python scripts/run_post_training.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, check=True)


def main() -> None:
    run([sys.executable, "scripts/convert_models.py"])
    run([sys.executable, "scripts/evaluate_models.py"])
    run([sys.executable, "scripts/update_results_doc.py"])
    run([sys.executable, "scripts/prepare_samples.py"])
    run(["npm", "run", "build"], cwd=WEB_DIR)
    print("\nPost-training pipeline complete.")


if __name__ == "__main__":
    main()
