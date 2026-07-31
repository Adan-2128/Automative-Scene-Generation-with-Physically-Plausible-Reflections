"""
Environment image discovery and loading utilities.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def discover_environment_images(directory: Path = config.ENV_IMAGES_DIR) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Environment image directory not found: {directory}")
    images = sorted(
        p for p in directory.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(
            f"No environment images found in {directory}. "
            f"Drop .jpg/.png files there before running the pipeline "
            f"(see README.md 'where to save what')."
        )
    return images