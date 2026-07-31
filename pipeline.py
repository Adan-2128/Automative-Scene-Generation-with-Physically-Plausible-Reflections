"""
End-to-end orchestration: environment image -> ResNet features ->
PCA compression -> Groq lighting inference -> car-invariance-locked
scene descriptor -> JSON written to outputs/scenes/.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import config
from environments.car_constraints import CarLockProfile
from environments.loader import discover_environment_images
from models.feature_extractor import ResNetFeatureExtractor
from models.lighting_engine import GroqLightingEngine
from models.pca_reducer import PCAReducer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_DIR / "pipeline.log"),
    ],
)
logger = logging.getLogger(__name__)


class ScenePipeline:
    def __init__(self, car_lock: CarLockProfile):
        self.car_lock = car_lock
        self.extractor = ResNetFeatureExtractor()
        self.lighting_engine = GroqLightingEngine()
        self.pca: Optional[PCAReducer] = None

    def fit_pca(self, image_paths: List[Path]) -> PCAReducer:
        logger.info("Extracting ResNet features for PCA fitting on %d images", len(image_paths))
        feats = self.extractor.extract(image_paths).numpy()
        reducer = PCAReducer()
        reducer.fit(feats)
        reducer.save()
        self.pca = reducer
        return reducer

    def _ensure_pca(self) -> PCAReducer:
        if self.pca is None:
            self.pca = PCAReducer.load()
        return self.pca

    def process_image(self, image_path: Path) -> dict:
        pca = self._ensure_pca()

        feat = self.extractor.extract_single(image_path).numpy().reshape(1, -1)
        pca_vector = pca.transform(feat)[0]

        logger.info("Requesting lighting profile from Groq for %s", image_path.name)
        lighting_profile = self.lighting_engine.infer_lighting(pca_vector)

        scene_descriptor = {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "environment_image": str(image_path),
            "car_lock": self.car_lock.as_locked_context(),
            "pca_dimensionality": int(pca_vector.shape[0]),
            "lighting_profile": lighting_profile.model_dump(),
        }

        out_path = config.SCENE_OUTPUT_DIR / f"{image_path.stem}_scene.json"
        out_path.write_text(json.dumps(scene_descriptor, indent=2))
        logger.info("Wrote scene descriptor -> %s", out_path)
        return scene_descriptor

    def run(self, fit_pca: bool = False) -> List[dict]:
        image_paths = discover_environment_images()

        if fit_pca or not config.PCA_STATE_PATH.exists():
            self.fit_pca(image_paths)

        results = []
        for path in image_paths:
            try:
                results.append(self.process_image(path))
            except Exception as e:
                logger.error("Failed processing %s: %s", path, e)
        return results