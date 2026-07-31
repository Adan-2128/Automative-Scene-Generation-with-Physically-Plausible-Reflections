"""
PCA-based dimensionality reduction for CNN feature vectors.

Compresses high-dimensional ResNet features (2048-d for ResNet-50) down
to a compact k-dimensional signature, cutting memory footprint and
speeding up all downstream vector operations (Groq prompt construction,
similarity search, caching) by roughly (2048 / k)x.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

import config

logger = logging.getLogger(__name__)


class PCAReducer:
    def __init__(
        self,
        n_components: int = config.PCA_N_COMPONENTS,
        random_state: int = config.PCA_RANDOM_STATE,
    ):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit(self, features: np.ndarray) -> "PCAReducer":
        n_samples = features.shape[0]
        if n_samples < self.n_components:
            raise ValueError(
                f"PCA requires at least n_components ({self.n_components}) samples, "
                f"got {n_samples}. Add more environment images or lower PCA_N_COMPONENTS."
            )
        self.pca.fit(features)
        self._fitted = True
        explained = self.pca.explained_variance_ratio_.sum()
        logger.info(
            "PCA fitted: %d -> %d dims, %.2f%% variance explained",
            features.shape[1], self.n_components, explained * 100,
        )
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PCAReducer.transform() called before fit()/load().")
        return self.pca.transform(features)

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        self.fit(features)
        return self.transform(features)

    def save(self, path: Path = config.PCA_STATE_PATH) -> None:
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted PCAReducer.")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pca": self.pca, "n_components": self.n_components}, path)
        logger.info("Saved PCA state to %s", path)

    @classmethod
    def load(cls, path: Path = config.PCA_STATE_PATH) -> "PCAReducer":
        if not path.exists():
            raise FileNotFoundError(
                f"No fitted PCA state found at {path}. Run the pipeline with "
                f"--fit-pca first (see main.py)."
            )
        state = joblib.load(path)
        reducer = cls(n_components=state["n_components"])
        reducer.pca = state["pca"]
        reducer._fitted = True
        logger.info("Loaded PCA state from %s", path)
        return reducer