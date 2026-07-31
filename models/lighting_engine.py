"""
Groq LLM lighting integration.

Takes the PCA-compressed environment feature signature and asks
llama-3.3-70b-versatile (via Groq, JSON mode) to infer a physically
plausible lighting + reflection profile: direction, intensity, color
temperature, and reflection parameters — all under the car invariance
lock defined in environments/car_constraints.py.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from groq import Groq
from pydantic import BaseModel, Field, ValidationError

import config
from environments.car_constraints import build_invariance_clause, validate_no_car_mutation

logger = logging.getLogger(__name__)


class LightingProfile(BaseModel):
    light_direction_xyz: List[float] = Field(..., min_length=3, max_length=3)
    light_intensity: float = Field(..., ge=0.0, le=10.0)
    color_temperature_kelvin: float = Field(..., ge=1000.0, le=20000.0)
    ambient_occlusion_strength: float = Field(..., ge=0.0, le=1.0)
    reflection_intensity: float = Field(..., ge=0.0, le=1.0)
    reflection_roughness: float = Field(..., ge=0.0, le=1.0)
    specular_highlight_sharpness: float = Field(..., ge=0.0, le=1.0)
    scene_mood: str = Field(default="neutral")
    rationale: str = Field(default="")


SYSTEM_PROMPT_TEMPLATE = """You are a physically-based rendering (PBR) lighting \
expert for automotive scene compositing. You will be given a compressed \
PCA feature signature summarizing an environment image's textures, \
lighting, and global composition. Infer a physically plausible lighting \
and reflection profile that would make a car rendered into this \
environment look photorealistic.

{invariance_clause}

Respond ONLY with a single JSON object matching exactly this schema, no \
prose, no markdown fences:
{{
  "light_direction_xyz": [float, float, float],
  "light_intensity": float,
  "color_temperature_kelvin": float,
  "ambient_occlusion_strength": float,
  "reflection_intensity": float,
  "reflection_roughness": float,
  "specular_highlight_sharpness": float,
  "scene_mood": string,
  "rationale": string
}}"""


class GroqLightingEngine:
    def __init__(self, api_key: str = config.GROQ_API_KEY, model: str = config.GROQ_MODEL):
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Export it or add it to a .env file "
                "before running the pipeline (see .env.example)."
            )
        self.client = Groq(api_key=api_key)
        self.model = model

    @staticmethod
    def _summarize_pca_vector(vector: np.ndarray, top_k: int = 8) -> Dict[str, Any]:
        """Turns a raw PCA float vector into a compact, LLM-friendly summary
        instead of dumping dozens of raw floats into the prompt.
        """
        top_k = min(top_k, vector.shape[0])
        idx_sorted = np.argsort(-np.abs(vector))[:top_k]
        return {
            "dimensionality": int(vector.shape[0]),
            "l2_norm": float(np.linalg.norm(vector)),
            "mean": float(vector.mean()),
            "std": float(vector.std()),
            "dominant_components": [
                {"index": int(i), "value": round(float(vector[i]), 4)} for i in idx_sorted
            ],
        }

    def infer_lighting(self, pca_vector: np.ndarray) -> LightingProfile:
        signature = self._summarize_pca_vector(pca_vector)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(invariance_clause=build_invariance_clause())
        user_prompt = (
            "Environment PCA feature signature:\n"
            f"{json.dumps(signature, indent=2)}\n\n"
            "Infer the lighting/reflection profile now."
        )

        last_err: Optional[Exception] = None
        for attempt in range(1, config.GROQ_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=config.GROQ_TEMPERATURE,
                    max_tokens=config.GROQ_MAX_TOKENS,
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content
                data = json.loads(raw)
                validate_no_car_mutation(data)
                return LightingProfile.model_validate(data)
            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                last_err = e
                logger.warning(
                    "Groq attempt %d/%d failed validation: %s",
                    attempt, config.GROQ_MAX_RETRIES, e,
                )
            except Exception as e:  # network / rate limit / API errors
                last_err = e
                logger.warning(
                    "Groq attempt %d/%d failed: %s", attempt, config.GROQ_MAX_RETRIES, e,
                )
            time.sleep(config.GROQ_RETRY_BACKOFF_SECONDS * attempt)

        raise RuntimeError(
            f"Groq lighting inference failed after {config.GROQ_MAX_RETRIES} attempts: {last_err}"
        )