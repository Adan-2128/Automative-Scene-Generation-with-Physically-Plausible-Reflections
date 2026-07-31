"""
Central configuration for the Automotive Scene Generation Pipeline.
All paths, model names, and hyperparameters live here so nothing
is hardcoded deep inside the pipeline logic.
"""
import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
ENVIRONMENTS_DIR = ROOT_DIR / "environments"
OUTPUTS_DIR = ROOT_DIR / "outputs"

ENV_IMAGES_DIR = ENVIRONMENTS_DIR / "images"          # raw input environment photos/renders
PCA_STATE_PATH = MODELS_DIR / "artifacts" / "pca_state.joblib"
FEATURE_CACHE_DIR = OUTPUTS_DIR / "feature_cache"
SCENE_OUTPUT_DIR = OUTPUTS_DIR / "scenes"
LOG_DIR = OUTPUTS_DIR / "logs"

for d in (ENV_IMAGES_DIR, MODELS_DIR / "artifacts", FEATURE_CACHE_DIR, SCENE_OUTPUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


RESNET_VARIANT = "resnet50"          # torchvision backbone
RESNET_PRETRAINED_WEIGHTS = "IMAGENET1K_V2"
FEATURE_LAYER = "avgpool"            # global pooled feature (2048-d for resnet50)
IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1" else "cpu"


PCA_N_COMPONENTS = int(os.environ.get("PCA_N_COMPONENTS", 1))
PCA_RANDOM_STATE = 42

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.2
GROQ_MAX_TOKENS = 1024
GROQ_MAX_RETRIES = 4
GROQ_RETRY_BACKOFF_SECONDS = 2.0


CAR_LOCK_FIELDS = (
    "geometry", "proportions", "paint", "wheels", "badge", "chassis", "silhouette"
)