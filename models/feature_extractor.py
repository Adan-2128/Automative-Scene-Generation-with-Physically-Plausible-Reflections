"""
ResNet-based CNN feature extractor for environment images.

Ingests an environment image (conceptually L x W x H x C: batch length,
width, height, channels) and produces a dense feature vector per image
by running it through a pretrained ResNet with the classification head
removed. The forward pass over a batch is O(L * W * H * C) with respect
to the convolutional stack, dominated by the early conv layers operating
on the full spatial resolution before pooling.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

import config

logger = logging.getLogger(__name__)


class ResNetFeatureExtractor:
    """Wraps a pretrained ResNet, strips the FC classification head, and
    exposes `.extract()` / `.extract_single()` for dense feature vectors.
    """

    def __init__(
        self,
        variant: str = config.RESNET_VARIANT,
        weights: str = config.RESNET_PRETRAINED_WEIGHTS,
        device: str = config.DEVICE,
        image_size: int = config.IMAGE_SIZE,
    ):
        self.device = torch.device(device)
        self.model = self._build_backbone(variant, weights).to(self.device).eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        logger.info("Loaded %s (weights=%s) on %s", variant, weights, self.device)

    @staticmethod
    def _build_backbone(variant: str, weights: str) -> nn.Module:
        if not hasattr(models, variant):
            raise ValueError(f"Unsupported ResNet variant: {variant}")
        weight_enum = getattr(models, f"{variant.capitalize()}_Weights", None)
        w = getattr(weight_enum, weights) if weight_enum is not None else None
        backbone = getattr(models, variant)(weights=w)
        # Strip classification head (fc). Keep everything up to avgpool,
        # yielding a (N, C, 1, 1) globally-pooled feature map (C=2048 for
        # resnet50) that encodes textures, lighting, and global composition.
        modules = list(backbone.children())[:-1]
        return nn.Sequential(*modules)

    @torch.no_grad()
    def _load_tensor(self, image_path: Union[str, Path]) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        return self.preprocess(img)

    @torch.no_grad()
    def extract(self, image_paths: List[Union[str, Path]], batch_size: int = 16) -> torch.Tensor:
        """Returns a (N, C) tensor of dense feature vectors, one row per image."""
        all_feats = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch = torch.stack([self._load_tensor(p) for p in batch_paths]).to(self.device)
            feats = self.model(batch)                   # (B, C, 1, 1)
            feats = torch.flatten(feats, start_dim=1)    # (B, C)
            all_feats.append(feats.cpu())
        return torch.cat(all_feats, dim=0)

    @torch.no_grad()
    def extract_single(self, image_path: Union[str, Path]) -> torch.Tensor:
        return self.extract([image_path])[0]