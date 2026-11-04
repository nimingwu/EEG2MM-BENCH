"""
Perceptual similarity metrics using pretrained feature extractors:
  - AlexNet(2): torchvision ``features.4`` two-way identification
  - AlexNet(5): torchvision ``features.11`` two-way identification
  - Inception:  avgpool two-way identification
  - SwAV:       paired feature correlation distance (LOWER is better)

The identification metrics intentionally operate on the full N x N feature
correlation matrix.  A paired cosine average is a different statistic and does
not reproduce the protocol used by the ATM paper.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, Callable, List, Dict
from pathlib import Path

from .base import Metric

log = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Lazy model loader (shared across instances)
# --------------------------------------------------------------------

_MODEL_CACHE: Dict[str, dict] = {}


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_image_tensor(img: Any, size: int = 224):
    """Load one image as an ImageNet-normalized ``[3,size,size]`` tensor."""
    from PIL import Image
    from torchvision import transforms

    if isinstance(img, (str, Path)):
        img = Image.open(img).convert("RGB")
    elif isinstance(img, np.ndarray):
        arr = img
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors. Flattens if multi-dim."""
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _global_avg_pool(features: np.ndarray) -> np.ndarray:
    """Global average pool a 4D feature map → 1D vector."""
    if features.ndim == 4:
        return features.mean(axis=(2, 3))
    return features.flatten()


def _extract_feature_batches(
    images: List[Any],
    extractor: Callable,
    *,
    size: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Load images and extract flattened features without retaining GPU inputs."""
    import torch

    outputs = []
    for start in range(0, len(images), batch_size):
        tensors = [_load_image_tensor(value, size=size)
                   for value in images[start:start + batch_size]]
        batch = torch.stack(tensors).to(_get_device())
        with torch.no_grad():
            features = extractor(batch)
        outputs.append(features.float().flatten(1).cpu().numpy())
    if not outputs:
        return np.empty((0, 0), dtype=np.float32)
    return np.concatenate(outputs, axis=0)


def two_way_identification(real_features: np.ndarray,
                           predicted_features: np.ndarray) -> float:
    """Paper two-way identification accuracy from an N x N correlation matrix."""
    real = np.asarray(real_features, dtype=np.float64)
    predicted = np.asarray(predicted_features, dtype=np.float64)
    if real.shape != predicted.shape or real.ndim != 2:
        raise ValueError(
            "Two-way identification requires matching [samples,features] arrays, "
            f"got {real.shape} and {predicted.shape}")
    if len(real) < 2:
        return float("nan")
    real -= real.mean(axis=1, keepdims=True)
    predicted -= predicted.mean(axis=1, keepdims=True)
    real /= np.linalg.norm(real, axis=1, keepdims=True).clip(min=1e-12)
    predicted /= np.linalg.norm(predicted, axis=1, keepdims=True).clip(min=1e-12)
    correlations = real @ predicted.T
    congruent = np.diag(correlations)
    success = correlations < congruent[None, :]
    return float(success.sum(axis=0).mean() / (len(real) - 1))


# --------------------------------------------------------------------
# AlexNet2
# --------------------------------------------------------------------

class AlexNet2(Metric):
    """AlexNet(2): low-level visual features. Higher is better."""

    @property
    def name(self) -> str:
        return "alexnet_2"

    @property
    def applicable(self) -> List[str]:
        return ["image_generation"]

    def _get_model(self):
        if "alexnet" in _MODEL_CACHE:
            return _MODEL_CACHE["alexnet"]
        import torch
        import torchvision.models as models
        device = _get_device()
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features.eval().to(device)
        _MODEL_CACHE["alexnet"] = {"model": model, "device": device}
        return _MODEL_CACHE["alexnet"]

    def _extract(self, img_tensor) -> np.ndarray:
        cache = self._get_model()
        model = cache["model"]
        import torch
        with torch.no_grad():
            x = model[:5](img_tensor)  # torchvision node features.4
        return x.float().flatten(1).cpu().numpy()

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        p = _load_image_tensor(predictions).unsqueeze(0).to(_get_device())
        g = _load_image_tensor(ground_truths).unsqueeze(0).to(_get_device())
        return _cosine_sim(self._extract(p), self._extract(g))

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        model = self._get_model()["model"]
        predicted = _extract_feature_batches(
            predictions, lambda value: model[:5](value), size=224)
        real = _extract_feature_batches(
            ground_truths, lambda value: model[:5](value), size=224)
        return two_way_identification(real, predicted)


# --------------------------------------------------------------------
# AlexNet5
# --------------------------------------------------------------------

class AlexNet5(Metric):
    """AlexNet(5): mid/high-level visual features. Higher is better."""

    @property
    def name(self) -> str:
        return "alexnet_5"

    @property
    def applicable(self) -> List[str]:
        return ["image_generation"]

    def _get_model(self):
        if "alexnet" in _MODEL_CACHE:
            return _MODEL_CACHE["alexnet"]
        import torch
        import torchvision.models as models
        device = _get_device()
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features.eval().to(device)
        _MODEL_CACHE["alexnet"] = {"model": model, "device": device}
        return _MODEL_CACHE["alexnet"]

    def _extract(self, img_tensor) -> np.ndarray:
        cache = self._get_model()
        model = cache["model"]
        import torch
        with torch.no_grad():
            x = model[:12](img_tensor)  # torchvision node features.11
        return x.float().flatten(1).cpu().numpy()

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        p = _load_image_tensor(predictions).unsqueeze(0).to(_get_device())
        g = _load_image_tensor(ground_truths).unsqueeze(0).to(_get_device())
        return _cosine_sim(self._extract(p), self._extract(g))

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        model = self._get_model()["model"]
        predicted = _extract_feature_batches(
            predictions, lambda value: model[:12](value), size=224)
        real = _extract_feature_batches(
            ground_truths, lambda value: model[:12](value), size=224)
        return two_way_identification(real, predicted)


# --------------------------------------------------------------------
# Inception
# --------------------------------------------------------------------

class InceptionSim(Metric):
    """Inception-v3 avgpool feature similarity. Higher is better."""

    @property
    def name(self) -> str:
        return "inception"

    @property
    def applicable(self) -> List[str]:
        return ["image_generation"]

    def _get_model(self):
        if "inception" in _MODEL_CACHE:
            return _MODEL_CACHE["inception"]
        import torch
        import torchvision.models as models
        device = _get_device()
        # Torchvision's pretrained Inception weights require aux_logits=True at
        # construction time. In eval mode the model still returns pool3/fc only.
        model = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        ).eval().to(device)
        # Replace fc with identity — keeps pool3 features
        model.fc = torch.nn.Identity()
        _MODEL_CACHE["inception"] = {"model": model, "device": device}
        return _MODEL_CACHE["inception"]

    def _extract(self, img_tensor) -> np.ndarray:
        # Inception-v3 expects 299x299
        import torch
        import torch.nn.functional as F
        img_299 = F.interpolate(img_tensor, size=(299, 299), mode="bilinear",
                                align_corners=False)
        cache = self._get_model()
        model = cache["model"]
        with torch.no_grad():
            x = model(img_299)             # [1, 2048, 1, 1]
        return x.cpu().numpy().flatten()

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        p_t = _load_image_tensor(predictions, size=299).unsqueeze(0).to(_get_device())
        g_t = _load_image_tensor(ground_truths, size=299).unsqueeze(0).to(_get_device())
        return _cosine_sim(self._extract(p_t), self._extract(g_t))

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        model = self._get_model()["model"]
        predicted = _extract_feature_batches(predictions, model, size=299)
        real = _extract_feature_batches(ground_truths, model, size=299)
        return two_way_identification(real, predicted)


# --------------------------------------------------------------------
# SwAV
# --------------------------------------------------------------------

class SwAVSim(Metric):
    """SwAV feature distance (LOWER is better, per literature convention).

    Uses the paired Pearson correlation distance from the released evaluation.
    """

    @property
    def name(self) -> str:
        return "swav"

    @property
    def applicable(self) -> List[str]:
        return ["image_generation"]

    def _get_model(self):
        if "swav" in _MODEL_CACHE:
            return _MODEL_CACHE["swav"]
        import torch
        import torch.nn as nn
        device = _get_device()
        model = torch.hub.load("facebookresearch/swav:main", "resnet50",
                               pretrained=True).eval().to(device)
        model.fc = nn.Identity()
        _MODEL_CACHE["swav"] = {"model": model, "device": device}
        return _MODEL_CACHE["swav"]

    def _extract(self, img_tensor) -> np.ndarray:
        cache = self._get_model()
        model = cache["model"]
        import torch
        with torch.no_grad():
            x = model(img_tensor)
        return x.cpu().numpy().flatten()

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        p_t = _load_image_tensor(predictions).unsqueeze(0).to(_get_device())
        g_t = _load_image_tensor(ground_truths).unsqueeze(0).to(_get_device())
        fp = self._extract(p_t)
        fg = self._extract(g_t)
        fp = fp - fp.mean()
        fg = fg - fg.mean()
        correlation = float(np.dot(fp, fg) / (
            np.linalg.norm(fp) * np.linalg.norm(fg) + 1e-12))
        return 1.0 - correlation

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("SwAV predictions and ground truths must have equal length")
        model = self._get_model()["model"]
        predicted = _extract_feature_batches(predictions, model, size=224)
        real = _extract_feature_batches(ground_truths, model, size=224)
        if not len(predicted):
            return float("nan")
        predicted -= predicted.mean(axis=1, keepdims=True)
        real -= real.mean(axis=1, keepdims=True)
        numerator = np.sum(predicted * real, axis=1)
        denominator = (
            np.linalg.norm(predicted, axis=1) * np.linalg.norm(real, axis=1)
        ).clip(min=1e-12)
        return float(np.mean(1.0 - numerator / denominator))
