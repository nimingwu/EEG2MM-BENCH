"""
SSIM — Structural Similarity Index between two images.
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Union
from pathlib import Path

from .base import Metric


class SSIM(Metric):
    """Structural Similarity Index.

    Higher is better. Measures luminance, contrast, and structure similarity.
    """

    def __init__(self, win_size: int = 11, channel_axis: int = -1):
        self.win_size = win_size
        self.channel_axis = channel_axis

    @property
    def name(self) -> str:
        return "ssim"

    @property
    def applicable(self) -> List[str]:
        return ["image_generation"]

    def _load(self, img: Any, target_size: tuple = None) -> np.ndarray:
        if isinstance(img, (str, Path)):
            from PIL import Image
            img = Image.open(img).convert("L")
            if target_size is not None:
                img = img.resize(target_size, Image.BILINEAR)
            return np.array(img, dtype=np.uint8)
        arr = np.asarray(img)
        if target_size is not None and arr.shape[:2] != target_size[::-1]:
            from PIL import Image
            arr = np.array(Image.fromarray(arr).resize(target_size, Image.BILINEAR))
        return arr

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        from skimage.metrics import structural_similarity
        # ATM evaluates grayscale SSIM after resizing both images to 425 px.
        p = self._load(predictions, target_size=(425, 425))
        g = self._load(ground_truths, target_size=(425, 425))
        if p.ndim == 3:
            p = np.asarray(p, dtype=np.float32).mean(axis=-1)
        if g.ndim == 3:
            g = np.asarray(g, dtype=np.float32).mean(axis=-1)
        score = structural_similarity(p, g, win_size=self.win_size,
                                       channel_axis=None,
                                       data_range=255)
        return float(score)

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        scores = []
        for p, g in zip(predictions, ground_truths):
            try:
                scores.append(self.compute(p, g))
            except Exception:
                continue
        return float(np.mean(scores)) if scores else 0.0
