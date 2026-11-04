"""
PixCorr — pixel-level correlation between two images.
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional, Union
from pathlib import Path

from .base import Metric


class PixCorr(Metric):
    """Pixel-wise Pearson correlation between generated and ground-truth images.

    Higher is better. Measures linear correspondence at the pixel level.
    """

    @property
    def name(self) -> str:
        return "pixcorr"

    @property
    def applicable(self) -> List[str]:
        return ["image_generation"]

    def _load_image(self, img: Any, target_size: tuple = None) -> np.ndarray:
        """Load image as flattened grayscale array.

        If target_size is given, resize the image to that size first.
        """
        if isinstance(img, (str, Path)):
            from PIL import Image
            img = Image.open(img).convert("L")  # grayscale
            if target_size is not None:
                img = img.resize(target_size, Image.BILINEAR)
            return np.array(img, dtype=np.float32).flatten()
        # Assume numpy array
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        if target_size is not None and arr.shape != target_size[::-1]:
            from PIL import Image
            arr = Image.fromarray(arr).resize(target_size, Image.BILINEAR)
            arr = np.array(arr, dtype=np.float32)
        return arr.flatten()

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        # The released ATM evaluation resizes both grayscale images to 425 px.
        p = self._load_image(predictions, target_size=(425, 425))
        g = self._load_image(ground_truths, target_size=(425, 425))
        corr = np.corrcoef(p, g)[0, 1]
        return float(corr) if not np.isnan(corr) else 0.0

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        scores = []
        for p, g in zip(predictions, ground_truths):
            try:
                scores.append(self.compute(p, g))
            except Exception:
                continue
        return float(np.mean(scores)) if scores else 0.0
