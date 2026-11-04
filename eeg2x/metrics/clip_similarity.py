"""
CLIP two-way identification
===========================
Full-set two-way identification between generated and ground-truth images.

Used as the primary quality metric for EEG-to-Image generation.
Also used for EEG-conditioned retrieval evaluation (embedding-space alignment).

Reference: Radford et al. (2021) — Learning Transferable Visual Models
           From Natural Language Supervision.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, List, Optional, Union
from pathlib import Path

from .base import Metric

log = logging.getLogger(__name__)
HF_CACHE = "/datas/hf-cache/hub"


class CLIPSimilarity(Metric):
    """CLIP cosine similarity between two images (or image + text).

    Parameters
    ----------
    model_name : str
        HuggingFace CLIP model id.
        Default: "openai/clip-vit-base-patch32" (fast, good enough).
        For higher quality: "openai/clip-vit-large-patch14".
    device : str
        "cpu" | "cuda". Auto-detected if None.

    Usage
    -----
    >>> m = CLIPSimilarity()
    >>> # Image generation: compare generated vs ground-truth images
    >>> m.compute("gen/img_0042.png", "gt/img_0042.png")            # 0.87
    >>> m.batch_compute(["gen/1.png","gen/2.png"], ["gt/1.png","gt/2.png"])

    >>> # Retrieval: compare retrieved vs ground-truth image
    >>> m.compute("retrieved/carousel.jpg", "gt/carousel_11s.jpg")
    """

    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 device: str = None):
        self.model_name = model_name
        self._device = device
        self._model = None
        self._processor = None

    @property
    def name(self) -> str:
        return "clip_similarity"

    @property
    def applicable(self) -> List[str]:
        return ["image_generation", "image_retrieval"]

    # ------------------------------------------------------------------
    def _lazy_load(self):
        """Lazy-import CLIP — only when actually used."""
        if self._model is not None:
            return
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers"
            )
        import torch

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Loading CLIP model '{self.model_name}' on {device} ...")
        self._model = CLIPModel.from_pretrained(
            self.model_name, cache_dir=HF_CACHE, local_files_only=True
        ).to(device).eval()
        self._processor = CLIPProcessor.from_pretrained(
            self.model_name, cache_dir=HF_CACHE, local_files_only=True
        )
        self._torch = torch
        self._device_str = device

    # ------------------------------------------------------------------
    def _embed_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """Return L2-normalised image embedding (numpy)."""
        self._lazy_load()
        if isinstance(image, (str, Path)):
            from PIL import Image
            image = Image.open(image).convert("RGB")

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device_str) for k, v in inputs.items()}
        with self._torch.no_grad():
            emb = self._model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def _embed_text(self, text: str) -> np.ndarray:
        """Return L2-normalised text embedding."""
        self._lazy_load()
        inputs = self._processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self._device_str) for k, v in inputs.items()}
        with self._torch.no_grad():
            emb = self._model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    # ------------------------------------------------------------------
    def compute(self,
                predictions: Any,
                ground_truths: Any,
                **kwargs) -> float:
        """Single-sample cosine similarity.

        predictions : str|np.ndarray  — generated image (path or RGB array)
        ground_truths : str|np.ndarray — ground-truth image (path or RGB array)
        """
        emb_pred = self._embed_image(predictions)
        emb_gt = self._embed_image(ground_truths)
        return float(np.dot(emb_pred, emb_gt))

    # ------------------------------------------------------------------
    def batch_compute(self,
                      predictions: List[Any],
                      ground_truths: List[Any],
                      **kwargs) -> float:
        """ATM-paper CLIP two-way identification over a complete batch.

        predictions : List[str|ndarray]  — generated images
        ground_truths : List[str|ndarray] — ground-truth images
        """
        self._lazy_load()
        from PIL import Image

        # Load images, skip invalid paths
        def _load(path_or_arr):
            if isinstance(path_or_arr, (str, Path)):
                if not Path(path_or_arr).exists():
                    return None
                return Image.open(path_or_arr).convert("RGB")
            return path_or_arr

        images_pred = [_load(p) for p in predictions]
        images_gt = [_load(g) for g in ground_truths]
        valid = [(i, p, g) for i, (p, g) in enumerate(zip(images_pred, images_gt))
                 if p is not None and g is not None]
        if not valid:
            log.warning("No valid image pairs found for CLIP similarity")
            return float("nan")

        _, valid_pred, valid_gt = zip(*valid)

        # Batch-encode predictions
        inp_pred = self._processor(images=list(valid_pred), return_tensors="pt")
        inp_pred = {k: v.to(self._device_str) for k, v in inp_pred.items()}
        inp_gt = self._processor(images=list(valid_gt), return_tensors="pt")
        inp_gt = {k: v.to(self._device_str) for k, v in inp_gt.items()}

        with self._torch.no_grad():
            emb_pred = self._model.get_image_features(**inp_pred)
            emb_pred = emb_pred / emb_pred.norm(dim=-1, keepdim=True)
            emb_gt = self._model.get_image_features(**inp_gt)
            emb_gt = emb_gt / emb_gt.norm(dim=-1, keepdim=True)

        from .perceptual_similarity import two_way_identification
        return two_way_identification(
            emb_gt.float().cpu().numpy(), emb_pred.float().cpu().numpy())

    # ------------------------------------------------------------------
    def compute_with_text(self,
                          images: List[Any],
                          texts: List[str]) -> float:
        """CLIP similarity between images and text labels (for retrieval).

        images : List[str|ndarray]  — generated/retrieved images
        texts  : List[str]           — concept labels, e.g. "a photo of a carousel"
        """
        self._lazy_load()
        from PIL import Image

        imgs = [
            Image.open(p).convert("RGB") if isinstance(p, (str, Path)) else p
            for p in images
        ]

        inp_img = self._processor(images=imgs, return_tensors="pt")
        inp_img = {k: v.to(self._device_str) for k, v in inp_img.items()}
        inp_txt = self._processor(text=texts, return_tensors="pt", padding=True)
        inp_txt = {k: v.to(self._device_str) for k, v in inp_txt.items()}

        with self._torch.no_grad():
            emb_img = self._model.get_image_features(**inp_img)
            emb_img = emb_img / emb_img.norm(dim=-1, keepdim=True)
            emb_txt = self._model.get_text_features(**inp_txt)
            emb_txt = emb_txt / emb_txt.norm(dim=-1, keepdim=True)

        sims = (emb_img * emb_txt).sum(dim=-1)
        return float(sims.mean().cpu().item())

    def reset(self):
        self._model = None
        self._processor = None
