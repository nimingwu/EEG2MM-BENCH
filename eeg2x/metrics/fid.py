"""
FID — Fréchet Inception Distance for evaluating generated image quality.

Lower is better. Compares the distribution of generated images vs real images
in the feature space of an Inception-v3 network.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, List, Optional, Union
from pathlib import Path

from .base import Metric

log = logging.getLogger(__name__)


class FID(Metric):
    """Fréchet Inception Distance.

    Uses Inception-v3 pool3 features. Requires at least ~1000 samples
    for a stable estimate. For smaller sets, use CLIP similarity instead.

    Parameters
    ----------
    batch_size : int
        Batch size for feature extraction.
    """

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self._model = None
        self._device = None

    @property
    def name(self) -> str:
        return "fid"

    @property
    def applicable(self) -> List[str]:
        return ["image_generation"]

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        import torchvision.models as models
        from torchvision import transforms

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # Torchvision 0.20 requires aux_logits=True when pretrained weights are
        # selected. Eval-mode forward returns the primary logits tensor only.
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        ).eval().to(self._device)
        # Replace fc with identity → keeps pool3 features [N, 2048, 1, 1]
        inception.fc = torch.nn.Identity()
        self._model = inception
        self._torch = torch
        # Use explicit transforms for cross-version compatibility
        self._to_tensor = transforms.Compose([
            transforms.Resize(342, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_images(self, paths: List[Any]) -> "torch.Tensor":
        """Load and preprocess a batch of images → [N, 3, 299, 299]."""
        import torch
        from PIL import Image

        tensors = []
        for p in paths:
            if isinstance(p, (str, Path)):
                img = Image.open(p).convert("RGB")
            elif isinstance(p, np.ndarray):
                arr = p
                if arr.dtype in (np.float32, np.float64):
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(arr)
            else:
                img = p
            tensors.append(self._to_tensor(img))
        return torch.stack(tensors).to(self._device)

    def _extract_features(self, images: "torch.Tensor") -> np.ndarray:
        """Extract Inception pool3 features → [N, 2048]."""
        import torch
        feats = []
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            with torch.no_grad():
                feat = self._model(batch)
                feat = feat.view(feat.size(0), -1)
            feats.append(feat.cpu().numpy())
        return np.concatenate(feats, axis=0)

    def _compute_fid(self, mu1, sigma1, mu2, sigma2) -> float:
        """Compute Fréchet distance between two Gaussians."""
        diff = mu1 - mu2
        # Product of covariance matrices
        from scipy.linalg import sqrtm
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)

    @staticmethod
    def _compute_fid_from_features(
        predicted_features: np.ndarray,
        ground_truth_features: np.ndarray,
    ) -> float:
        """Compute the exact empirical FID without a 2048 x 2048 square root.

        ThingsEEG has 200 test concepts, so both empirical covariance matrices
        are rank deficient in Inception's 2048-dimensional feature space.  If
        ``A`` and ``B`` are centered features scaled by ``sqrt(n - 1)``, then
        ``trace(sqrt((A.T A) (B.T B)))`` is the nuclear norm of ``A B.T``.
        This is algebraically equivalent to the standard FID expression while
        reducing the expensive decomposition to at most ``n_pred x n_gt``.
        """
        predicted = np.asarray(predicted_features, dtype=np.float64)
        ground_truth = np.asarray(ground_truth_features, dtype=np.float64)
        if predicted.ndim != 2 or ground_truth.ndim != 2:
            raise ValueError("FID features must be two-dimensional")
        if predicted.shape[1] != ground_truth.shape[1]:
            raise ValueError(
                "FID feature dimensions must match, got "
                f"{predicted.shape} and {ground_truth.shape}")
        if len(predicted) < 2 or len(ground_truth) < 2:
            raise ValueError("FID requires at least two images in each set")

        mean_predicted = predicted.mean(axis=0)
        mean_ground_truth = ground_truth.mean(axis=0)
        centered_predicted = (
            predicted - mean_predicted
        ) / np.sqrt(len(predicted) - 1)
        centered_ground_truth = (
            ground_truth - mean_ground_truth
        ) / np.sqrt(len(ground_truth) - 1)
        covariance_trace = float(np.square(centered_predicted).sum())
        covariance_trace += float(np.square(centered_ground_truth).sum())
        cross = centered_predicted @ centered_ground_truth.T
        covariance_root_trace = float(
            np.linalg.svd(cross, compute_uv=False).sum())
        mean_distance = float(
            np.square(mean_predicted - mean_ground_truth).sum())
        # Roundoff can produce a tiny negative result for identical inputs.
        return max(
            0.0,
            mean_distance + covariance_trace - 2.0 * covariance_root_trace,
        )
    
    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        """FID over a single pair — not meaningful. Use batch_compute."""
        raise NotImplementedError("FID requires a batch. Use batch_compute.")
    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        """Compute FID between two lists of images."""
        self._lazy_load()
        if len(predictions) < 50:
            log.warning(f"FID with {len(predictions)} samples may be unstable")

        log.info(f"  Extracting features for {len(predictions)} pred + "
                 f"{len(ground_truths)} GT images ...")
        try:
            p_feats = self._extract_features(self._load_images(predictions))
            g_feats = self._extract_features(self._load_images(ground_truths))
        except Exception as exc:
            log.warning(f"FID feature extraction failed: {exc}")
            return float("nan")

        return self._compute_fid_from_features(p_feats, g_feats)
