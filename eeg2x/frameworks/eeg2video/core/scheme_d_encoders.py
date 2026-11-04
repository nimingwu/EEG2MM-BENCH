"""Scheme D with a swappable EEG encoder.

Scheme D's original predictor is ``EEGNetBackbone`` + a projection head to
``[B, 77, 768]`` (``scheme_d.py``).  This module keeps that shape contract and
generalises the encoder slot, so LaBraM / REVE / LUNA can replace EEGNet
without touching the training loop, the loss, the CLIP targets or the
AnimateDiff generator.

What changes with the encoder, and why it must change
-----------------------------------------------------
An encoder's *window geometry* is not cosmetic -- it is set by the model's own
patch size, which was fixed during pretraining:

===========  =========  =========  =======  ==================================
encoder      window     stride     windows  why
===========  =========  =========  =======  ==================================
``eegnet``   100        50         7        EEG2Video paper, unchanged
``labram``   200        100        3        LaBraM's patch is 200 samples @200 Hz
``reve``     200        100        3        REVE's patch is 200 samples @200 Hz
``luna``     240        120        3        LUNA's patch is 40 samples @256 Hz
===========  =========  =========  =======  ==================================

LaBraM and REVE also refuse anything shorter than one whole patch -- REVE's
tokenizer is a bare ``unfold`` and LaBraM reshapes to ``(B, C, A, 200)`` --
so the original 100-sample window is not merely suboptimal for them, it is
impossible.  EEGNet is the only one of the four that can consume the paper's
7 x 100 samples, and it keeps them exactly.

The window geometry is stored in the HDF5 by the loader
(:data:`core.preprocessing_profiles.SEED_DV_ENCODER_WINDOWS`) and read back
here, so loader and model cannot drift apart.

Aggregation
-----------
Every encoder sees all windows of a clip independently and the resulting
feature vectors are averaged, exactly as Scheme D averages its seven EEGNet
windows.  That keeps "one clip -> one 77x768 semantic embedding" intact
regardless of how many windows the encoder's patch size produces.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn

from .scheme_d import (
    CLIP_HIDDEN_SIZE,
    CLIP_SEQUENCE_LENGTH,
    SchemeDSemanticPredictor,
)

def _build_encoder(*args, **kwargs):
    """``model.encoders.build_encoder``, imported on first use.

    Imported lazily on purpose: ``model.encoders`` pulls in this package while
    ``model/__init__.py`` is still running, so a module-level import here makes
    the result depend on which of the two was imported first.
    """

    from .encoders import build_encoder

    return build_encoder(*args, **kwargs)

#: Per-encoder window geometry.  Mirrors
#: ``core.preprocessing_profiles.SEED_DV_ENCODER_WINDOWS``; the h5 is the
#: authority, this is the fallback when a file predates the branch.
ENCODER_WINDOW_SPECS: Dict[str, Dict[str, int]] = {
    "eegnet": {"window_samples": 100, "stride_samples": 50, "n_windows": 7},
    "labram": {"window_samples": 200, "stride_samples": 100, "n_windows": 3},
    "reve": {"window_samples": 200, "stride_samples": 100, "n_windows": 3},
    "luna": {"window_samples": 240, "stride_samples": 120, "n_windows": 3},
    # EEGVid's modified-NICE encoder (arXiv:2505.21385) consumes a whole trace,
    # not short patches -- the paper feeds it NICE's kind of input (2 s at
    # 200 Hz).  One window spanning the epoch; must match the loader's
    # ``SEED_DV_ENCODER_WINDOWS['eegvid_nice']``.
    "eegvid_nice": {"window_samples": 400, "stride_samples": 400, "n_windows": 1},
}

#: Encoders whose input is already model-ready when it leaves the loader, so
#: Scheme D's in-model standardisation must not run a second time.
PREPROCESSED_ENCODERS = ("labram", "luna")


def windows_from_epoch(epochs: np.ndarray, spec: Dict[str, int]) -> np.ndarray:
    """``(n, C, T)`` clip -> ``(n, n_windows, C, window_samples)``.

    Same slicing rule as the paper's ``_windows_from_epoch`` (fixed stride,
    whole windows only, no padding); the geometry comes from ``spec``.
    """

    window = int(spec["window_samples"])
    stride = int(spec["stride_samples"])
    expected = int(spec["n_windows"])
    starts = np.arange(0, epochs.shape[-1] - window + 1, stride)
    if len(starts) != expected:
        raise ValueError(
            f"window spec {spec} yields {len(starts)} windows from "
            f"{epochs.shape[-1]} samples, expected {expected}"
        )
    return np.stack(
        [epochs[..., s:s + window] for s in starts], axis=1
    ).astype(np.float32)


class SchemeDEncoderPredictor(nn.Module):
    """``EEGNetBackbone`` replaced by any registered benchmark encoder.

    Drop-in for :class:`~eeg2video_core.scheme_d.SchemeDSemanticPredictor`:
    same input contract ``[B, n_windows, C, window_samples]``, same output
    contract ``[B, 77, 768]``, same MSE objective against frozen CLIP text
    embeddings.

    ``encoder="eegnet"`` is deliberately routed to the original
    :class:`SchemeDSemanticPredictor` instead of being re-expressed here, so
    the baseline's arithmetic stays byte-identical to ``scheme_d.py``.
    """

    def __init__(
        self,
        encoder: str = "eegnet",
        n_chans: int = 62,
        window_spec: Optional[Dict[str, int]] = None,
        channel_names: Optional[Sequence[str]] = None,
        input_clip: Optional[float] = None,
    ) -> None:
        super().__init__()
        if encoder not in ENCODER_WINDOW_SPECS:
            raise KeyError(
                f"unknown encoder {encoder!r}; expected one of "
                f"{list(ENCODER_WINDOW_SPECS)}"
            )
        self.encoder_name = encoder
        spec = dict(window_spec or ENCODER_WINDOW_SPECS[encoder])
        self.window_spec = spec
        self.n_chans = int(n_chans)
        self.n_windows = int(spec["n_windows"])
        self.window_samples = int(spec["window_samples"])
        #: REVE's released recipe clamps its (standardised) input at +/-15.
        self.input_clip = float(input_clip) if input_clip else 0.0

        # The paper's SchemeDSemanticPredictor hard-codes [.,7,62,100] and
        # asserts it, so it can only serve the SEED-DV EEGNet baseline it was
        # written for.  Any other geometry (ds005662 runs 2x64x100) goes
        # through the benchmark's shared EEGNet encoder instead -- same
        # architecture, but built for the branch's real shape.
        paper_geometry = (
            self.n_windows, self.n_chans, self.window_samples
        ) == (7, 62, 100)
        if encoder == "eegnet" and paper_geometry:
            # Unchanged code path: the paper's model, verbatim.
            self.baseline = SchemeDSemanticPredictor()
            self.backbone = self.baseline.eegnet
            self.projection = self.baseline.projection
            self.feature_dim = int(self.baseline.eegnet.feature_dim)
            self._uses_baseline = True
            return

        self._uses_baseline = False
        self.baseline = None
        self.backbone = _build_encoder(
            encoder, n_chans=self.n_chans, n_samples=self.window_samples,
            channel_names=list(channel_names) if channel_names else None,
        )
        self.feature_dim = int(self.backbone.feature_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, CLIP_SEQUENCE_LENGTH * CLIP_HIDDEN_SIZE)
        )

    def forward(self, eeg_windows: torch.Tensor) -> torch.Tensor:
        if self._uses_baseline:
            return self.baseline(eeg_windows)
        if eeg_windows.ndim != 4:
            raise ValueError(
                f"expected [B,{self.n_windows},{self.n_chans},"
                f"{self.window_samples}], got {tuple(eeg_windows.shape)}"
            )
        if tuple(eeg_windows.shape[1:]) != (
            self.n_windows, self.n_chans, self.window_samples
        ):
            raise ValueError(
                f"expected [B,{self.n_windows},{self.n_chans},"
                f"{self.window_samples}], got {tuple(eeg_windows.shape)}"
            )
        if self.input_clip > 0:
            eeg_windows = eeg_windows.clamp(-self.input_clip, self.input_clip)
        batch = eeg_windows.shape[0]
        flat = eeg_windows.reshape(
            batch * self.n_windows, self.n_chans, self.window_samples
        )
        features = self.backbone(flat).reshape(batch, self.n_windows, -1)
        clip_feature = features.mean(dim=1)
        return self.projection(clip_feature).reshape(
            batch, CLIP_SEQUENCE_LENGTH, CLIP_HIDDEN_SIZE
        )

    def describe(self) -> Dict[str, Any]:
        return {
            "encoder": self.encoder_name,
            "feature_dim": self.feature_dim,
            "window_samples": self.window_samples,
            "stride_samples": int(self.window_spec["stride_samples"]),
            "n_windows": self.n_windows,
            "n_chans": self.n_chans,
            "input_clip": self.input_clip,
            "window_aggregation": "mean over windows",
            "loss": "torch.nn.functional.mse_loss",
        }


def build_predictor(encoder: str, **kwargs) -> SchemeDEncoderPredictor:
    """Factory used by the training loop to stay encoder-agnostic."""

    return SchemeDEncoderPredictor(encoder=encoder, **kwargs)
