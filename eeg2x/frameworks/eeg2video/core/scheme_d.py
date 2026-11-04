"""Scheme D EEG condition models.

Scheme D replaces the paper's DE-feature MLP Semantic Predictor with a
time-domain EEGNet backbone.  The rest of the semantic target contract is
kept unchanged: the predictor returns the Stable-Diffusion-compatible
``[batch, 77, 768]`` tensor and is trained against the frozen CLIP text
encoder output with elementwise MSE.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as functional
from torch import nn


EEG_CHANNELS = 62
EEG_WINDOW_SAMPLES = 100
EEG_TOKENS = 7
CLIP_SEQUENCE_LENGTH = 77
CLIP_HIDDEN_SIZE = 768


class EEGNetBackbone(nn.Module):
    """EEG2X-style EEGNet-8,2 for one time-domain EEG window.

    Input:
        ``[batch, channels, samples]`` = ``[B, 62, 100]``
    Output:
        ``[batch, 48]`` for the default 100-sample window.
    """

    def __init__(
        self,
        n_chans: int = EEG_CHANNELS,
        n_samples: int = EEG_WINDOW_SAMPLES,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kern_length: int = 64,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        if n_samples < 32:
            raise ValueError(f"EEG window is too short: {n_samples}")
        self.n_chans = n_chans
        self.n_samples = n_samples
        self.F1, self.D, self.F2 = F1, D, F2
        self.features = nn.Sequential(
            nn.Conv2d(
                1, F1, (1, kern_length), padding="same", bias=False
            ),
            nn.BatchNorm2d(F1),
            nn.Conv2d(
                F1, D * F1, (n_chans, 1), groups=F1, bias=False
            ),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                D * F1,
                D * F1,
                (1, 16),
                groups=D * F1,
                padding="same",
                bias=False,
            ),
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )
        with torch.no_grad():
            probe = self.features(torch.zeros(1, 1, n_chans, n_samples))
        self.feature_dim = int(probe.flatten(1).shape[1])
        if self.feature_dim != F2 * (n_samples // 32):
            raise RuntimeError(
                "EEGNet feature-dimension formula disagrees with the actual "
                f"topology: actual={self.feature_dim}, expected={F2 * (n_samples // 32)}"
            )

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3:
            raise ValueError(f"Expected [B,C,T], got {tuple(eeg.shape)}")
        if tuple(eeg.shape[1:]) != (self.n_chans, self.n_samples):
            raise ValueError(
                f"Expected [B,{self.n_chans},{self.n_samples}], "
                f"got {tuple(eeg.shape)}"
            )
        return self.features(eeg.unsqueeze(1)).flatten(1)


class SchemeDSemanticPredictor(nn.Module):
    """EEGNet plus a trainable projection to the original semantic shape."""

    def __init__(self) -> None:
        super().__init__()
        self.eegnet = EEGNetBackbone()
        self.projection = nn.Sequential(
            nn.Linear(
                self.eegnet.feature_dim,
                CLIP_SEQUENCE_LENGTH * CLIP_HIDDEN_SIZE,
            )
        )

    def forward(self, eeg_windows: torch.Tensor) -> torch.Tensor:
        if eeg_windows.ndim != 4:
            raise ValueError(
                f"Expected [B,7,62,100], got {tuple(eeg_windows.shape)}"
            )
        if tuple(eeg_windows.shape[1:]) != (
            EEG_TOKENS,
            EEG_CHANNELS,
            EEG_WINDOW_SAMPLES,
        ):
            raise ValueError(
                "Unexpected Scheme D EEG shape: "
                f"{tuple(eeg_windows.shape)}"
            )
        batch, tokens, channels, samples = eeg_windows.shape
        window_features = self.eegnet(
            eeg_windows.reshape(batch * tokens, channels, samples)
        ).reshape(batch, tokens, -1)
        clip_feature = window_features.mean(dim=1)
        return self.projection(clip_feature).reshape(
            batch, CLIP_SEQUENCE_LENGTH, CLIP_HIDDEN_SIZE
        )


class SchemeDLinearAdapter(nn.Module):
    """Stage-two tokenwise Linear adapter preserving ``[B,77,768]``."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(CLIP_HIDDEN_SIZE, CLIP_HIDDEN_SIZE)

    def forward(self, semantic_embedding: torch.Tensor) -> torch.Tensor:
        if semantic_embedding.ndim != 3 or tuple(semantic_embedding.shape[1:]) != (
            CLIP_SEQUENCE_LENGTH,
            CLIP_HIDDEN_SIZE,
        ):
            raise ValueError(
                "Expected semantic embedding [B,77,768], got "
                f"{tuple(semantic_embedding.shape)}"
            )
        return self.linear(semantic_embedding)


def scheme_d_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """The same elementwise MSE objective used by the original predictor."""

    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target mismatch: {tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )
    return functional.mse_loss(prediction, target)


def parameter_groups(model: SchemeDSemanticPredictor) -> Tuple[nn.Parameter, ...]:
    """Return all stage-one parameters, including the projection head."""

    return tuple(model.parameters())
