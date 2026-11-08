"""Low-level VAE-latent EEG decoder from the released ATM implementation."""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class LowLevelEEGEncoder(nn.Module):
    """Map [B, C, T] EEG directly to an unscaled SDXL VAE latent.

    The released ATM implementation is written for the ThingsEEG2 geometry
    (C=63, T=250); both dimensions are parameters so the same branch can serve
    datasets with another channel count or epoch length.  The defaults reproduce
    the original architecture (and therefore its checkpoints) exactly.
    """

    def __init__(self, n_chans: int = 63, n_samples: int = 250):
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_samples = int(n_samples)
        self.subject_wise_linear = nn.ModuleList([nn.Linear(self.n_samples, 128)])
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.upsampler = nn.Sequential(
            nn.ConvTranspose2d(self.n_chans * 128, 1024, 4, 2, 1),
            nn.BatchNorm2d(1024), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 4, 1),
        )

    def forward(self, eeg: Tensor) -> Tensor:
        if eeg.ndim != 3 or eeg.shape[1:] != (self.n_chans, self.n_samples):
            raise ValueError(
                f"Low-level encoder expects [batch,{self.n_chans},{self.n_samples}], "
                f"got {tuple(eeg.shape)}")
        value = self.subject_wise_linear[0](eeg)
        return self.upsampler(
            value.reshape(value.shape[0], self.n_chans * 128, 1, 1))


DEFAULT_LOW_LEVEL_EPOCHS = 200
