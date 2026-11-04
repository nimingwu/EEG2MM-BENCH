"""LUNA-specific, protocol-visible EEG input adaptation."""
from __future__ import annotations

from typing import Dict

import numpy as np


LUNA_SAMPLING_RATE = 256
LUNA_VALID_SAMPLES = 256
LUNA_PATCH_SIZE = 40
LUNA_INPUT_SAMPLES = 280
LUNA_PAD_RIGHT = LUNA_INPUT_SAMPLES - LUNA_VALID_SAMPLES
LUNA_ZSCORE_EPS = 1e-8


def luna_input_state() -> Dict[str, object]:
    """Return the machine-readable post-MNN input adaptation contract."""
    return {
        "sampling_rate_hz": LUNA_SAMPLING_RATE,
        "valid_samples": LUNA_VALID_SAMPLES,
        "patch_size_samples": LUNA_PATCH_SIZE,
        "output_samples": LUNA_INPUT_SAMPLES,
        "normalization": "per_sample_per_channel_zscore",
        "std_correction": 1,
        "epsilon": LUNA_ZSCORE_EPS,
        "padding": {"side": "right", "samples": LUNA_PAD_RIGHT, "value": 0.0},
    }


def prepare_luna_epochs(epochs: np.ndarray) -> np.ndarray:
    """Z-score each valid 1 s channel, then right-pad to seven 40-point patches."""
    value = np.asarray(epochs, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != LUNA_VALID_SAMPLES:
        raise ValueError(
            "LUNA adaptation requires [epochs, channels, 256] before padding, "
            f"got {tuple(value.shape)}"
        )
    if not np.isfinite(value).all():
        raise ValueError("LUNA adaptation received non-finite EEG")
    mean = value.mean(axis=-1, keepdims=True)
    # Match torch.std() in the official ChannelWiseNormalize (correction=1).
    std = value.std(axis=-1, keepdims=True, ddof=1)
    normalized = (value - mean) / (std + LUNA_ZSCORE_EPS)
    padded = np.pad(
        normalized,
        ((0, 0), (0, 0), (0, LUNA_PAD_RIGHT)),
        mode="constant",
        constant_values=0.0,
    )
    if padded.shape[-1] % LUNA_PATCH_SIZE:
        raise RuntimeError("LUNA padded length is not divisible by its pretrained patch size")
    return np.ascontiguousarray(padded, dtype=np.float32)


__all__ = [
    "LUNA_INPUT_SAMPLES",
    "LUNA_PAD_RIGHT",
    "LUNA_PATCH_SIZE",
    "LUNA_SAMPLING_RATE",
    "LUNA_VALID_SAMPLES",
    "luna_input_state",
    "prepare_luna_epochs",
]
