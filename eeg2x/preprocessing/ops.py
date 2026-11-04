"""通用的 EEG 预处理原语：重采样、切 epoch、MNN 白化。

这些函数与数据集无关；ThingsEEG、NOD、Grootswagers 等模块共用它们。
"""
from __future__ import annotations

from typing import List

import numpy as np
import scipy

from eeg2x.protocol import normalize_channel_name, resample_eeg

TARGET_SR = 250


def _norm_ch(name: str) -> str:
    return normalize_channel_name(name)

def _keep_all_eeg(data: np.ndarray, src_channels: List[str]) -> np.ndarray:
    """Keep all EEG channels as-is (NICE uses all 63 channels)."""
    return data.astype(np.float32, copy=False)

def _resample(data: np.ndarray, orig_sr: float, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample on the exact declared EEG clock."""
    return resample_eeg(data, orig_sr, target_sr)

def _epoch_around_events(data: np.ndarray, event_samples: np.ndarray,
                         t_pre: float, t_post: float, sr: float) -> np.ndarray:
    n_pre = int(round(abs(t_pre) * sr))
    n_post = int(round(t_post * sr))
    epoch_len = n_pre + n_post
    n_chan = data.shape[0]
    epochs_list = []
    for es in event_samples:
        start = es - n_pre
        stop = es + n_post
        if start >= 0 and stop <= data.shape[1]:
            epochs_list.append(data[:, start:stop])
    if not epochs_list:
        return np.zeros((0, n_chan, 0), dtype=np.float32)
    return np.stack(epochs_list, axis=0).astype(np.float32)

def _valid_event_mask(n_samples: int, event_samples: np.ndarray,
                      t_pre: float, t_post: float, sr: float) -> np.ndarray:
    """Return the row mask used for epoching so metadata stays aligned."""
    n_pre = int(round(abs(t_pre) * sr))
    n_post = int(round(t_post * sr))
    samples = np.asarray(event_samples, dtype=np.int64)
    return (samples - n_pre >= 0) & (samples + n_post <= n_samples)

def _compute_mnn_whitener(epochs, concept_nums):
    """Compute MNN whitening matrix from data grouped by concept_num (NICE §4.1).
    Uses Ledoit-Wolf shrinkage covariance per epoch, averaged across conditions."""
    from sklearn.covariance import LedoitWolf
    n_ch = epochs.shape[1]
    unique_c = np.unique(concept_nums)
    sigma_cond = np.empty((len(unique_c), n_ch, n_ch))
    lw = LedoitWolf()
    for i, c in enumerate(unique_c):
        idx = np.where(concept_nums == c)[0]
        cond_epochs = epochs[idx]
        sigma_cond[i] = np.mean([
            lw.fit(cond_epochs[e].T).covariance_
            for e in range(len(cond_epochs))
        ], axis=0)
    sigma_tot = sigma_cond.mean(axis=0)
    return scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)

def _apply_whiten(epochs, sigma_inv):
    """Apply MNN whitening: epochs (N, ch, t) → (N, ch, t)."""
    return (epochs.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)
