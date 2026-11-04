"""
MCD — Mel Cepstral Distortion for speech reconstruction quality.

Lower is better. Measures the distance between mel-cepstral coefficients
of generated and ground-truth audio.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, List, Optional, Union
from pathlib import Path

from .base import Metric

log = logging.getLogger(__name__)


class MCD(Metric):
    """Mel Cepstral Distortion.

    Computes mel-cepstral coefficients via pysptk, then calculates the
    root-mean-square distance between frames (DTW-aligned if lengths differ).

    Parameters
    ----------
    n_mels : int
        Order of mel-cepstral coefficients (default 24, per literature).
    use_dtw : bool
        Whether to DTW-align before computing frame-wise distance.
    """

    def __init__(self, n_mels: int = 24, use_dtw: bool = True):
        self.n_mels = n_mels
        self.use_dtw = use_dtw

    @property
    def name(self) -> str:
        return "mcd"

    @property
    def applicable(self) -> List[str]:
        return ["speech_reconstruction"]

    def _load_audio(self, audio: Any, sr: int = 16000) -> np.ndarray:
        """Load audio as mono float array."""
        import scipy.io.wavfile as wavfile

        if isinstance(audio, (str, Path)):
            orig_sr, data = wavfile.read(str(audio))
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float64)
            # Resample if needed
            if orig_sr != sr:
                from scipy.signal import resample_poly
                import fractions
                ratio = fractions.Fraction(sr, orig_sr)
                data = resample_poly(data, ratio.numerator, ratio.denominator)
        elif isinstance(audio, np.ndarray):
            data = audio.astype(np.float64)
        else:
            data = np.asarray(audio, dtype=np.float64)

        return data

    def _mcd_frame(self, mcep_ref: np.ndarray, mcep_syn: np.ndarray) -> float:
        """MCD between two frames (single frame or mean over frames)."""
        diff = mcep_ref - mcep_syn
        return 10.0 / np.log(10) * np.sqrt(2 * (diff @ diff))

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        """MCD between two audio signals (single utterance)."""
        sr = kwargs.get("sample_rate", 16000)
        ref = self._load_audio(ground_truths, sr)
        syn = self._load_audio(predictions, sr)

        import pysptk

        # Extract mel-cepstral coefficients
        # Both must have same sr
        mcep_ref = pysptk.mcep(ref, order=self.n_mels, fs=sr)
        mcep_syn = pysptk.mcep(syn, order=self.n_mels, fs=sr)

        if self.use_dtw:
            # DTW alignment
            from pysptk import dtw
            dist, _ = dtw(mcep_ref, mcep_syn, distance=lambda x, y: np.linalg.norm(x - y))
            return float(dist / len(mcep_ref))
        else:
            # Simple frame-wise average (requires equal length)
            min_len = min(len(mcep_ref), len(mcep_syn))
            mcep_ref = mcep_ref[:min_len]
            mcep_syn = mcep_syn[:min_len]
            frames = [self._mcd_frame(mcep_ref[i], mcep_syn[i])
                      for i in range(min_len)]
            return float(np.mean(frames))

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        scores = []
        for p, g in zip(predictions, ground_truths):
            try:
                scores.append(self.compute(p, g, **kwargs))
            except Exception as exc:
                log.warning(f"MCD failed: {exc}")
        return float(np.mean(scores)) if scores else 0.0
