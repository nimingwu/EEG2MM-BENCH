"""
STOI — Short-Time Objective Intelligibility for speech reconstruction.

Higher is better (range 0-1). Measures the intelligibility of generated
speech compared to ground-truth reference.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, List, Optional, Union
from pathlib import Path

from .base import Metric

log = logging.getLogger(__name__)


class STOI(Metric):
    """Short-Time Objective Intelligibility.

    Uses the pystoi implementation. Higher values indicate better
    speech intelligibility. Typical range: 0 (unintelligible) to 1 (perfect).

    Parameters
    ----------
    sr : int
        Sample rate (must match both reference and degraded audio).
    extended : bool
        Use extended STOI (better correlation with human judgment).
    """

    def __init__(self, sr: int = 16000, extended: bool = True):
        self.sr = sr
        self.extended = extended

    @property
    def name(self) -> str:
        return "stoi"

    @property
    def applicable(self) -> List[str]:
        return ["speech_reconstruction"]

    def _load(self, audio: Any) -> np.ndarray:
        import scipy.io.wavfile as wavfile

        if isinstance(audio, (str, Path)):
            orig_sr, data = wavfile.read(str(audio))
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float64)
            if orig_sr != self.sr:
                from scipy.signal import resample_poly
                import fractions
                ratio = fractions.Fraction(self.sr, orig_sr)
                data = resample_poly(data, ratio.numerator, ratio.denominator)
        elif isinstance(audio, np.ndarray):
            data = audio.astype(np.float64)
        else:
            data = np.asarray(audio, dtype=np.float64)
        return data

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        from pystoi import stoi
        ref = self._load(ground_truths)
        syn = self._load(predictions)
        # pystoi handles resampling internally but we already match sr
        score = stoi(ref, syn, self.sr, extended=self.extended)
        return float(score)

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        scores = []
        for p, g in zip(predictions, ground_truths):
            try:
                scores.append(self.compute(p, g, **kwargs))
            except Exception as exc:
                log.warning(f"STOI failed: {exc}")
        return float(np.mean(scores)) if scores else 0.0
