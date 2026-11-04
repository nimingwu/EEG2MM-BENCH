"""
CER — Character Error Rate for speech-to-text evaluation.

Lower is better. Measures character-level edit distance between
ASR-transcribed generated speech and ground-truth transcription.

Used in EEG-to-Speech papers (e.g. Park et al.) to evaluate whether
reconstructed speech preserves linguistic content.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, List, Optional

from .base import Metric

log = logging.getLogger(__name__)


class CER(Metric):
    """Character Error Rate.

    Similar to WER but operates at the character level. Useful for
    languages with character-based writing or for fine-grained
    speech reconstruction evaluation.

    Lower is better. 0 = perfect character-level match.
    """

    @property
    def name(self) -> str:
        return "cer"

    @property
    def applicable(self) -> List[str]:
        return ["speech_reconstruction", "sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        import jiwer
        ref = str(ground_truths)
        hyp = str(predictions)
        return float(jiwer.cer(ref, hyp))

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        import jiwer
        refs = [str(g) for g in ground_truths]
        hyps = [str(p) for p in predictions]
        return float(jiwer.cer(refs, hyps))
