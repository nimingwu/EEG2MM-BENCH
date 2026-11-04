"""
Text generation metrics: BLEU, ROUGE-L, METEOR, WER.

All are well-established NLG / ASR metrics with available reference libraries.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, List, Optional

from .base import Metric

log = logging.getLogger(__name__)


# ===================================================================
# BLEU
# ===================================================================

class BLEU(Metric):
    """BLEU-N score for text generation.

    Parameters
    ----------
    max_order : int
        Maximum n-gram order (1-4). Default 4 (reports BLEU-1..4 average).
        To compute only BLEU-1 specifically, set max_order=1.

    Usage
    -----
    >>> m = BLEU(max_order=4)
    >>> m.compute("the cat is on the mat", "the cat is on the mat")
    """

    def __init__(self, max_order: int = 4):
        self.max_order = max_order

    @property
    def name(self) -> str:
        return f"bleu_{self.max_order}"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def _tokenize(self, text: str) -> List[str]:
        return text.strip().lower().split()

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref = self._tokenize(ground_truths)
        hyp = self._tokenize(predictions)
        weights = [1.0 / self.max_order] * self.max_order
        smoothie = SmoothingFunction().method4
        return float(sentence_bleu([ref], hyp, weights=weights,
                                    smoothing_function=smoothie))

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        scores = []
        for p, g in zip(predictions, ground_truths):
            try:
                scores.append(self.compute(p, g))
            except Exception as exc:
                log.warning(f"BLEU failed for pair: {exc}")
        return float(np.mean(scores)) if scores else 0.0


# ===================================================================
# ROUGE-L
# ===================================================================

class ROUGEL(Metric):
    """ROUGE-L F1 score — longest common subsequence based.

    Higher is better.
    """

    @property
    def name(self) -> str:
        return "rouge_l"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(ground_truths, predictions)
        return float(scores["rougeL"].fmeasure)

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        scores = []
        for p, g in zip(predictions, ground_truths):
            try:
                scores.append(self.compute(p, g))
            except Exception as exc:
                log.warning(f"ROUGE-L failed for pair: {exc}")
        return float(np.mean(scores)) if scores else 0.0


# ===================================================================
# METEOR
# ===================================================================

class METEOR(Metric):
    """METEOR — Metric for Evaluation of Translation with ORdering.

    Higher is better. Considers synonyms and stemming via WordNet.
    """

    @property
    def name(self) -> str:
        return "meteor"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        from nltk.translate.meteor_score import meteor_score
        ref_tokens = predictions.strip().lower().split()
        hyp_tokens = ground_truths.strip().lower().split()
        return float(meteor_score([ref_tokens], hyp_tokens))

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        scores = []
        for p, g in zip(predictions, ground_truths):
            try:
                scores.append(self.compute(p, g))
            except Exception as exc:
                log.warning(f"METEOR failed for pair: {exc}")
        return float(np.mean(scores)) if scores else 0.0


# ===================================================================
# WER — Word Error Rate
# ===================================================================

class WordErrorRate(Metric):
    """Word Error Rate — commonly used in ASR.

    Lower is better. Based on Levenshtein distance at the word level.
    """

    @property
    def name(self) -> str:
        return "wer"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation", "word_decoding", "speech_reconstruction"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        import jiwer
        return float(jiwer.wer(str(ground_truths), str(predictions)))

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        import jiwer
        refs = [str(g) for g in ground_truths]
        hyps = [str(p) for p in predictions]
        return float(jiwer.wer(refs, hyps))
