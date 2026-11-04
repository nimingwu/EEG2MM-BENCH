"""
BERTScore — Semantic similarity via BERT embeddings for text generation.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any, List, Optional

from .base import Metric

log = logging.getLogger(__name__)


class BERTScore(Metric):
    """BERTScore F1 — semantic similarity using BERT contextual embeddings.

    Higher is better. More robust than BLEU/ROUGE for free-form text.

    Parameters
    ----------
    model_type : str
        BERT model to use. Default "microsoft/deberta-xlarge-mnli"
        (best correlation with human judgment per BERTScore paper).
        For speed: "bert-base-uncased".
    batch_size : int
        Batch size for BERT encoding.
    """

    def __init__(self, model_type: str = "bert-base-uncased",
                 batch_size: int = 32):
        self.model_type = model_type
        self.batch_size = batch_size
        self._scorer = None

    @property
    def name(self) -> str:
        return "bertscore"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def _lazy_load(self):
        if self._scorer is not None:
            return
        from bert_score import BERTScorer
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log.info(f"Loading BERTScore model '{self.model_type}' on {device} ...")
        self._scorer = BERTScorer(
            model_type=self.model_type,
            lang="en",
            device=device,
            batch_size=self.batch_size,
        )

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        self._lazy_load()
        p, r, f1 = self._scorer.score([str(predictions)], [str(ground_truths)])
        return float(f1.item())

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        self._lazy_load()
        preds = [str(p) for p in predictions]
        refs = [str(g) for g in ground_truths]
        p, r, f1 = self._scorer.score(preds, refs)
        return float(f1.mean().item())
