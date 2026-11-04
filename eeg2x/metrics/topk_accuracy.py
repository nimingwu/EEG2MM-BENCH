"""
Top-K Accuracy
==============
Standard metric for retrieval and classification tasks.

- Retrieval:   is the correct image among the top-K ranked candidates?
- Classification: is the correct class among the top-K predicted classes?
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional

from .base import Metric


class TopKAccuracy(Metric):
    """Top-K accuracy.

    Parameters
    ----------
    k : int  (default 1 for standard accuracy, 5 for top-5)
    subject_agg : bool  (default False)
        When True and ``subject_ids`` are passed to ``batch_compute``, compute
        the metric per subject and macro-average across subjects
        (mirrors NeuralBench's ``*_subject-agg`` retrieval variants).

    Usage
    -----
    >>> m = TopKAccuracy(k=1)
    >>> # Retrieval: predictions = ranked_indices per sample, gt = correct_index
    >>> m.compute(predicted_ranks=[0, 2, 0], ground_truths=0)      # → 0.667
    >>> m.batch_compute(predictions=[[0,3,1], [2,0,1]], gts=[0, 1])  # gt index in top-1?
    """

    def __init__(self, k: int = 1, subject_agg: bool = False):
        self.k = k
        self.subject_agg = subject_agg
        self._name = f"top{k}_accuracy"
        self._correct = 0
        self._total = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def applicable(self) -> List[str]:
        # 仅图像检索/分类；speech 检索用专用实例级指标 speech_*_accuracy
        return ["image_retrieval", "concept_classification"]

    def _batch_score(self,
                     predictions: List[Any],
                     ground_truths: List[Any]) -> float:
        correct = 0
        for pred, gt in zip(predictions, ground_truths):
            if gt in pred[:self.k]:
                correct += 1
        return correct / max(len(ground_truths), 1)

    # ------------------------------------------------------------------
    def compute(self,
                predictions: Any,
                ground_truths: Any,
                **kwargs) -> float:
        """Single-sample.

        predictions : list[int] of length ≥ K  (ranked indices, best first)
        ground_truths : int  (the correct index)
        """
        if ground_truths in predictions[:self.k]:
            return 1.0
        return 0.0

    # ------------------------------------------------------------------
    def batch_compute(self,
                      predictions: List[Any],
                      ground_truths: List[Any],
                      **kwargs) -> float:
        """Batch.

        predictions :  List[List[int]]  — each inner list is ranked candidate indices
        ground_truths : List[int]       — correct index per sample
        subject_ids :  List[str]        — optional per-sample subject (for
                                          subject_agg=True, macro-average)
        """
        subject_ids = kwargs.get("subject_ids")
        if self.subject_agg and subject_ids is not None:
            from collections import defaultdict
            groups = defaultdict(lambda: ([], []))
            for pred, gt, sid in zip(predictions, ground_truths, subject_ids):
                groups[sid][0].append(pred)
                groups[sid][1].append(gt)
            if not groups:
                return 0.0
            scores = [self._batch_score(ps, gs) for ps, gs in groups.values()]
            return float(np.mean(scores))
        return self._batch_score(predictions, ground_truths)

    # ------------------------------------------------------------------
    def reset(self):
        self._correct = 0
        self._total = 0

    def update(self, predictions: List[Any], ground_truths: List[Any]):
        """Accumulate batch results (for streaming evaluation)."""
        for pred, gt in zip(predictions, ground_truths):
            self._total += 1
            if gt in pred[:self.k]:
                self._correct += 1

    def finalize(self) -> float:
        """Return accumulated accuracy."""
        if self._total == 0:
            return 0.0
        return self._correct / self._total
