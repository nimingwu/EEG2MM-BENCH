"""
mAP — mean Average Precision for retrieval tasks.
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional

from .base import Metric


class MeanAveragePrecision(Metric):
    """mean Average Precision (mAP) for ranked retrieval.

    Evaluates the quality of the ranking: a high mAP means the correct
    item is ranked near the top AND all relevant items are retrieved.

    Usage
    -----
    >>> m = MeanAveragePrecision()
    >>> predictions = [[0, 3, 1], [2, 0, 1]]      # ranked indices per query
    >>> ground_truths = [0, 0]                       # correct index per query
    >>> m.batch_compute(predictions, ground_truths)
    """

    @property
    def name(self) -> str:
        return "map"

    @property
    def applicable(self) -> List[str]:
        # 仅图像检索；speech 检索用专用实例级指标 speech_map
        return ["image_retrieval"]

    def _ap(self, ranked: List[int], correct: int) -> float:
        """Average precision for a single query (single-relevant-item case)."""
        for k, idx in enumerate(ranked, 1):
            if idx == correct:
                return 1.0 / k
        return 0.0

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return self._ap(predictions, ground_truths)

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        aps = []
        for pred, gt in zip(predictions, ground_truths):
            aps.append(self._ap(pred, gt))
        return float(np.mean(aps)) if aps else 0.0

    def reset(self):
        pass
