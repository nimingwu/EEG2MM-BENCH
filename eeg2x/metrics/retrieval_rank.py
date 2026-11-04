"""
Retrieval Rank
==============
Rank family metrics for retrieval tasks (mirrors NeuralBench's ``Rank``
metrics: median / mean / std of the ground-truth item's rank).

The rank of a query is the 1-based position of its ground-truth candidate in
the model's ranked list; a query whose GT is not retrieved scores
``len(ranked)`` (worse than every candidate).

Subject aggregation
-------------------
With ``subject_agg=True`` and per-sample ``subject_ids`` passed to
``batch_compute``, the reduction is applied per subject first, then the
per-subject values are macro-averaged (mirrors NeuralBench's
``*_subject-agg`` variants).

Usage
-----
>>> m = RetrievalRank("median")
>>> m.batch_compute(predictions=[["a","b","c"], ["b","a"]], gts=["b","a"])  # 2, 1 -> 1.5
>>> m_subj = RetrievalRank("median", subject_agg=True)
>>> m_subj.batch_compute([["a"], ["b"]], ["a", "b"], subject_ids=["S01", "S02"])
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional

from .base import Metric


class RetrievalRank(Metric):
    """Retrieval rank with median / mean / std reduction.

    Parameters
    ----------
    reduction : str  ("median" | "mean" | "std")
    subject_agg : bool  (default False) — per-subject macro-average
    """

    _REDUCTIONS = ("median", "mean", "std")

    def __init__(self, reduction: str = "median", subject_agg: bool = False):
        if reduction not in self._REDUCTIONS:
            raise ValueError(f"reduction must be one of {self._REDUCTIONS}")
        self.reduction = reduction
        self.subject_agg = subject_agg
        self._name = f"{reduction}_retrieval_rank"

    @property
    def name(self) -> str:
        return self._name

    @property
    def applicable(self) -> List[str]:
        # 仅图像检索；speech 检索用专用实例级指标 speech_*_retrieval_rank
        return ["image_retrieval"]

    # ------------------------------------------------------------------
    def _rank(self, ranked: List[Any], correct: Any,
              pool_size: Optional[int] = None) -> int:
        """1-based rank of the correct candidate; pool_size if not found.

        空排序（模型没有预测）不返回 0 —— 那会被当作"优于一切真实排名"。
        按原则 4/7：空预测应在评测入口被过滤为 invalid；若漏网，这里
        显式抛错而不是静默给出错误分数。
        """
        if not ranked:
            raise ValueError(
                "empty ranked list — prediction should have been rejected "
                "as invalid (principle 4: no silent degradation)"
            )
        for k, item in enumerate(ranked, 1):
            if item == correct:
                return k
        return pool_size if pool_size is not None else len(ranked)

    def _reduce(self, ranks: List[int]) -> float:
        if not ranks:
            return 0.0
        if self.reduction == "median":
            return float(np.median(ranks))
        if self.reduction == "mean":
            return float(np.mean(ranks))
        return float(np.std(ranks))

    # ------------------------------------------------------------------
    def compute(self,
                predictions: Any,
                ground_truths: Any,
                **kwargs) -> float:
        """Single-sample: predictions = ranked candidate list, gt = correct item."""
        return float(self._rank(predictions, ground_truths))

    def batch_compute(self,
                      predictions: List[Any],
                      ground_truths: List[Any],
                      **kwargs) -> float:
        """Batch: predictions = list of ranked lists, gts = correct items.

        subject_ids :  List[str]  — optional per-sample subject (for
                                    subject_agg=True, macro-average)
        """
        pool_size = kwargs.get("pool_size")
        ranks = [self._rank(p, g, pool_size=pool_size)
                 for p, g in zip(predictions, ground_truths)]
        subject_ids = kwargs.get("subject_ids")
        if self.subject_agg and subject_ids is not None:
            from collections import defaultdict
            groups = defaultdict(list)
            for r, sid in zip(ranks, subject_ids):
                groups[sid].append(r)
            if not groups:
                return 0.0
            return float(np.mean([self._reduce(v) for v in groups.values()]))
        return self._reduce(ranks)

    # ------------------------------------------------------------------
    def reset(self):
        pass
