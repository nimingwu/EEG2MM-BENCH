"""
Base metric protocol.

All metrics inherit from this class so the benchmark runner can call them
uniformly without knowing implementation details.

To add a new metric:
  1. Create metrics/<name>.py with a class inheriting Metric.
  2. Register it in metrics/__init__.py → METRIC_REGISTRY.
  3. Done — the runner picks it up automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class Metric(ABC):
    """Abstract metric with a uniform call signature.

    Subclasses must implement:
      - name         → short string key, e.g. "clip_similarity"
      - applicable   → which tasks this metric is relevant for
      - compute      → the actual computation
      - reset        → clear accumulated state (for batched / streaming eval)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique key, e.g. 'top1_accuracy'."""

    @property
    @abstractmethod
    def applicable(self) -> List[str]:
        """List of task types this metric applies to.
        e.g. ['image_retrieval', 'image_generation', 'concept_classification']
        """

    @abstractmethod
    def compute(self,
                predictions: Any,
                ground_truths: Any,
                **kwargs) -> float:
        """Compute the metric given predictions and ground truths.

        Parameters
        ----------
        predictions : model output (type varies by task).
            - retrieval:    list of ranked indices / candidate ids
            - generation:   list of generated image paths or tensors
            - classification: list of (predicted_class, class_probs) tuples
        ground_truths : ground-truth labels (type varies by task).
        **kwargs : optional extra context (e.g. candidate_pool, image_root).

        Returns
        -------
        score : float
        """

    @abstractmethod
    def batch_compute(self,
                      predictions: List[Any],
                      ground_truths: List[Any],
                      **kwargs) -> float:
        """Compute metric over a batch of samples at once (preferred).

        The default implementation calls ``compute`` in a loop — override
        if the metric benefits from batch processing (e.g. CLIP can
        encode a stack of images in one forward pass).
        """

    def reset(self):
        """Reset accumulated state. Override in stateful metrics."""
        pass

    def __repr__(self) -> str:
        return f"Metric({self.name})"
