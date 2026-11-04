#!/usr/bin/env python
"""Task interface for ``video_generation`` (EEG2Video / SEED-DV).

Single source of truth (README principle 3)
-------------------------------------------
``sample_id`` is minted by ``benchmark_eval.extract_ground_truths`` and is
``"<h5 file stem>_clip-<row:04d>"``.  :meth:`VideoSample.from_h5_row` derives the
same string from the same inputs, so the interface and the evaluator can never
drift apart.

Prediction format
-----------------
A model returns one dict per sample::

    {"path": "<file>.npy#<row>"}          # one generated clip, or
    {"path": "<file>.npy", ...}           # the whole stacked array (row picked by sample_id)
    {"data_ref": "h5://<file>#<dataset>[<row>]"}

``path`` is what ``TASK_REGISTRY["video_generation"]["pred_required"]``
declares, and it is what ``benchmark_eval.generate_report`` validates before
scoring.  Missing keys are reported and counted, never silently dropped.

Ground truth
------------
The ground-truth clip lives inside the test HDF5 as ``gt_frames``; the evaluator
hands the metrics an ``h5://<file>#gt_frames[<row>]`` reference, which
``metrics/video_generation.py`` resolves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


TASK_ID = "video_generation"
DATASET_ID = "seed_dv"

#: ``epoch_type`` written by the loader for every epoch of this task.
EPOCH_TYPE = "video_clip"
#: Number of frames per clip, matching the released EEG2Video protocol.
N_FRAMES = 6
FRAME_HEIGHT = 288
FRAME_WIDTH = 512
#: Accepted ground-truth classes used by the N-way metrics.
GT_TOP_K = 3


def make_sample_id(h5_stem: str, row: int) -> str:
    """The authoritative sample id, identical to ``benchmark_eval``'s."""

    return f"{h5_stem}_clip-{row:04d}"


@dataclass
class VideoSample:
    """One EEG epoch plus the identity of the clip it was recorded against."""

    sample_id: str
    subject_id: str
    split: str
    eeg: np.ndarray                      # (n_channels, n_samples)
    channel_names: List[str]
    concept: Optional[int] = None
    caption: str = ""
    stimulus_id: str = ""
    stimulus_uri: str = ""
    stimulus_transform: Dict[str, Any] = field(default_factory=dict)
    gt_ref: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_h5_row(cls, h5: Any, row: int, *, h5_path: str | Path) -> "VideoSample":
        """Build a sample from one row of a loader-written HDF5.

        The row order of the HDF5 and the row order of the stimulus manifest are
        guaranteed identical by ``write_stimulus_manifest``; this method relies
        on that and asserts it rather than re-deriving alignment.
        """

        import json

        n_epochs = int(h5.attrs["n_epochs"])
        if not 0 <= row < n_epochs:
            raise IndexError(f"row {row} outside the {n_epochs} epochs of {h5_path}")
        for name in ("stimulus_id", "stimulus_uri", "stimulus_transform_json"):
            if name not in h5:
                raise KeyError(
                    f"{h5_path} has no stimulus manifest column {name!r}; "
                    "the file was not written by the v2 loader"
                )

        def _text(name: str, default: str = "") -> str:
            if name not in h5:
                return default
            value = h5[name][row]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        stem = Path(h5_path).stem
        eeg_field = "data" if "data" in h5 else "eeg"
        gt_ref = (
            f"h5://{Path(h5_path).resolve()}#gt_frames[{row}]"
            if "gt_frames" in h5 else ""
        )
        return cls(
            sample_id=make_sample_id(stem, row),
            subject_id=str(h5.attrs["subject_id"]),
            split=str(h5.attrs["split"]),
            eeg=np.asarray(h5[eeg_field][row], dtype=np.float32),
            channel_names=[
                c.decode("utf-8") if isinstance(c, bytes) else str(c)
                for c in h5["channel_names"][:]
            ],
            concept=int(h5["event_concept"][row]) if "event_concept" in h5 else None,
            caption=_text("event_caption"),
            stimulus_id=_text("stimulus_id"),
            stimulus_uri=_text("stimulus_uri"),
            stimulus_transform=json.loads(
                _text("stimulus_transform_json") or "{}"),
            gt_ref=gt_ref,
            meta={
                "dataset": str(h5.attrs["dataset_id"]),
                "subject": str(h5.attrs["subject_id"]),
                "split": str(h5.attrs["split"]),
            },
        )

    @property
    def n_channels(self) -> int:
        return int(self.eeg.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.eeg.shape[1])


@dataclass
class VideoGenerationOutput:
    """One generated clip, in the format ``benchmark_eval`` expects."""

    sample_id: str
    path: str = ""
    data_ref: str = ""
    query_embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_prediction(self) -> Dict[str, Any]:
        pred: Dict[str, Any] = {"sample_id": self.sample_id}
        if self.path:
            pred["path"] = self.path
        if self.data_ref:
            pred["data_ref"] = self.data_ref
        if self.query_embedding is not None:
            pred["query_embedding"] = self.query_embedding
        pred.update(self.meta)
        return pred


def validate_video_output(value: Any) -> Dict[str, Any]:
    """Return the prediction dict, or raise with an actionable message.

    Follows README principle 7: an output that cannot be scored is an error,
    not something to quietly drop.
    """

    if not isinstance(value, dict):
        raise TypeError(
            f"video_generation prediction must be a dict, got {type(value).__name__}"
        )
    if "path" not in value and "data_ref" not in value:
        raise KeyError(
            "video_generation prediction needs 'path' or 'data_ref' "
            f"(got keys: {sorted(value)})"
        )
    for key in ("path", "data_ref"):
        if key in value and not isinstance(value[key], str):
            raise TypeError(f"prediction['{key}'] must be a string")
    return value


def load_video_schema_registry() -> Dict[str, Any]:
    """Expose the task's registry entry for tests and tooling."""

    from core.registry import get_task_config

    return get_task_config(TASK_ID)
