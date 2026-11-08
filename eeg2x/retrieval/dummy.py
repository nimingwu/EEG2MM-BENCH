"""Retrieval Dummy baseline: canonical EEGBench results and their metrics.

The Dummy experiment is not a trained EEG model.  EEGBench builds NeuralBench's
``DummyPredictor(mode="auto", random_state=33)`` from the training targets
followed by the validation targets, predicts the same mean vector for every
test EEG, and then runs the normal ``full_retrieval`` evaluation.  The canonical
per-dataset result JSONs live in ``/EEG_bench_open/Dummy/runs``.

This module reads those canonical results.  It does not retrain or recompute
them, because the prepared caches used by the upstream Dummy run are not part
of this repository; ``EEG2X_DUMMY_ROOT`` can point at another copy of the
Dummy deliverable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .evaluate import METRIC_KEYS

DUMMY_MODEL = "dummy"

DUMMY_ROOT = Path(
    os.environ.get("EEG2X_DUMMY_ROOT", "/EEG_bench_open/Dummy")
).resolve()
DUMMY_RUNS = DUMMY_ROOT / "runs"

#: v3.1 retrieval dataset key -> filename stem in EEGBench Dummy/runs.
DATASET_TO_DUMMY = {
    "things_eeg": "things_eeg",
    "nod_eeg": "nod_eeg",
    "grootswagers2019_5hz": "grootswagers2019_5hz",
    "grootswagers2019_20hz": "grootswagers2019_20hz",
    "grootswagers2025": "grootswagers2025",
    "zuco1": "zuco1",
    "zuco2": "zuco2",
    "chisco": "chisco_author_imagined_text",
    "tmnred": "tmnred",
    "cofett": "cofett",
    "ds005662": "ds005662",
    "seed_dv": "seed_dv",
    "emd": "emd",
    "eeg_video_action": "eeg_video_action",
    "brennan2019": "brennan2019",
}


def dummy_result_path(dataset: str) -> Path:
    """Canonical Dummy result JSON for a v3.1 retrieval dataset key."""
    if dataset not in DATASET_TO_DUMMY:
        raise KeyError(
            f"no Dummy result for dataset {dataset!r}; "
            f"known: {sorted(DATASET_TO_DUMMY)}")
    path = DUMMY_RUNS / f"{DATASET_TO_DUMMY[dataset]}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"no Dummy result at {path}; set EEG2X_DUMMY_ROOT to the "
            "EEG_bench_open Dummy deliverable")
    return path


def evaluate_dummy(dataset: str, *, path: str | Path | None = None) -> Dict[str, Any]:
    """Load one canonical Dummy result and return its retrieval metrics."""
    source = Path(path) if path is not None else dummy_result_path(dataset)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Dummy result is not complete: {source}")
    missing = [key for key in METRIC_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Dummy result {source} is missing metrics: {missing}")
    return {
        "run_dir": str(source),
        "dataset": dataset,
        "model": DUMMY_MODEL,
        "method": payload.get("method"),
        "mode": payload.get("mode"),
        "random_state": payload.get("random_state"),
        "fit_row_order": payload.get("fit_row_order"),
        "fit_rows": payload.get("fit_rows"),
        "test_rows": payload.get("test_rows"),
        "reference_run": payload.get("reference_run"),
        "prediction_vector_sha256": payload.get("prediction_vector_sha256"),
        **{key: float(payload[key]) for key in METRIC_KEYS},
        "n_candidates": payload.get("n_candidates"),
        "n_subject_target_queries": payload.get("n_subject_target_queries"),
    }


def list_dummy_results() -> List[Tuple[str, Path]]:
    """Available (v3.1 dataset key, canonical result path) pairs."""
    rows: List[Tuple[str, Path]] = []
    for dataset, stem in DATASET_TO_DUMMY.items():
        path = DUMMY_RUNS / f"{stem}.json"
        if path.is_file():
            rows.append((dataset, path))
    return sorted(rows)


__all__ = [
    "DATASET_TO_DUMMY", "DUMMY_MODEL", "DUMMY_ROOT", "dummy_result_path",
    "evaluate_dummy", "list_dummy_results",
]
