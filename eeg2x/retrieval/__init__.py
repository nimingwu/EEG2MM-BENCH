"""检索任务：EEGBench 轨道的评测、推理适配与 Dummy 基线。"""
from .adapter import infer
from .dummy import (
    DUMMY_MODEL, evaluate_dummy, list_dummy_results,
)
from .evaluate import evaluate_run
from .registry import DATASETS, MODELS, iter_runs, resolve_run

#: Learned retrieval models plus the fixed-target Dummy baseline.
ALL_MODELS = (*MODELS, DUMMY_MODEL)

__all__ = [
    "ALL_MODELS", "DATASETS", "DUMMY_MODEL", "MODELS", "evaluate_dummy",
    "evaluate_run", "infer", "iter_runs", "list_dummy_results", "resolve_run",
]
