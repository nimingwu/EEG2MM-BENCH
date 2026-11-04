"""检索任务：EEGBench 轨道的评测与推理适配。"""
from .adapter import infer
from .evaluate import evaluate_run
from .registry import DATASETS, MODELS, iter_runs, resolve_run

__all__ = ["DATASETS", "MODELS", "evaluate_run", "infer", "iter_runs", "resolve_run"]
