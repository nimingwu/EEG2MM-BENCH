"""评测入口。

v2 把 ``evaluate_predictions`` 放在 **runner** ``run_atm_benchmark.py`` 里，结果
16 个模块都反过来依赖一个 runner 脚本——这是 v2 最别扭的耦合之一。v3 把它提到
中立的 ``evaluation`` 层，逻辑逐条不变。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

from eeg2x.data import write_json
from eeg2x.rawdata import require_benchmark_output

from .evaluator import (
    discover_h5_files, extract_ground_truths, generate_report, print_report,
)
from .registry import get_task_config
from eeg2x.h5_schema import validate_h5_file

#: 带 EEG 输入契约的图像子任务。检索/分类当前没有接线，但契约推导对它们同样成立。
_IMAGE_TASKS = ("image_retrieval", "image_generation", "concept_classification")


def evaluate_predictions(*, task: str, h5_dir: str | Path,
                         predictions_path: str | Path,
                         output_path: str | Path,
                         metrics: Sequence[str] | None = None) -> Path:
    """Evaluate with the unchanged EEG2X schema and metric registry."""
    h5_root = require_benchmark_output(h5_dir)
    prediction_file = require_benchmark_output(predictions_path)
    output = require_benchmark_output(output_path)
    h5_files = discover_h5_files(str(h5_root), task)
    if not h5_files:
        raise ValueError(f"No benchmark HDF5 files for {task} in {h5_root}")
    task_config = get_task_config(task)
    selected_metrics = (list(metrics) if metrics is not None
                        else list(task_config.get("default_metrics", ())))
    violations: List[str] = []
    for info in h5_files:
        ok, errors = validate_h5_file(info["path"], task_config["h5_schema"])
        if not ok:
            violations.append(f"{info['filename']}: {'; '.join(errors)}")
    if violations:
        raise ValueError("Benchmark HDF5 schema violations: " + " | ".join(violations))
    raw = json.loads(prediction_file.read_text(encoding="utf-8"))
    ground_truths, sample_meta = extract_ground_truths(h5_files, task)
    contract = _task_contract(task, h5_files, selected_metrics)
    report = generate_report(
        task=task, ground_truths=ground_truths,
        predictions=raw.get("predictions", raw), h5_files=h5_files,
        metric_names=selected_metrics,
        candidate_pool_ids=raw.get("candidate_pool_ids", []),
        candidate_embeddings=raw.get("candidate_embeddings"),
        sample_meta=sample_meta)
    if contract is not None:
        report["task_contract"] = contract
    print_report(report)
    return write_json(output, report)


def _task_contract(task: str, h5_files: List[dict],
                   metrics: Sequence[str]) -> dict | None:
    """把 ``eeg2x.tasks`` 的任务契约落到这次评测上。

    这是 v2 缺失的那条连线：任务接口在 v2 里是一份没人读的契约。v3 在评测时
    从产物 HDF5 推出真实的 EEG 几何（采样率、通道列表、时间窗、预处理步骤），
    与任务声明一起写进报告，使「这份报告是用哪套契约算出来的」可复查。
    """
    if task not in _IMAGE_TASKS:
        return None
    from eeg2x.tasks.image import TASK_REGISTRY, eeg_spec_from_h5_files

    spec = eeg_spec_from_h5_files([info["path"] for info in h5_files])
    task_spec = TASK_REGISTRY[task]
    return {
        "task": task,
        "task_id": task_spec.task_id,
        "output_type": task_spec.output_type,
        "metrics": list(metrics),
        "lower_is_better": list(task_spec.lower_is_better),
        "eeg": spec.to_dict(),
    }


__all__ = ["evaluate_predictions"]
