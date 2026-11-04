"""检索评测：从 EEGBench 的轨道产物重算 subject-agg 指标。

EEGBench 的 full retrieval 是：同一被试同一目标的所有重复先取均值，然后对
该被试的候选目标按余弦相似度排序，统计 top1/5/10 与中位排名。本模块逐条
复现这一计算，并在存在 ``metrics.json`` 时逐项对照。

它不需要 NeuralBench，也不导入 torch：只读 ``predictions.npz`` 与
``test_samples.jsonl``，因此可以在现在的任何解释器里运行。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np


REFERENCE_TOLERANCE = 2e-3  # 参考实现的分组/并列处理未公开，留出浮点与并列误差

METRIC_KEYS = (
    "test/full_retrieval/top1_acc_subject-agg",
    "test/full_retrieval/top5_acc_subject-agg",
    "test/full_retrieval/top10_acc_subject-agg",
    "test/full_retrieval/median_rank_subject-agg",
)


def _load(run_dir: Path):
    data = np.load(run_dir / "predictions.npz")
    predictions = np.asarray(data["predictions"], dtype=np.float64)
    targets = np.asarray(data["targets"], dtype=np.float64)
    rows = [json.loads(line) for line in
            (run_dir / "test_samples.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if len(rows) != len(predictions) or predictions.shape != targets.shape:
        raise ValueError("predictions, targets and sample rows are not aligned")
    return predictions, targets, rows


def subject_agg_metrics(predictions: np.ndarray, targets: np.ndarray,
                        rows: list[dict]) -> Dict[str, Any]:
    """复现 EEGBench 的 subject-agg full retrieval 指标。

    每个 (subject, target_id) 的重复先取均值；目标候选集是**全局**的
    target_id 集合（同一目标在不同被试间的 target 向量取均值），这样缺少数
    个 subject-target 组合时，候选集不会缩小。相似度用余弦。
    """
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    per_target: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(str(row["subject"]), str(row["target_id"]))].append(index)
        per_target[str(row["target_id"])].append(index)

    target_ids = sorted(per_target)
    column = {target_id: i for i, target_id in enumerate(target_ids)}
    target_matrix = np.stack([targets[per_target[t]].mean(axis=0)
                              for t in target_ids])
    target_matrix /= np.maximum(
        np.linalg.norm(target_matrix, axis=1, keepdims=True), 1e-12)

    subjects = sorted({subject for subject, _ in groups})
    top1 = top5 = top10 = 0.0
    ranks: list[int] = []
    per_subject = {}
    for subject in subjects:
        items = sorted((target_id, indices) for (subj, target_id), indices
                       in groups.items() if subj == subject)
        pred = np.stack([predictions[idx].mean(axis=0) for _, idx in items])
        pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
        similarity = pred @ target_matrix.T
        order = np.argsort(-similarity, axis=1)
        subject_ranks = []
        for row_index, (target_id, _) in enumerate(items):
            subject_ranks.append(int(np.where(
                order[row_index] == column[target_id])[0][0]))
        ranks.extend(subject_ranks)
        rank = np.asarray(subject_ranks)
        per_subject[subject] = {
            "n_queries": len(items),
            "top1": float((rank == 0).mean()),
            "top5": float((rank < 5).mean()),
            "top10": float((rank < 10).mean()),
            "median_rank": float(np.median(rank)),
        }
        top1 += (rank == 0).mean()
        top5 += (rank < 5).mean()
        top10 += (rank < 10).mean()
    n = max(1, len(subjects))
    return {
        "test/full_retrieval/top1_acc_subject-agg": top1 / n,
        "test/full_retrieval/top5_acc_subject-agg": top5 / n,
        "test/full_retrieval/top10_acc_subject-agg": top10 / n,
        "test/full_retrieval/median_rank_subject-agg": float(np.median(ranks)) if ranks else float("nan"),
        "n_candidates": len(target_ids),
        "n_subject_target_queries": len(groups),
        "per_subject": per_subject,
    }


def evaluate_run(run_dir: str | Path, *, compare: bool = True) -> Dict[str, Any]:
    """重算一条轨道，并与同目录的 ``metrics.json`` 对照。"""
    run_dir = Path(run_dir)
    predictions, targets, rows = _load(run_dir)
    got = subject_agg_metrics(predictions, targets, rows)
    result: Dict[str, Any] = {"run_dir": str(run_dir), **got}
    reference_path = run_dir / "metrics.json"
    if compare and reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        diffs = {}
        for key in METRIC_KEYS:
            if key in reference:
                diffs[key] = abs(float(got[key]) - float(reference[key]))
        result["reference"] = {key: reference.get(key) for key in METRIC_KEYS}
        result["max_abs_diff"] = max(diffs.values()) if diffs else None
        result["matches_reference"] = all(
            value <= REFERENCE_TOLERANCE for value in diffs.values()) if diffs else None
    return result


__all__ = ["METRIC_KEYS", "evaluate_run", "subject_agg_metrics"]
