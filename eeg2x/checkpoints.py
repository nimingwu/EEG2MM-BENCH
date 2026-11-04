"""checkpoint 读取。

与 v2 的区别：v2 的 ``load_state_checkpoint`` 用 ``require_benchmark_output`` 断言路径
在仓库内——那条守卫是为**写**设计的。v3 这一阶段按计划**复用 v2 的训练产物**，
所以读的时候要允许 v2 根目录；写仍然只允许 v3 根目录（``paths.require_v3_output``）。

允许的读取根目录可用 ``EEG2X_CHECKPOINT_ROOTS`` 覆盖（``os.pathsep`` 分隔）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import torch
from torch import nn

from .paths import V2_ROOT, V3_ROOT


def allowed_roots() -> tuple[Path, ...]:
    override = os.environ.get("EEG2X_CHECKPOINT_ROOTS")
    if override:
        return tuple(Path(p).expanduser().resolve()
                     for p in override.split(os.pathsep) if p)
    return (V3_ROOT, V2_ROOT)


def _require_readable(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not any(resolved == root or root in resolved.parents
               for root in allowed_roots()):
        raise ValueError(
            f"checkpoint {resolved} is outside the allowed roots "
            f"{[str(r) for r in allowed_roots()]}; set EEG2X_CHECKPOINT_ROOTS to "
            "extend this")
    if not resolved.is_file():
        raise FileNotFoundError(f"checkpoint not found: {resolved}")
    return resolved


def load_state_checkpoint(model: nn.Module, checkpoint_path: str | Path, *,
                          expected_type: str | None = None) -> Dict[str, object]:
    """加载 v2/v3 的 checkpoint，并校验 artifact_type 与派生输入标记。

    语义逐条沿用 v2 ``model/atm/training.py:load_state_checkpoint``，只把路径守卫
    从「必须在仓库内」放宽为「必须在 :func:`allowed_roots` 之内」。
    """
    checkpoint = _require_readable(checkpoint_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid ATM checkpoint: {checkpoint}")
    if expected_type is not None and payload.get("artifact_type") != expected_type:
        raise ValueError(
            f"Checkpoint {checkpoint} is {payload.get('artifact_type')!r}, "
            f"expected {expected_type!r}")
    if bool(payload.get("derived_inputs_used", True)):
        raise ValueError(
            f"Checkpoint used forbidden external derived inputs: {checkpoint}")
    model.load_state_dict(payload["model_state"])
    return payload


__all__ = ["allowed_roots", "load_state_checkpoint"]
