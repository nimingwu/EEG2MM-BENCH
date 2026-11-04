"""EEGBench 推理适配器：用 EEGBench 自己的解释器重跑一条轨道的推理与评测。

EEGBench 建在 NeuralBench 上，依赖 Python 3.12 与 neuroai 的四个包。本仓库
不复制它的模型层，也不替它安装依赖：通过 ``EEG2X_EEGBENCH_PYTHON`` 指定一个
已经可用的解释器，然后调用它自己的 ``eegbench.launch ... evaluate``。

``evaluate`` 会从 ``weights.pt`` 重新加载模型、对测试集做一次前向，再重算
检索指标，因此是推理而不是训练。若解释器或 NeuralBench 不可用，这里给出
明确错误并指向 ``environments/retrieval.txt``，不会静默退回。
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict

from .registry import EEGBENCH_CODE, EEGBENCH_ROOT, launch_task, resolve_run


def eegbench_python() -> Path:
    value = os.environ.get("EEG2X_EEGBENCH_PYTHON") or os.environ.get("EEGBENCH_PYTHON")
    if not value:
        raise RuntimeError(
            "EEGBench inference needs its own interpreter; set "
            "EEG2X_EEGBENCH_PYTHON=/path/to/python (see environments/retrieval.txt)")
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"EEG2X_EEGBENCH_PYTHON does not exist: {path}")
    return path


def _probe(python: Path) -> None:
    probe = subprocess.run(
        [str(python), "-c",
         "import neuralbench, neuraltrain, neuralset, neuralfetch"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(
            "EEGBench runtime is not importable with "
            f"{python}:\n{probe.stderr.strip()[-1200:]}\n"
            "Restore the pinned neuroai packages (see environments/retrieval.txt); "
            "no inference was run.")


def _pythonpath() -> str:
    parts = [str(EEGBENCH_CODE / "src")]
    vendor = os.environ.get("EEG2X_NEUROAI_VENDOR")
    if vendor:
        parts.append(vendor)
    old = os.environ.get("PYTHONPATH")
    if old:
        parts.append(old)
    return os.pathsep.join(parts)


def infer(dataset: str, model: str, seed: int, *, device: str = "cuda") -> Path:
    """在 EEGBench 自己的环境里重跑一条轨道的推理与评测。"""
    spec = resolve_run(dataset, model, seed)
    python = eegbench_python()
    _probe(python)
    task = launch_task(dataset)
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env.setdefault("PYTHONHASHSEED", "2000")
    command = [str(python), "-m", "eegbench.launch", task, "evaluate",
               str(spec.directory), "--device", device]
    completed = subprocess.run(command, cwd=str(EEGBENCH_CODE), env=env,
                               capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"EEGBench inference failed ({completed.returncode}):\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}")
    return spec.directory


def describe() -> Dict[str, Any]:
    return {
        "root": str(EEGBENCH_ROOT),
        "code": str(EEGBENCH_CODE),
        "python": os.environ.get("EEG2X_EEGBENCH_PYTHON"),
        "vendor": os.environ.get("EEG2X_NEUROAI_VENDOR"),
    }


__all__ = ["describe", "eegbench_python", "infer"]
