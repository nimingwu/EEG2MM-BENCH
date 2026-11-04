"""GLIM Orig. baseline：同一个 GLIM 框架，domain prompt 全开。

它与框架里的 glim 变体共用同一份模型代码，区别只在于 checkpoint 的
``use_null_prompt=False`` 与 prompt dropout 全零。这里把它作为 baseline
单独登记，避免在 ``baselines/`` 里看不到它。
"""
from __future__ import annotations

from pathlib import Path

from eeg2x.datasets import DatasetSpec
from eeg2x.frameworks.glim.pipeline import (
    resolve_paths as _resolve_text_paths,
    run_inference as _run_text,
)


def run_inference(dataset: DatasetSpec, *, device: str = "cuda:0",
                  noise: str = "none", noise_seed: int = 2000,
                  overwrite: bool = False) -> Path:
    paths = _resolve_text_paths(dataset, "glim_orig")
    return _run_text(paths, device=device, noise=noise,
                     noise_seed=noise_seed, overwrite=overwrite)


__all__ = ["run_inference"]
