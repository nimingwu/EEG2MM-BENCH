"""AVDE baseline 的 v3 入口：**只推理**。

AVDE 是两阶段自回归：stage1 用 LaBraM 做对比微调，stage2 训 VAR。它的 pipeline
自带 checkpoint 复用：

    stage1: if overwrite or not (checkpoint-best.pth or checkpoint-final.pth): 训练
    stage2: if overwrite or not ar-ckpt-best.pth:                              训练

所以只推理的路径与 ENIGMA 一致：把 v2 的四个 checkpoint 放进 v3 输出目录，
再以 ``overwrite=False`` 调用。

**输入分支**：AVDE 原配的编码器是 LaBraM，所以它读的是 **LaBraM 的 HDF5 分支**
（v2 的 ``run_avde_seed2000.py`` 也是这么配的），不是 ATM 分支。

**权重**：``avde/pretrained``（1.7 GB）在 v3 里是**指向 v2 的软链接**，不复制。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from eeg2x.datasets import DatasetSpec
from eeg2x.paths import artifact_dir, h5_dir

from .pipeline import run_avde

log = logging.getLogger("eeg2x.baselines.avde")

#: 需要从 v2 搬过来的 checkpoint（相对 v2 产物目录）。
CHECKPOINTS = (
    "stage1/checkpoint-best.pth",
    "stage1/checkpoint-final.pth",
    "stage2/ar-ckpt-best.pth",
    "stage2/ar-ckpt-last.pth",
)


def run_inference(dataset: DatasetSpec, subject: str, *,
                  device: str = "cuda:0",
                  overwrite_outputs: bool = False,
                  noise: str = "none", noise_seed: int = 2000,
                  distinct_images: int | None = None) -> Path:
    """从 v2 checkpoint 重跑 AVDE 的推理全链路并评测。

    ``noise="gaussian"`` 时四个 checkpoint 仍读干净根目录，只有测试集 EEG 被替换，
    生成与评测产物写到 ``<干净根>/noise_control/``。
    """
    source = artifact_dir(dataset.name, "avde", subject)
    output = artifact_dir(dataset.name, "avde", subject, layout="v3")
    feature_store = source / "features" / "clip_vit_h_seed2000.h5"
    if not feature_store.is_file():
        raise FileNotFoundError(f"missing CLIP feature store: {feature_store}")

    staged = 0
    for relative in CHECKPOINTS:
        src = source / relative
        if not src.is_file():
            continue
        dst = output / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file():
            shutil.copy2(src, dst)
            staged += 1
    if not (output / "stage2/ar-ckpt-best.pth").is_file():
        raise FileNotFoundError(
            f"no trained AVDE checkpoint under {source}; the inference-only "
            "milestone reads v2 artifacts and never trains")
    log.info("staged %d v2 checkpoint(s) -> %s", staged, output)

    if overwrite_outputs and output.is_dir():
        for child in output.iterdir():
            if child.name in ("stage1", "stage2"):
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()

    return run_avde(
        dataset=dataset.name, subject=subject,
        # AVDE 原配 LaBraM：读 LaBraM 的 HDF5 分支。
        profile_id=dataset.profile_for("labram"),
        h5_dir=h5_dir(dataset.name, "labram", subject),
        artifact_root=output,
        feature_store=feature_store,
        device=device, seed=2000,
        overwrite=False,        # 关键：已有 checkpoint 就跳过两个阶段的训练
        noise=noise, noise_seed=noise_seed,
        distinct_images=distinct_images,
    )


__all__ = ["CHECKPOINTS", "run_inference"]
