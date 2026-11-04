"""ENIGMA baseline 的 v3 入口：**只推理**。

ENIGMA 的 `run_enigma` 是训练+推理一体的，但它有一个 checkpoint 复用分支：

    if overwrite or not checkpoint.is_file():  <训练>
    else:                                      <load_state_dict>

所以只推理的路径是：把 v2 训练好的 ``model/final.pt`` 放到 v3 的输出目录，
再用 ``overwrite=False`` 调用。**这正是 baseline 与框架的关键区别**——baseline
自带固定的编码器和完整训练流程，不经过 `eeg2x.encoders`，也不接受 ``--encoder``。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from eeg2x import encoders as encoder_registry
from eeg2x.datasets import DatasetSpec
from eeg2x.paths import artifact_dir, h5_dir

from .pipeline import run_enigma

log = logging.getLogger("eeg2x.baselines.enigma")


def run_inference(dataset: DatasetSpec, subject: str, *,
                  device: str = "cuda:0",
                  overwrite_outputs: bool = False,
                  noise: str = "none", noise_seed: int = 2000,
                  distinct_images: int | None = None) -> Path:
    """从 v2 checkpoint 重跑 ENIGMA 的推理全链路并评测。

    ``noise="gaussian"`` 时 checkpoint 仍读干净根目录，产物写到
    ``<干净根>/noise_control/``（见 :mod:`eeg2x.noise_control`）。
    """
    source = artifact_dir(dataset.name, "enigma", subject)          # v2：训练产物
    output = artifact_dir(dataset.name, "enigma", subject, layout="v3")
    checkpoint = source / "model" / "final.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"no trained ENIGMA checkpoint at {checkpoint}; the inference-only "
            "milestone reads v2 artifacts and never trains")
    feature_store = source / "features" / "clip_vit_h_seed2000.h5"
    if not feature_store.is_file():
        raise FileNotFoundError(f"missing CLIP feature store: {feature_store}")

    # 只清理输出，不碰复制进来的 checkpoint。
    if overwrite_outputs and output.is_dir():
        for child in output.iterdir():
            if child.name == "model":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()

    staged = output / "model" / "final.pt"
    staged.parent.mkdir(parents=True, exist_ok=True)
    if not staged.is_file():
        shutil.copy2(checkpoint, staged)
        log.info("staged v2 checkpoint -> %s", staged)

    return run_enigma(
        dataset=dataset.name, subject=subject,
        profile_id=dataset.profile_for("enigma_cnn"),
        # ENIGMA 的输入契约与 ATM 相同，复用 ATM 的 HDF5（见 EncoderSpec.h5_encoder）。
        h5_dir=h5_dir(dataset.name, _h5_owner(), subject),
        artifact_root=output,
        feature_store=feature_store,
        device=device, seed=2000,
        overwrite=False,          # 关键：False 表示「已有 checkpoint 就跳过训练」
        noise=noise, noise_seed=noise_seed,
        distinct_images=distinct_images,
    )


def _h5_owner() -> str:
    spec = encoder_registry.resolve("enigma_cnn")
    return spec.h5_encoder or spec.name


__all__ = ["run_inference"]
