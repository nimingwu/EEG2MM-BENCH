"""embed 阶段：把 HDF5 的 EEG 过一遍框架模型，写出 1024 维 embedding。

逐字沿用 v2 ``model/atm/training.py:extract_atm_embeddings`` 的产物格式
（``eeg_embedding`` / ``clip_target`` / ``stimulus_id`` / ``event_label``，
float16 存储，附带 manifest），差别只在：

* 模型是 v3 的 ``OnlyClipModel``，``forward(eeg)`` 不接收被试（构造时已绑定）；
* 支持只跑一个 split（本阶段只做推理，默认 ``test``）。
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Sequence

import h5py
import numpy as np
import torch

from eeg2x import noise_control
from eeg2x.data import (
    ATMH5Dataset, artifact_manifest, discover_subject_h5, write_json,
)
from eeg2x.protocol import sha256_file
from eeg2x.rawdata import require_benchmark_output

log = logging.getLogger("eeg2x.only_clip.embed")

SEED = 2000


def embed_split(model: torch.nn.Module, *, subject_id: str, h5_dir: str | Path,
                feature_store: str | Path, checkpoint: str | Path,
                output_dir: str | Path, split: str = "test",
                profile_id: str, batch_size: int = 256,
                device: str = "cuda:0", overwrite: bool = False,
                noise: str = "none", noise_seed: int = SEED) -> Path:
    """把一个 split 的 EEG 编码成 embedding HDF5。

    ``noise="gaussian"`` 是噪声对照（见 :mod:`eeg2x.noise_control`）：只替换
    **test** split 的 EEG，权重保持冻结。这里的注入点是框架级的那一个——五个
    可插拔编码器全部经过本函数，所以新增编码器时噪声条件自动生效。
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    output = require_benchmark_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = Path(output) / f"{split}_embeddings.h5"
    if destination.is_file() and not overwrite:
        # 嵌入必须与 checkpoint 对应：训练阶段换过权重后，旧嵌入不能复用，
        # 否则投影头会拟合到上一版权重的输出上。
        with h5py.File(destination, "r") as existing:
            previous = str(existing.attrs.get("atm_checkpoint_sha256", ""))
        if previous == sha256_file(checkpoint):
            log.info("reusing embeddings: %s", destination)
            return destination
        log.info("embeddings at %s came from another checkpoint; recomputing",
                 destination)

    paths = discover_subject_h5(h5_dir, subject_id, profile_id=profile_id, split=split)
    dataset = ATMH5Dataset(paths, feature_store, profile_id=profile_id,
                           cache_eeg=True, cache_features=True)
    target_device = torch.device(device)
    model = model.eval().to(target_device)
    temporary = destination.with_suffix(".h5.tmp")
    if temporary.exists():
        temporary.unlink()
    string_dtype = h5py.string_dtype(encoding="utf-8")
    try:
        with h5py.File(temporary, "w") as h5:
            eeg_ds = h5.create_dataset(
                "eeg_embedding", shape=(len(dataset), 1024), dtype=np.float16,
                chunks=(min(256, len(dataset)), 1024))
            clip_ds = h5.create_dataset(
                "clip_target", shape=(len(dataset), 1024), dtype=np.float16,
                chunks=(min(256, len(dataset)), 1024))
            ids = h5.create_dataset("stimulus_id", shape=(len(dataset),), dtype=string_dtype)
            labels = h5.create_dataset("event_label", shape=(len(dataset),), dtype=string_dtype)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            # 噪声对照：只替换 test 侧；生成器在循环外建一次，同一份噪声由
            # (种子, batch size, 数据顺序) 决定，与干净条件逐行对齐。
            generator = (noise_control.noise_generator(noise_seed)
                         if noise_control.noise_enabled(noise) and split == "test"
                         else None)
            # 结构性零填充不参与替换（目前只有 LUNA：280 = 256 有效 + 24 填充）。
            valid_samples = getattr(getattr(model, "encoder", None),
                                   "valid_samples", None)
            offset = 0
            with torch.inference_mode():
                for batch in loader:
                    eeg = noise_control.inject_noise(
                        batch["eeg"].to(target_device), generator,
                        valid_samples=valid_samples)
                    embeddings = model(eeg).float().cpu().numpy()
                    stop = offset + len(embeddings)
                    eeg_ds[offset:stop] = embeddings.astype(np.float16)
                    clip_ds[offset:stop] = batch["target"].numpy().astype(np.float16)
                    ids[offset:stop] = np.asarray(batch["stimulus_id"], dtype=object)
                    labels[offset:stop] = np.asarray(batch["label"], dtype=object)
                    offset = stop
            h5.attrs["schema"] = "eeg2x-atm-eeg-embeddings-v1"
            h5.attrs["subject_id"] = subject_id
            h5.attrs["split"] = split
            h5.attrs["source_h5_json"] = json.dumps(dataset.fingerprints, sort_keys=True)
            h5.attrs["atm_checkpoint"] = str(Path(checkpoint).resolve())
            h5.attrs["atm_checkpoint_sha256"] = sha256_file(checkpoint)
            h5.attrs["derived_inputs_used"] = False
            h5.attrs["noise"] = noise
            if noise_control.noise_enabled(noise):
                h5.attrs["noise_seed"] = int(noise_seed)
        os.replace(temporary, destination)
        write_json(destination.with_suffix(".h5.manifest.json"), artifact_manifest(
            artifact_type="atm_eeg_embeddings",
            source_h5=dataset.fingerprints,
            config={"subject_id": subject_id, "split": split,
                    "batch_size": batch_size, "profile_id": profile_id,
                    **({"noise": noise, "noise_seed": int(noise_seed)}
                       if noise_control.noise_enabled(noise) else {})},
            extra={"artifact": str(destination),
                   "artifact_sha256": sha256_file(destination),
                   "atm_checkpoint": str(Path(checkpoint).resolve()),
                   "feature_store": str(Path(feature_store).resolve())}))
        return destination
    finally:
        if temporary.exists():
            temporary.unlink()
        dataset.close()


__all__ = ["embed_split"]
