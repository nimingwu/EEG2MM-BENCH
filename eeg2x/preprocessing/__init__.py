"""预处理层：**一个数据集一个文件**，文件内按 profile 分支。

这是对 v2 结构的直接修正。v2 里预处理散在项目根目录的 4 个脚本 + 一个 1091 行的
单体 loader 里，且脚本之间互相 import 私有函数。v3 的约定：

* 每个数据集一个模块：``thingseeg.py`` / ``nod.py`` / ``grootswagers.py`` /
  ``grootswagers2025.py``；
* 每个模块暴露原生签名（数据集特有的参数由它自己决定）；
* **本文件**提供统一入口 :func:`preprocess`，按 ``profile_id`` 查几何参数并分派。

``common.py`` 是从 v2 的 ``eeg2image_benchmark_loader.py`` 原样迁来的**辅助函数库**
（MNN 白化、OSF 读取、H5 写入等）。它**不再是入口**——v2 的 loader 作为「统一入口」
的定位被这里的 4 个数据集模块取代。

几何参数（采样率 / 样本数 / 是否 LUNA 适配）由 profile 唯一确定：同一个编码器在不同
数据集上要求的窗口不同，所以这张表按 ``profile_id`` 索引，而不是按编码器。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict

from eeg2x.datasets import DatasetSpec

log = logging.getLogger("eeg2x.preprocessing")

#: profile_id -> 预处理几何。与 v2 ``run_rsvp_atm_seed2000.py:GEOMETRY`` 一致。
GEOMETRY: Dict[str, Dict[str, object]] = {
    # ThingsEEG：ATM 论文 epoch 裁到 [0,1.0) s
    "thingseeg-image-atm-v1":          {"output_prefix": "thingseeg_atm", "target_sr": 250.0, "output_samples": 250, "luna": False},
    "thingseeg-image-labram-atm-v1":   {"output_prefix": "thingseeg_atm", "target_sr": 200.0, "output_samples": 200, "luna": False},
    "thingseeg-image-reve-atm-v1":     {"output_prefix": "thingseeg_atm", "target_sr": 200.0, "output_samples": 200, "luna": False},
    "thingseeg-image-luna-atm-v1":     {"output_prefix": "thingseeg_atm", "target_sr": 256.0, "output_samples": 280, "luna": True},
    # ds004018：[-0.1, 1.0] s -> 276
    "grootswagers-image-atm-v1":        {"output_prefix": "grootswagers_atm", "target_sr": 250.0, "output_samples": 276, "luna": False},
    "grootswagers-image-labram-atm-v1": {"output_prefix": "grootswagers_labram_atm", "target_sr": 200.0, "output_samples": 200, "luna": False},
    "grootswagers-image-reve-atm-v1":   {"output_prefix": "grootswagers_reve_atm", "target_sr": 200.0, "output_samples": 200, "luna": False},
    "grootswagers-image-luna-atm-v1":   {"output_prefix": "grootswagers_luna_atm", "target_sr": 256.0, "output_samples": 280, "luna": True},
    # ds005106：[-0.1, 0.8] s -> 226
    "grootswagers2025-image-atm-paper-v1":        {"output_prefix": "grootswagers2025_atm", "target_sr": 250.0, "output_samples": 226, "luna": False},
    "grootswagers2025-image-labram-atm-paper-v1": {"output_prefix": "grootswagers2025_labram_atm", "target_sr": 200.0, "output_samples": 200, "luna": False},
    "grootswagers2025-image-reve-atm-paper-v1":   {"output_prefix": "grootswagers2025_reve_atm", "target_sr": 200.0, "output_samples": 200, "luna": False},
    "grootswagers2025-image-luna-atm-paper-v1":   {"output_prefix": "grootswagers2025_luna_atm", "target_sr": 256.0, "output_samples": 280, "luna": True},
    # NOD：[-0.1, 0.8] s -> 226
    "nod-image-atm-v1":        {"output_prefix": "nod_atm", "target_sr": 250.0, "output_samples": 226, "luna": False},
    "nod-image-labram-atm-v1": {"output_prefix": "nod_labram_atm", "target_sr": 200.0, "output_samples": 200, "luna": False},
    "nod-image-reve-atm-v1":   {"output_prefix": "nod_reve_atm", "target_sr": 200.0, "output_samples": 200, "luna": False},
    "nod-image-luna-atm-v1":   {"output_prefix": "nod_luna_atm", "target_sr": 256.0, "output_samples": 280, "luna": True},
}


def geometry_for(profile_id: str) -> Dict[str, object]:
    if profile_id not in GEOMETRY:
        raise KeyError(
            f"no preprocessing geometry for profile {profile_id!r}; "
            f"known: {sorted(GEOMETRY)}")
    return dict(GEOMETRY[profile_id])


def _thingseeg(subject, out_dir, profile_id, data_root, *, overwrite, **geom) -> int:
    """ThingsEEG 的签名只收 ``target_sr``，不收 ``output_samples`` / ``luna``。

    这不是遗漏：ThingsEEG 的 ATM epoch 裁剪和 LUNA 适配是在脚本内部按
    ``profile_id`` 分支完成的，几何参数不外包。所以这里**显式**只传它认识的。
    """
    from .thingseeg import preprocess_subject
    # things_root 必须留空：脚本会断言它等于仓库内的 ThingsEEG 符号链接，
    # 传入解析后的原始路径会被拒绝（v2 的契约，沿用）。
    return preprocess_subject(subject, out_dir, things_root=None,
                              profile_id=profile_id,
                              target_sr=geom["target_sr"],
                              overwrite=overwrite)


def _grootswagers(subject, out_dir, profile_id, data_root, *, overwrite, **geom) -> int:
    # ds004018 的脚本没有 overwrite 开关（总是重写），几何参数全部显式传入。
    from .grootswagers import process_grootswagers
    return process_grootswagers(subject, data_root, out_dir,
                                profile_id=profile_id,
                                target_sr=geom["target_sr"],
                                output_samples=geom["output_samples"],
                                luna=geom["luna"],
                                output_prefix=geom["output_prefix"],
                                layout="per_session")


def _grootswagers2025(subject, out_dir, profile_id, data_root, *, overwrite, **geom) -> int:
    from .grootswagers2025 import process_subject
    return process_subject(subject, data_root, out_dir,
                           profile_id=profile_id,
                           target_sr=geom["target_sr"],
                           output_samples=geom["output_samples"],
                           luna=geom["luna"],
                           output_prefix=geom["output_prefix"],
                           layout="per_session")


def _nod(subject, out_dir, profile_id, data_root, *, overwrite, **geom) -> int:
    from .nod import process_nod_atm
    return process_nod_atm(subject, data_root, out_dir,
                           profile_id=profile_id,
                           target_sr=geom["target_sr"],
                           output_samples=geom["output_samples"],
                           luna=geom["luna"],
                           output_prefix=geom["output_prefix"])


#: dataset name -> 适配器（统一签名）。
_ADAPTERS: Dict[str, Callable[..., int]] = {
    "thingseeg": _thingseeg,
    "nod": _nod,
    "grootswagers": _grootswagers,
    "grootswagers2025": _grootswagers2025,
}


# ---------------------------------------------------------------------------
# 视频数据集
# ---------------------------------------------------------------------------
#
# 视频 loader 的输入契约与图像不同：几何（窗口长度、采样率、是否需要零填充）
# 由「数据集 × 编码器」的 profile 分支自己决定，loader 内部已经按分支读完，
# 不需要 GEOMETRY 再给一份 target_sr / output_samples。所以视频用单独一张
# 适配表，签名也不同——多收一个 encoder，少收 profile_id 与几何参数。

def _seed_dv(subject, out_dir, encoder, data_root, *, overwrite):
    from .video import build_branch, process_subject
    number = int(str(subject).split("-")[-1])
    return process_subject(Path(data_root), Path(out_dir), number,
                           branches=[build_branch(encoder)])


def _ds005662(subject, out_dir, encoder, data_root, *, overwrite):
    from .video import ds005662_build_branch, ds005662_process_subject
    return ds005662_process_subject(Path(data_root), Path(out_dir), subject,
                                    branches=[ds005662_build_branch(encoder)])


def _ds008257(subject, out_dir, encoder, data_root, *, overwrite):
    from .video import (
        ds008257_build_branch, ds008257_process_subject,
        load_ds008257_captions, load_ds008257_stimuli,
    )
    return ds008257_process_subject(
        Path(data_root), load_ds008257_stimuli(), load_ds008257_captions(),
        Path(out_dir), subject, [ds008257_build_branch(encoder)])


_VIDEO_ADAPTERS: Dict[str, Callable[..., object]] = {
    "seed_dv": _seed_dv,
    "ds005662": _ds005662,
    "ds008257": _ds008257,
}


def preprocess(dataset: DatasetSpec, encoder: str, subject: str, *,
               out_dir: str | Path, data_root: str | Path | None = None,
               overwrite: bool = False, **extra) -> object:
    """把一个 (数据集, 编码器, 被试) 预处理成 HDF5。

    参数由 profile 决定（见 :data:`GEOMETRY`）；``extra`` 透传给数据集模块，
    用于确实只属于某个数据集的开关（如 ds004018 的 ``speeds``）。
    视频数据集走单独的分派表，见 ``_VIDEO_ADAPTERS``。
    """
    root = Path(data_root) if data_root is not None else Path(dataset.root)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    profile_id = dataset.profile_for(encoder)

    if dataset.name in _VIDEO_ADAPTERS:
        log.info("%s/%s/%s: profile=%s (video branch)", dataset.name, encoder,
                 subject, profile_id)
        return _VIDEO_ADAPTERS[dataset.name](
            subject, target, encoder, root, overwrite=overwrite)

    if dataset.name not in _ADAPTERS:
        raise KeyError(
            f"no preprocessing adapter for {dataset.name!r}; "
            f"known: {sorted(_ADAPTERS)} + video {sorted(_VIDEO_ADAPTERS)}")
    geometry = geometry_for(profile_id)
    log.info("%s/%s/%s: profile=%s sr=%s samples=%s luna=%s",
             dataset.name, encoder, subject, profile_id,
             geometry["target_sr"], geometry["output_samples"], geometry["luna"])
    return _ADAPTERS[dataset.name](
        subject, target, profile_id, root, overwrite=overwrite, **geometry, **extra)


__all__ = ["GEOMETRY", "geometry_for", "preprocess"]
