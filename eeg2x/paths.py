"""路径解析：v3 新布局，以及只读 v2 产物的适配层。

v2 的命名**不统一**，是历史包袱：

| 用途 | v2 路径 |
| --- | --- |
| 框架 + ThingsEEG | ``artifacts/only_clip_linear_seed2000/<enc>/<sub-XX>/`` |
| 框架 + 其余数据集 | ``artifacts/<dataset>_<enc>_atm_sub<tag>_seed2000/`` |
| 完整 ATM | ``artifacts/<dataset>_atm_full_sub<tag>_seed2000/``（NOD 例外，见下） |
| ENIGMA / AVDE / Dummy | ``artifacts/<model>_<dataset>_sub<tag>_seed2000/`` |

其中 ``tag = subject.replace("-", "")``。**NOD 的完整 ATM 是例外**：
``artifacts/nod_sub01_atm_full_seed2000/``（主语在前）。

v3 的新布局把这些统一成 ``artifacts/<dataset>/<model>/<subject>[/<encoder>]``。
本阶段用 ``layout="v2"`` 读取既有产物；``layout="v3"`` 供将来重跑使用。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

#: v3 仓库根（EEG2X-Benchmark-v3/）
V3_ROOT = Path(__file__).resolve().parents[1]

#: v2 仓库根，可通过环境变量覆盖（比如挪到别处）。
V2_ROOT = Path(os.environ.get(
    "EEG2X_V2_ROOT",
    V3_ROOT.parent / "EEG2X-Benchmark-v2-test",
)).resolve()

#: 视频仓库（EEG2X-Benchmark-v2 of fuxian-eeg2video-1）。产物在 runs/ 下，
#: HDF5 在 h5_seed_dv / h5_ds005662 / h5_ds008257 下。
V2_VIDEO_ROOT = Path(os.environ.get(
    "EEG2X_V2_VIDEO_ROOT",
    V3_ROOT.parents[1] / "fuxian-eeg2video-1" / "EEG2X-Benchmark-v2",
)).resolve()

#: 文本仓库（fuxian-GLIM 的 EEG2X-Benchmark-v2-GS）。产物在 outputs/ 下，
#: HDF5 在 zuco1_*_h5 / zuco2_*_h5 / chisco_*_h5 / tmnred_*_h5 下。
V2_TEXT_ROOT = Path(os.environ.get(
    "EEG2X_V2_TEXT_ROOT",
    V3_ROOT.parents[1] / "fuxian-GLIM" / "EEG2X-Benchmark-v2-GS",
)).resolve()

#: 框架的 v2 实验名（ThingsEEG 专用）。
V2_FRAMEWORK_EXPERIMENT = "only_clip_linear_seed2000"

#: 视频数据集 -> (v2 runs 子目录, v2 h5 子目录, dataset_id)。
_V2_VIDEO_LAYOUT = {
    "seed_dv": ("eeg2video", "h5_seed_dv", "seed_dv"),
    "ds005662": ("eeg2video_ds005662", "h5_ds005662", "ds005662"),
    "ds008257": ("eeg2video_ds008257", "h5_ds008257", "ds008257"),
}


def video_artifact_dir(dataset: str, encoder: str, subject: str) -> Path:
    """视频统一框架在 v2 仓库里的轨道目录。"""
    if dataset not in _V2_VIDEO_LAYOUT:
        raise KeyError(f"unknown video dataset {dataset!r}; "
                       f"known: {sorted(_V2_VIDEO_LAYOUT)}")
    runs, _h5, _dataset_id = _V2_VIDEO_LAYOUT[dataset]
    return V2_VIDEO_ROOT / "runs" / runs / encoder / subject


def video_h5_dir(dataset: str, encoder: str, *,
                 layout: Layout = "v2") -> Path:
    """视频统一框架某个编码器分支的 HDF5 目录（含全部被试）。

    v2 的布局是 ``h5_<dataset>/<encoder>/``；v3 重跑后是
    ``V3_ROOT/h5/<dataset>/<encoder>/``。两者都按被试命名文件，不按被建目录。
    """
    if layout == "v3":
        return V3_ROOT / "h5" / dataset / encoder
    if layout != "v2":
        raise ValueError(f"layout must be 'v2' or 'v3', got {layout!r}")
    if dataset not in _V2_VIDEO_LAYOUT:
        raise KeyError(f"unknown video dataset {dataset!r}; "
                       f"known: {sorted(_V2_VIDEO_LAYOUT)}")
    _runs, h5, _dataset_id = _V2_VIDEO_LAYOUT[dataset]
    return V2_VIDEO_ROOT / h5 / encoder


def video_dataset_id(dataset: str) -> str:
    """HDF5 文件名前缀与 event_concept 命名空间。"""
    try:
        return _V2_VIDEO_LAYOUT[dataset][2]
    except KeyError as exc:
        raise KeyError(f"unknown video dataset {dataset!r}") from exc

Layout = Literal["v2", "v3"]

#: ThingsEEG 的框架目录直接用编码器原名（``only_clip_linear_seed2000/luna``）。
_V2_THINGSEEG_SLUG = {
    "atm_s": "atm",
    "labram": "labram",
    "reve": "reve",
    "luna": "luna",
    "enigma_cnn": "enigma",
}

#: 其余数据集把编码器写进扁平目录名（``grootswagers_labram_atm_sub01_...``）。
_V2_FLAT_SLUG = {
    "atm_s": "atm",
    "labram": "labram_atm",
    "reve": "reve_atm",
    "luna": "luna_atm",
    "enigma_cnn": "enigma_atm",
}


def subject_tag(subject: str) -> str:
    """``sub-01`` -> ``sub01``（v2 的扁平命名用这个）。"""
    return str(subject).replace("-", "")


def _check_layout(layout: str) -> Layout:
    if layout not in ("v2", "v3"):
        raise ValueError(f"layout must be 'v2' or 'v3', got {layout!r}")
    return layout  # type: ignore[return-value]


def h5_dir(dataset: str, encoder: str, subject: str, *,
           layout: Layout = "v2") -> Path:
    """某个 (数据集, 编码器, 被试) 的 HDF5 目录。"""
    _check_layout(layout)
    if layout == "v3":
        return V3_ROOT / "h5" / dataset / encoder / subject

    tag = subject_tag(subject)
    if dataset == "thingseeg":
        # ThingsEEG 的框架 HDF5 用 ``<enc>/<sub-XX>`` 两级。
        slug = _V2_THINGSEEG_SLUG.get(encoder, encoder)
        return V2_ROOT / "h5" / V2_FRAMEWORK_EXPERIMENT / slug / subject
    slug = _V2_FLAT_SLUG.get(encoder, encoder)
    if dataset == "nod":
        if encoder == "atm_s":
            # NOD 的 ATM-S 框架行走 atm_linear 命名空间。
            return V2_ROOT / "h5" / f"nod_{tag}_atm_linear_seed2000"
        return V2_ROOT / "h5" / f"nod_{slug}_{tag}_seed2000"
    return V2_ROOT / "h5" / f"{dataset}_{slug}_{tag}_seed2000"


def artifact_dir(dataset: str, model: str, subject: str, *,
                 encoder: str | None = None, layout: Layout = "v2") -> Path:
    """某个 (数据集, 模型[, 编码器], 被试) 的产物目录。

    ``model`` 是 ``only_clip`` / ``atm_full`` / ``avde`` / ``enigma`` / ``dummy``。
    框架必须给 ``encoder``；baseline 传了就报错（它们不换编码器）。
    """
    _check_layout(layout)
    if layout == "v3":
        base = V3_ROOT / "artifacts" / dataset / model / subject
        return base / encoder if encoder else base

    tag = subject_tag(subject)
    if model == "only_clip":
        if not encoder:
            raise ValueError("model 'only_clip' needs an encoder")
        if dataset == "thingseeg":
            slug = _V2_THINGSEEG_SLUG.get(encoder, encoder)
            return V2_ROOT / "artifacts" / V2_FRAMEWORK_EXPERIMENT / slug / subject
        slug = _V2_FLAT_SLUG.get(encoder, encoder)
        if dataset == "nod" and encoder == "atm_s":
            # NOD 的 ATM-S 框架行走 atm_linear 命名空间。
            return V2_ROOT / "artifacts" / f"nod_{tag}_atm_linear_seed2000"
        return V2_ROOT / "artifacts" / f"{dataset}_{slug}_{tag}_seed2000"
    if encoder is not None:
        raise ValueError(
            f"baseline {model!r} must not carry an encoder (its encoder is fixed); "
            f"got {encoder!r}")
    if model == "atm_full":
        # NOD 是例外：主语在前。
        if dataset == "nod":
            return V2_ROOT / "artifacts" / f"nod_{tag}_atm_full_seed2000"
        return V2_ROOT / "artifacts" / f"{dataset}_atm_full_{tag}_seed2000"
    return V2_ROOT / "artifacts" / f"{model}_{dataset}_{tag}_seed2000"


def require_v3_output(path: Path) -> Path:
    """断言要**写入**的路径在 v3 仓库内（沿用 v2 的产物必须可追溯原则）。

    只对写生效；读 v2 产物走 :func:`artifact_dir` / :func:`h5_dir`，不经过这里。
    """
    resolved = Path(path).resolve()
    if V3_ROOT not in resolved.parents and resolved != V3_ROOT:
        raise ValueError(
            f"benchmark outputs must live under {V3_ROOT}, got {resolved}")
    return resolved


__all__ = [
    "Layout",
    "V2_ROOT",
    "V2_VIDEO_ROOT",
    "V3_ROOT",
    "artifact_dir",
    "h5_dir",
    "require_v3_output",
    "subject_tag",
    "video_artifact_dir",
    "video_dataset_id",
    "video_h5_dir",
]
