"""数据集的注册表：被试、各编码器对应的 versioned profile、原始数据根。

**profile 是 (数据集 x 编码器) 的预处理契约**，不是编码器的属性：同一个编码器
在不同数据集上要读几何不同的 H5。所以这张表放在数据集侧，而不是
``EncoderSpec`` 里。profile 的内容与 v2 ``core/preprocessing_profiles.py`` 一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .paths import V2_ROOT


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    subjects: tuple[str, ...]
    profiles: Mapping[str, str]      # 编码器 -> profile_id
    root: str                        # 原始数据根（预处理用；本阶段只读 H5）
    convention_subject: str = ""     # baseline 的惯例被试（v2 用 sub-08 / sub-01）
    task: str = "image_generation"   # 数据集所属任务，决定可用编码器与指标

    def profile_for(self, encoder: str) -> str:
        if encoder not in self.profiles:
            raise KeyError(
                f"dataset {self.name!r} has no profile for encoder {encoder!r}; "
                f"known: {sorted(self.profiles)}")
        return self.profiles[encoder]


def _subs(start: int, end: int) -> tuple[str, ...]:
    return tuple(f"sub-{n:02d}" for n in range(start, end + 1))


#: ThingsEEG2：10 被试，每被试 4 个 session。
THINGSEEG = DatasetSpec(
    name="thingseeg",
    subjects=_subs(1, 10),
    profiles={
        "atm_s": "thingseeg-image-atm-v1",
        "enigma_cnn": "thingseeg-image-atm-v1",
        "labram": "thingseeg-image-labram-atm-v1",
        "reve": "thingseeg-image-reve-atm-v1",
        "luna": "thingseeg-image-luna-atm-v1",
    },
    root="/datas/eeg_benchmark_project/ThingsEEG",
    convention_subject="sub-08",
)

#: NOD-EEG：**19 被试**（sub-01..14, 24, 26, 27, 29, 30），不是 20。
NOD = DatasetSpec(
    name="nod",
    subjects=(*(f"sub-{n:02d}" for n in range(1, 15)),
              "sub-24", "sub-26", "sub-27", "sub-29", "sub-30"),
    profiles={
        "atm_s": "nod-image-atm-v1",
        "enigma_cnn": "nod-image-atm-v1",
        "labram": "nod-image-labram-atm-v1",
        "reve": "nod-image-reve-atm-v1",
        "luna": "nod-image-luna-atm-v1",
    },
    root="/datas/eeg_benchmark_project/NOD-EEG",
    convention_subject="sub-01",
)

#: Grootswagers2019（ds004018）：16 被试。
GROOTSWAGERS = DatasetSpec(
    name="grootswagers",
    subjects=_subs(1, 16),
    profiles={
        "atm_s": "grootswagers-image-atm-v1",
        "enigma_cnn": "grootswagers-image-atm-v1",
        "labram": "grootswagers-image-labram-atm-v1",
        "reve": "grootswagers-image-reve-atm-v1",
        "luna": "grootswagers-image-luna-atm-v1",
    },
    root="/datas/eeg_benchmark_project/openneurodownload/ds004018-download",
    convention_subject="sub-01",
)

#: Grootswagers2025（ds005106，婴儿）：42 被试，编号不连续。
GROOTSWAGERS2025 = DatasetSpec(
    name="grootswagers2025",
    subjects=tuple(f"sub-{n:03d}" for n in (
        1, 2, 3, 4, 5, 7, 8, 9, *range(11, 29), 30, 31, 32,
        35, 36, 37, 39, 40, *range(42, 47), 48, 49, 51)),
    profiles={
        "atm_s": "grootswagers2025-image-atm-paper-v1",
        "enigma_cnn": "grootswagers2025-image-atm-paper-v1",
        "labram": "grootswagers2025-image-labram-atm-paper-v1",
        "reve": "grootswagers2025-image-reve-atm-paper-v1",
        "luna": "grootswagers2025-image-luna-atm-paper-v1",
    },
    root="/datas/eeg_benchmark_project/openneurodownload/ds005106-download",
    convention_subject="sub-001",
)

#: SEED-DV：20 被试，视频刺激，每段 2 秒、200 Hz。
SEED_DV = DatasetSpec(
    name="seed_dv",
    subjects=_subs(1, 20),
    profiles={
        "eegnet": "seed-dv-video-v1",
        "labram": "seed-dv-video-labram-v1",
        "reve": "seed-dv-video-reve-v1",
        "luna": "seed-dv-video-luna-v1",
        "eegvid_nice": "seed-dv-video-eegvid-nice-v1",
    },
    root="/datas/eeg_benchmark_project/SEED-DV",
    convention_subject="sub-01",
    task="video_generation",
)

#: Smit2025（OpenNeuro ds005662）：20 被试，600 毫秒 epoch、64 通道。
SMIT2025 = DatasetSpec(
    name="ds005662",
    subjects=_subs(1, 20),
    profiles={
        "eegnet": "ds005662-video-eegnet-v1",
        "labram": "ds005662-video-labram-v1",
        "reve": "ds005662-video-reve-v1",
        "luna": "ds005662-video-luna-v1",
        "eegvid_nice": "ds005662-video-eegvid-nice-v1",
    },
    root="/datas/eeg_benchmark_project/ds005662-download",
    convention_subject="sub-01",
    task="video_generation",
)

#: EMD（OpenNeuro ds008257）：6 被试。EEGVid Orig. 只用 sub-02_k1250。
EMD = DatasetSpec(
    name="ds008257",
    subjects=_subs(1, 6),
    profiles={
        "eegnet": "ds008257-video-eegnet-v1",
        "labram": "ds008257-video-labram-v1",
        "reve": "ds008257-video-reve-v1",
        "luna": "ds008257-video-luna-v1",
        "eegvid_nice": "ds008257-video-eegvid-nice-v1",
    },
    root="/datas/eeg_benchmark_project/openneurodownload/ds008257",
    convention_subject="sub-01",
    task="video_generation",
)

#: ZuCo 2：18 名被试，三字母被试编码。
ZUCO2 = DatasetSpec(
    name="zuco2",
    subjects=("YAC", "YAG", "YAK", "YDG", "YDR", "YFR", "YFS", "YHS", "YIS",
              "YLS", "YMD", "YMS", "YRH", "YRK", "YRP", "YSD", "YSL", "YTL"),
    profiles={
        "glim": "zuco2-glim-raw-v1",
        "labram": "zuco2-labram-raw-v1",
        "luna": "zuco2-luna-raw-v1",
        "reve": "zuco2-reve-raw-v1",
    },
    root="/datas/eeg_benchmark_project/zuco_2",
    convention_subject="YAC",
    task="sentence_generation",
)

#: Chisco（ds005170）：5 名被试。
CHISCO = DatasetSpec(
    name="chisco",
    subjects=_subs(1, 5),
    profiles={
        "glim": "chisco-glim-raw-v1",
        "labram": "chisco-labram-raw-v1",
        "luna": "chisco-luna-raw-v1",
        "reve": "chisco-reve-raw-v1",
    },
    root="/datas/eeg_benchmark_project/openneurodownload/ds005170-download",
    convention_subject="sub-01",
    task="sentence_generation",
)

DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (THINGSEEG, NOD, GROOTSWAGERS, GROOTSWAGERS2025,
                 SEED_DV, SMIT2025, EMD, ZUCO2, CHISCO)
}


def resolve(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    return DATASETS[name]


__all__ = ["DATASETS", "DatasetSpec", "resolve"]
