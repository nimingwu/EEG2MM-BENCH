"""编码器注册表：跨任务共享，只实现一遍。

框架拿到的只是一个满足 :class:`~eeg2x.encoders.base.EEGEncoder` 的模块；
"换编码器"= 换这里的 ``SPEC``。baseline 不经过本表——它们固定自己的编码器。

用法::

    from eeg2x.encoders import ENCODERS, build
    spec = ENCODERS["luna"]
    model = build("luna", n_chans=63, n_samples=280,
                  config={"channel_names": [...], "subject_id": "sub-01"})

注意 ``atm_s`` 需要 ``subject_id``（被试在构造时绑定），其余编码器不需要；
多余的 config 键会被各编码器自己的 ``_CONFIG_KEYS`` 过滤掉。
"""
from __future__ import annotations

from typing import Dict, Mapping

from torch import nn

from .base import BaseEEGEncoder, EEGEncoder, EncoderSpec, OptimizerSpec, build as _build

# 注意：luna 不在此处 import 实现（它会加载 rotary_embedding_torch）。只取 SPEC。
from . import atm_s, enigma, labram, luna, reve

#: 名字 -> 契约。新增编码器只需在这里加一行。
ENCODERS: Dict[str, EncoderSpec] = {
    spec.name: spec
    for spec in (
        atm_s.SPEC,
        labram.SPEC,
        reve.SPEC,
        luna.SPEC,
        enigma.SPEC,
    )
}

#: 编码器别名（v2 的 runner 里同时出现过 atm / enigma 两个名字）。
ALIASES: Mapping[str, str] = {
    "atm": "atm_s",
    "enigma": "enigma_cnn",
}


def resolve(name: str) -> EncoderSpec:
    """把别名解析成正式名字，未知名字**显式报错**。"""
    canonical = ALIASES.get(str(name), str(name))
    if canonical not in ENCODERS:
        raise KeyError(
            f"unknown encoder {name!r}; available: {sorted(ENCODERS)} "
            f"(aliases: {sorted(ALIASES)})")
    return ENCODERS[canonical]


def build(name: str, *, n_chans: int, n_samples: int,
          config: dict | None = None) -> nn.Module:
    """按注册表构造编码器。"""
    return _build(resolve(name), n_chans=n_chans, n_samples=n_samples, config=config)


def list_encoders() -> list[str]:
    return sorted(ENCODERS)


__all__ = [
    "ALIASES",
    "ENCODERS",
    "BaseEEGEncoder",
    "EEGEncoder",
    "EncoderSpec",
    "OptimizerSpec",
    "build",
    "list_encoders",
    "resolve",
]
