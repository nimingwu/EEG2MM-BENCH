"""LUNA 编码器（BioFoundation, NeurIPS 2025）。

模型代码逐字来自 v2（``model/luna_encoder.py`` + ``model/luna/official.py`` +
``model/luna_input.py``），数学未改。

训练设置：LUNA 使用与其他编码器相同的框架统一配方（AdamW、平坦学习率、
无 warmup、无学习率调度、相同 epoch/batch/seed）。唯一例外是 ThingsEEG 上
学习率取 1e-4，而不是框架默认的 3e-4；其他数据集仍用 3e-4。此前 LUNA 论文
Table 6 的 min-lr、warmup、weight decay 与逐层衰减配方已移除。

实现方式是在 :class:`~eeg2x.encoders.base.EncoderSpec` 上声明学习率覆盖，
训练循环仍然只有一条代码路径；见 ``docs/LUNA_TRAINING_DEVIATION.md``。
"""
from __future__ import annotations

from torch import nn

from ..base import EncoderSpec

# 注意：不在这里 import .encoder。它一加载就要 rotary_embedding_torch，那样缺包时
# 会抛裸 ImportError，绕过 EncoderSpec.require_packages 的明确报错。

CONFIG_KEYS = (
    "sampling_rate", "channel_names", "repo_id", "weight_filename",
    "cache_dir", "revision", "local_files_only", "freeze",
)

#: ThingsEEG 上 LUNA 的学习率；其余数据集沿用框架默认 3e-4。
LUNA_LEARNING_RATE = 1e-4
LUNA_LEARNING_RATE_DATASETS = ("thingseeg",)


def _builder(*, n_chans: int, n_samples: int, config: dict) -> nn.Module:
    from .encoder import LunaEncoder       # 懒加载：见文件顶部说明

    kwargs = {k: config[k] for k in CONFIG_KEYS if k in config}
    return LunaEncoder(n_chans=n_chans, n_samples=n_samples, **kwargs)


SPEC = EncoderSpec(
    name="luna",
    builder=_builder,
    feature_dim=256,
    learning_rate=LUNA_LEARNING_RATE,
    learning_rate_datasets=LUNA_LEARNING_RATE_DATASETS,
    required_packages=("rotary_embedding_torch", "timm", "mne"),
    needs_channel_names=True,
    description="Pretrained LUNA-Base with official 3D electrode positions and "
                "attention pooling; requires 256 Hz plus explicit right padding. "
                "Uses the framework recipe; only the ThingsEEG learning rate is "
                "overridden to 1e-4.",
)

__all__ = [
    "LUNA_LEARNING_RATE", "LUNA_LEARNING_RATE_DATASETS", "SPEC", "build",
]


def build(*, n_chans: int, n_samples: int, config: dict):
    """构造 LUNA；需要时再导入实现。"""
    return _builder(n_chans=n_chans, n_samples=n_samples, config=config)
