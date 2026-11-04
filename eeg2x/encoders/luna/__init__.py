"""LUNA 编码器（BioFoundation, NeurIPS 2025）与它自己的微调配方。

模型代码逐字来自 v2（``model/luna_encoder.py`` + ``model/luna/official.py`` +
``model/luna_input.py``），数学未改。

**这里是"框架只写一遍"与"LUNA 要特殊学习率"两个要求的交汇点。**
LUNA 在框架统一配方（AdamW 平坦 lr=3e-4）下会塌缩成常函数——残差流沿输入无关
方向线性增长，末端 LayerNorm 把刺激相关成分除没了。修复方式是改用其论文
Table 6 的微调优化器，而这件事**不写在训练循环里**，而是由编码器在这里
声明成 :class:`~eeg2x.encoders.base.OptimizerSpec`。训练器读 spec，未声明就用
框架默认——只有一条代码路径。

代价（不可淡化）：LUNA 行因此不再与其他编码器「同配方」，不能并列排名。
完整诊断、证据与适用范围见 ``EEG2X-Benchmark-v2-test/LUNA_TRAINING_DEVIATION.md``。
"""
from __future__ import annotations

from torch import nn

from ..base import EncoderSpec, OptimizerSpec

# 注意：不在这里 import .encoder。它一加载就要 rotary_embedding_torch，那样缺包时
# 会抛裸 ImportError，绕过 EncoderSpec.require_packages 的明确报错。

CONFIG_KEYS = (
    "sampling_rate", "channel_names", "repo_id", "weight_filename",
    "cache_dir", "revision", "local_files_only", "freeze",
)

#: LUNA 论文 Table 6（下游微调，p.22）。**有意偏离框架统一配方。**
#: epoch / batch / seed 仍与其余编码器一致，只改优化器。
#:
#: **适用范围只有 ThingsEEG。** 在 Grootswager2019（ds004018）上用同一配方
#: 重跑后结果更差（clip_similarity 0.4988 对旧配方 0.5358，15 个被试里 11 个
#: 变差），已回退。机制是数据量：ds004018 单被试 14400 个训练试次，只有
#: ThingsEEG 66160 的 1/4.6，layer_decay 0.5 冻住主干的代价超过了收益。因此
#: 这个例外不能按「LUNA 需要特殊配方」推广到其它数据集，见
#: ``实验报告12.txt`` 第九节。
LUNA_OPTIMIZER = OptimizerSpec(
    peak_lr=1e-4,
    min_lr=5e-6,
    warmup_epochs=5,
    weight_decay=0.05,
    layer_decay=0.5,
)

#: 例外生效的数据集。其余数据集上 LUNA 与其他编码器用同一套统一配方。
LUNA_OPTIMIZER_DATASETS = ("thingseeg",)


def _builder(*, n_chans: int, n_samples: int, config: dict) -> nn.Module:
    from .encoder import LunaEncoder       # 懒加载：见文件顶部说明

    kwargs = {k: config[k] for k in CONFIG_KEYS if k in config}
    return LunaEncoder(n_chans=n_chans, n_samples=n_samples, **kwargs)


SPEC = EncoderSpec(
    name="luna",
    builder=_builder,
    feature_dim=256,
    optimizer=LUNA_OPTIMIZER,
    optimizer_datasets=LUNA_OPTIMIZER_DATASETS,
    required_packages=("rotary_embedding_torch", "timm", "mne"),
    needs_channel_names=True,
    description="Pretrained LUNA-Base with official 3D electrode positions and "
                "attention pooling; requires 256 Hz plus explicit right padding. "
                "Declares its own fine-tuning optimizer (see LUNA_OPTIMIZER).",
)

__all__ = ["LUNA_OPTIMIZER", "LUNA_OPTIMIZER_DATASETS", "SPEC", "build"]


def build(*, n_chans: int, n_samples: int, config: dict):
    """构造 LUNA；需要时再导入实现。"""
    return _builder(n_chans=n_chans, n_samples=n_samples, config=config)
