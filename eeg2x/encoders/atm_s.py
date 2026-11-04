"""ATM-S 编码器（2403.07721v7）：通道注意力 + 时空卷积。

模型代码逐字来自 v2 ``model/atm/model.py`` 的 ``ATMConfig`` / ``ChannelTransformer`` /
``TemporalSpatialConv``，数学未改。

**与 v2 的一处接口差异（行为等价）**：v2 的 ``ATMS.forward(eeg, subject_ids)`` 把被试
索引作为参数传进来；v3 的生产 ``forward(eeg)`` 不接收额外参数，所以在**构造时**把
被试绑定好（``subject_id``）。benchmark 的每次运行都是单被试，两者等价。
被试索引沿用 v2 的 ``subject_number()``（``sub-08`` -> 8，1 起算）。

这个编码器是**从随机初始化训练**的，没有预训练权重，因此不声明 ``optimizer``
（走框架统一配方）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from .base import EncoderSpec

# 注意：``subject_layers`` 的三个模块在 import 时会拉起 reformer_pytorch，
# 那是图像侧才需要的依赖。把它们放在 ``ChannelTransformer.__init__`` 里，
# 使 ``eeg2x.encoders`` 可以在视频环境（没有 reformer_pytorch）里被导入。


def subject_number(subject_id: str) -> int:
    """``sub-08`` -> 8。逐字沿用 v2 ``model/atm/data.py:subject_number``。"""
    match = re.fullmatch(r"sub-(\d+)", subject_id)
    if match is None:
        raise ValueError(f"Invalid subject ID: {subject_id!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class ATMConfig:
    task_name: str = "classification"
    seq_len: int = 250
    pred_len: int = 250
    output_attention: bool = False
    d_model: int = 250
    embed: str = "timeF"
    freq: str = "h"
    dropout: float = 0.25
    factor: int = 1
    n_heads: int = 4
    e_layers: int = 1
    d_ff: int = 256
    activation: str = "gelu"
    enc_in: int = 63


class ChannelTransformer(nn.Module):
    """Paper channel-wise attention, including its subject-token behavior."""

    def __init__(self, config: ATMConfig, num_subjects: int = 10):
        super().__init__()
        from .subject_layers.Embed import DataEmbedding
        from .subject_layers.SelfAttention_Family import (
            AttentionLayer, FullAttention,
        )
        from .subject_layers.Transformer_EncDec import Encoder, EncoderLayer

        self.n_chans = int(config.enc_in)
        self.enc_embedding = DataEmbedding(
            config.seq_len,
            config.d_model,
            config.embed,
            config.freq,
            config.dropout,
            joint_train=False,
            num_subjects=num_subjects,
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            config.factor,
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            norm_layer=nn.LayerNorm(config.d_model),
        )

    def forward(self, eeg: Tensor, subject_ids: Tensor) -> Tensor:
        encoded = self.enc_embedding(eeg, None, subject_ids)
        encoded, _ = self.encoder(encoded, attn_mask=None)
        # This deliberately preserves the released implementation: the subject
        # token is retained and the final EEG token is discarded.
        return encoded[:, :self.n_chans, :]


class TemporalSpatialConv(nn.Module):
    def __init__(self, n_chans: int = 63, n_samples: int = 250,
                 embedding_size: int = 40):
        super().__init__()
        if n_chans < 1 or n_samples < 25:
            raise ValueError("ATM-S input is too short for temporal convolution")
        temporal_after_conv = n_samples - 25 + 1
        temporal_after_pool = (temporal_after_conv - 51) // 5 + 1
        if temporal_after_pool < 1 or n_chans < 1:
            raise ValueError("ATM-S input is too short for temporal-spatial pooling")
        self.output_dim = int(embedding_size * temporal_after_pool)
        self.network = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (n_chans, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Conv2d(40, embedding_size, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value.unsqueeze(1))




class AtmSEncoder(nn.Module):
    """``ChannelTransformer`` + ``TemporalSpatialConv`` 合成一个池化编码器。

    ``forward(eeg) -> [B, output_dim]``，被试在构造时绑定。
    """

    def __init__(self, *, n_chans: int, n_samples: int, subject_id: str,
                 num_subjects: int = 10):
        super().__init__()
        config = ATMConfig(
            seq_len=int(n_samples), pred_len=int(n_samples),
            d_model=int(n_samples),
            n_heads=4 if int(n_samples) == 250 else 2,
            enc_in=int(n_chans),
        )
        self.config = config
        self.n_chans = int(n_chans)
        self.n_samples = int(n_samples)
        self.subject_id = str(subject_id)
        self.subject_index = subject_number(self.subject_id)
        if self.subject_index >= num_subjects:
            raise ValueError(
                f"subject {subject_id!r} needs index {self.subject_index} but the "
                f"subject embedding table has {num_subjects} rows")
        self.encoder = ChannelTransformer(config, num_subjects=num_subjects)
        self.temporal_spatial = TemporalSpatialConv(
            n_chans=self.n_chans, n_samples=self.n_samples)
        self.feature_dim = int(self.temporal_spatial.output_dim)

    def forward(self, eeg: Tensor) -> Tensor:
        if eeg.ndim != 3 or eeg.shape[1:] != (self.n_chans, self.n_samples):
            raise ValueError(
                f"ATM-S expects [batch, {self.n_chans}, {self.n_samples}], "
                f"got {tuple(eeg.shape)}")
        subjects = torch.full((eeg.shape[0],), self.subject_index,
                              dtype=torch.long, device=eeg.device)
        encoded = self.encoder(eeg, subjects)
        return self.temporal_spatial(encoded).flatten(start_dim=1)

    def pretrained_metadata(self) -> dict:
        return {"pretrained_loaded": False, "subject_id": self.subject_id}


_CONFIG_KEYS = ("subject_id", "num_subjects")


def _builder(*, n_chans: int, n_samples: int, config: dict) -> nn.Module:
    if "subject_id" not in config:
        raise ValueError(
            "ATM-S needs subject_id at construction (v3 binds the subject there "
            "instead of passing subject_ids on every forward)")
    kwargs = {k: config[k] for k in _CONFIG_KEYS if k in config}
    return AtmSEncoder(n_chans=n_chans, n_samples=n_samples, **kwargs)


SPEC = EncoderSpec(
    name="atm_s",
    builder=_builder,
    feature_dim=None,          # 随 n_samples 变化，由 AtmSEncoder 决定
    required_packages=(),
    needs_channel_names=False,
    description="ATM-S channel transformer + temporal-spatial conv, trained from "
                "scratch. Subject identity is bound at construction.",
)
