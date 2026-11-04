"""ENIGMA 时空 CNN 编码器（Kneeland et al., 2026）。

模型代码逐字来自 v2 ``model/encoders.py`` 的 ``EnigmaEncoder``（结构逐层复制自
上游 ``model/enigma/source/models.py`` 的 ``Spatio_Temporal_CNN``），数学未改。

预训练侧没有权重——它随框架一起从随机初始化训练，因此不声明 ``optimizer``。
可选 ``subjects`` 用于每个被试的 temporal alignment 层。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from .base import EncoderSpec

# v2 默认几何（ThingsEEG NICE 口径），仅作为构造函数的默认值；
# 实际值一律由框架按 (数据集 x 编码器) 的 profile 显式传入。
N_CHANNELS = 63
N_TIMES = 250

# ===================================================================
# EnigmaEncoder — ENIGMA 时空 CNN（Kneeland et al., 2026）
# ===================================================================

ENIGMA_EMB_SIZE = 4          # 上游 Spatio_Temporal_CNN(emb_size=4)
ENIGMA_CONV1_KERNEL = (1, 5)
ENIGMA_POOL1_KERNEL = (1, 17)
ENIGMA_POOL1_STRIDE = (1, 5)


class EnigmaEncoder(nn.Module):
    """ENIGMA 的 EEG 侧时空 CNN（结构逐层复制自上游源码）。

    上游: ``model/enigma/source/models.py`` 的 ``Spatio_Temporal_CNN``，由
    ``ENIGMA.__init__`` 以 (emb_size=4, conv1_kernel=(1,5), pool1_kernel=(1,17),
    pool1_stride=(1,5), conv2_kernel=(n_chans,1)) 构造：

        Conv2d(1, 40, (1,5)) → AvgPool2d((1,17),(1,5)) → BN → ELU
        → Conv2d(40, 40, (n_chans,1)) → BN → ELU → Dropout(0.5)
        → Conv2d(40, emb_size, (1,1)) → (B, tokens, emb_size) → flatten

    与上游的唯一结构差异：上游 ``MLP_Projector(hidden_dim → 1024)`` 由调用方的
    投影头（统一框架的 Linear(1024,1024)）替代，因此这里只输出展平特征
    ``feature_dim = emb_size * n_tokens``，其中
    ``n_tokens = (n_samples - 21) // 5 + 1``（即上游 ``hidden_dim // emb_size``）。

    ``subjects`` 非空时插入上游的 per-subject 时间对齐层 ``subject_wise_linear``
    （``nn.Linear(T, T)``，沿时间轴混合）。本 benchmark 每次只训练一个被试，通常
    只登记一个 subject，forward 时对整批生效；也可显式传入 ``subject_ids`` 逐行
    路由。ModuleDict 的 key 用下标（``subject_0``）而非被试名——PyTorch 的模块名
    不允许含 "."，而 ``sub-01`` 这类名字需要能同时落到 checkpoint 的 state_dict。
    """

    def __init__(self, n_chans: int = N_CHANNELS, n_samples: int = N_TIMES,
                 subjects: Optional[Sequence[str]] = None,
                 emb_size: int = ENIGMA_EMB_SIZE):
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_samples = int(n_samples)
        self.emb_size = int(emb_size)
        self.subjects = tuple(str(subject) for subject in (subjects or ()))
        # 被试名 → ModuleDict key（保序，checkpoint 重建时映射一致）
        self.subject_index = {
            subject: f"subject_{position}"
            for position, subject in enumerate(self.subjects)}
        self.subject_wise_linear = nn.ModuleDict({
            self.subject_index[subject]: nn.Linear(self.n_samples, self.n_samples)
            for subject in self.subjects})
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, ENIGMA_CONV1_KERNEL, stride=(1, 1)),
            nn.AvgPool2d(ENIGMA_POOL1_KERNEL, ENIGMA_POOL1_STRIDE),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (self.n_chans, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, self.emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        n_tokens = (self.n_samples - 21) // 5 + 1
        if n_tokens <= 0:
            raise ValueError(
                f"EnigmaEncoder: n_samples={n_samples} too short "
                f"(temporal tokens={n_tokens} ≤ 0)")
        self.feature_dim = self.emb_size * n_tokens

    def forward(self, x: torch.Tensor,
                subject_ids: Optional[Sequence[str]] = None) -> torch.Tensor:
        # x: (B, C, T) → (B, feature_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.n_chans, self.n_samples):
            raise ValueError(
                f"ENIGMA expects [batch, {self.n_chans}, {self.n_samples}], "
                f"got {tuple(x.shape)}")
        if self.subject_wise_linear:
            x = self._align_subjects(x, subject_ids)
        x = x.unsqueeze(1)           # (B, 1, C, T)
        x = self.tsconv(x)           # (B, 40, 1, n_tokens)
        x = self.projection(x)       # (B, n_tokens, emb_size)
        return x.flatten(1)          # (B, emb_size * n_tokens)

    def _align_subjects(self, x: torch.Tensor,
                        subject_ids: Optional[Sequence[str]]) -> torch.Tensor:
        """上游 ``ENIGMA.forward`` 的 per-subject 线性对齐（按被试分组做整块 matmul）。"""
        if subject_ids is None:
            if len(self.subjects) != 1:
                raise ValueError(
                    "EnigmaEncoder was built with "
                    f"{len(self.subjects)} subjects; forward needs explicit "
                    "subject_ids (the encoder contract carries none)")
            return self.subject_wise_linear[self.subject_index[self.subjects[0]]](x)
        requested = [str(subject) for subject in subject_ids]
        if len(requested) != x.shape[0]:
            raise ValueError(
                f"subject_ids has {len(requested)} entries for a batch of "
                f"{x.shape[0]}")
        aligned = torch.empty_like(x)
        for subject in dict.fromkeys(requested):
            if subject not in self.subject_index:
                raise KeyError(
                    f"subject {subject!r} was not registered; known: "
                    f"{list(self.subjects)}")
            rows = [index for index, value in enumerate(requested)
                    if value == subject]
            selector = torch.as_tensor(rows, device=x.device)
            aligned = aligned.index_copy(
                0, selector,
                self.subject_wise_linear[self.subject_index[subject]](
                    x.index_select(0, selector)))
        return aligned




_CONFIG_KEYS = ("subjects", "emb_size")


def _builder(*, n_chans: int, n_samples: int, config: dict) -> nn.Module:
    kwargs = {k: config[k] for k in _CONFIG_KEYS if k in config}
    return EnigmaEncoder(n_chans=n_chans, n_samples=n_samples, **kwargs)


SPEC = EncoderSpec(
    name="enigma_cnn",
    builder=_builder,
    feature_dim=None,          # emb_size * ((n_samples - 21) // 5 + 1)
    required_packages=(),
    needs_channel_names=False,
    h5_encoder="atm_s",     # 输入契约与 ATM 一致，复用同一份 HDF5
    description="ENIGMA spatio-temporal CNN, trained from scratch; optional "
                "per-subject temporal alignment via subjects.",
)
