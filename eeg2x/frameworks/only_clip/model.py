"""Only-CLIP 框架的模型：编码器 + ATM 头。

**这是"框架只写一遍"的落点。** 框架拿到的编码器只是一个满足
``eeg2x.encoders.EEGEncoder`` 的模块；这里把编码器输出接到 ATM 投影头上，
再映射到 1024 维 CLIP 空间。

与 v2 ``model/atm/model.py:ATMS`` 的关系：数学完全一致（``ATMProjector``、
``ClipLoss``、两个对齐损失逐字复制），差别只在**编码器的注入方式**：

* v2 的 ``ATMS.forward(eeg, subject_ids)`` 按名字分支：``atm_s`` 走特例、
  其余走 registry；``subject_ids`` 只传给 ``atm_s``。
* v3 的 ``OnlyClipModel.forward(eeg)`` 不接收被试——被试在构造编码器时绑定
  （见 ``eeg2x.encoders.atm_s``），因此所有编码器的调用方式一致。

checkpoint 兼容性：v2 的 ``ATMS`` state_dict 是 ``encoder.* / projector.* /
logit_scale``（atm_s 还有未使用的 ``subject_wise_linear.*``）。v3 的
``OnlyClipModel`` 是 ``encoder.* / projector.* / logit_scale``——同样的键，
所以 v2 的 checkpoint 可以直接灌进来；``subject_wise_linear.*`` 是 v2 为
state-dict 兼容保留的死参数，加载时忽略。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ResidualAdd(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, value: Tensor) -> Tensor:
        return value + self.module(value)


class ATMProjector(nn.Sequential):
    def __init__(self, input_dim: int = 1440, output_dim: int = 1024):
        super().__init__(
            nn.Linear(input_dim, output_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(), nn.Linear(output_dim, output_dim), nn.Dropout(0.5))),
            nn.LayerNorm(output_dim),
        )


class ClipLoss(nn.Module):
    """The released symmetric CLIP loss for a single-process benchmark run."""

    def forward(self, eeg_features: Tensor, image_features: Tensor,
                logit_scale: Tensor) -> Tensor:
        logits = logit_scale * eeg_features @ image_features.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) +
                F.cross_entropy(logits.T, labels)) / 2


def generation_alignment_loss(predicted: Tensor, target: Tensor,
                              logit_scale: Tensor, clip_loss: ClipLoss) -> Tensor:
    """Released generation objective: 0.9*MSE*10 + 0.1*CLIP*10."""
    return (0.9 * F.mse_loss(predicted, target) * 10.0 +
            0.1 * clip_loss(predicted, target, logit_scale) * 10.0)


def retrieval_alignment_loss(predicted: Tensor, target: Tensor,
                             logit_scale: Tensor, clip_loss: ClipLoss) -> Tensor:
    return clip_loss(predicted, target, logit_scale)


class OnlyClipModel(nn.Module):
    """编码器 + ATM 投影头 + 可学习温度。

    ``encoder`` 是任何满足 ``EEGEncoder`` 协议的模块；它的 ``feature_dim``
    决定投影头首层宽度。**换编码器 = 换这一个属性。**
    """

    embedding_dim = 1024

    def __init__(self, encoder: nn.Module):
        super().__init__()
        feature_dim = getattr(encoder, "feature_dim", None)
        if feature_dim is None:
            raise TypeError(
                f"{type(encoder).__name__} must declare feature_dim before it can "
                "feed the ATM projector")
        self.encoder = encoder
        self.encoder_name = getattr(encoder, "name", type(encoder).__name__)
        self.projector = ATMProjector(input_dim=int(feature_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, eeg: Tensor) -> Tensor:
        return self.projector(self.encoder(eeg))

    def extra_repr(self) -> str:
        return f"encoder={self.encoder_name}, input_dim={self.projector[0].in_features}"


__all__ = [
    "ATMProjector",
    "ClipLoss",
    "OnlyClipModel",
    "generation_alignment_loss",
    "retrieval_alignment_loss",
]
