"""完整 ATM baseline 的 `ATMS` 兼容垫片。

**为什么需要它**：完整 ATM 是一个 baseline，代码逐字继承自 v2，它按 v2 的
`ATMS(encoder=..., encoder_config=...)` / `model(eeg, subject_ids)` 约定编写。
v3 的框架把「被试」挪到编码器构造时绑定（`AtmSEncoder(subject_id=...)`），所以
直接复用 `OnlyClipModel` 会签名不符。

这个垫片保留 v2 的签名，内部换成 v3 的编码器与投影头：
* 编码器：`eeg2x.encoders.build`（atm_full 默认 `atm_s`，也支持其它注册编码器）
* 投影头：复用 `eeg2x.frameworks.only_clip.model.ATMProjector`
* `forward(eeg, subject_ids)`：**忽略 subject_ids**——benchmark 每次只跑一个被试，
  身份已经在编码器里绑定，行为与 v2 等价。

被试从哪来：调用方通过 `encoder_config["subject_id"]` 注入（checkpoint 的
``config.subject_id`` 里有），见 `eeg2x/baselines/atm_full/training.py`。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from eeg2x import encoders as encoder_registry
from eeg2x.frameworks.only_clip.model import ATMProjector, ClipLoss


class ATMS(nn.Module):
    """v2 ``ATMS`` 的签名兼容层，内部用 v3 的编码器。"""

    embedding_dim = 1024

    def __init__(self, num_subject_tokens: int = 10, *,
                 encoder: str = "atm_s",
                 encoder_config: dict | None = None):
        super().__init__()
        cfg = dict(encoder_config or {})
        if "subject_id" not in cfg:
            raise ValueError(
                "v3's ATMS shim needs encoder_config['subject_id'] (the encoder "
                "binds the subject at construction; v2 passed it on every forward)")
        self.encoder_name = str(encoder)
        self.encoder_config = cfg
        self.n_chans = int(cfg.get("n_chans", 63))
        self.n_samples = int(cfg.get("n_samples", 250))
        self.encoder = encoder_registry.build(
            self.encoder_name, n_chans=self.n_chans, n_samples=self.n_samples,
            config=cfg)
        self.projector = ATMProjector(input_dim=int(self.encoder.feature_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = ClipLoss()

    def forward(self, eeg: Tensor, subject_ids: Tensor | None = None) -> Tensor:
        # subject_ids 被刻意忽略：身份已在构建编码器时绑定（单被试运行等价）。
        return self.projector(self.encoder(eeg))

    def checkpoint_model_config(self) -> dict:
        return {
            "encoder": self.encoder_name,
            "encoder_config": {**self.encoder_config,
                               "n_chans": self.n_chans,
                               "n_samples": self.n_samples},
        }

    def extra_repr(self) -> str:
        return f"encoder={self.encoder_name}"


__all__ = ["ATMS"]
