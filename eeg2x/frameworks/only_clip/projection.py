"""Only-CLIP 的投影层：单层 ``Linear(1024, 1024)``。

逐字来自 v2 ``model/atm/diffusion_prior.py``（它**直接继承 ``nn.Linear``**，
所以 checkpoint 的键是顶层的 ``weight`` / ``bias``，没有嵌套前缀）。
**那个文件名是历史遗留的谎话**——它早已不含 diffusion prior；v3 按实际内容重命名。
"""
from __future__ import annotations

from torch import nn


class EEGToImageProjection(nn.Linear):
    """One affine 1024-to-1024 map, with no hidden layers or normalization."""

    embedding_dim = 1024

    def __init__(self) -> None:
        super().__init__(
            in_features=self.embedding_dim,
            out_features=self.embedding_dim,
            bias=True,
        )


__all__ = ["EEGToImageProjection"]
