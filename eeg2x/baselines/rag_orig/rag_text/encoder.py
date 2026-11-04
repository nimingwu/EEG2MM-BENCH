"""EEG 编码器：EEGChannelNet 风格的时域-空间卷积网络 → 768 维句向量空间。

对应论文 arXiv:2605.17503 §II-E（"EEG Encoder and Embedding Alignment"）的四句话：

    (i)   temporal depthwise convolutions with multiple dilation factors
    (ii)  spatial convolutions to model inter-channel relationships
    (iii) residual blocks
    (iv)  global average pooling → linear projection to the 768-d sentence space

论文**没有给**任何具体宽度、膨胀率、卷积核长度或残差块层数（那是一篇 6 页会议稿，且
没有开源代码），所以下面这些数字是**我们的具体化**，不是原作者的配置：

    width=64, kernel=9, dilations=(1,2,4,8), n_res_blocks=3, dropout=0.3

这一点必须写进 run config 的 deviations 与 README，不能假装是论文的设置。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_DILATIONS = (1, 2, 4, 8)
DEFAULT_WIDTH = 64
DEFAULT_KERNEL = 9


class ResidualBlock(nn.Module):
    """两层 Conv1d + BN + ELU + 恒等跳连（论文 §II-E 的 (iii)）。"""

    def __init__(self, width: int, kernel: int = 7, dropout: float = 0.0):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv1d(width, width, kernel, padding=pad)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(width, width, kernel, padding=pad)
        self.bn2 = nn.BatchNorm1d(width)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.elu(x + h)


class EEGChannelNetEncoder(nn.Module):
    """(B, C, T) EEG → (B, out_dim) 的句向量嵌入（未归一化，归一化在 forward 里做）。

    输入布局是 **(B, C, T)**（通道在前），与 `ZuCoH5Dataset` 交付的 (T, C) 不同 ——
    调用方负责转置，见 `model/rag_text/dataset.py`。
    """

    def __init__(self, n_channels: int, n_times: int, out_dim: int = 768,
                 width: int = DEFAULT_WIDTH, kernel: int = DEFAULT_KERNEL,
                 dilations=DEFAULT_DILATIONS, n_res_blocks: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_times = int(n_times)
        self.out_dim = int(out_dim)

        # (i) 时域 depthwise 卷积：每种膨胀率一个分组卷积（groups=C，即每个通道独立），
        #     多尺度结果相加 —— 这是"depthwise + multiple dilation"的直接实现。
        self.temporal = nn.ModuleList([
            nn.Conv1d(self.n_channels, self.n_channels, kernel, dilation=d,
                      padding=d * (kernel - 1) // 2, groups=self.n_channels)
            for d in dilations
        ])
        self.temporal_bn = nn.BatchNorm1d(self.n_channels)

        # (ii) 空间卷积：把通道轴当成空间维一次压掉（kernel=(C,1)），输出 F 个滤波器。
        self.spatial = nn.Conv2d(1, width, kernel_size=(self.n_channels, 1))
        self.spatial_bn = nn.BatchNorm2d(width)
        self.drop = nn.Dropout(dropout)

        # (iii) 残差块
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout=dropout * 0.5)
                                      for _ in range(n_res_blocks)])

        # (iv) 时域 GAP → 线性投影到句向量维度
        self.proj = nn.Linear(width, self.out_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """eeg: (B, C, T) float → (B, out_dim) 的 **L2 归一化**嵌入。"""
        if eeg.dim() != 3 or eeg.shape[1] != self.n_channels:
            raise ValueError(
                f"EEGChannelNetEncoder 期望 (B, {self.n_channels}, T)，"
                f"收到 {tuple(eeg.shape)}")
        x = eeg.float()
        h = sum(conv(x) for conv in self.temporal)
        h = F.elu(self.temporal_bn(h))                      # (B, C, T)

        h = h.unsqueeze(1)                                   # (B, 1, C, T)
        h = F.elu(self.spatial_bn(self.spatial(h)))          # (B, F, 1, T)
        h = self.drop(h.squeeze(2))                          # (B, F, T)

        h = self.blocks(h)                                   # (B, F, T)
        h = h.mean(dim=-1)                                   # 时域 GAP → (B, F)
        return F.normalize(self.proj(h), dim=-1)


__all__ = ["EEGChannelNetEncoder", "ResidualBlock",
           "DEFAULT_DILATIONS", "DEFAULT_WIDTH", "DEFAULT_KERNEL"]
