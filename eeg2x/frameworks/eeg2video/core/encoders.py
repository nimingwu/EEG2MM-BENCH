"""
EEG 编码器注册表 — 可替换编码器（模型槽位）
============================================
将「EEG 编码器」从模型中解耦为独立可注册组件（甲乙丙），任何模型（A/B/C）
都可经 build_encoder() 使用任意编码器，例如：

    NICERetrieval(encoder="eegnet")      # NICE 模型 × EEGNet 编码器
    EEGNetRetrieval(encoder="tsconv")    # EEGNet 模型 × TSConv 编码器
    --encoder conformer                  # CLI 指定

编码器契约（所有编码器遵守）：
  - 构造:  Encoder(n_chans=..., n_samples=..., **kwargs)
  - 输出:  forward(x: (B, C, T)) → (B, feature_dim)  展平特征向量
  - 属性:  feature_dim  int  输出特征维度（投影头据此建 Linear 首层）

架构来源：
  - tsconv:    NICE (Song et al., ICLR 2024) TSConv 骨干
  - eegnet:    EEGNet-8,2 (Lawhern et al., 2018, arl-eegmodels)
  - conformer: EEG-Conformer (Song et al., 2023, IEEE TNSRE)
  - labram:    预训练 LaBraM (Jiang et al., ICLR 2024) 去 probe 部分
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from einops.layers.torch import Rearrange
    from einops import rearrange
except ImportError:
    raise ImportError("PyTorch + einops required. pip install torch einops")

log = logging.getLogger("encoders")

# ===================================================================
# Constants
# ===================================================================

N_CHANNELS = 63              # NICE 论文: 保留全部 EEG 通道
N_TIMES = 250                # NICE 论文: [0, 1000ms] @ 250Hz
NICE_FEATURE_DIM = 1440      # NICE Table 11: TSConv → 40 × 36 = 1440
CONFORMER_FEATURE_DIM = 440  # Conformer: tokens * emb = 11 * 40 = 440
CONV_K = 40                  # temporal/spatial filters
EMB_SIZE = 40                # token embedding dim
DEPTH = 6                    # transformer layers
NUM_HEADS = 10               # attention heads

# LaBraM 预训练规格（braindecode/labram-pretrained）
LABRAM_PRETRAINED = "braindecode/labram-pretrained"
LABRAM_N_TIMES = 3000        # 预训练输入 3000 样本（@200Hz，15 patches）
LABRAM_N_CHANS = 128         # 预训练 128 通道模板
LABRAM_PATCH_SIZE = 200      # 200Hz × 1s
LABRAM_EMBED_DIM = 200       # LaBraM hidden dim

# labram.json 通道映射路径（数字通道 '1'-'61' → LABRAM_CHANNEL_ORDER 名）
_CH_MAP_PATH = Path(__file__).resolve().parent / "channel_mappings" / "labram.json"


# ===================================================================
# TSConvEncoder — NICE 默认骨干
# ===================================================================

class TSConvEncoder(nn.Module):
    """NICE default TSConv (Table 1, Song et al. ICLR 2024).

    temporal conv → avg pool → BN → ELU → spatial conv → BN → ELU → Dropout
    """

    def __init__(self, n_chans: int = N_CHANNELS, n_samples: int = N_TIMES):
        super().__init__()
        # n_samples 接受但不使用（TSConv 池化拓扑固定为 250 样本设计）
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),          # (B, 40, C, 226)
            nn.AvgPool2d((1, 51), (1, 5)),                # (B, 40, C,  36)
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (n_chans, 1), (1, 1)),     # (B, 40, 1,  36)
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        # 时间维随 n_samples 变化（conv 1×25 valid → avgpool 51/5）:
        # T=250 → 36 tokens → 1440（NICE 原文值）
        n_tokens = (n_samples - 25 + 1 - 51) // 5 + 1
        self.feature_dim = 40 * n_tokens
        if self.feature_dim <= 0:
            raise ValueError(
                f"TSConvEncoder: n_samples={n_samples} too short "
                f"(temporal tokens={n_tokens} ≤ 0)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, feature_dim)
        x = x.unsqueeze(1)           # (B, 1, C, T)
        x = self.tsconv(x)           # (B, 40, 1, n_tokens)
        x = self.projection(x)       # (B, n_tokens, 40)
        return x.flatten(1)          # (B, 40 * n_tokens)


# ===================================================================
# EEGNetEncoder — EEGNet-8,2（arl-eegmodels 对齐）
# ===================================================================

class EEGNetEncoder(nn.Module):
    """EEGNet-8,2 backbone — 输出 flatten 特征。

    架构与 arl-eegmodels/EEGModels.py 中的 EEGNet 一致:
      Block1: Conv2D → DepthwiseConv2D → ELU → AvgPool → Dropout
      Block2: SeparableConv2D → ELU → AvgPool → Dropout → Flatten
    图像检索（NICE/ATM）与语音检索（NeuralBench/braindecode）共用此实现；
    max_norm 语义在训练循环中实施（见 eegnet_speech.apply_max_norm_constraints）。
    """

    def __init__(self, n_chans: int = N_CHANNELS, n_samples: int = N_TIMES,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 kern_length: int = 64, dropout_rate: float = 0.25):
        super().__init__()
        self.F1, self.D, self.F2 = F1, D, F2

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        # DepthwiseConv2D((Chans,1), depth_multiplier=D) —— max_norm(1.) 约束
        # 在训练循环中实施
        self.depthwise = nn.Conv2d(F1, D * F1, (n_chans, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(D * F1)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: SeparableConv = Depthwise + Pointwise
        self.depthwise2 = nn.Conv2d(D * F1, D * F1, (1, 16),
                                    groups=D * F1, padding="same", bias=False)
        self.pointwise = nn.Conv2d(D * F1, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        self.feature_dim = F2 * (n_samples // 32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → (B, feature_dim)"""
        x = x.unsqueeze(1)  # (B, 1, C, T)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.depthwise2(x)
        x = self.pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(1)


# ===================================================================
# ConformerEncoder — EEG-Conformer（原论文骨干）
# ===================================================================

class ConvModule(nn.Module):
    """Conformer convolution module (paper Table I).
    temporal conv(padding='valid') → spatial conv → BN → ELU → AvgPool → Dropout → project
    """

    def __init__(self, n_chans: int = N_CHANNELS, emb_size: int = EMB_SIZE):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, CONV_K, (1, 25), (1, 1)),            # temporal
            nn.Conv2d(CONV_K, CONV_K, (n_chans, 1), (1, 1)),   # spatial
            nn.BatchNorm2d(CONV_K),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),                     # kernel 75, stride 15
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(CONV_K, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, C, T) → (B, n_tokens, emb_size)
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = EMB_SIZE, num_heads: int = NUM_HEADS,
                 dropout: float = 0.5):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.keys(x),    "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(self.values(x),  "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd,bhkd->bhqk', q, k)
        scaling = self.emb_size ** 0.5
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal,bhlv->bhav', att, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.projection(out)


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size: int = EMB_SIZE, expansion: int = 4,
                 drop_p: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size: int = EMB_SIZE, num_heads: int = NUM_HEADS,
                 dropout: float = 0.5, forward_drop_p: float = 0.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attn = MultiHeadAttention(emb_size, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_size)
        self.ffn = FeedForwardBlock(emb_size, expansion=4, drop_p=forward_drop_p)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class ConformerEncoder(nn.Module):
    """Original EEG-Conformer backbone: conv module + 6-layer Transformer.
    Output: flattened features (440 dim).
    """

    def __init__(self, n_chans: int = N_CHANNELS, n_samples: int = N_TIMES):
        super().__init__()
        self.conv = ConvModule(n_chans, EMB_SIZE)
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(EMB_SIZE, NUM_HEADS) for _ in range(DEPTH)
        ])
        # 时间维随 n_samples 变化（conv 1×25 valid → avgpool 75/15）:
        # T=250 → 11 tokens → 440（Conformer 原文值）
        n_tokens = (n_samples - 25 + 1 - 75) // 15 + 1
        self.feature_dim = EMB_SIZE * n_tokens
        if self.feature_dim <= 0:
            raise ValueError(
                f"ConformerEncoder: n_samples={n_samples} too short "
                f"(temporal tokens={n_tokens} ≤ 0)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, feature_dim)
        x = x.unsqueeze(1)           # (B, 1, C, T)
        x = self.conv(x)             # (B, n_tokens, emb)
        x = self.transformer(x)      # (B, n_tokens, emb)
        return x.flatten(1)          # (B, emb * n_tokens)


# ===================================================================
# LaBraM 编码器 — 预训练 LaBraM 去 probe（对齐 neuraltrain + braindecode）
# ===================================================================

def _adapt_pretrained_dimensions(model: nn.Module, n_times: int) -> Tuple[int, int]:
    """适配预训练 LaBraM 到不同 n_times（对齐 neuraltrain 同款实现）。

    词窗 600 样本 = 3 patches（600 % 200 == 0，无需 pad/truncate）；
    temporal_embedding 从预训练 (1, 15+1, 200) 截断到 (1, 3+1, 200)，
    patch_embed 元数据同步。返回 (pad_right, truncate_right)。
    """
    for attr in ("patch_size", "n_times", "patch_embed"):
        if not hasattr(model, attr):
            raise AttributeError(
                f"braindecode Labram has no attribute {attr!r} — braindecode "
                f"internal layout changed?"
            )
    patch_embed_first = model.patch_embed[0]
    if not hasattr(patch_embed_first, "n_patchs"):
        raise AttributeError("model.patch_embed[0] has no 'n_patchs'")

    patch_size: int = model.patch_size
    pad_right = truncate_right = 0
    if n_times < patch_size:
        pad_right = patch_size - n_times
    elif n_times % patch_size != 0:
        truncate_right = n_times % patch_size
    effective_n_times = n_times + pad_right - truncate_right

    actual_n_patches = effective_n_times // patch_size
    pretrained_n_patches: int = patch_embed_first.n_patchs

    if actual_n_patches != pretrained_n_patches:
        log.info(f"Adapting pretrained LaBraM from n_times={model.n_times} "
                 f"({pretrained_n_patches} patches) to n_times={effective_n_times} "
                 f"({actual_n_patches} patches)")
        if hasattr(model, "temporal_embedding"):
            if actual_n_patches < pretrained_n_patches:
                n_keep = actual_n_patches + 1
                model.temporal_embedding = nn.Parameter(
                    model.temporal_embedding.data[:, :n_keep, :])
            else:
                old_emb = model.temporal_embedding.data
                cls_token = old_emb[:, :1, :]
                patch_tokens = old_emb[:, 1:, :].permute(0, 2, 1)
                patch_tokens = F.interpolate(
                    patch_tokens, size=actual_n_patches, mode="linear",
                    align_corners=False).permute(0, 2, 1)
                model.temporal_embedding = nn.Parameter(
                    torch.cat([cls_token, patch_tokens], dim=1))
        model._n_times = effective_n_times
        model.n_path = actual_n_patches
        patch_embed_first.n_patchs = actual_n_patches
        patch_embed_first.n_times = effective_n_times

    return pad_right, truncate_right


def _load_channel_mapping() -> Dict[str, str]:
    """labram.json：数字 easycap 通道名 → LABRAM_CHANNEL_ORDER 标准名。"""
    with open(_CH_MAP_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


class LabramChannelWrapper(nn.Module):
    """通道名包装：Brennan2019 60 数字通道 → LaBraM 标准名。

    对齐 neuraltrain _LabramChannelWrapper 的结果语义：全部 60 通道经
    labram.json 显式映射且均命中 LABRAM_CHANNEL_ORDER（已验证 60/60），
    且 easycap-M10 数字通道全部有标准位置（NB 侧位置检查恒通过）——
    因此不需要动态位置掩码，forward 直接传 ch_names。
    """

    def __init__(self, model: nn.Module, union_ch_names: List[str],
                 pad_right: int = 0, truncate_right: int = 0):
        super().__init__()
        self.model = model
        self._pad_right = pad_right
        self._truncate_right = truncate_right
        ch_map = _load_channel_mapping()
        self._labram_names: List[str] = [
            ch_map.get(name, name) for name in union_ch_names]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._pad_right > 0:
            x = F.pad(x, (0, self._pad_right))
        elif self._truncate_right > 0:
            x = x[..., :x.shape[-1] - self._truncate_right]
        return self.model(x, ch_names=self._labram_names,
                          return_all_tokens=True)


class Mean(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class LabramEncoder(nn.Module):
    """预训练 LaBraM 编码器（去 probe 部分）→ (B, LABRAM_EMBED_DIM)。

    forward: (B, C, T) → Labram(return_all_tokens) → (B, n_tokens, 200)
             → Mean(dim=1) → (B, 200)
    channel_names 必填（braindecode 按名选择位置嵌入，Brennan2019 为
    60 个 easycap 数字通道名）。作为其他模型的编码器槽位使用时，
    由其下游（如 eegnet_speech 的 "labram" 选项）从 h5/checkpoint 提供。
    """

    def __init__(self, n_chans: int = 60, n_samples: int = 600,
                 channel_names: Optional[List[str]] = None):
        super().__init__()
        if channel_names is None:
            raise ValueError(
                "LabramEncoder requires channel_names (easycap 数字通道名) — "
                "braindecode 按名选择位置嵌入;请由下游模型从 h5/checkpoint 提供"
            )
        from braindecode.models import Labram

        # 1. 预训练权重（构造参数匹配预训练形状 n_times=3000/n_chans=128）
        model = Labram.from_pretrained(
            LABRAM_PRETRAINED,
            n_times=LABRAM_N_TIMES, n_chans=LABRAM_N_CHANS, n_outputs=0,
        )
        # 2. 时间维适配到词窗
        pad_right, truncate_right = _adapt_pretrained_dimensions(model, n_samples)
        # 3. 通道名包装（h5 通道名 → LaBraM 名）
        if channel_names is not None:
            model = LabramChannelWrapper(
                model, channel_names, pad_right, truncate_right)
        self.wrapped = model
        self.mean_pool = Mean(dim=1)
        self.feature_dim = LABRAM_EMBED_DIM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.wrapped(x)          # (B, n_tokens, 200) — 含 CLS 的所有 token
        return self.mean_pool(out)     # (B, 200) — Mean(dim=1)


# ===================================================================
# 编码器注册表
# ===================================================================

# -------------------------------------------------------------------
# Vendored EEG foundation-model encoders (LaBraM / REVE / LUNA)
# -------------------------------------------------------------------
# These three live under ``model/eeg2video_core/pretrained`` because the EEG2Video
# task is what needs them, and because vendoring means copying the released
# modelling code rather than depending on braindecode / trust_remote_code.
# ``model/`` goes on sys.path so ``eeg2video_core`` resolves the same way
# ``model/eeg2video.py`` already imports it -- one module object, not two.
from .pretrained.encoders_pretrained import (  # noqa: E402
    PRETRAINED_ENCODER_REGISTRY,
    LabramOfficialEncoder,
    LunaEncoder,
    ReveEncoder,
)

def _build_eegvid_nice(n_chans: int, n_samples: int, **kwargs: Any) -> nn.Module:
    """EEGVid's modified-NICE encoder, imported lazily.

    Kept as a function rather than a class reference so that ``torch_geometric``
    -- which only this encoder needs -- is not a hard dependency of importing
    ``model.encoders``.
    """

    from .eegvid.encoder import EEGVidEncoder

    return EEGVidEncoder(n_chans=n_chans, n_samples=n_samples, **kwargs)


ENCODER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "tsconv": {
        "builder": TSConvEncoder,
        "extra_kwargs": (),     # 仅 n_chans
        "description": "NICE TSConv (Song et al., ICLR 2024), feature_dim=1440",
    },
    "eegnet": {
        "builder": EEGNetEncoder,
        "extra_kwargs": ("F1", "D", "F2", "kern_length", "dropout_rate"),
        "description": "EEGNet-8,2 (Lawhern et al., 2018), "
                       "feature_dim = F2 * (n_samples // 32)",
    },
    "conformer": {
        "builder": ConformerEncoder,
        "extra_kwargs": (),     # 仅 n_chans
        "description": "EEG-Conformer (Song et al., 2023), feature_dim=440",
    },
    # The braindecode-based LaBraM wrapper, kept and still registered.  The
    # canonical ``labram`` key now resolves to the *reference* LaBraM
    # implementation vendored from the LaBraM project (no braindecode
    # dependency); this entry preserves the original behaviour for callers
    # that want it.  See the merge block below.
    "labram_braindecode": {
        "builder": LabramEncoder,
        "extra_kwargs": ("channel_names",),
        "description": "Pretrained LaBraM (Jiang et al., ICLR 2024) minus probe, "
                       "braindecode implementation, feature_dim=200; "
                       "requires channel_names",
    },
    "eegvid_nice": {
        "builder": _build_eegvid_nice,
        "extra_kwargs": ("feature_dim", "emb_size", "gat_dropout", "conv_dropout"),
        "description": "EEGVid's modified NICE encoder (arXiv:2505.21385): "
                       "channel GAT -> temporal conv -> linear, unit-normalised, "
                       "feature_dim=1024. Randomly initialised -- the paper "
                       "releases no weights -- so it must be trained.",
    },
}

# Install the vendored LaBraM / REVE / LUNA encoders.  ``labram`` is
# deliberately overridden by the vendored reference implementation so that the
# name means what the LaBraM paper's model means; the previous entry stays
# reachable as ``labram_braindecode``.
ENCODER_REGISTRY.update(PRETRAINED_ENCODER_REGISTRY)


def build_encoder(name: str, n_chans: int, n_samples: int,
                  config: Optional[Dict[str, Any]] = None,
                  **extra: Any) -> nn.Module:
    """按注册表构建编码器（n_chans/n_samples 必传，编码器专属参数经 config 透传）。

    只透传 ENCODER_REGISTRY 中该编码器声明的 extra_kwargs（如 eegnet 的
    F1/D/kern_length、labram 的 channel_names），其余参数忽略——避免把
    其他编码器的参数误传给本编码器。优先级: 显式 **extra > config dict >
    编码器类默认值。未知编码器显式 KeyError（原则 7: 失败必须可见）。
    """
    if name not in ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown encoder '{name}'. Available: {list_encoders()}"
        )
    entry = ENCODER_REGISTRY[name]
    kwargs: Dict[str, Any] = {"n_chans": n_chans, "n_samples": n_samples}
    if isinstance(config, dict):
        for k in entry["extra_kwargs"]:
            if k in config:
                kwargs[k] = config[k]
    for k in entry["extra_kwargs"]:
        if k in extra:
            kwargs[k] = extra[k]
    return entry["builder"](**kwargs)


def list_encoders() -> List[str]:
    return list(ENCODER_REGISTRY.keys())


# ===================================================================
# 旧名别名（向后兼容: from model.encoders import ... 与旧代码引用）
# ===================================================================

TSConvBackbone = TSConvEncoder
EEGNetBackbone = EEGNetEncoder
ConformerBackbone = ConformerEncoder

__all__ = [
    "build_encoder", "list_encoders", "ENCODER_REGISTRY",
    "TSConvEncoder", "EEGNetEncoder", "ConformerEncoder", "LabramEncoder",
    "TSConvBackbone", "EEGNetBackbone", "ConformerBackbone",
    "ConvModule", "MultiHeadAttention", "FeedForwardBlock",
    "TransformerEncoderBlock", "Mean",
    "LabramChannelWrapper", "LabramEncoder",
    "_adapt_pretrained_dimensions", "_load_channel_mapping",
    "N_CHANNELS", "N_TIMES", "NICE_FEATURE_DIM", "CONFORMER_FEATURE_DIM",
    "LABRAM_PRETRAINED", "LABRAM_N_TIMES", "LABRAM_N_CHANS",
    "LABRAM_PATCH_SIZE", "LABRAM_EMBED_DIM",
]
