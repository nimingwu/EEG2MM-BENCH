"""EEG 编码器契约。

**所有编码器只实现一遍，跨任务（image / video / text）共享。** 框架拿到的只是
一个满足 :class:`EEGEncoder` 的模块；"换编码器"= 换一个 :class:`EncoderSpec`。

模型数学部分逐字来自 EEG2X-Benchmark-v2-test，未改；这里定义的是接口与注册。

设计要点
--------
``EncoderSpec.learning_rate``
    所有编码器共用框架统一配方（AdamW 平坦学习率，无 warmup、无调度）。
    唯一的数据集级差异是 LUNA 在 ThingsEEG 上把学习率设为 1e-4，其余设置
    完全相同；``learning_rate_datasets`` 限定该覆盖只在该数据集生效。训练
    循环只有一条代码路径，差异是**数据**而不是埋在循环里的
    ``if encoder == "luna"``。见 ``docs/LUNA_TRAINING_DEVIATION.md``。

``forward`` / ``forward_tokens``
    前者返回池化后的 ``[B, D]``，后者返回时序表征 ``[B, T', D]``。image 任务
    只用前者；video/text 接入时用后者，框架无需改动。

``profile`` 不在这里
    采样率、窗口长度、padding 属于 **(数据集 × 编码器)** 的预处理 profile，
    由 ``eeg2x.registry.DatasetSpec.profiles`` 决定，不在编码器内部。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import torch
from torch import Tensor, nn


@runtime_checkable
class EEGEncoder(Protocol):
    """所有 EEG 编码器实现的协议。

    不继承 ``nn.Module`` 也能通过 ``isinstance`` 检查（``runtime_checkable`` 只
    检查属性/方法存在）；实际实现都是 ``nn.Module``。
    """

    feature_dim: int

    def forward(self, eeg: Tensor) -> Tensor:
        """``[B, C, T]`` -> ``[B, feature_dim]``（池化表征）。"""
        ...

    def forward_tokens(self, eeg: Tensor) -> Tensor:
        """``[B, C, T]`` -> ``[B, T', feature_dim]``（时序表征，供 video/text）。"""
        ...

    def pretrained_metadata(self) -> dict:
        """预训练来源、权重哈希、通道名等，写进 checkpoint 供溯源。"""
        ...


class BaseEEGEncoder(nn.Module):
    """方便实现：默认把 ``forward_tokens`` 的时序维求均值作为 ``forward``。

    子类可覆盖任意一个。不需要时序输出的编码器可以只实现 ``forward``，
    ``forward_tokens`` 会明确报错而不是静默返回错的东西。
    """

    feature_dim: int

    def forward_tokens(self, eeg: Tensor) -> Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a token sequence; "
            "implement forward_tokens() before wiring this encoder into a "
            "task that needs temporal representations")

    def pretrained_metadata(self) -> dict:
        return {"pretrained_loaded": False}

    def extra_repr(self) -> str:
        return f"feature_dim={self.feature_dim}"


@dataclass(frozen=True)
class EncoderSpec:
    """编码器的构造与元数据契约。

    ``builder`` 的签名统一为 ``builder(n_chans, n_samples, config) -> nn.Module``；
    具体参数由各编码器自己从 ``config`` 里取（比如通道名、权重缓存路径）。
    """

    name: str
    builder: Callable[..., nn.Module]
    feature_dim: int | None          # None 表示随 n_samples 变化，由 builder 决定
    #: 框架统一配方下的学习率覆盖；``None`` 表示用框架默认 3e-4。
    learning_rate: float | None = None
    #: ``learning_rate`` 适用的数据集；``None`` = 所有数据集。LUNA 的覆盖只在
    #: ThingsEEG 上成立（其他数据集仍用 3e-4）。
    learning_rate_datasets: tuple[str, ...] | None = None
    required_packages: tuple[str, ...] = ()
    needs_channel_names: bool = False
    #: 非空表示该编码器的**输入契约与另一个编码器完全相同**，直接复用后者的
    #: HDF5，不重复预处理。ENIGMA 就是这种情况（它的输入与 ATM 一致）。
    h5_encoder: str | None = None
    description: str = ""

    def learning_rate_for(self, dataset: str) -> float | None:
        """该数据集上实际生效的学习率覆盖；``None`` 表示框架默认。"""
        if self.learning_rate is None:
            return None
        if self.learning_rate_datasets is None:
            return self.learning_rate
        return self.learning_rate if dataset in self.learning_rate_datasets else None

    def require_packages(self) -> None:
        """缺依赖时**明确报错**，而不是在构造到一半时抛 ImportError。

        这个仓库里出现过「以为跑的是 LUNA，其实是没装 rotary_embedding_torch」
        这类问题，所以把检查提到最前面。
        """
        import importlib.util

        missing = [p for p in self.required_packages
                   if importlib.util.find_spec(p) is None]
        if missing:
            raise RuntimeError(
                f"encoder {self.name!r} needs package(s) {missing} which are not "
                "installed in this environment. See environments/README.md and "
                "install the matching requirements file.")


def build(spec: EncoderSpec, n_chans: int, n_samples: int,
          config: dict | None = None, **extra) -> nn.Module:
    """按 spec 构造编码器，并校验 n_chans/n_samples 与产物一致。"""
    spec.require_packages()
    cfg = dict(config or {})
    cfg.update(extra)
    if spec.needs_channel_names and not cfg.get("channel_names"):
        raise ValueError(
            f"encoder {spec.name!r} needs explicit channel_names (it maps HDF5 "
            "channel names onto its own montage)")
    model = spec.builder(n_chans=n_chans, n_samples=n_samples, config=cfg)
    for attr in ("feature_dim",):
        if not hasattr(model, attr):
            raise TypeError(
                f"encoder {spec.name!r} built {type(model).__name__} without "
                f"{attr!r}; every encoder must declare it")
    return model


__all__ = [
    "BaseEEGEncoder",
    "EEGEncoder",
    "EncoderSpec",
    "build",
]
