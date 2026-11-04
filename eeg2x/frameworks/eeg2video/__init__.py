"""EEG-to-Video 框架（Scheme D + AnimateDiff）：编码器可插拔。

这是与 Only-CLIP 并列的第二套框架。它与图像框架共享编码器契约，但目标空间
不同：图像预测 1024 维 CLIP 图像嵌入再交给 SDXL；视频预测 ``[B, 77, 768]``
的 CLIP 文本上下文再交给 AnimateDiff。两者不能合并成一套阶段表。

迁自视频仓库的 ``model/eeg2video.py`` 与 ``model/eeg2video_core/``。编码器
实现暂时保留视频仓库自带的 vendored LaBraM / REVE / LUNA，因为它们与视频
checkpoint 的权重布局一致；与 ``eeg2x.encoders`` 的合并需要先做权重与输出
等价性验证，尚未进行。

**导入是惰性的**：``core.config`` 在 import 时快照目录环境变量，而调用方
（``pipeline._set_track_env``）必须先设好这些变量再导入模型。所以这里不在
模块顶层 import ``.model``。
"""
from __future__ import annotations

__all__ = ["EEG2VideoModel"]


def __getattr__(name: str):
    if name == "EEG2VideoModel":
        from .model import EEG2VideoModel

        return EEG2VideoModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
