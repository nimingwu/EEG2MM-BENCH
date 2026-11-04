"""视频数据集共享的小工具与资源目录。

``_resize_rgb`` 被 SEED-DV 与 ds005662 共用；``RESOURCES`` 指向随包分发的
刺激标注 JSON。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RESOURCES = Path(__file__).resolve().parent / "resources"
FRAME_WIDTH = 512
FRAME_HEIGHT = 288


def _resize_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR source frame -> RGB, resized to the released output geometry."""

    import cv2

    return cv2.resize(
        frame[..., ::-1], (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

