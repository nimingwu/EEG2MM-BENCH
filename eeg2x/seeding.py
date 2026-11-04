"""唯一的进程级播种入口。

``eeg2x.protocol`` 定义种子协议（``SEED`` 与 ``derive_seed``），本模块负责把
种子真正施加到 ``random``、``numpy``、``torch`` 与 ``PYTHONHASHSEED`` 上。
单独成文件是为了让不依赖 torch 的模块（例如 ``eeg2x.protocol``）保持轻量；
torch 缺失时本模块跳过相应部分，CPU-only 的预处理仍可运行。
"""
from __future__ import annotations

import os
import random

from eeg2x.protocol import SEED


def seed_everything(seed: int = SEED) -> int:
    """播种 benchmark 会触及的每一个随机源，并返回使用的种子。

    关闭 cuDNN 自动调优，避免卷积算法被非确定性地选择；同时固定
    ``PYTHONHASHSEED``。同一进程内需要第二条独立随机流时，用
    :func:`eeg2x.protocol.derive_seed`。
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import numpy as np

    # derive_seed 返回 64 位；numpy 的 legacy RandomState 只接受 32 位。
    # 对 SEED 本身取模是无操作，因此不会改变既有运行的随机流。
    np.random.seed(int(seed) % (2 ** 32))
    try:
        import torch
    except ImportError:  # torch 对 CPU-only 的 loader 是可选的
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def seed_worker(worker_id: int) -> None:
    """确定性的 DataLoader worker 初始化。"""
    worker_seed = (SEED + int(worker_id)) % (2 ** 32)
    random.seed(worker_seed)
    import numpy as np

    np.random.seed(worker_seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(worker_seed)


__all__ = ["seed_everything", "seed_worker"]
