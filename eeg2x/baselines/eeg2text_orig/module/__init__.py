"""EEG-To-Text（Wang & Ji, AAAI 2022）—— 词级输入 baseline。

注意与隔壁 `model/eeg2text` 区分：
  * `eegtotext`   = 本文（AAAI 2022，arXiv 2112.02690）：**词级** 840 维特征 → BART
  * `eeg2text`    = arXiv 2405.02165：**句子级**时域信号 → conv-transformer → BART
两者是前后作关系（后者的编码器源于前者），但输入表示与接线都不同，所以各写一份。
"""
from __future__ import annotations

from .model import (
    EEGToText, EEGToTextModule, WordLevelEncoder,
    MAX_WORDS, N_BANDS, N_CHANNELS, N_FEATURES,
)

__all__ = ["EEGToText", "EEGToTextModule", "WordLevelEncoder",
           "MAX_WORDS", "N_BANDS", "N_CHANNELS", "N_FEATURES"]
