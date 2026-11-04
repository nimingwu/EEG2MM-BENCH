"""
Tokenisation for the text metrics, made language-aware.

Why this exists
---------------
Every text metric in this benchmark tokenises on whitespace -- either explicitly
(`text.split()` in ``text_metrics.py``) or implicitly inside a library
(torchmetrics' `bleu_score`, `jiuse`/jiwer's WER, `rouge_score`).  That is
correct for English and **catastrophic for Chinese**, which has no inter-word
spaces: an entire Chinese sentence becomes a single token, so two different
sentences share no n-grams at all.

Measured on `加拿大真是个好地方` vs `加拿大是个好地方。`:

    metric            whitespace      character-level
    BLEU-1              0.0000            0.5969
    WER                 1.0000            0.2222      (1.0 is the worst possible)

Nothing was wrong with the data or the model -- the evaluator simply could not
see inside a Chinese sentence.

The fix
-------
When a string contains CJK, tokenise **character by character** (dropping
whitespace), which is standard practice for Chinese NLG evaluation.  Otherwise
fall back to the original whitespace behaviour, so English results are
bit-identical to before.

Two entry points:
  * ``tokenize(text)``    -> list[str], for callers that do their own counting
  * ``retokenize(text)``  -> str with single spaces, for libraries that
    tokenise on whitespace internally; splitting this again reproduces
    ``tokenize(text)`` exactly.
"""

from __future__ import annotations

import os
import re
from typing import List

# NLTK corpora search path.  `punkt` lives in /root/nltk_data, but `wordnet` and
# `omw-1.4` (needed by METEOR) are only present in the SemKey checkout -- and on
# 2026-09-21 /root/nltk_data/corpora was found emptied, which silently breaks
# METEOR for English too, not just Chinese.  Both are tried, in order.
NLTK_DATA_CANDIDATES = (
    os.environ.get("EEG2X_NLTK_DATA", ""),
    "/root/nltk_data",
    "/datas/eeg_benchmark_project/fuxian/fuxian-SEMKEY/SemKey/"
    "data/zuco_preprocessed_dataframe/nltk_data",
)


def ensure_nltk() -> None:
    """Make the known NLTK data directories visible to `nltk.data.find`."""
    import nltk
    for path in NLTK_DATA_CANDIDATES:
        if path and os.path.isdir(path) and path not in nltk.data.path:
            nltk.data.path.insert(0, path)

# CJK Unified Ideographs + Extension A, Hiragana, Katakana, Hangul, and the
# fullwidth/CJK punctuation blocks.  The extras matter because Chisco sentences
# carry fullwidth commas and periods.
_CJK = re.compile(
    r'[一-鿿'      # CJK Unified Ideographs
    r'㐀-䶿'       # Extension A
    r'぀-ヿ'       # Hiragana + Katakana
    r'가-힯'       # Hangul syllables
    r'＀-￯'       # fullwidth forms + CJK punctuation
    r'　-〿]'      # CJK symbols and punctuation
)


def contains_cjk(text: str) -> bool:
    """True when the string has any CJK character (i.e. needs char tokenising)."""
    return bool(_CJK.search(text or ""))


def tokenize(text: str) -> List[str]:
    """Tokens for one string.

    CJK text -> one token per non-space character.
    Anything else -> the original whitespace split (lowercased), unchanged.
    """
    text = "" if text is None else str(text)
    if contains_cjk(text):
        # Whitespace is dropped rather than emitted as a token: it carries no
        # meaning in Chinese and would otherwise inflate every n-gram count.
        return [ch for ch in text.strip().lower() if not ch.isspace()]
    return text.strip().lower().split()


def retokenize(text: str) -> str:
    """`tokenize(text)` re-joined with single spaces.

    Use this when handing text to a library that will split on whitespace
    itself -- the library then recovers exactly `tokenize(text)`.
    """
    return " ".join(tokenize(text))
