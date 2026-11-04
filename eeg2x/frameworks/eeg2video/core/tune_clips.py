"""Clip-identity deduplication for the Tune-A-Video fine-tune's training set.

Tune-A-Video never sees EEG.  It fine-tunes on the stimulus clips alone, and
the benchmark's manifests are indexed by *trial*: a dataset that shows each
stimulus to several trials -- or, in ds008257's case, to a structure that
repeats a stimulus within one subject -- therefore hands the trainer the same
(frames, caption) pair once per showing.

Measured on this benchmark:

    ds008257   5722 training rows -> 1000 distinct clips  (groups of 6/5/4/3)
    ds005662    630 training rows ->   90 distinct clips  (every group of 7)

The duplicates are byte-identical down to the caption, so the trainer was
rebuilding the same example thousands of times.  Deduplicating the training
set is not an approximation: it removes rows that carry no information the
retained row does not already carry.

:func:`distinct_clip_indices` returns the first occurrence of each distinct
(frames, caption) pair, in manifest order, so callers keep their existing
indexing by adding one indirection.  ``EEG2VIDEO_TUNE_DEDUP=0`` restores the
original trial-counted behaviour.
"""

from __future__ import annotations

import hashlib
import os
from typing import Sequence

import numpy as np

#: Set to ``0`` to train on every manifest row, as the chains did before the
#: redundancy was measured.
DEDUP_ENV = "EEG2VIDEO_TUNE_DEDUP"


def dedup_enabled() -> bool:
    return os.environ.get(DEDUP_ENV, "1").strip() not in ("0", "false", "no")


def distinct_clip_indices(
    frames: np.ndarray,
    captions: Sequence[str],
    *,
    label: str = "Tune-A-Video",
) -> np.ndarray:
    """Indices of the first occurrence of each distinct (frames, caption) pair.

    The key is the clip's actual content -- the frame bytes the trainer will
    read, plus the caption it will be conditioned on -- rather than a manifest
    id.  Identity of the *example* is what makes a row redundant, and a content
    key cannot drift when a dataset's ids mean something slightly different.

    ``frames`` may be a ``np.load(..., mmap_mode="r")`` handle; rows are read
    one at a time and never held.
    """

    if len(frames) < len(captions):
        raise ValueError(
            f"frames has {len(frames)} rows but there are {len(captions)} "
            "captions; the two must line up one-to-one"
        )
    if not dedup_enabled():
        print(
            f"{label}: dedup disabled ({DEDUP_ENV}=0), training on all "
            f"{len(captions)} rows",
            flush=True,
        )
        return np.arange(len(captions), dtype=np.int64)

    seen: set[bytes] = set()
    keep: list[int] = []
    for index, caption in enumerate(captions):
        digest = hashlib.blake2b(
            np.ascontiguousarray(frames[index]).tobytes(), digest_size=16)
        digest.update(str(caption).encode("utf-8"))
        key = digest.digest()
        if key in seen:
            continue
        seen.add(key)
        keep.append(index)

    kept = np.asarray(keep, dtype=np.int64)
    if not len(kept):
        raise ValueError("deduplication kept no training clips")
    print(
        f"{label}: {len(captions)} training rows -> {len(kept)} distinct clips "
        f"({len(captions) / len(kept):.2f}x redundancy removed; "
        f"set {DEDUP_ENV}=0 to keep the originals)",
        flush=True,
    )
    return kept
