"""Fast/Slow labels for the Dynamic GLMNet, from optical flow.

The paper (Sec. 3) defines the label source:

    "Optical Flow Score: the optical flow score (OFS) of each 24 FPS
     two-second video clip obtained by **averaging the length of the optical
     flow vectors**, ranging from 0.008 (almost static) to 6.252 (rapidly
     changing).  ... We choose the **median OFS of 1.799** as the threshold to
     make sure the label is balanced."

SEED-DV ships the result as ``All_video_optical_flow_score.npy``; ds005662
ships nothing equivalent, and **no code in either EEG2Video project computes
it** -- the file is only ever read.  So the score has to be reproduced here.

What the paper fixes and what it does not
-----------------------------------------
Fixed: the score is the mean optical-flow magnitude over the clip, and the
Fast/Slow split is at the **median**.
Not fixed: which optical-flow estimator.  Any standard dense estimator
satisfies "the length of the optical flow vectors"; Farneback is used here
because it is in OpenCV and needs no model download.

Consequence, stated plainly: ds005662's scores are **not comparable in
absolute value** to SEED-DV's.  The clips differ (600 ms / 25 fps / 15 frames
against 2 s / 24 fps / 48 frames), and the estimator is unspecified.  What
*is* comparable is the rule the paper actually relies on -- threshold at the
median, which makes the label balanced by construction and is robust to both
differences.  SEED-DV's own 1.799 is its median, so the same rule is applied
here rather than the constant.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG = logging.getLogger("eeg2video_paper.optical_flow")

#: Caps the frame count fed to the estimator.  A 600 ms clip at 25 fps has 15
#: frames, so this only bites on longer stimuli; it keeps the cost bounded if
#: one ever appears.
MAX_FRAMES = 64


def _read_gray(path: Path) -> List[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stimulus video {path}")
    frames: List[np.ndarray] = []
    while len(frames) < MAX_FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    if len(frames) < 2:
        raise RuntimeError(f"{path} yielded {len(frames)} frames; need >= 2")
    return frames


def optical_flow_score(path: Path) -> float:
    """Mean optical-flow magnitude over one clip, in pixels per frame.

    This is the paper's OFS: the optical flow is estimated between every
    consecutive frame pair and the magnitudes are averaged over all pairs and
    all pixels.
    """

    import cv2

    frames = _read_gray(Path(path))
    magnitudes = []
    for previous, current in zip(frames[:-1], frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0,
        )
        magnitudes.append(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
    return float(np.mean(magnitudes))


def fast_slow_labels(
    stimuli: Sequence[Path],
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], Dict[str, int], float]:
    """OFS and Fast/Slow labels for a stimulus set.

    ``threshold`` defaults to the **median** OFS of the set -- the paper's own
    rule for balancing the two classes -- rather than SEED-DV's 1.799, which is
    that dataset's median and not a general constant.
    """

    scores: Dict[str, float] = {}
    for path in sorted(stimuli):
        scores[path.name] = optical_flow_score(path)
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if threshold is None:
        threshold = float(np.median(values))
        LOG.info("OFS median threshold for this stimulus set: %.4f "
                 "(SEED-DV's published value is 1.799)", threshold)
    labels = {name: int(score >= threshold) for name, score in scores.items()}
    fast = sum(labels.values())
    LOG.info("Fast/Slow split: %d fast / %d slow (threshold %.4f)",
             fast, len(labels) - fast, threshold)
    return scores, labels, float(threshold)
