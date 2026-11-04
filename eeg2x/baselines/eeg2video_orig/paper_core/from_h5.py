"""Materialise the paper chain's arrays from the benchmark HDF5.

``data.prepare()`` in the original project rebuilds its caches by reading
SEED-DV's ``EEG/subNN.npy`` and the block videos directly.  Inside a benchmark
that would bypass the loader's single source of truth -- the seed protocol,
the train/test boundary, the sample ordering and the stimulus manifest all
live in the HDF5, not in the raw dataset.  So this module produces the *same
files* from the HDF5 instead, and everything downstream in ``data.py`` runs
unchanged.

The mapping is one-to-one and is why the swap is cheap:

============  ==========================================================
paper chain   benchmark HDF5
============  ==========================================================
segments      ``data`` -- already ``(N, 62, 400)``, the paper's 2 s clip
windows       7 x 100 at stride 50, cut from the segments
feature       3 x 200 at stride 100, cut from the segments
DE            5-band DE of the feature windows (``data._extract_de``)
captions      ``event_caption`` (BLIP, same as SEED-DV ships)
frames        test: ``gt_frames``; train: re-decoded from ``stimulus_uri``
              at the ``frame_indices`` the loader recorded
split         blocks 1-6 train, block 7 test -- identical in both
============  ==========================================================

Row order is the HDF5's: train first, then test, which is exactly the paper's
``block_index * samples_per_block + ...`` ordering because both walk blocks
1-7 and, inside a block, ``class_position`` then ``repeat``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SETTINGS

LOG = logging.getLogger("eeg2video_paper.from_h5")

#: Which dataset the benchmark HDF5 came from.  The two differ in file naming,
#: channel count, trial length and where the Dynamic GLMNet's Fast/Slow labels
#: come from, so the preparation branches on it; the geometry itself is already
#: handled by ``config.SETTINGS``, which reads the EEG2VIDEO_* overrides.
DATASET_ID = os.environ.get("EEG2VIDEO_DATASET", "seed_dv")

#: ds005662's stimulus videos, all four flip variants; the chain keeps the
#: unflipped ``base`` variant, matching the loader's branch.
DS005662_STIMULI_DIR = Path(
    "/datas/eeg_benchmark_project/ds005662-download/code/experiment_files"
    "/stimuli/videos_short_600ms")

#: ds008257's stimulus videos, keyed by the filename each manifest row cites.
DS008257_STIMULI_DIR = Path(
    "/datas/eeg_benchmark_project/openneurodownload/ds008257"
    "/stimulus_set/stimulus_set/mp4_h264")

#: Dataset -> the directory holding *that dataset's* stimulus videos.  The
#: Dynamic GLMNet's Fast/Slow labels are a property of the videos, so computing
#: them from the wrong dataset's directory is not a crash -- it is a silent
#: substitution of one stimulus set's optical flow for another's.  Every
#: dataset that needs them is listed here, and an unlisted one raises instead.
STIMULI_DIRS: Dict[str, Path] = {
    "ds005662": DS005662_STIMULI_DIR,
    "ds008257": DS008257_STIMULI_DIR,
}


def _window(segments: np.ndarray, window: int, stride: int,
            count: int) -> np.ndarray:
    """``(n, C, T)`` -> ``(n, count, C, window)`` at a fixed stride."""

    starts = np.arange(0, segments.shape[-1] - window + 1, stride)
    if len(starts) < count:
        raise ValueError(
            f"{segments.shape[-1]} samples yield {len(starts)} windows of "
            f"{window} at stride {stride}, need {count}"
        )
    starts = starts[:count]
    return np.stack(
        [segments[..., s:s + window] for s in starts], axis=1
    ).astype(np.float32)


def _extract_frames_for_video(
    uri: str, requests: Sequence[Tuple[int, Sequence[int]]],
    shape: Tuple[int, int, int],
) -> Dict[int, np.ndarray]:
    """Decode one stimulus video once and cut every requested clip from it.

    Same OpenCV read, BGR->RGB and resize as
    ``eeg2video_benchmark_loader.extract_gt_frames``, so the paper model trains
    on the very pixels the benchmark scores against.  The whole video is
    scanned once and every wanted frame is captured on the way past:
    re-opening per clip would rescan from frame 0 every time, which is
    O(clips x frames) and made the equivalent loader path take ~20 minutes per
    subject instead of ~45 seconds.
    """

    import cv2

    height, width, n_frames = shape
    wanted: Dict[int, List[Tuple[int, int]]] = {}
    for row, indices in requests:
        for slot, index in enumerate(indices):
            wanted.setdefault(index, []).append((row, slot))

    out = {row: np.zeros((n_frames, height, width, 3), dtype=np.uint8)
           for row, _ in requests}
    capture = cv2.VideoCapture(str(uri))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stimulus video {uri}")
    found, cursor, last = 0, 0, max(wanted)
    while cursor <= last and found < len(wanted):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"{uri} ended before frame {last}")
        slots = wanted.get(cursor)
        if slots is not None:
            resized = cv2.resize(frame[..., ::-1], (width, height),
                                 interpolation=cv2.INTER_LINEAR)
            for row, slot in slots:
                out[row][slot] = resized
            found += 1
        cursor += 1
    capture.release()
    if found != len(wanted):
        raise RuntimeError(
            f"read {found}/{len(wanted)} distinct frames from {uri}")
    return out


def _frame_indices(transform: Dict[str, Any], uri: str) -> List[int]:
    """Which source frames to cut for this clip.

    SEED-DV records them explicitly, because its clips are offsets into a
    13-minute block video.  ds005662's stimulus *is* the 600 ms clip, so the
    frames are an even spread over the whole file -- the same rule the
    ds005662 loader applied when it wrote ``gt_frames``, which matters because
    test rows are scored against exactly those pixels.
    """

    if "frame_indices" in transform:
        return list(transform["frame_indices"])
    count = int(transform["n_frames"])
    import cv2

    capture = cv2.VideoCapture(str(uri))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stimulus video {uri}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if total < count:
        raise ValueError(f"{uri} has {total} frames, need {count}")
    return [int(round(i * (total - 1) / (count - 1))) for i in range(count)]


def _read_split(h5_path: Path) -> Dict[str, Any]:
    import h5py

    with h5py.File(h5_path, "r") as h5:
        n = int(h5.attrs["n_epochs"])
        out = {
            "segments": np.asarray(h5["data"][:], dtype=np.float32),
            "block": np.asarray(h5["event_block"][:]),
            "concept": np.asarray(h5["event_concept"][:]),
            "caption": [
                c.decode("utf-8") if isinstance(c, bytes) else str(c)
                for c in h5["event_caption"][:]
            ],
            "uri": [
                c.decode("utf-8") if isinstance(c, bytes) else str(c)
                for c in h5["stimulus_uri"][:]
            ],
            "transform": [
                json.loads(c.decode("utf-8") if isinstance(c, bytes) else str(c))
                for c in h5["stimulus_transform_json"][:]
            ],
            "gt_frames": (np.asarray(h5["gt_frames"][:], dtype=np.uint8)
                          if "gt_frames" in h5 else None),
            "n_epochs": n,
        }
        if out["segments"].shape[0] != n:
            raise ValueError(f"{h5_path}: data has {out['segments'].shape[0]} "
                             f"rows but declares {n}")
        return out


def _seed_dv_flow_labels(data_root: Path) -> np.ndarray:
    """SEED-DV's shipped OFS, thresholded at the paper's 1.799 (its median)."""

    path = data_root / "Video" / "meta-info" / "All_video_optical_flow_score.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing optical-flow labels: {path}")
    flow = np.load(path)
    expected = (7, SETTINGS.samples_per_block)
    if flow.shape != expected:
        raise ValueError(f"{path} has shape {flow.shape}, expected {expected}")
    return (flow >= SETTINGS.optical_flow_threshold).astype(np.int64).reshape(-1)


def _flow_by_stimulus(dataset_id: str) -> Dict[str, int]:
    """Fast/Slow per stimulus video, thresholded at that set's OFS median.

    SEED-DV ships the scores; the others do not, so they are computed here with
    ``optical_flow.optical_flow_score`` and split at the median -- the paper's
    own rule, applied to each set rather than to SEED-DV's constant, because
    the clips differ in length and frame rate and the paper never names its
    estimator.  See ``optical_flow`` for the full argument.

    Keyed by the video *filename*, which is what the manifest rows cite.
    """

    from .optical_flow import fast_slow_labels

    directory = STIMULI_DIRS.get(dataset_id)
    if directory is None:
        raise KeyError(
            f"no stimulus directory is registered for dataset {dataset_id!r}, "
            f"so the Fast/Slow labels cannot be computed for it.  Register it "
            f"in STIMULI_DIRS -- falling back to another dataset's videos "
            f"would silently label these trials with theirs.  Known: "
            f"{sorted(STIMULI_DIRS)}")
    if not directory.is_dir():
        raise FileNotFoundError(
            f"missing {dataset_id} stimuli at {directory}")
    videos = sorted(directory.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no .mp4 under {directory}")
    _scores, labels, _threshold = fast_slow_labels(videos)
    return labels


def prepare_from_h5(h5_dir: Path, run_root: Path, shared_root: Path,
                    subject: str, data_root: Path,
                    force: bool = False) -> Dict[str, Any]:
    """Write every cache ``data.py`` expects, sourced from the HDF5."""

    marker = run_root / "prepare.complete.json"
    if marker.is_file() and not force:
        LOG.info("prepare already complete: %s", marker)
        return json.loads(marker.read_text(encoding="utf-8"))
    run_root.mkdir(parents=True, exist_ok=True)
    shared_root.mkdir(parents=True, exist_ok=True)

    train_h5 = h5_dir / f"{DATASET_ID}_{subject}_train_epoched.h5"
    test_h5 = h5_dir / f"{DATASET_ID}_{subject}_test_epoched.h5"
    for path in (train_h5, test_h5):
        if not path.is_file():
            raise FileNotFoundError(f"missing benchmark HDF5 {path}")

    train, test = _read_split(train_h5), _read_split(test_h5)
    n_train, n_test = train["n_epochs"], test["n_epochs"]
    if (n_train, n_test) != (SETTINGS.train_samples, SETTINGS.test_samples):
        raise ValueError(
            f"h5 holds {n_train}+{n_test} epochs, the paper chain expects "
            f"{SETTINGS.train_samples}+{SETTINGS.test_samples}"
        )

    segments = np.concatenate([train["segments"], test["segments"]], axis=0)
    blocks = np.concatenate([train["block"], test["block"]])
    concepts = np.concatenate([train["concept"], test["concept"]])
    captions = train["caption"] + test["caption"]
    uris = train["uri"] + test["uri"]
    transforms = train["transform"] + test["transform"]
    if DATASET_ID == "seed_dv":
        flow = _seed_dv_flow_labels(data_root)
    else:
        flow = None  # ds005662 labels are per stimulus, resolved below

    np.save(run_root / "eeg_segments.float32.npy", segments)
    windows = _window(segments, SETTINGS.eeg_window_samples,
                      SETTINGS.eeg_stride_samples, SETTINGS.eeg_tokens)
    np.save(run_root / "eeg_windows.float32.npy", windows)
    feature_windows = _window(segments, SETTINGS.feature_window_samples,
                              SETTINGS.feature_stride_samples,
                              SETTINGS.feature_windows)
    np.save(run_root / "eeg_feature_windows.float32.npy", feature_windows)

    from .data import _extract_de  # the paper's own DE, unchanged

    # ``_extract_de`` returns a memmap onto ``data.DE_PATH`` and has already
    # written it.  Saving it again would open that same file for writing and
    # truncate it while the array being saved is still mapped to it, so the
    # write reads back zero bytes and fails -- which is exactly what happened
    # the first time this ran.
    de = _extract_de(feature_windows)

    # ---- video frames: test rows come from gt_frames, train rows are re-decoded
    frames = np.zeros((n_train + n_test, SETTINGS.video_frames,
                       SETTINGS.video_height, SETTINGS.video_width, 3),
                      dtype=np.uint8)
    if test["gt_frames"] is not None:
        frames[n_train:] = test["gt_frames"]
    else:
        LOG.warning("test h5 has no gt_frames; re-decoding the test stimuli")
    by_video: Dict[str, List[Tuple[int, Sequence[int]]]] = {}
    for row in range(n_train):
        by_video.setdefault(uris[row], []).append(
            (row, _frame_indices(transforms[row], uris[row])))
    for uri, requests in by_video.items():
        first = transforms[requests[0][0]]
        decoded = _extract_frames_for_video(
            uri, requests,
            (first["height"], first["width"], first["n_frames"]))
        for row, clip in decoded.items():
            frames[row] = clip
        LOG.info("decoded %d clips from %s", len(requests), Path(uri).name)
    if test["gt_frames"] is None:
        by_video = {}
        for row in range(n_test):
            index = n_train + row
            by_video.setdefault(uris[index], []).append(
                (index, _frame_indices(transforms[index], uris[index])))
        for uri, requests in by_video.items():
            first = transforms[requests[0][0]]
            decoded = _extract_frames_for_video(
                uri, requests,
                (first["height"], first["width"], first["n_frames"]))
            for row, clip in decoded.items():
                frames[row] = clip
    np.save(shared_root / "video_frames.uint8.npy", frames)

    # ---- manifest: one row per trial, in the HDF5's own order ----
    #
    # The dynamic label indexes differently per dataset.  SEED-DV's OFS file is
    # per (block, slot) -- one score per presentation.  ds005662 has no such
    # file, so the scores are computed from the stimulus videos and are a
    # property of the *video*, shared by every presentation of it.  The slot a
    # row occupies inside its block is ``class_position * repeats + repeat``,
    # and the loader writes rows in exactly that order, so a running counter
    # per block reproduces it without re-deriving it from the concept id.
    stimulus_labels = (_flow_by_stimulus(DATASET_ID)
                       if DATASET_ID != "seed_dv" else None)

    rows: List[Dict[str, Any]] = []
    seen: Dict[int, int] = {}
    for row in range(n_train + n_test):
        block = int(blocks[row])
        within = seen.get(block, 0)
        seen[block] = within + 1
        slot = (block - 1) * SETTINGS.samples_per_block + within
        if flow is not None:
            if not 0 <= slot < flow.size:
                raise ValueError(
                    f"row {row}: block {block} slot {within} maps to flow "
                    f"index {slot}, outside 0..{flow.size - 1}")
            label = int(flow[slot])
        else:
            name = Path(uris[row]).name
            if name not in stimulus_labels:
                raise KeyError(
                    f"no Fast/Slow label for stimulus {name!r}; the HDF5 "
                    "references a video outside the labelled stimulus set")
            label = int(stimulus_labels[name])
        rows.append({
            "sample_id": row,
            "block_index": block,
            "class_position": int(concepts[row]),
            # The SEED-DV manifest builder writes ``concept_one_based`` (its
            # 1-based class id) and ``concept`` (that minus one).  This path
            # only wrote ``concept``, so ``inference.infer``'s metadata block
            # -- which reads ``concept_one_based`` -- raised KeyError *after*
            # generating every clip.  ``event_concept`` is already 1-based
            # here (1..90), so it is the value that field wants; ``concept`` is
            # left as it is because the trained predictors' ``dynamic_label``
            # and ``class_position`` group on it.
            "concept_one_based": int(concepts[row]),
            "slot_in_block": within,
            "caption": captions[row],
            "stimulus_uri": uris[row],
            "dynamic_label": label,
        })

    (run_root / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    summary = {
        "source": "benchmark_hdf5",
        "h5_dir": str(h5_dir),
        "subject": subject,
        "train_samples": n_train,
        "test_samples": n_test,
        "segments": list(segments.shape),
        "windows": list(windows.shape),
        "feature_windows": list(feature_windows.shape),
        "de_features": list(de.shape),
        "video_frames": list(frames.shape),
        "dataset_id": DATASET_ID,
        "fast_slow_counts": {
            "fast": sum(1 for r in rows if r["dynamic_label"] == 1),
            "slow": sum(1 for r in rows if r["dynamic_label"] == 0),
        },
    }
    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("prepared from h5: %s", summary)
    return summary
