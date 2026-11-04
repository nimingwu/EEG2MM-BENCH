"""EEG-Video-Action 视频预处理：EEGLAB 记录 -> 2 秒片段。"""
from __future__ import annotations

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


from eeg2x.protocol import (
    StimulusRecord,
    canonical_json,
    make_stimulus_id,
    normalize_channel_names,
    resample_eeg,
    seconds_to_sample,
    sha256_file,
    write_common_h5_metadata,
    write_stimulus_manifest,
)
from eeg2x.h5_schema import require_valid_h5_file
from eeg2x.profiles_video import (
    DS005662_BAND_HZ,
    DS005662_ENCODER_CLIP,
    DS005662_ENCODER_MODEL_NORM,
    DS005662_ENCODER_PROFILES,
    DS005662_ENCODER_RATE,
    DS005662_ENCODER_UNITS,
    DS005662_ENCODER_WINDOWS,
    DS005662_FRAME_HEIGHT,
    DS005662_FRAME_WIDTH,
    DS005662_NATIVE_RATE,
    DS005662_NATIVE_SAMPLES,
    DS005662_N_FRAMES,
    DS005662_PADDED_SAMPLES,
    DS005662_SOURCE_CHANNELS,
    DS008257_CLIP_SECONDS,
    DS008257_ENCODER_MODEL_NORM,
    DS008257_ENCODER_PROFILES,
    DS008257_ENCODER_RATE,
    DS008257_ENCODER_UNITS,
    DS008257_ENCODER_WINDOWS,
    DS008257_NATIVE_RATE,
    DS008257_SOURCE_BAND_HZ,
    DS008257_SOURCE_CHANNELS,
    DS008257_SOURCE_REFERENCE,
    DS008257_TEST_ID_MIN,
    DS008257_WINDOW_START_SEC,
    DS008257_WINDOW_STOP_SEC,
    EEG_VIDEO_ACTION_BAND_HZ,
    EEG_VIDEO_ACTION_CLIP_SECONDS,
    EEG_VIDEO_ACTION_ENCODER_MODEL_NORM,
    EEG_VIDEO_ACTION_ENCODER_PROFILES,
    EEG_VIDEO_ACTION_ENCODER_RATE,
    EEG_VIDEO_ACTION_ENCODER_UNITS,
    EEG_VIDEO_ACTION_ENCODER_WINDOWS,
    EEG_VIDEO_ACTION_EOG_CHANNELS,
    EEG_VIDEO_ACTION_FRAME_HEIGHT,
    EEG_VIDEO_ACTION_FRAME_WIDTH,
    EEG_VIDEO_ACTION_FRAME_RULE,
    EEG_VIDEO_ACTION_NATIVE_RATE,
    EEG_VIDEO_ACTION_NOTCH_HZ,
    EEG_VIDEO_ACTION_N_FRAMES,
    EEG_VIDEO_ACTION_SOURCE_CHANNELS,
    EEG_VIDEO_ACTION_TEST_VIDEOS,
    SEED_DV_ENCODER_MODEL_NORM,
    SEED_DV_ENCODER_PROFILES,
    SEED_DV_ENCODER_RATE,
    SEED_DV_ENCODER_UNITS,
    SEED_DV_ENCODER_WINDOWS,
    get_preprocessing_profile,
    seed_dv_window_spec,
)
from eeg2x.evaluation.registry import get_task_config
from eeg2x.protocol import SEED
from eeg2x.seeding import seed_everything
from .video_common import RESOURCES

# ===========================================================================
# EEG-Video-Action  (Yao et al., J Neural Eng 21:016018, 2024;
#                     Zenodo record 10512414, "Single-Shot" subset)
# ===========================================================================
#
# Third dataset for the same ``video_generation`` task.  Like the other two it
# is added here rather than in a separate module: the benchmark keeps one
# loader per task and routes datasets through it (ADDING_A_DATASET.md).
#
# The release ships one EEGLAB ``.set`` per (subject, stimulus) at 2048 Hz with
# 68 channels (64 EEG + 4 EOG), and no videos (copyright) -- only a YouTube
# link and a start/end timestamp per stimulus, in the record's description.
# Those timestamps are the alignment; ``eeg_video_action_stimuli.json`` holds
# them, built and cross-checked by
# ``scripts/build_eeg_video_action_stimuli.py``.
#
# Three consequences shape this branch:
#
#   * The EEG segment *is* the played clip, so an epoch is a slice of a long
#     recording rather than a trial.  Epochs tile it at 2 s -- SEED-DV's clip
#     length -- and the tail remainder is dropped.  The epoch index is defined
#     in **time**, not in samples, so every encoder branch (200 Hz and LUNA's
#     256 Hz alike) produces the same epoch count and epoch ``e`` always covers
#     the same two seconds of stimulus.
#
#   * The video must be decoded at a *seeked* position: the played clip starts
#     ``start_sec`` into a source file that can be minutes long.  Frames are
#     read for the test split only, one sequential pass per stimulus.
#
#   * Not every source video is 16:9 (three are 5:4 or 4:3) while the task's
#     ground-truth geometry is fixed at 288x512.  Those frames are letterboxed
#     -- scaled to fit and padded to the centre -- so no content is cropped and
#     no axis is stretched; the padding is recorded per stimulus.

EVA_DATASET_ID = "eeg_video_action"
#: Same task as the other two -- the HDF5 schema, the evaluator and the five
#: metrics are shared; only the dataset id and the numbers inside differ.
EVA_TASK_ID = "video_generation"

EVA_STIMULI_PATH = RESOURCES / "eeg_video_action_stimuli.json"
EVA_CAPTIONS_PATH = RESOURCES / "eeg_video_action_stimulus_captions.json"
EVA_DEFAULT_ROOT = Path("/datas/eeg_benchmark_project/EEG-Video-Action")
#: The release unzips to ``SingleShot/SingleShot`` and ``video/video``; the
#: single-level spellings are accepted so a re-unzip with one less nesting
#: still works.
EVA_EEG_DIR_CANDIDATES = ("SingleShot/SingleShot", "SingleShot")
EVA_VIDEO_DIR_CANDIDATES = ("video/video", "video")

#: Downscaling to the task's geometry, so area averaging (not bilinear, which
#: samples a 2x2 neighbourhood regardless of the ratio).  Recorded in every
#: stimulus transform, because it is part of how the ground truth was made.
EVA_RESIZE_INTERPOLATION = "area"

BAD_CHANNEL_RE = re.compile(
    r"Potential bad channels?:\s*(.+?)\.?\s*$", re.IGNORECASE | re.MULTILINE)


# ---------------------------------------------------------------------------
# source facts
# ---------------------------------------------------------------------------

def load_eva_stimuli(path: Path = EVA_STIMULI_PATH) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/build_eeg_video_action_stimuli.py "
            f"first (it parses the Zenodo record and checks every row against "
            f"the EEG)"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != EVA_DATASET_ID:
        raise ValueError(
            f"{path} declares dataset_id={payload.get('dataset_id')!r}, "
            f"expected {EVA_DATASET_ID!r}"
        )
    if not payload.get("videos"):
        raise ValueError(f"{path} lists no stimuli")
    return payload


def load_eva_captions(path: Path = EVA_CAPTIONS_PATH) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/caption_eeg_video_action_stimuli.py "
            f"first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def eva_resolve_dir(root: Path, candidates) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path
    raise FileNotFoundError(f"none of {candidates} exist under {root}")


def eva_stimulus_path(video_dir: Path, row: Dict[str, Any]) -> Path:
    """The local ``.mp4`` for a stimulus row, by id (filename case tolerant)."""

    path = video_dir / row["file"]
    if path.is_file():
        return path
    wanted = str(row["file"]).lower()
    for candidate in sorted(video_dir.glob("*.mp4")):
        if candidate.name.lower() == wanted:
            return candidate
    raise FileNotFoundError(f"missing stimulus video {path}")


def eva_bad_channels(eeg_dir: Path, subject_name: str) -> List[str]:
    """The release's per-subject "potential bad channels" note, as a list.

    Recorded, never applied: the release calls them *potential* and says users
    should apply their own criteria, and interpolation is a fitted transform.
    Empty when the subject's README has no such line.
    """

    readme = eeg_dir / subject_name / "README.txt"
    if not readme.is_file():
        return []
    match = BAD_CHANNEL_RE.search(readme.read_text(encoding="utf-8"))
    if not match:
        return []
    return [name.strip() for name in match.group(1).split(",") if name.strip()]


def eva_discover_subjects(eeg_dir: Path) -> List[str]:
    subjects = sorted(p.name for p in eeg_dir.glob("SUB_*") if p.is_dir())
    if not subjects:
        raise FileNotFoundError(f"no SUB_* directories under {eeg_dir}")
    return subjects


def eva_subject_id(subject_name: str) -> str:
    """``SUB_01`` -> ``sub-01`` -- the id every other track uses."""

    digits = subject_name.split("_", 1)[-1]
    return f"sub-{int(digits):02d}"


def eva_epoch_count(seconds: float) -> int:
    return int(seconds // EEG_VIDEO_ACTION_CLIP_SECONDS)


def eva_frame_indices(row: Dict[str, Any], epoch_index: int) -> List[int]:
    """The six source-frame indices of one epoch, on the mp4's own timeline.

    ``k * clip / n_frames`` into the epoch, endpoint excluded -- the released
    EEG2Video rule, and the same 1/3 s step SEED-DV's branch uses.  Indices are
    absolute in the source video, so they are recorded verbatim in the
    manifest and the ground truth can be rebuilt without this code.
    """

    fps = float(row["fps"])
    start = float(row["start_sec"]) + epoch_index * EEG_VIDEO_ACTION_CLIP_SECONDS
    indices = [
        int(round((start + k * EEG_VIDEO_ACTION_CLIP_SECONDS
                   / EEG_VIDEO_ACTION_N_FRAMES) * fps))
        for k in range(EEG_VIDEO_ACTION_N_FRAMES)
    ]
    last = int(row["n_frames"]) - 1
    if indices[-1] > last:
        raise ValueError(
            f"{row['file']} epoch {epoch_index}: frames reach {indices[-1]} "
            f"but the file has {last}"
        )
    return indices


def letterbox_geometry(source_width: int, source_height: int,
                       height: int, width: int) -> Dict[str, int]:
    """Where a source of this size lands inside a ``height`` x ``width`` target.

    Pure, so the *predicted* geometry can be written into a training row's
    manifest without decoding anything -- and the test split checks the
    measurement against it, which is what makes the padding auditable rather
    than merely asserted.

    Shared by every dataset whose stimulus is not already the target aspect
    ratio; the target is passed in rather than read from a constant so two
    datasets with different geometries cannot silently inherit each other's.
    """

    scale = min(width / source_width, height / source_height)
    new_w = max(1, int(round(source_width * scale)))
    new_h = max(1, int(round(source_height * scale)))
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    return {
        "source_width": int(source_width), "source_height": int(source_height),
        "scaled_width": new_w, "scaled_height": new_h,
        "pad_top": int(top), "pad_left": int(left),
        "pad_bottom": int(height - new_h - top),
        "pad_right": int(width - new_w - left),
    }


def letterbox_frame(frame: np.ndarray, height: int, width: int,
                    ) -> Tuple[np.ndarray, Dict[str, int]]:
    """BGR frame -> RGB (height, width, 3), scaled to fit and centred.

    Cropping would drop content and stretching would distort it; padding keeps
    the frame intact and is recorded instead, so the manifest says exactly how
    many columns or rows were added.
    """

    import cv2

    source_h, source_w = frame.shape[:2]
    geometry = letterbox_geometry(source_w, source_h, height, width)
    resized = cv2.resize(
        frame, (geometry["scaled_width"], geometry["scaled_height"]),
        interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    top, left = geometry["pad_top"], geometry["pad_left"]
    canvas[top:top + geometry["scaled_height"],
           left:left + geometry["scaled_width"]] = resized
    return canvas[..., ::-1], geometry


def eva_letterbox_geometry(source_width: int, source_height: int) -> Dict[str, int]:
    """EEG-Video-Action's letterbox geometry -- see :func:`letterbox_geometry`."""

    return letterbox_geometry(source_width, source_height,
                              EEG_VIDEO_ACTION_FRAME_HEIGHT,
                              EEG_VIDEO_ACTION_FRAME_WIDTH)


def eva_letterbox(frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    """BGR frame -> RGB (288, 512, 3), scaled to fit and centred with padding."""

    return letterbox_frame(frame, EEG_VIDEO_ACTION_FRAME_HEIGHT,
                           EEG_VIDEO_ACTION_FRAME_WIDTH)


def eva_decode_groups(video_path: Path, index_groups: List[List[int]],
                      ) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """All ground-truth clips of one stimulus, in one seeked pass.

    The clips live ``start_sec`` into a source file that can be minutes long,
    so the reader seeks to the first wanted frame and then advances
    sequentially -- OpenCV's ``CAP_PROP_POS_FRAMES`` was verified to return the
    exact frame on all thirteen sources.  Returns the frames *and* the
    letterbox measurement for each group, because the geometry differs per
    stimulus and has to reach the manifest.
    """

    import cv2

    if not index_groups:
        return (np.zeros((0, EEG_VIDEO_ACTION_N_FRAMES,
                          EEG_VIDEO_ACTION_FRAME_HEIGHT,
                          EEG_VIDEO_ACTION_FRAME_WIDTH, 3), dtype=np.uint8), [])
    wanted: Dict[int, List[Tuple[int, int]]] = {}
    for group, indices in enumerate(index_groups):
        if len(indices) != EEG_VIDEO_ACTION_N_FRAMES:
            raise ValueError(f"clip {group} has {len(indices)} frame indices")
        for slot, index in enumerate(indices):
            wanted.setdefault(index, []).append((group, slot))

    out = np.zeros(
        (len(index_groups), EEG_VIDEO_ACTION_N_FRAMES,
         EEG_VIDEO_ACTION_FRAME_HEIGHT, EEG_VIDEO_ACTION_FRAME_WIDTH, 3),
        dtype=np.uint8,
    )
    geometry: List[Optional[Dict[str, int]]] = [None] * len(index_groups)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stimulus video {video_path}")
    first = min(wanted)
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    cursor = first
    last = max(wanted)
    while cursor <= last:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"{video_path} ended before frame {last}")
        slots = wanted.pop(cursor, None)
        if slots is not None:
            letterboxed, measured = eva_letterbox(frame)
            for group, slot in slots:
                out[group, slot] = letterboxed
                geometry[group] = measured
        cursor += 1
    capture.release()
    if any(item is None for item in geometry):
        raise RuntimeError(f"{video_path}: some clips got no frames")
    return out, [item for item in geometry if item is not None]


# ---------------------------------------------------------------------------
# EEG
# ---------------------------------------------------------------------------

def read_eva_segment(eeg_dir: Path, subject_name: str, session: str,
                     ) -> Tuple[np.ndarray, List[str], float]:
    """One subject's EEG for one stimulus: ``(64, n)`` float64 volts.

    Applies the two stateless steps the branch declares before the band-pass:
    channel selection (the 4 EOG channels are dropped, not regressed -- see the
    profile's comment) and the average rereference the release's README asks
    for.  Reading is the only step that touches the ``.fdt``.
    """

    import mne

    mne.set_log_level("ERROR")
    path = eeg_dir / subject_name / f"{session}_eeg.set"
    if not path.is_file():
        raise FileNotFoundError(f"missing EEG segment {path}")
    raw = mne.io.read_raw_eeglab(str(path), preload=True, verbose="ERROR")
    if abs(float(raw.info["sfreq"]) - EEG_VIDEO_ACTION_NATIVE_RATE) > 1e-6:
        raise ValueError(f"{path} runs at {raw.info['sfreq']} Hz, expected "
                         f"{EEG_VIDEO_ACTION_NATIVE_RATE}")
    channels = [c for c in raw.ch_names if not c.upper().startswith("EXG")]
    dropped = len(raw.ch_names) - len(channels)
    if dropped != EEG_VIDEO_ACTION_EOG_CHANNELS:
        raise ValueError(
            f"{path} has {len(raw.ch_names)} channels, {dropped} of them "
            f"non-EEG; expected {EEG_VIDEO_ACTION_SOURCE_CHANNELS} channels "
            f"with {EEG_VIDEO_ACTION_EOG_CHANNELS} to drop"
        )
    # DATA_PROTOCOL rule 2: normalise the montage names and refuse duplicates
    # once normalised.  A1..B32 is already canonical, so this is a check rather
    # than a rename -- but it is the check that would catch a re-released file
    # with, say, both "A1" and "a1".
    channels = normalize_channel_names(channels)
    raw.pick(channels)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    data = raw.get_data().astype(np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f"{path} holds non-finite samples")
    return data, channels, float(raw.info["sfreq"])


def band_limit_eva(data: np.ndarray, rate: float,
                   ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The branch's 0.1-100 Hz band-pass and 50 Hz notch, plus a measurement.

    Both are zero-phase FIRs at the native clock, so nothing above the stored
    Nyquist survives the decimation that follows.  The notch is a separate
    filter because 50 Hz is *inside* the passband -- the band-pass cannot stand
    in for it, and the release's README asks for it explicitly.  The RMS before
    and after is recorded rather than glossed: this recording is DC-coupled and
    the README warns about very strong mains, so the filter moves the signal a
    long way, and the number says how far.
    """

    import mne

    low, high = EEG_VIDEO_ACTION_BAND_HZ
    record: Dict[str, Any] = {
        "declared_band_hz": [low, high],
        "notch_hz": [EEG_VIDEO_ACTION_NOTCH_HZ],
        "applied_at_hz": rate,
    }
    before = float(np.sqrt(np.mean(np.square(data))))
    filtered = mne.filter.filter_data(
        data, sfreq=rate, l_freq=low, h_freq=high, verbose=False)
    filtered = mne.filter.notch_filter(
        filtered, Fs=rate, freqs=[EEG_VIDEO_ACTION_NOTCH_HZ], verbose=False)
    after = float(np.sqrt(np.mean(np.square(filtered))))
    record["rms_before"] = before
    record["rms_after"] = after
    record["rms_ratio"] = after / before if before else float("nan")
    record["highpass_note"] = (
        f"{low} Hz on a {data.shape[-1] / rate:.1f}s continuous segment; no "
        f"epoch-length caveat as on ds005662"
    )
    if not np.isfinite(filtered).all():
        raise ValueError("band-pass produced non-finite EEG values")
    return filtered.astype(np.float32), record


# ---------------------------------------------------------------------------
# encoder branch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaBranch:
    """What one encoder's branch does to the already band-limited recording.

    Unlike the other two datasets the filtering happens once per subject on the
    continuous signal, so a branch only moves the clock, rescales, and cuts the
    recording into epochs.  The epoch count is passed in, not derived from the
    branch's own sample count: it is a property of the *stimulus* timeline, so
    LUNA's 256 Hz and EEGNet's 200 Hz must agree on it.
    """

    encoder: str
    profile_id: str
    window: Dict[str, int]
    sampling_rate: float
    eeg_units: str
    model_input_normalization: str
    scale_factor: Optional[float] = None

    @property
    def clip_samples(self) -> int:
        return int(round(EEG_VIDEO_ACTION_CLIP_SECONDS * self.sampling_rate))

    def apply(self, continuous: np.ndarray, n_epochs: int) -> np.ndarray:
        out = np.asarray(continuous, dtype=np.float32)
        if self.sampling_rate != EEG_VIDEO_ACTION_NATIVE_RATE:
            # ``resample_eeg`` is the protocol's resampler: exact rational
            # ratio, polyphase anti-aliasing, and a linear-extrapolation edge
            # pad rather than zeros.  The pad matters here in a way it does not
            # for a short trial -- this is a continuous 200-400 s recording, and
            # a zero pad would drag the first and last epoch toward silence.
            out = resample_eeg(out, EEG_VIDEO_ACTION_NATIVE_RATE,
                               self.sampling_rate, axis=-1)
        if self.scale_factor is not None:
            out = out * np.float32(self.scale_factor)
        need = n_epochs * self.clip_samples
        if out.shape[-1] < need:
            raise ValueError(
                f"branch {self.profile_id}: {out.shape[-1]} samples cover "
                f"{out.shape[-1] / self.sampling_rate:.3f}s, but {n_epochs} "
                f"epochs need {need / self.sampling_rate:.3f}s"
            )
        # (channels, n_epochs * clip) -> (n_epochs, channels, clip)
        epochs = out[:, :need].reshape(
            out.shape[0], n_epochs, self.clip_samples).transpose(1, 0, 2)
        if not np.isfinite(epochs).all():
            raise ValueError(f"branch {self.profile_id} produced non-finite EEG")
        return np.ascontiguousarray(epochs)


def eva_build_branch(encoder: str) -> EvaBranch:
    if encoder not in EEG_VIDEO_ACTION_ENCODER_PROFILES:
        raise KeyError(
            f"unknown encoder {encoder!r}; expected "
            f"{sorted(EEG_VIDEO_ACTION_ENCODER_PROFILES)}"
        )
    profile = get_preprocessing_profile(
        EEG_VIDEO_ACTION_ENCODER_PROFILES[encoder])
    scale_factor = None
    window: Dict[str, int] = {}
    for step in profile.steps:
        if step.name == "scale":
            scale_factor = float(step.parameters["factor"])
        elif step.name == "epoch":
            window = {
                "window_samples": int(step.parameters["window_samples"]),
                "stride_samples": int(step.parameters["stride_samples"]),
                "n_windows": int(step.parameters["n_windows"]),
            }
    if not window:
        raise KeyError(f"profile {profile.profile_id!r} declares no 'epoch' step")
    return EvaBranch(
        encoder=encoder,
        profile_id=profile.profile_id,
        window=window,
        sampling_rate=EEG_VIDEO_ACTION_ENCODER_RATE[encoder],
        eeg_units=EEG_VIDEO_ACTION_ENCODER_UNITS[encoder],
        model_input_normalization=EEG_VIDEO_ACTION_ENCODER_MODEL_NORM[encoder],
        scale_factor=scale_factor,
    )


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def write_eeg_video_action_split_h5(
    *, out_path: Path, subject: str, split: str, eeg: np.ndarray,
    channel_names: Sequence[str], rows: Sequence[Dict[str, Any]],
    captions: Sequence[str], records: Sequence[StimulusRecord],
    gt_frames: Optional[np.ndarray], branch: EvaBranch,
    filter_record: Dict[str, Any], bad_channels: Sequence[str],
) -> int:
    import h5py

    profile = get_preprocessing_profile(branch.profile_id)
    n_epochs, n_channels, _ = eeg.shape
    # A caption is a training target, not decoration: a row that lost its
    # caption would train against a neighbour's, and the manifest's own length
    # check would not notice because ``event_caption`` is not a stimulus
    # column.  So it is checked here.
    for name, values in (("rows", rows), ("captions", captions),
                         ("records", records)):
        if len(values) != n_epochs:
            raise ValueError(
                f"{out_path}: {len(values)} {name} for {n_epochs} epochs")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    concepts = [int(str(row["video_id"]).split("_", 1)[0]) for row in rows]
    blocks = [1 if str(row["source_bdf"]).upper().startswith("P1") else 2
              for row in rows]

    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("data", data=eeg, dtype=np.float32)
        h5.create_dataset("channel_names",
                          data=np.asarray(list(channel_names), dtype=object),
                          dtype=string_dtype)
        h5.create_dataset("event_concept", data=np.asarray(concepts, np.int32))
        h5.create_dataset("event_block", data=np.asarray(blocks, np.int32))
        h5.create_dataset("event_caption",
                          data=np.asarray(list(captions), dtype=object),
                          dtype=string_dtype)
        if gt_frames is not None:
            h5.create_dataset("gt_frames", data=gt_frames, dtype=np.uint8)

        h5.attrs["dataset"] = EVA_DATASET_ID
        h5.attrs["subject_id"] = subject
        h5.attrs["sampling_rate"] = float(branch.sampling_rate)
        h5.attrs["n_channels"] = int(n_channels)
        h5.attrs["n_epochs"] = int(n_epochs)

        h5.attrs["encoder_id"] = branch.encoder
        h5.attrs["encoder_window_specs_json"] = canonical_json(
            EEG_VIDEO_ACTION_ENCODER_WINDOWS)
        h5.attrs["window_samples"] = int(branch.window["window_samples"])
        h5.attrs["window_stride_samples"] = int(branch.window["stride_samples"])
        h5.attrs["n_windows"] = int(branch.window["n_windows"])
        h5.attrs["model_input_normalization"] = branch.model_input_normalization
        h5.attrs["bandpass_json"] = canonical_json(filter_record)
        h5.attrs["bad_channels_json"] = canonical_json(list(bad_channels))
        h5.attrs["preprocessing_branch_json"] = canonical_json({
            "encoder": branch.encoder,
            "profile_id": branch.profile_id,
            "source_rate_hz": EEG_VIDEO_ACTION_NATIVE_RATE,
            "resampled_to_hz": branch.sampling_rate,
            "clip_seconds": EEG_VIDEO_ACTION_CLIP_SECONDS,
            "n_frames": EEG_VIDEO_ACTION_N_FRAMES,
            "frame_height": EEG_VIDEO_ACTION_FRAME_HEIGHT,
            "frame_width": EEG_VIDEO_ACTION_FRAME_WIDTH,
            "resize_interpolation": EVA_RESIZE_INTERPOLATION,
            "eog_regression": "none",
            "bad_channel_interpolation": "none",
            "test_videos": list(EEG_VIDEO_ACTION_TEST_VIDEOS),
            "model_input_normalization": branch.model_input_normalization,
            "encoder_window_specs": EEG_VIDEO_ACTION_ENCODER_WINDOWS,
        })

        write_common_h5_metadata(
            h5,
            dataset_id=EVA_DATASET_ID,
            task_id=EVA_TASK_ID,
            subject_id=subject,
            sampling_rate=branch.sampling_rate,
            source_sampling_rate=EEG_VIDEO_ACTION_NATIVE_RATE,
            n_channels=n_channels,
            n_epochs=n_epochs,
            split=split,
            epoch_type="video_clip",
            eeg_units=branch.eeg_units,
            profile=profile,
            preprocessing_state={
                # Nothing here is fitted: the band-pass and notch are
                # stateless, and the EOG channels were dropped rather than
                # regressed out.  So no step is scoped to the source, and the
                # test file is allowed to contain all of them.
                "fitted_split": "none",
                "encoder": branch.encoder,
                "bandpass": filter_record,
                "resampled_to_hz": branch.sampling_rate,
                "eog_channels_dropped": EEG_VIDEO_ACTION_EOG_CHANNELS,
                "eog_regression": "none",
                "bad_channel_interpolation": "none",
                "bad_channels_recorded": list(bad_channels),
                "model_input_normalization": branch.model_input_normalization,
                "seed": SEED,
            },
        )
        write_stimulus_manifest(h5, records)

    schema = get_task_config(EVA_TASK_ID)["h5_schema"]
    require_valid_h5_file(str(out_path), schema)
    return n_epochs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def eeg_video_action_process_subject(
    stimuli: Dict[str, Any], captions: Dict[str, str], out_dir: Path,
    subject_name: str, branches: Sequence[EvaBranch],
) -> Dict[str, Dict[str, int]]:
    seed_everything(SEED)
    eeg_dir = Path(stimuli["eeg_root"])
    video_dir = Path(stimuli["video_root"])
    subject = eva_subject_id(subject_name)
    bad_channels = eva_bad_channels(eeg_dir, subject_name)
    videos = stimuli["videos"]

    # Read and band-limit each stimulus once; every branch is cut from these.
    prepared: Dict[str, Dict[str, Any]] = {}
    filter_record: Dict[str, Any] = {}
    channel_names: List[str] = []
    for video_id in sorted(videos):
        row = videos[video_id]
        data, names, rate = read_eva_segment(eeg_dir, subject_name, row["session"])
        if not channel_names:
            channel_names = names
        elif names != channel_names:
            raise ValueError(f"{subject}: {video_id} channel order differs")
        filtered, measured = band_limit_eva(data, rate)
        filter_record[video_id] = measured
        seconds = filtered.shape[-1] / EEG_VIDEO_ACTION_NATIVE_RATE
        n_epochs = eva_epoch_count(seconds)
        if n_epochs == 0:
            raise ValueError(f"{subject}/{video_id}: {seconds:.3f}s is shorter "
                             f"than one {EEG_VIDEO_ACTION_CLIP_SECONDS}s epoch")
        prepared[video_id] = {"filtered": filtered, "n_epochs": n_epochs}
        LOG.info("%s %s: %.1fs -> %d epochs", subject, video_id, seconds,
                 n_epochs)

    written: Dict[str, Dict[str, int]] = {b.encoder: {} for b in branches}
    for split, video_ids in (
        ("train", tuple(sorted(set(videos) - set(EEG_VIDEO_ACTION_TEST_VIDEOS)))),
        ("test", tuple(sorted(set(videos) & set(EEG_VIDEO_ACTION_TEST_VIDEOS)))),
    ):
        if not video_ids:
            raise ValueError(f"{subject}: no stimuli for split {split}")
        rows, captions_for_rows, records = [], [], []
        gt_parts: List[np.ndarray] = []
        for video_id in video_ids:
            row = videos[video_id]
            n_epochs = prepared[video_id]["n_epochs"]
            path = eva_stimulus_path(video_dir, row)
            for epoch in range(n_epochs):
                rows.append({**row, "video_id": video_id, "epoch": epoch})
                key = f"{video_id}#{epoch}"
                if key not in captions:
                    raise ValueError(f"no caption for {key} in "
                                     f"{EVA_CAPTIONS_PATH.name}")
                captions_for_rows.append(captions[key])
                records.append(StimulusRecord(
                    stimulus_id=make_stimulus_id(
                        EVA_DATASET_ID, f"{video_id}-e{epoch:04d}"),
                    modality="video",
                    uri=str(path),
                    sha256=str(row["sha256"]),
                    # Onset on the *source* video's timeline, so the record's
                    # own start/end numbers reconstruct this row without any
                    # benchmark code.
                    onset_sec=round(float(row["start_sec"])
                                    + epoch * EEG_VIDEO_ACTION_CLIP_SECONDS, 4),
                    duration_sec=EEG_VIDEO_ACTION_CLIP_SECONDS,
                    alignment_source="zenodo_segment_table",
                    transform={
                        "n_frames": EEG_VIDEO_ACTION_N_FRAMES,
                        "height": EEG_VIDEO_ACTION_FRAME_HEIGHT,
                        "width": EEG_VIDEO_ACTION_FRAME_WIDTH,
                        "fps": float(row["fps"]),
                        "source_start_sec": float(row["start_sec"]),
                        "source_end_sec": float(row["end_sec"]),
                        "youtube": str(row["youtube"]),
                        "frame_rule": EEG_VIDEO_ACTION_FRAME_RULE,
                        "frame_indices": eva_frame_indices(row, epoch),
                        "resize_interpolation": EVA_RESIZE_INTERPOLATION,
                        "letterbox_geometry": eva_letterbox_geometry(
                            int(row["width"]), int(row["height"])),
                    },
                ))
            if split == "test":
                groups = [eva_frame_indices(row, epoch)
                          for epoch in range(n_epochs)]
                frames, geometry = eva_decode_groups(path, groups)
                gt_parts.append(frames)
                # The measured padding must match what the manifest claims for
                # every row of this stimulus, or the record is describing
                # frames that were not made that way.
                expected = canonical_json(eva_letterbox_geometry(
                    int(row["width"]), int(row["height"])))
                if any(canonical_json(g) != expected for g in geometry):
                    raise ValueError(
                        f"{path}: measured letterbox geometry {geometry[0]} does "
                        f"not match the recorded "
                        f"{eva_letterbox_geometry(int(row['width']), int(row['height']))}")
        gt_frames = np.concatenate(gt_parts, axis=0) if gt_parts else None

        for branch in branches:
            per_video = [branch.apply(prepared[v]["filtered"], prepared[v]["n_epochs"])
                         for v in video_ids]
            eeg = np.concatenate(per_video, axis=0)
            branch_dir = out_dir / branch.encoder
            branch_dir.mkdir(parents=True, exist_ok=True)
            out_path = (branch_dir
                        / f"{EVA_DATASET_ID}_{subject}_{split}_epoched.h5")
            written[branch.encoder][split] = write_eeg_video_action_split_h5(
                out_path=out_path, subject=subject, split=split, eeg=eeg,
                channel_names=channel_names, rows=rows,
                captions=captions_for_rows, records=records,
                gt_frames=gt_frames, branch=branch,
                filter_record=filter_record, bad_channels=bad_channels,
            )
            LOG.info("wrote %s (%d epochs, %d samples)", out_path,
                     written[branch.encoder][split], eeg.shape[-1])
    return written


def eeg_video_action_run_loader(
    data_root: str, out_dir: str,
    subjects: Optional[Sequence[str]] = None,
    encoders: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, int]]:
    root = Path(data_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stimuli = load_eva_stimuli()
    captions = load_eva_captions()
    eeg_dir = eva_resolve_dir(root, EVA_EEG_DIR_CANDIDATES)

    if subjects:
        names = [s if s.upper().startswith("SUB_") else f"SUB_{int(s.split('-')[-1]):02d}"
                 for s in subjects]
    else:
        names = eva_discover_subjects(eeg_dir)
    # ``eeg_root``/``video_root`` in the table are absolute paths from build
    # time; re-resolve against the requested root so a moved dataset is either
    # used correctly or refused loudly.
    stimuli = {**stimuli, "eeg_root": str(eeg_dir),
               "video_root": str(eva_resolve_dir(root, EVA_VIDEO_DIR_CANDIDATES))}

    branches = [eva_build_branch(name) for name in (encoders or ["eegnet"])]
    totals: Dict[str, Dict[str, int]] = {b.encoder: {} for b in branches}
    for name in names:
        written = eeg_video_action_process_subject(
            stimuli, captions, out, name, branches)
        for encoder_name, splits in written.items():
            for split, count in splits.items():
                totals[encoder_name][split] = (
                    totals[encoder_name].get(split, 0) + count
                )
    return totals


# ===========================================================================
# ds008257  (EEG Moments Dataset; Gifford et al., arXiv:2608.28768)
# ===========================================================================
#
# Fourth dataset for the ``video_generation`` task.  The tidiest of the four to
# load, because the release has done the hard part:
#
#   * the EEG is **already epoched** -- one row per trial, with a
#     ``stimulus_id`` naming the video that trial showed -- so there is no
#     alignment to derive and no stimulus table to transcribe.  Contrast
#     EEG-Video-Action, where the link between a recording and a video had to
#     be reconstructed from the Zenodo record.
#   * the stimuli ship with the dataset, so every manifest row points at a file
#     that exists.
#
# What the branch does is therefore only the two things this benchmark always
# decides for itself: **which slice of a trial is scored** (the middle two
# seconds of the three-second stimulus, which resamples to exactly SEED-DV's
# 400 samples) and **which trials are the test set** (the release's own 102
# test videos, one trial each).

LOG = logging.getLogger("eeg2x.preprocessing.eeg_video_action")


