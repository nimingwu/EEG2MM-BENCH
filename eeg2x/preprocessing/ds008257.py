"""EMD（ds008257）视频预处理：OpenNeuro H5 记录 -> 关键帧片段。"""
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

DS008257_DATASET_ID = "ds008257"
DS008257_TASK_ID = "video_generation"

DS008257_STIMULI_PATH = RESOURCES / "ds008257_stimuli.json"
DS008257_CAPTIONS_PATH = RESOURCES / "ds008257_stimulus_captions.json"
DS008257_DEFAULT_ROOT = Path("/datas/eeg_benchmark_project/openneurodownload/ds008257")
#: ``<root>/EMD/eeg/sub-XX/sub-XX_ses-YY_preprocessed_eeg.h5``.
DS008257_EEG_SUBDIR = ("EMD", "eeg")

#: Downscaling to the task's geometry; recorded in every stimulus transform.
DS008257_RESIZE_INTERPOLATION = "area"
DS008257_NATIVE_RATE = 500.0
DS008257_SOURCE_CHANNELS = 127
DS008257_STIMULUS_SECONDS = 3.0
DS008257_N_FRAMES = 6
DS008257_FRAME_HEIGHT = 288
DS008257_FRAME_WIDTH = 512


# ---------------------------------------------------------------------------
# source facts
# ---------------------------------------------------------------------------

def load_ds008257_stimuli(path: Path = DS008257_STIMULI_PATH) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/build_ds008257_stimuli.py first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_id") != DS008257_DATASET_ID:
        raise ValueError(
            f"{path} declares dataset_id={payload.get('dataset_id')!r}, "
            f"expected {DS008257_DATASET_ID!r}")
    if not payload.get("videos"):
        raise ValueError(f"{path} lists no stimuli")
    return payload


def load_ds008257_captions(path: Path = DS008257_CAPTIONS_PATH) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/caption_ds008257_stimuli.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def ds008257_window_slice(epoch_start_sec: float,
                          native_rate: float = DS008257_NATIVE_RATE) -> slice:
    """The scored window inside a released epoch, as a sample slice.

    ``epoch_start_sec`` is the time of the epoch's *first* sample, which the
    release sets to -0.1 s -- the window is 0.5-2.5 s **after stimulus onset**,
    so it lands at samples 300..1300, not 250..1250.  Taking the offset from
    the file's own ``times`` array rather than assuming epochs start at zero is
    the difference between the right two seconds and a silently shifted clip.
    """

    first = int(round((DS008257_WINDOW_START_SEC - epoch_start_sec) * native_rate))
    last = int(round((DS008257_WINDOW_STOP_SEC - epoch_start_sec) * native_rate))
    if first < 0 or last <= first:
        raise ValueError(
            f"window {DS008257_WINDOW_START_SEC}-{DS008257_WINDOW_STOP_SEC}s "
            f"falls outside an epoch starting at {epoch_start_sec}s "
            f"(samples {first}..{last})")
    return slice(first, last)


def ds008257_keyframe_indices(n_frames: int) -> List[int]:
    """The six keyframes of the scored window, as indices into the video.

    Frame ``i`` of an ``n``-frame three-second video sits at ``3i / n``, so a
    time ``t`` is frame ``round(t * n / 3)``.  That is the same construction
    SEED-DV's branch uses (``start_seconds * SOURCE_FPS``), with the frame rate
    derived from the file instead of declared -- the container's own reported
    rate is not trustworthy on this set.
    """

    if n_frames <= 0:
        raise ValueError(f"n_frames must be positive, got {n_frames}")
    per_second = n_frames / DS008257_STIMULUS_SECONDS
    indices = [
        int(round((DS008257_WINDOW_START_SEC
                   + k * DS008257_CLIP_SECONDS / DS008257_N_FRAMES) * per_second))
        for k in range(DS008257_N_FRAMES)
    ]
    if indices[-1] >= n_frames:
        raise ValueError(
            f"keyframes {indices} reach past the last of {n_frames} frames")
    return indices


def ds008257_frame_indices(row: Dict[str, Any]) -> List[int]:
    """The keyframes of one stimulus, refusing a video the window overruns.

    The index rule scales with the frame count, so it cannot notice that a
    stimulus is simply too short: a one-second video would happily yield six
    indices and the "window" would silently be a different two seconds of
    something else.  The duration is what has to be checked.
    """

    length = float(row.get("seconds", DS008257_STIMULUS_SECONDS))
    if length + 1e-9 < DS008257_WINDOW_STOP_SEC:
        raise ValueError(
            f"{row.get('video_id', '?')}: the stimulus is {length}s long but "
            f"the scored window ends at {DS008257_WINDOW_STOP_SEC}s")
    return ds008257_keyframe_indices(int(row["n_frames"]))


def ds008257_middle_frame_index(n_frames: int) -> int:
    """The caption frame: literally the middle of the six keyframes.

    Defined as one of them rather than by its own formula, so the caption can
    never describe a frame the ground truth does not also score.
    """

    return ds008257_keyframe_indices(n_frames)[DS008257_N_FRAMES // 2]


def ds008257_discover_subjects(root: Path) -> List[str]:
    eeg_dir = root.joinpath(*DS008257_EEG_SUBDIR)
    if not eeg_dir.is_dir():
        raise FileNotFoundError(
            f"missing {eeg_dir} -- is {root} the ds008257 download?")
    subjects = sorted(p.name for p in eeg_dir.glob("sub-*") if p.is_dir())
    if not subjects:
        raise FileNotFoundError(f"no sub-* directories under {eeg_dir}")
    return subjects


# ---------------------------------------------------------------------------
# EEG
# ---------------------------------------------------------------------------

def read_ds008257_subject(root: Path, subject: str) -> Dict[str, Any]:
    """One subject's scored windows, in session/trial order.

    Returns the windowed EEG ``(N, 127, T)`` float32, each trial's stimulus id
    and session number, and the montage.  Only the scored slice of each epoch
    is read out of the HDF5 -- the released files are ~930 MB per session and
    the other 800 samples of every trial are never used.
    """

    import h5py

    eeg_dir = root.joinpath(*DS008257_EEG_SUBDIR) / subject
    metadata = eeg_dir / f"{subject}_eeg_metadata.npy"
    if not metadata.is_file():
        raise FileNotFoundError(f"missing EEG metadata {metadata}")
    meta = np.load(metadata, allow_pickle=True).item()
    channel_names = [str(name) for name in meta["ch_names"]]
    if len(channel_names) != DS008257_SOURCE_CHANNELS:
        raise ValueError(
            f"{subject}: {len(channel_names)} channels, expected "
            f"{DS008257_SOURCE_CHANNELS}")
    times = np.asarray(meta["times"], dtype=np.float64)
    if abs(float(times[1] - times[0]) - 1.0 / DS008257_NATIVE_RATE) > 1e-9:
        raise ValueError(
            f"{subject}: the released epochs run at "
            f"{1.0 / float(times[1] - times[0]):.1f} Hz, expected "
            f"{DS008257_NATIVE_RATE:g}")
    if DS008257_WINDOW_STOP_SEC > float(times[-1]):
        raise ValueError(
            f"{subject}: the scored window ends at "
            f"{DS008257_WINDOW_STOP_SEC}s but the epochs end at "
            f"{float(times[-1]):.3f}s")
    window = ds008257_window_slice(float(times[0]))

    parts, stimulus_ids, sessions = [], [], []
    for key in sorted(meta["stimulus_id"], key=lambda k: k.split("-")[-1]):
        session = int(key.split("-")[-1])
        path = eeg_dir / f"{subject}_ses-{session:02d}_preprocessed_eeg.h5"
        if not path.is_file():
            raise FileNotFoundError(f"missing session file {path}")
        with h5py.File(path, "r") as h5:
            block = np.asarray(h5["eeg"][:, :, window], dtype=np.float32)
        ids = np.asarray(meta["stimulus_id"][key], dtype=np.int64)
        if block.shape[0] != len(ids):
            raise ValueError(
                f"{path}: {block.shape[0]} EEG rows but {len(ids)} stimulus ids")
        if block.shape[1] != len(channel_names):
            raise ValueError(
                f"{path}: {block.shape[1]} channels but the montage lists "
                f"{len(channel_names)}")
        parts.append(block)
        stimulus_ids.append(ids)
        sessions.append(np.full(len(ids), session, dtype=np.int32))

    eeg = np.concatenate(parts, axis=0)
    ids = np.concatenate(stimulus_ids)
    session_ids = np.concatenate(sessions)
    if not np.isfinite(eeg).all():
        raise ValueError(f"{subject}: non-finite samples in the released EEG")
    expected = int(round(DS008257_CLIP_SECONDS * DS008257_NATIVE_RATE))
    if eeg.shape[-1] != expected:
        raise ValueError(
            f"{subject}: window is {eeg.shape[-1]} samples, expected {expected}")
    return {"eeg": eeg, "stimulus_id": ids, "session": session_ids,
            "channel_names": channel_names}


def ds008257_split(ids: np.ndarray, test_id_min: int) -> Tuple[List[int], List[int]]:
    """``(train rows, test rows)`` -- the release's split, one test trial each.

    Training gets every retained trial of every training video.  The test set
    gets **one** trial per held-out video: the first one, in session/trial
    order, so the choice is deterministic and reproducible.  Every session
    shows all 102 test videos, so the first occurrence is a session-1 trial for
    each of them -- scoring all ~2,300 test trials would cost roughly eight
    hours of evaluation per track and would not add a single class.
    """

    first_of: Dict[int, int] = {}
    for row, stimulus in enumerate(ids.tolist()):
        if stimulus >= test_id_min and stimulus not in first_of:
            first_of[stimulus] = row
    if not first_of:
        raise ValueError(f"no trial with stimulus_id >= {test_id_min}")
    train = [row for row, stimulus in enumerate(ids.tolist())
             if stimulus < test_id_min]
    test = [first_of[stimulus] for stimulus in sorted(first_of)]
    return train, test


# ---------------------------------------------------------------------------
# stimulus video
# ---------------------------------------------------------------------------

def ds008257_stack_gt(clips: Sequence[np.ndarray], n_rows: int) -> np.ndarray:
    """``(n_rows, n_frames, H, W, 3)`` from one clip per epoch.

    ``stack``, not ``concatenate``: each clip already carries its own frame
    axis, so concatenating would flatten those into the epoch axis and produce
    ``6N`` rows -- which the evaluator rejects, because ``gt_frames`` must have
    exactly one row per epoch.  Written as a function with the shape asserted
    because the mistake is a silent one until evaluation.
    """

    stacked = np.stack([np.asarray(clip) for clip in clips], axis=0)
    expected = (n_rows, DS008257_N_FRAMES, DS008257_FRAME_HEIGHT,
                DS008257_FRAME_WIDTH, 3)
    if stacked.shape != expected:
        raise ValueError(
            f"gt_frames {stacked.shape}, expected {expected} -- one clip per "
            f"epoch, each carrying its own {DS008257_N_FRAMES} frames")
    return stacked


def ds008257_decode_keyframes(mp4: Path, indices: List[int],
                              ) -> Tuple[np.ndarray, Dict[str, int]]:
    """The six keyframes of one stimulus, letterboxed to the task's geometry.

    Every stimulus is one trial's worth of ground truth and every trial of a
    video scores the same six frames, so this runs once per stimulus rather
    than once per row: 102 decodes for the whole test set.
    """

    import cv2

    out = np.zeros((len(indices), DS008257_FRAME_HEIGHT, DS008257_FRAME_WIDTH, 3),
                   dtype=np.uint8)
    geometry: Optional[Dict[str, int]] = None
    wanted = {index: slot for slot, index in enumerate(indices)}
    capture = cv2.VideoCapture(str(mp4))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stimulus video {mp4}")
    cursor, last = 0, max(indices)
    while cursor <= last:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"{mp4} ended before frame {last}")
        slot = wanted.get(cursor)
        if slot is not None:
            letterboxed, measured = letterbox_frame(
                frame, DS008257_FRAME_HEIGHT, DS008257_FRAME_WIDTH)
            if geometry is None:
                geometry = measured
            elif geometry != measured:
                raise ValueError(f"{mp4}: letterbox geometry changed mid-file")
            out[slot] = letterboxed
        cursor += 1
    capture.release()
    if geometry is None:
        raise RuntimeError(f"{mp4}: no keyframes decoded")
    return out, geometry


# ---------------------------------------------------------------------------
# encoder branch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ds008257Branch:
    """What one encoder's branch does to the already-windowed trials.

    The epochs arrive aligned one-per-trial, so a branch only moves the clock
    and rescales -- there is no tiling step, unlike EEG-Video-Action's
    continuous recordings.
    """

    encoder: str
    profile_id: str
    window: Dict[str, int]
    sampling_rate: float
    eeg_units: str
    model_input_normalization: str
    scale_factor: Optional[float] = None
    #: Channels to keep, by name, or None for the whole montage.  Only LaBraM
    #: sets it: its positional table cannot place 50 of EMD's extended 10-05
    #: names, so its branch stores the 77 it can.
    channels: Optional[Tuple[str, ...]] = None

    def apply(self, windows: np.ndarray,
              channel_names: Sequence[str]) -> np.ndarray:
        out = np.asarray(windows, dtype=np.float32)
        if self.channels is not None:
            index = []
            for name in self.channels:
                if name not in channel_names:
                    raise ValueError(
                        f"branch {self.profile_id} keeps channel {name!r}, "
                        f"which this subject does not have")
                index.append(list(channel_names).index(name))
            out = out[:, index, :]
        if self.sampling_rate != DS008257_NATIVE_RATE:
            out = resample_eeg(out, DS008257_NATIVE_RATE, self.sampling_rate,
                               axis=-1)
        if self.scale_factor is not None:
            out = out * np.float32(self.scale_factor)
        expected = int(round(DS008257_CLIP_SECONDS * self.sampling_rate))
        if out.shape[-1] != expected:
            raise ValueError(
                f"branch {self.profile_id}: {out.shape[-1]} samples, expected "
                f"{expected} at {self.sampling_rate:g} Hz")
        if not np.isfinite(out).all():
            raise ValueError(f"branch {self.profile_id} produced non-finite EEG")
        return np.ascontiguousarray(out)


def ds008257_build_branch(encoder: str) -> Ds008257Branch:
    if encoder not in DS008257_ENCODER_PROFILES:
        raise KeyError(
            f"unknown encoder {encoder!r}; expected "
            f"{sorted(DS008257_ENCODER_PROFILES)}")
    profile = get_preprocessing_profile(DS008257_ENCODER_PROFILES[encoder])
    scale_factor, window, channels = None, {}, None
    for step in profile.steps:
        if step.name == "scale":
            scale_factor = float(step.parameters["factor"])
        elif step.name == "select_encoder_channels":
            channels = tuple(step.parameters["channels"])
        elif step.name == "epoch":
            window = {
                "window_samples": int(step.parameters["window_samples"]),
                "stride_samples": int(step.parameters["stride_samples"]),
                "n_windows": int(step.parameters["n_windows"]),
            }
    if not window:
        raise KeyError(f"profile {profile.profile_id!r} declares no 'epoch' step")
    return Ds008257Branch(
        encoder=encoder, profile_id=profile.profile_id, window=window,
        sampling_rate=DS008257_ENCODER_RATE[encoder],
        eeg_units=DS008257_ENCODER_UNITS[encoder],
        model_input_normalization=DS008257_ENCODER_MODEL_NORM[encoder],
        scale_factor=scale_factor,
        channels=channels,
    )


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def write_ds008257_split_h5(
    *, out_path: Path, subject: str, split: str, eeg: np.ndarray,
    channel_names: Sequence[str], stimulus_ids: Sequence[int],
    session_ids: Sequence[int], captions: Sequence[str],
    records: Sequence[StimulusRecord], gt_frames: Optional[np.ndarray],
    branch: Ds008257Branch,
) -> int:
    import h5py

    profile = get_preprocessing_profile(branch.profile_id)
    n_epochs, n_channels, _ = eeg.shape
    for name, values in (("stimulus_ids", stimulus_ids),
                         ("session_ids", session_ids),
                         ("captions", captions), ("records", records)):
        if len(values) != n_epochs:
            raise ValueError(
                f"{out_path}: {len(values)} {name} for {n_epochs} epochs")
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("data", data=eeg, dtype=np.float32)
        h5.create_dataset("channel_names",
                          data=np.asarray(list(channel_names), dtype=object),
                          dtype=string_dtype)
        h5.create_dataset("event_concept",
                          data=np.asarray(list(stimulus_ids), np.int32))
        h5.create_dataset("event_block",
                          data=np.asarray(list(session_ids), np.int32))
        h5.create_dataset("event_caption",
                          data=np.asarray(list(captions), dtype=object),
                          dtype=string_dtype)
        if gt_frames is not None:
            h5.create_dataset("gt_frames", data=gt_frames, dtype=np.uint8)

        h5.attrs["dataset"] = DS008257_DATASET_ID
        h5.attrs["subject_id"] = subject
        h5.attrs["sampling_rate"] = float(branch.sampling_rate)
        h5.attrs["n_channels"] = int(n_channels)
        h5.attrs["n_epochs"] = int(n_epochs)

        h5.attrs["encoder_id"] = branch.encoder
        h5.attrs["encoder_window_specs_json"] = canonical_json(
            DS008257_ENCODER_WINDOWS)
        h5.attrs["window_samples"] = int(branch.window["window_samples"])
        h5.attrs["window_stride_samples"] = int(branch.window["stride_samples"])
        h5.attrs["n_windows"] = int(branch.window["n_windows"])
        h5.attrs["model_input_normalization"] = branch.model_input_normalization
        h5.attrs["preprocessing_branch_json"] = canonical_json({
            "encoder": branch.encoder,
            "profile_id": branch.profile_id,
            "source_rate_hz": DS008257_NATIVE_RATE,
            "resampled_to_hz": branch.sampling_rate,
            "window_start_sec": DS008257_WINDOW_START_SEC,
            "window_stop_sec": DS008257_WINDOW_STOP_SEC,
            "clip_seconds": DS008257_CLIP_SECONDS,
            "n_frames": DS008257_N_FRAMES,
            "frame_height": DS008257_FRAME_HEIGHT,
            "frame_width": DS008257_FRAME_WIDTH,
            "resize_interpolation": DS008257_RESIZE_INTERPOLATION,
            "source_reference": DS008257_SOURCE_REFERENCE,
            "source_band_hz": list(DS008257_SOURCE_BAND_HZ),
            "band_filter_reapplied": False,
            "rereferenced_by_benchmark": False,
            "test_id_min": DS008257_TEST_ID_MIN,
            "test_trials_per_video": 1,
            "model_input_normalization": branch.model_input_normalization,
            "encoder_window_specs": DS008257_ENCODER_WINDOWS,
        })

        write_common_h5_metadata(
            h5,
            dataset_id=DS008257_DATASET_ID,
            task_id=DS008257_TASK_ID,
            subject_id=subject,
            sampling_rate=branch.sampling_rate,
            source_sampling_rate=DS008257_NATIVE_RATE,
            n_channels=n_channels,
            n_epochs=n_epochs,
            split=split,
            epoch_type="video_clip",
            eeg_units=branch.eeg_units,
            profile=profile,
            preprocessing_state={
                # Every step is stateless: the publisher's filter is declared
                # and not re-run, the window is a slice, the resampling is
                # polyphase, and nothing is fitted on the source.
                "fitted_split": "none",
                "encoder": branch.encoder,
                "source_band_hz": list(DS008257_SOURCE_BAND_HZ),
                "band_filter_reapplied": False,
                "source_reference": DS008257_SOURCE_REFERENCE,
                "rereferenced_by_benchmark": False,
                "window_sec": [DS008257_WINDOW_START_SEC,
                               DS008257_WINDOW_STOP_SEC],
                "resampled_to_hz": branch.sampling_rate,
                "model_input_normalization": branch.model_input_normalization,
                "seed": SEED,
            },
        )
        write_stimulus_manifest(h5, records)

    schema = get_task_config(DS008257_TASK_ID)["h5_schema"]
    require_valid_h5_file(str(out_path), schema)
    return n_epochs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def ds008257_process_subject(
    root: Path, stimuli: Dict[str, Any], captions: Dict[str, str],
    out_dir: Path, subject: str, branches: Sequence[Ds008257Branch],
) -> Dict[str, Dict[str, int]]:
    import h5py  # noqa: F401  (kept next to its use for readability)

    seed_everything(SEED)
    loaded = read_ds008257_subject(root, subject)
    eeg, ids = loaded["eeg"], loaded["stimulus_id"]
    session_ids = loaded["session"]
    videos = stimuli["videos"]
    test_id_min = int(stimuli["test_id_min"])

    unknown = sorted({int(s) for s in ids.tolist()
                      if f"{int(s):04d}" not in videos})
    if unknown:
        raise ValueError(f"{subject}: stimulus ids {unknown[:5]} are not in "
                         f"{DS008257_STIMULI_PATH.name}")

    train_rows, test_rows = ds008257_split(ids, test_id_min)
    LOG.info("%s: %d trials, %d train / %d test (%d test classes)",
             subject, len(ids), len(train_rows), len(test_rows),
             len({int(ids[r]) for r in test_rows}))

    written: Dict[str, Dict[str, int]] = {b.encoder: {} for b in branches}
    for split, rows in (("train", train_rows), ("test", test_rows)):
        windowed = eeg[rows]
        stimulus_ids = [int(ids[r]) for r in rows]
        sessions = [int(session_ids[r]) for r in rows]
        rows_captions = []
        for stimulus in stimulus_ids:
            video_id = f"{stimulus:04d}"
            if video_id not in captions:
                raise ValueError(
                    f"no caption for {video_id} in {DS008257_CAPTIONS_PATH.name}")
            rows_captions.append(captions[video_id])

        records, gt_parts = [], []
        if split == "test":
            # One row per test video, so decode each stimulus once.
            for row, stimulus in zip(rows, stimulus_ids):
                video_id = f"{stimulus:04d}"
                video = videos[video_id]
                mp4 = Path(video["mp4"])
                indices = ds008257_frame_indices(video)
                frames, geometry = ds008257_decode_keyframes(mp4, indices)
                expected = ds008257_letterbox_geometry(video)
                if geometry != expected:
                    raise ValueError(
                        f"{video_id}: measured letterbox {geometry} does not "
                        f"match the recorded {expected}")
                gt_parts.append(frames)
                records.append(_ds008257_record(
                    video, mp4, indices, subject=subject, row=row))
        else:
            for row, stimulus in zip(rows, stimulus_ids):
                video_id = f"{stimulus:04d}"
                video = videos[video_id]
                records.append(_ds008257_record(
                    video, Path(video["mp4"]), ds008257_frame_indices(video),
                    subject=subject, row=row))

        gt_frames = ds008257_stack_gt(gt_parts, len(rows)) if gt_parts else None
        for branch in branches:
            branch_eeg = branch.apply(windowed, loaded["channel_names"])
            # The branch may have kept a subset (LaBraM), so the file must
            # declare the montage it actually stored, not the subject's.
            branch_channels = (list(branch.channels)
                               if branch.channels is not None
                               else loaded["channel_names"])
            branch_dir = out_dir / branch.encoder
            branch_dir.mkdir(parents=True, exist_ok=True)
            out_path = (branch_dir
                        / f"{DS008257_DATASET_ID}_{subject}_{split}_epoched.h5")
            written[branch.encoder][split] = write_ds008257_split_h5(
                out_path=out_path, subject=subject, split=split, eeg=branch_eeg,
                channel_names=branch_channels,
                stimulus_ids=stimulus_ids, session_ids=sessions,
                captions=rows_captions, records=records, gt_frames=gt_frames,
                branch=branch,
            )
            LOG.info("wrote %s (%d epochs, %d samples)", out_path,
                     written[branch.encoder][split], branch_eeg.shape[-1])
    return written


def ds008257_letterbox_geometry(video: Dict[str, Any]) -> Dict[str, int]:
    """Where this stimulus lands in the task's frame geometry.

    Read from the stimulus table's measured width and height rather than from
    a constant: the stimulus set happens to be uniformly 268x268 today, and a
    hard-coded 268 would silently mis-describe the padding of any file that
    is not.
    """

    return letterbox_geometry(int(video["width"]), int(video["height"]),
                              DS008257_FRAME_HEIGHT, DS008257_FRAME_WIDTH)


def _ds008257_record(video: Dict[str, Any], mp4: Path, indices: List[int],
                     *, subject: str, row: int) -> StimulusRecord:
    """One trial's stimulus manifest row."""

    return StimulusRecord(
        stimulus_id=make_stimulus_id(
            DS008257_DATASET_ID, f"video{video['video_id']}"),
        modality="video",
        uri=str(mp4),
        sha256=str(video["sha256"]),
        onset_sec=DS008257_WINDOW_START_SEC,
        duration_sec=DS008257_CLIP_SECONDS,
        alignment_source="released_stimulus_id",
        transform={
            "fps": float(video["n_frames"]) / DS008257_STIMULUS_SECONDS,
            "n_frames": DS008257_N_FRAMES,
            "height": DS008257_FRAME_HEIGHT,
            "width": DS008257_FRAME_WIDTH,
            "stimulus_seconds": DS008257_STIMULUS_SECONDS,
            "stimulus_n_frames": int(video["n_frames"]),
            "frame_rule": "round((window_start + k * clip / n_frames) * "
                          "stimulus_n_frames / stimulus_seconds)",
            "frame_indices": indices,
            "resize_interpolation": DS008257_RESIZE_INTERPOLATION,
            "source_width": int(video["width"]),
            "source_height": int(video["height"]),
            "letterbox_geometry": ds008257_letterbox_geometry(video),
        },
    )


def ds008257_run_loader(
    data_root: str, out_dir: str,
    subjects: Optional[Sequence[str]] = None,
    encoders: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, int]]:
    root = Path(data_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stimuli = load_ds008257_stimuli()
    captions = load_ds008257_captions()
    names = list(subjects) if subjects else ds008257_discover_subjects(root)
    branches = [ds008257_build_branch(name) for name in (encoders or ["eegnet"])]
    totals: Dict[str, Dict[str, int]] = {b.encoder: {} for b in branches}
    for subject in names:
        written = ds008257_process_subject(
            root, stimuli, captions, out, subject, branches)
        for encoder_name, splits in written.items():
            for split, count in splits.items():
                totals[encoder_name][split] = (
                    totals[encoder_name].get(split, 0) + count)
    return totals


LOG = logging.getLogger("eeg2x.preprocessing.ds008257")


