"""SEED-DV 视频预处理：一条被试的 7 个 block -> 每个编码器分支的 HDF5。"""
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
from .video_common import _resize_rgb

DATASET_ID = "seed_dv"
TASK_ID = "video_generation"
#: The EEGNet profile id, kept as a module constant for callers that predate
#: the per-encoder branch.  Other branches are resolved through
#: ``SEED_DV_ENCODER_PROFILES``.
PROFILE_ID = "seed-dv-video-v1"

SAMPLING_RATE = 200.0
N_CHANNELS = 62
N_BLOCKS = 7
CONCEPTS = 40
REPEATS = 5
SAMPLES_PER_BLOCK = CONCEPTS * REPEATS          # 200
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(SAMPLING_RATE * CLIP_SECONDS)  # 400
SLOT_SECONDS = 13
SLOT_OFFSET_SECONDS = 3
REPEAT_SECONDS = 2
TRAIN_BLOCKS = 6
TEST_BLOCK = 7

VIDEO_FPS = 3
N_FRAMES = 6
FRAME_HEIGHT = 288
FRAME_WIDTH = 512
#: The released extraction indexes the nominal 24 FPS source timeline.
SOURCE_FPS = 24.0
SOURCE_FRAMES_PER_OUTPUT_FRAME = 8

BLOCK_NAMES = ("1st_10min", "2nd_10min", "3rd_10min", "4th_10min",
               "5th_10min", "6th_10min", "7th_10min")

LOG = logging.getLogger("eeg2x.preprocessing.seed_dv")


def load_channel_names(dataset_root: Path) -> List[str]:
    """Read the 62 ESI channel names, in EEG row order."""

    for candidate in ("channel_62_pos.locs", "../dataset/channel_62_pos.locs"):
        path = (dataset_root / candidate).resolve()
        if path.is_file():
            names = []
            for line in path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    names.append(parts[-1])
            if len(names) != N_CHANNELS:
                raise ValueError(
                    f"{path} lists {len(names)} channels, expected {N_CHANNELS}"
                )
            return normalize_channel_names(names)
    raise FileNotFoundError(
        f"channel_62_pos.locs not found under {dataset_root}"
    )


def read_block_labels(dataset_root: Path) -> np.ndarray:
    labels = np.load(dataset_root / "Video" / "meta-info" / "All_video_label.npy")
    if labels.shape != (N_BLOCKS, CONCEPTS):
        raise ValueError(f"Unexpected label shape {labels.shape}")
    return labels


def read_block_captions(dataset_root: Path, block_index: int) -> List[str]:
    path = dataset_root / "Video" / "BLIP-caption" / f"{BLOCK_NAMES[block_index]}.txt"
    captions = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    if len(captions) != SAMPLES_PER_BLOCK:
        raise ValueError(f"{path} holds {len(captions)} captions, expected 200")
    return captions


def clip_start_seconds(class_position: int, repeat: int) -> float:
    return float(class_position * SLOT_SECONDS + SLOT_OFFSET_SECONDS
                 + repeat * REPEAT_SECONDS)


def clip_frame_indices(start_seconds: float) -> List[int]:
    """Frame indices on the nominal 24 FPS source timeline."""

    return [
        int(start_seconds * SOURCE_FPS + SOURCE_FRAMES_PER_OUTPUT_FRAME * i)
        for i in range(N_FRAMES)
    ]


# ---------------------------------------------------------------------------
# EEG
# ---------------------------------------------------------------------------

def read_subject_eeg(dataset_root: Path, subject: int) -> np.ndarray:
    path = dataset_root / "EEG" / f"sub{subject}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"missing EEG for subject {subject}: {path}")
    array = np.load(path, mmap_mode="r")
    expected = (N_BLOCKS, N_CHANNELS, 104000)
    if array.shape != expected:
        raise ValueError(f"{path} has shape {array.shape}, expected {expected}")
    return array


def extract_clip(block: np.ndarray, start_seconds: float) -> np.ndarray:
    """One 2-second clip, ``(n_channels, clip_samples)`` float32."""

    start = seconds_to_sample(start_seconds, SAMPLING_RATE)
    stop = start + CLIP_SAMPLES
    if stop > block.shape[-1]:
        raise ValueError(
            f"clip at {start_seconds}s reaches sample {stop} of {block.shape[-1]}"
        )
    clip = np.asarray(block[:, start:stop], dtype=np.float32)
    if not np.isfinite(clip).all():
        raise ValueError(f"non-finite EEG in clip at {start_seconds}s")
    return clip


# ---------------------------------------------------------------------------
# stimulus video
# ---------------------------------------------------------------------------


def extract_gt_frames(video_path: Path, frame_indices: List[int]) -> np.ndarray:
    """The released OpenCV extraction: read frames, BGR->RGB, resize."""

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stimulus video {video_path}")
    wanted = {index: slot for slot, index in enumerate(frame_indices)}
    frames = np.zeros((N_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    found = 0
    cursor = 0
    last = max(frame_indices)
    while cursor <= last:
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"{video_path} ended before frame {last}")
        if cursor in wanted:
            frames[wanted[cursor]] = _resize_rgb(frame)
            found += 1
        cursor += 1
    capture.release()
    if found != N_FRAMES:
        raise RuntimeError(f"read {found}/{N_FRAMES} frames from {video_path}")
    return frames


def extract_gt_frame_groups(
    video_path: Path, index_groups: List[List[int]],
) -> np.ndarray:
    """All ground-truth clips of one block, in one sequential pass.

    Semantically identical to calling :func:`extract_gt_frames` once per group
    -- same frames, same resize, same row order -- but it decodes the source
    video once instead of once per clip.  The per-clip form scans from frame 0
    every time, which is O(n_clips x n_frames) and dominates the loader on the
    13-minute SEED-DV blocks; this is O(n_frames) and produces the same bytes.
    """

    import cv2

    if not index_groups:
        return np.zeros((0, N_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    wanted: Dict[int, List[Tuple[int, int]]] = {}
    for group, indices in enumerate(index_groups):
        if len(indices) != N_FRAMES:
            raise ValueError(f"clip {group} has {len(indices)} frame indices")
        for slot, index in enumerate(indices):
            wanted.setdefault(index, []).append((group, slot))

    out = np.zeros(
        (len(index_groups), N_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stimulus video {video_path}")
    last = max(wanted)
    found = 0
    cursor = 0
    while cursor <= last and found < len(wanted):
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"{video_path} ended before frame {last}")
        slots = wanted.get(cursor)
        if slots is not None:
            resized = _resize_rgb(frame)
            for group, slot in slots:
                out[group, slot] = resized
            found += 1
        cursor += 1
    capture.release()
    if found != len(wanted):
        raise RuntimeError(
            f"read {found}/{len(wanted)} distinct frames from {video_path}"
        )
    return out


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def _stimulus_records(
    *, dataset_root: Path, block_index: int, block_rows: List[Dict[str, Any]],
    video_sha: str, uri: str,
) -> List[StimulusRecord]:
    records = []
    for row in block_rows:
        start = clip_start_seconds(row["class_position"], row["repeat"])
        records.append(
            StimulusRecord(
                stimulus_id=make_stimulus_id(
                    DATASET_ID,
                    f"{BLOCK_NAMES[block_index]}-c{row['class_position']:02d}"
                    f"-r{row['repeat']}",
                ),
                modality="video",
                uri=uri,
                sha256=video_sha,
                onset_sec=start,
                duration_sec=CLIP_SECONDS,
                alignment_source="block_concept_repeat_manifest",
                transform={
                    "fps": VIDEO_FPS,
                    "n_frames": N_FRAMES,
                    "height": FRAME_HEIGHT,
                    "width": FRAME_WIDTH,
                    "source_fps": SOURCE_FPS,
                    "frame_indices": clip_frame_indices(start),
                },
            )
        )
    return records


def write_split_h5(
    *, out_path: Path, subject: int, split: str, eeg: np.ndarray,
    channel_names: List[str], concept_ids: np.ndarray, block_ids: np.ndarray,
    captions: List[str], records: List[StimulusRecord],
    gt_frames: np.ndarray | None, source_sampling_rate: float,
    branch: "EncoderBranch", sampling_rate: float,
) -> int:
    import h5py

    profile = get_preprocessing_profile(branch.profile_id)
    n_epochs, n_channels, n_samples = eeg.shape
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("data", data=eeg, dtype=np.float32)
        h5.create_dataset(
            "channel_names",
            data=np.asarray(channel_names, dtype=object), dtype=string_dtype,
        )
        h5.create_dataset("event_concept", data=np.asarray(concept_ids, np.int32))
        h5.create_dataset("event_block", data=np.asarray(block_ids, np.int32))
        h5.create_dataset(
            "event_caption",
            data=np.asarray(captions, dtype=object), dtype=string_dtype,
        )
        if gt_frames is not None:
            h5.create_dataset("gt_frames", data=gt_frames, dtype=np.uint8)

        # legacy attrs kept for models that predate the v2 protocol
        h5.attrs["dataset"] = DATASET_ID
        h5.attrs["subject_id"] = f"sub-{subject:02d}"
        h5.attrs["sampling_rate"] = float(sampling_rate)
        h5.attrs["n_channels"] = int(n_channels)
        h5.attrs["n_epochs"] = int(n_epochs)

        # ---- the encoder branch, stated in the file itself ----------------
        # The model reads its window geometry, its montage and whether the
        # stored array is already model-ready from these attrs, so the loader
        # and the model cannot disagree about what the branch produced.
        h5.attrs["encoder_id"] = branch.encoder
        h5.attrs["encoder_window_specs_json"] = canonical_json(
            SEED_DV_ENCODER_WINDOWS)
        h5.attrs["window_samples"] = int(branch.window["window_samples"])
        h5.attrs["window_stride_samples"] = int(branch.window["stride_samples"])
        h5.attrs["n_windows"] = int(branch.window["n_windows"])
        h5.attrs["model_input_normalization"] = branch.model_input_normalization
        h5.attrs["preprocessing_branch_json"] = canonical_json({
            "encoder": branch.encoder,
            "profile_id": branch.profile_id,
            "resampled": branch.resample_hz is not None,
            "scale_factor": branch.scale_factor,
            "model_input_normalization": branch.model_input_normalization,
            "encoder_window_specs": SEED_DV_ENCODER_WINDOWS,
        })

        write_common_h5_metadata(
            h5,
            dataset_id=DATASET_ID,
            task_id=TASK_ID,
            subject_id=f"sub-{subject:02d}",
            sampling_rate=sampling_rate,
            source_sampling_rate=source_sampling_rate,
            n_channels=n_channels,
            n_epochs=n_epochs,
            split=split,
            epoch_type="video_clip",
            eeg_units=branch.eeg_units,
            profile=profile,
            preprocessing_state={
                "fitted_split": "none",
                "encoder": branch.encoder,
                "scale_factor": branch.scale_factor,
                "resampled_to_hz": branch.resample_hz,
                "model_input_normalization": branch.model_input_normalization,
                "note": (
                    "profile has no fitted step; standardisation is model-side"
                    if branch.model_input_normalization != "none"
                    else "branch already produced a model-ready representation; "
                         "model standardisation is disabled"
                ),
                "seed": SEED,
            },
        )
        write_stimulus_manifest(h5, records)

    schema = get_task_config(TASK_ID)["h5_schema"]
    require_valid_h5_file(str(out_path), schema)
    return n_epochs


# ---------------------------------------------------------------------------
# encoder preprocessing branch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncoderBranch:
    """What one encoder's preprocessing branch does to the stored EEG.

    SEED-DV is the same recordings for every encoder; only three things can
    differ, and each is set explicitly here rather than inferred:

    ``scale_factor``  amplitude rescaling (LaBraM's volts -> microvolts)
    ``resample_hz``   a different clock (LUNA trains at 256 Hz, SEED-DV is 200)
    ``window``        the geometry the model will slice, from its own profile

    ``model_input_normalization`` says whether the array that comes out is
    already model-ready -- if it is, the model must not standardise it again.
    """

    encoder: str
    profile_id: str
    window: Dict[str, int]
    sampling_rate: float
    eeg_units: str
    model_input_normalization: str
    scale_factor: float | None = None
    resample_hz: float | None = None

    def apply(self, clips: np.ndarray) -> np.ndarray:
        """Apply the branch's data-level steps to one split's clips."""

        out = np.asarray(clips, dtype=np.float32)
        if self.scale_factor is not None:
            out = out * np.float32(self.scale_factor)
        if self.resample_hz is not None:
            from scipy.signal import resample_poly

            source = SAMPLING_RATE
            target = float(self.resample_hz)
            up, down = _ratio(target, source)
            out = resample_poly(out, up, down, axis=-1).astype(np.float32)
        if not np.isfinite(out).all():
            raise ValueError(
                f"branch {self.profile_id} produced non-finite EEG values"
            )
        return out


def _ratio(target_hz: float, source_hz: float) -> Tuple[int, int]:
    """Integer up/down sampling ratio for ``resample_poly`` (e.g. 256/200 -> 32/25)."""

    from math import gcd

    target = Fraction(target_hz).limit_denominator(10000)
    source = Fraction(source_hz).limit_denominator(10000)
    num = target.numerator * source.denominator
    den = target.denominator * source.numerator
    divisor = gcd(num, den)
    return num // divisor, den // divisor


def build_branch(encoder: str) -> EncoderBranch:
    """Assemble one encoder's branch from the profile it declares."""

    if encoder not in SEED_DV_ENCODER_PROFILES:
        raise KeyError(
            f"unknown encoder {encoder!r}; expected one of "
            f"{sorted(SEED_DV_ENCODER_PROFILES)}"
        )
    profile = get_preprocessing_profile(SEED_DV_ENCODER_PROFILES[encoder])
    scale_factor = None
    resample_hz = None
    for step in profile.steps:
        if step.name == "scale":
            scale_factor = float(step.parameters["factor"])
        elif step.name == "resample":
            resample_hz = float(step.parameters["target_hz"])
    return EncoderBranch(
        encoder=encoder,
        profile_id=profile.profile_id,
        window=seed_dv_window_spec(encoder),
        sampling_rate=SEED_DV_ENCODER_RATE[encoder],
        eeg_units=SEED_DV_ENCODER_UNITS[encoder],
        model_input_normalization=SEED_DV_ENCODER_MODEL_NORM[encoder],
        scale_factor=scale_factor,
        resample_hz=resample_hz,
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def process_subject(dataset_root: Path, out_dir: Path, subject: int,
                    branch: Optional["EncoderBranch"] = None,
                    branches: Optional[List["EncoderBranch"]] = None,
                    ) -> Dict[str, Dict[str, int]]:
    """Materialise one subject's HDF5s for one or more encoder branches.

    The clips and the ground-truth frames are identical for every branch --
    only the branch's data-level steps and the declared metadata differ -- so
    the stimulus video is decoded once and every branch is written from that
    one pass.  Returns ``{encoder: {split: n_epochs}}``.
    """

    seed_everything(SEED)
    if branches is None:
        branches = [branch] if branch is not None else [build_branch("eegnet")]
    channel_names = load_channel_names(dataset_root)
    labels = read_block_labels(dataset_root)
    source = read_subject_eeg(dataset_root, subject)
    # SEED-DV is natively 200 Hz; a branch may move the clock (LUNA does), and
    # then the stored array really is at a different rate.
    source_sampling_rate = SAMPLING_RATE
    written: Dict[str, Dict[str, int]] = {b.encoder: {} for b in branches}

    for split, blocks in (("train", range(TRAIN_BLOCKS)), ("test", [TEST_BLOCK - 1])):
        eeg_parts, concepts, block_ids, captions, records, gt_parts = (
            [], [], [], [], [], []
        )
        for block_index in blocks:
            block_name = BLOCK_NAMES[block_index]
            video_path = dataset_root / "Video" / f"{block_name}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(f"missing stimulus video {video_path}")
            uri = str(video_path)
            video_sha = sha256_file(video_path)
            block_captions = read_block_captions(dataset_root, block_index)
            block = source[block_index]
            block_rows = []
            block_starts = []
            for class_position in range(CONCEPTS):
                for repeat in range(REPEATS):
                    within = class_position * REPEATS + repeat
                    start = clip_start_seconds(class_position, repeat)
                    eeg_parts.append(extract_clip(block, start))
                    concepts.append(int(labels[block_index, class_position]))
                    block_ids.append(block_index + 1)
                    captions.append(block_captions[within])
                    block_rows.append(
                        {"class_position": class_position, "repeat": repeat}
                    )
                    block_starts.append(start)
            # One sequential decode of the stimulus video covers every clip of
            # the block; the released per-clip loop re-scanned from frame 0
            # each time, which was ~20 minutes per subject.
            if split == "test":
                gt_parts.append(
                    extract_gt_frame_groups(
                        video_path,
                        [clip_frame_indices(start) for start in block_starts],
                    )
                )
            records.extend(
                _stimulus_records(
                    dataset_root=dataset_root, block_index=block_index,
                    block_rows=block_rows, video_sha=video_sha, uri=uri,
                )
            )
            LOG.info("subject %02d %s block %d done", subject, split, block_index + 1)

        eeg = np.stack(eeg_parts).astype(np.float32)
        # Each entry already holds a whole block's clips, so concatenate rather
        # than stack: the result must be (n_epochs, N_FRAMES, H, W, 3) for the
        # eval-side ``gt_frames[<row>]`` lookup.
        gt_frames = np.concatenate(gt_parts, axis=0) if gt_parts else None
        for encoder_branch in branches:
            # The branch's data-level steps (amplitude scaling, resampling)
            # happen here, before anything is written, so the HDF5 holds
            # exactly what the encoder will consume.
            branch_eeg = encoder_branch.apply(eeg)
            branch_dir = out_dir / encoder_branch.encoder
            branch_dir.mkdir(parents=True, exist_ok=True)
            out_path = (
                branch_dir / f"{DATASET_ID}_sub-{subject:02d}_{split}_epoched.h5"
            )
            written[encoder_branch.encoder][split] = write_split_h5(
                out_path=out_path, subject=subject, split=split, eeg=branch_eeg,
                channel_names=channel_names,
                concept_ids=np.asarray(concepts, np.int32),
                block_ids=np.asarray(block_ids, np.int32), captions=captions,
                records=records, gt_frames=gt_frames,
                source_sampling_rate=source_sampling_rate,
                branch=encoder_branch, sampling_rate=encoder_branch.sampling_rate,
            )
            LOG.info("wrote %s (%d epochs, %d samples)", out_path,
                     written[encoder_branch.encoder][split], branch_eeg.shape[-1])
    return written


def run_loader(data_root: str, out_dir: str, subjects: List[int] | None = None,
               allow_partial: bool = False,
               encoders: List[str] | None = None) -> Dict[str, Dict[str, int]]:
    """Write one HDF5 set per requested encoder branch.

    Each branch gets its own ``<out_dir>/<encoder>/`` directory, so a
    ``seed_dv_*_epoched.h5`` glob inside one directory selects exactly one
    encoder's track.  All branches of a subject share a single decode of the
    stimulus video.
    """

    dataset_root = Path(data_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    subjects = subjects or list(range(1, 6))
    encoders = encoders or ["eegnet"]
    branches = [build_branch(name) for name in encoders]
    totals: Dict[str, Dict[str, int]] = {b.encoder: {} for b in branches}
    failures = []
    for subject in subjects:
        try:
            written = process_subject(dataset_root, out, subject, branches=branches)
            for encoder_name, splits in written.items():
                for split, count in splits.items():
                    totals[encoder_name][split] = (
                        totals[encoder_name].get(split, 0) + count
                    )
        except Exception as exc:  # noqa: BLE001 - reported, then re-raised
            LOG.error("subject %02d failed: %s", subject, exc)
            failures.append((subject, str(exc)))
            if not allow_partial:
                raise
    if failures and not allow_partial:
        raise RuntimeError(f"{len(failures)} subject(s) failed: {failures}")
    return totals



