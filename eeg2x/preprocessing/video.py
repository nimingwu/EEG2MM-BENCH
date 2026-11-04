#!/usr/bin/env python
"""Video-task dataset loader: source data -> v2 HDF5.

Three datasets feed the ``video_generation`` task, one section each below:

==================  ==========================  ==============================
dataset             source                      shape of a trial
==================  ==========================  ==============================
``seed_dv``         SEED-DV ``.npy`` blocks     2 s clip, 62 ch @ 200 Hz
``ds005662``        OpenNeuro MNE derivatives   600 ms epoch, 64 ch @ 200 Hz
``eeg_video_action``  EEGLAB ``.set``            2 s slice of a recording,
                    recordings                  64 ch @ 2048 -> 200 Hz
==================  ==========================  ==============================

They differ in montage, clock and trial geometry, so each has its own
preprocessing profiles and its own encoder-branch builder; what they share is
the recording convention -- one HDF5 set per (subject, split) per encoder
branch, a stimulus manifest aligned row-for-row with the EEG, and a strict
failure when either is missing.  ``ADDING_A_DATASET.md`` is the checklist; the
``--dataset`` flag selects the section.

There is one writer function per dataset rather than a shared one, because the
legacy fields they emit differ (SEED-DV and ds005662 carry concept ids from
different id spaces) and the fields the evaluator reads are decided by the
registry, not by the writer.

SEED-DV section
---------------
Source dataset
--------------
``/datas/eeg_benchmark_project/SEED-DV``

* ``EEG/sub<N>.npy``      ``(7, 62, 104000)`` float64, one array per subject.
  Block ``b`` holds 40 ``class_position`` slots of 13 s each; inside a slot the
  first 3 s are skipped and five consecutive 2 s clips follow.
* ``Video/<block>_10min.mp4``  the stimulus videos.  Clip ``(class_position,
  repeat)`` starts at ``class_position * 13 + 3 + repeat * 2`` seconds and is
  sampled at 3 FPS into six frames at 288x512 (released OpenCV path).
* ``Video/meta-info/All_video_label.npy``  ``(7, 40)`` concept id per slot.
* ``Video/BLIP-caption/<block>.txt``       200 captions per block, one per clip,
  in the same row order as the clips.

Declared facts (ADDING_A_DATASET step 1)
----------------------------------------
* EEG clock        : 200 Hz, already the model's clock, so no resampling step
* EEG units        : dataset-native scale (~1e-4); the files do not prove a
                     physical unit, so ``eeg_units=dataset_native`` per
                     DATA_PROTOCOL rule 5
* channel type     : 62 EEG channels, ESI montage, read from
                     ``dataset/channel_62_pos.locs``
* event clock      : video timeline in seconds inside each 10-minute block
* stimulus format  : MP4 at 3 FPS, six frames per clip, 288x512

Splits
------
Blocks 1-6 are ``train`` (1200 epochs), block 7 is ``test`` (200 epochs).
This mirrors the released EEG2Video protocol exactly.

Output
------
``seed_dv_sub-01_train_epoched.h5`` and ``seed_dv_sub-01_test_epoched.h5``.
The test file additionally carries ``gt_frames`` so the evaluator does not have
to re-decode the source video.

Strict failure
--------------
Following DATA_PROTOCOL, a missing subject, a missing stimulus or a schema
violation makes the process exit non-zero.  ``--allow_partial`` exists only for
diagnosis and never produces a releasable artefact.
"""

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

LOG = logging.getLogger("eeg2video_loader")


# ---------------------------------------------------------------------------
# source facts
# ---------------------------------------------------------------------------

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

def _resize_rgb(frame: np.ndarray) -> np.ndarray:
    """BGR source frame -> RGB, resized to the released output geometry."""

    import cv2

    return cv2.resize(
        frame[..., ::-1], (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )


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



# ===========================================================================
# ds005662  (OpenNeuro ds005662 v1.0.3; Smit et al., Sci Data 2026)
# ===========================================================================
#
# Second dataset for the same ``video_generation`` task.  Added here rather
# than in a separate module because the benchmark keeps one loader per task and
# routes datasets through it; see ADDING_A_DATASET.md.

#: Where the BLIP captions for the 384 stimulus videos live (produced by
#: ``scripts/caption_ds005662_stimuli.py``).
CAPTIONS_PATH = Path(__file__).resolve().parent / "ds005662_stimulus_captions.json"
DEFAULT_DATA_ROOT = Path("/datas/eeg_benchmark_project/ds005662-download")

DS005662_DATASET_ID = "ds005662"
#: Same task as SEED-DV -- the HDF5 schema, the evaluator and the five metrics
#: are shared; only the dataset id and the numbers inside differ.
DS005662_TASK_ID = "video_generation"

#: 90 distinct touch videos.  The full stimulus set is 8 blocks x 90 x 4 flip
#: variants = 2880 trials, exactly one trial per combination (verified).  Only
#: the unflipped ``base`` variant is kept: the flips are the same video
#: mirrored and a BLIP caption cannot express orientation, so flipped test
#: clips would be unanswerable in principle.
DS005662_CLASSES = 90
DS005662_VARIANTS_KEPT = 1
DS005662_TRAIN_BLOCKS = tuple(range(7))
DS005662_TEST_BLOCK = 7

# ---------------------------------------------------------------------------
# source facts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Trial:
    eventnumber: int
    class_id: int
    block: int
    variant: str
    stimulus_name: str
    stimulus_path: Path
    caption: str
    caption_alt: str


def _variant_of(stim: str) -> str:
    text = str(stim).replace("\\", "/").lower()
    for name in ("horvertflip", "horflip", "vertflip"):
        if name in text:
            return name
    return "base"


def _clean_caption_alt(stimulus_name: str) -> str:
    """The filename restated as a sentence.

    Recorded as ``event_caption_alt``, never as the training target.  The
    released BLIP captions describe the *scene* ("a person is holding a black
    object"), which this dataset deliberately holds constant; the filename
    describes the *action*, which varies.  Keeping both lets the report state
    how degenerate the BLIP target is without changing the protocol.
    """

    import re

    stem = re.sub(r"^\d+\.", "", Path(stimulus_name).stem)
    stem = re.sub(r"_short(?:_(?:hor|vert|horvert)flip)?_short_600ms$", "", stem)
    return "a video of " + stem.replace("_", " ")


def load_captions(path: Path = CAPTIONS_PATH) -> Dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/caption_ds005662_stimuli.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def stimulus_path(data_root: Path, stim: str, stimulus_name: str) -> Path:
    variant = _variant_of(stim)
    directory = "videos_short_600ms" + ("" if variant == "base" else f"_{variant}")
    path = (data_root / "code" / "experiment_files" / "stimuli" / directory
            / stimulus_name)
    if not path.is_file():
        raise FileNotFoundError(f"stimulus video missing: {path}")
    return path


def load_trials(data_root: Path, subject: str, captions: Dict[str, str],
                keep_variant: str = "base") -> List[Trial]:
    """The subject's (block, class, variant) trials, base variant only."""

    import csv

    tsv = data_root / subject / "eeg" / f"{subject}_task-video_events.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(f"missing events file {tsv}")
    rows = list(csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"))
    trials: List[Trial] = []
    for row in rows:
        if str(row["istarget"]) != "0":
            continue
        variant = _variant_of(row["stim"])
        if variant != keep_variant:
            continue
        name = str(row["stimname"])
        prefix = name.split(".", 1)[0]
        if not prefix.isdigit():
            raise ValueError(f"cannot parse stimulus id from {name!r}")
        class_id = int(prefix)
        if not 1 <= class_id <= DS005662_CLASSES:
            raise ValueError(f"class id {class_id} outside 1..{DS005662_CLASSES}")
        block = int(row["blocksequencenumber"])
        trials.append(Trial(
            eventnumber=int(row["eventnumber"]),
            class_id=class_id,
            block=block,
            variant=variant,
            stimulus_name=name,
            stimulus_path=stimulus_path(data_root, row["stim"], name),
            caption=captions.get(name, ""),
            caption_alt=_clean_caption_alt(name),
        ))
    if not trials:
        raise ValueError(f"no {keep_variant} trials for {subject}")
    seen = {(t.block, t.class_id) for t in trials}
    expected = (DS005662_TRAIN_BLOCKS[-1] + 2) * DS005662_CLASSES
    if len(seen) != expected or len(trials) != expected:
        raise ValueError(
            f"{subject}: expected {expected} unique (block, class) pairs, "
            f"found {len(trials)} trials / {len(seen)} pairs"
        )
    if not all(t.caption for t in trials):
        missing = [t.stimulus_name for t in trials if not t.caption][:3]
        raise ValueError(f"no BLIP caption for {missing}")
    trials.sort(key=lambda t: (t.block, t.class_id))
    return trials


def read_epochs(data_root: Path, subject: str) -> np.ndarray:
    """``(n_trials, 64, 180)`` in microvolts, in the fif's own order."""

    import mne

    mne.set_log_level("ERROR")
    fif = data_root / "derivatives" / "mne" / f"{subject}_mne_epo.fif"
    if not fif.is_file():
        raise FileNotFoundError(f"missing epochs {fif}")
    epochs = mne.read_epochs(str(fif), preload=True, verbose=False)
    data = epochs.get_data()
    if data.shape[1:] != (DS005662_SOURCE_CHANNELS, DS005662_NATIVE_SAMPLES):
        raise ValueError(
            f"{fif} holds {data.shape[1:]}, expected "
            f"({DS005662_SOURCE_CHANNELS}, {DS005662_NATIVE_SAMPLES})"
        )
    if abs(float(epochs.info["sfreq"]) - DS005662_NATIVE_RATE) > 1e-6:
        raise ValueError(f"{fif} runs at {epochs.info['sfreq']} Hz")
    if abs(float(epochs.info["highpass"]) - DS005662_BAND_HZ[0]) > 1e-6:
        raise ValueError(
            f"{fif} declares highpass={epochs.info['highpass']}, expected "
            f"{DS005662_BAND_HZ[0]}"
        )
    code_to_index = {
        int(code): index for index, code in enumerate(epochs.events[:, 2])
    }
    return data, code_to_index, [str(c) for c in epochs.ch_names]


def bandpass(x: np.ndarray, rate: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply the declared 0.1-100 Hz band-pass and report what it did.

    The source files are already band-passed (MNE records ``highpass=0.1``,
    ``lowpass=100``), so this is a second application.  Two facts are recorded
    rather than glossed over:

    * the 100 Hz low-pass sits exactly at Nyquist for 200 Hz data, so it cannot
      be applied at all (MNE raises) and would be a no-op if it could;
    * the 0.1 Hz high-pass has a ~1.6 s time constant against a 0.9 s epoch, so
      MNE warns "filter_length (6601) is longer than the signal (180),
      distortion is likely".
    """

    import mne

    l_freq, h_freq = DS005662_BAND_HZ
    nyquist = rate / 2.0
    record: Dict[str, Any] = {
        "declared_band_hz": [l_freq, h_freq],
        "nyquist_hz": nyquist,
        "lowpass_applied": False,
        "lowpass_note": (
            f"{h_freq} Hz equals Nyquist at {rate:g} Hz; the source data is "
            "already band-limited to it, so no low-pass is executed"
        ),
    }
    filtered = mne.filter.filter_data(
        np.asarray(x, dtype=np.float64), sfreq=rate,
        l_freq=l_freq, h_freq=None, verbose=False,
    )
    before = float(np.sqrt(np.mean(np.square(x))))
    after = float(np.sqrt(np.mean(np.square(filtered))))
    record["rms_before"] = before
    record["rms_after"] = after
    record["rms_ratio"] = after / before if before else float("nan")
    record["highpass_warning"] = (
        "MNE reports filter_length 6601 > signal 180 samples; distortion likely"
    )
    return filtered.astype(np.float32), record


# ---------------------------------------------------------------------------
# stimulus video
# ---------------------------------------------------------------------------

def stimulus_frame_indices(path: Path, transform: Dict[str, Any]) -> List[int]:
    """The ground-truth frame indices of one stimulus row, whatever the rule.

    Two rules exist and which one applies is declared by the manifest, not
    guessed:

    * SEED-DV (and EEG-Video-Action) write an explicit ``frame_indices``: the
      clip's position inside a long stimulus video.
    * ds005662 does not.  Its stimuli are short 600 ms clips and the rule is
      "six frames spread evenly over the whole file" -- the same rule
      :func:`stimulus_frames` applies to build that dataset's ground truth.

    The rule lives here, next to the function that applies it, so a model that
    needs the keyframes cannot drift from the bytes the evaluator scores.
    ``model/eeg2video_mindcine/prepare.py`` learned this the hard way: its
    ``frames_for_row`` reads ``frame_indices`` when present and falls back to
    :func:`stimulus_frames` otherwise.  This is the index-level analogue, for
    consumers that need positions rather than pixels.
    """

    if "frame_indices" in transform:
        return [int(v) for v in transform["frame_indices"]]

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stimulus video {path}")
    total = 0
    while capture.read()[0]:
        total += 1
    capture.release()
    if total == 0:
        raise RuntimeError(f"no readable frames in {path}")
    # Counted by reading, exactly as ``stimulus_frames`` does, rather than
    # trusting CAP_PROP_FRAME_COUNT: the two must not disagree.
    return [int(v) for v in
            np.linspace(0, total - 1, DS005662_N_FRAMES).round().astype(int)]


def stimulus_frames(path: Path) -> np.ndarray:
    """Six evenly spaced RGB frames, resized to the released output geometry.

    The source is 256x144 (16:9) and the target 512x288 (16:9), so the resize
    is an exact 2x scale with no aspect distortion -- the same landscape
    geometry the SEED-DV branch stores.
    """

    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open stimulus video {path}")
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"no readable frames in {path}")
    indices = np.linspace(0, len(frames) - 1, DS005662_N_FRAMES).round().astype(int)
    out = np.zeros(
        (DS005662_N_FRAMES, DS005662_FRAME_HEIGHT, DS005662_FRAME_WIDTH, 3),
        dtype=np.uint8,
    )
    for slot, index in enumerate(indices):
        out[slot] = cv2.resize(
            frames[index], (DS005662_FRAME_WIDTH, DS005662_FRAME_HEIGHT),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return out


def _frame_cache(paths: Sequence[Path]) -> Dict[Path, np.ndarray]:
    return {path: stimulus_frames(path) for path in sorted(set(paths))}


# ---------------------------------------------------------------------------
# encoder branch
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ds005662Branch:
    encoder: str
    profile_id: str
    window: Dict[str, int]
    sampling_rate: float
    eeg_units: str
    model_input_normalization: str
    pad_to: int
    resample_hz: Optional[float]

    def apply(self, epochs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """band-pass -> optional resample -> right zero-pad.

        Returns the array **and** the band-pass's measurement, because this
        dataclass is frozen and the report has to reach the HDF5.
        """

        out, report = bandpass(epochs, DS005662_NATIVE_RATE)
        if self.resample_hz is not None and self.resample_hz != DS005662_NATIVE_RATE:
            from scipy.signal import resample_poly

            up, down = _ratio(self.resample_hz, DS005662_NATIVE_RATE)
            out = resample_poly(out, up, down, axis=-1).astype(np.float32)
        if out.shape[-1] > self.pad_to:
            raise ValueError(
                f"branch produced {out.shape[-1]} samples, more than the "
                f"declared pad target {self.pad_to}"
            )
        if out.shape[-1] < self.pad_to:
            out = np.pad(out, ((0, 0), (0, 0), (0, self.pad_to - out.shape[-1])))
        if not np.isfinite(out).all():
            raise ValueError(f"branch {self.profile_id} produced non-finite EEG")
        return out, report


def ds005662_build_branch(encoder: str) -> Ds005662Branch:
    if encoder not in DS005662_ENCODER_PROFILES:
        raise KeyError(
            f"unknown encoder {encoder!r}; expected "
            f"{sorted(DS005662_ENCODER_PROFILES)}"
        )
    profile = get_preprocessing_profile(DS005662_ENCODER_PROFILES[encoder])
    resample_hz = None
    for step in profile.steps:
        if step.name == "resample":
            resample_hz = float(step.parameters["target_hz"])
        if step.name == "epoch":
            window = {
                "window_samples": int(step.parameters["window_samples"]),
                "stride_samples": int(step.parameters["stride_samples"]),
                "n_windows": int(step.parameters["n_windows"]),
            }
    return Ds005662Branch(
        encoder=encoder,
        profile_id=profile.profile_id,
        window=window,
        sampling_rate=DS005662_ENCODER_RATE[encoder],
        eeg_units=DS005662_ENCODER_UNITS[encoder],
        model_input_normalization=DS005662_ENCODER_MODEL_NORM[encoder],
        pad_to=DS005662_PADDED_SAMPLES[encoder],
        resample_hz=resample_hz,
    )


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

# NOTE: this is the ds005662 writer.  It used to be called
# ``write_split_h5`` too, exactly like the SEED-DV writer above, which
# meant the later definition silently shadowed the earlier one and any
# attempt to rebuild a SEED-DV branch died with
# "write_split_h5() got an unexpected keyword argument 'concept_ids'".
# Renamed so both writers coexist.
def write_ds005662_split_h5(*, out_path: Path, subject: str, split: str, eeg: np.ndarray,
                            channel_names: Sequence[str], trials: Sequence[Trial],
                   gt_frames: Optional[np.ndarray], branch: Ds005662Branch,
                   filter_record: Dict[str, Any]) -> int:
    import h5py

    profile = get_preprocessing_profile(branch.profile_id)
    n_epochs, n_channels, _ = eeg.shape
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("data", data=eeg, dtype=np.float32)
        h5.create_dataset("channel_names",
                          data=np.asarray(list(channel_names), dtype=object),
                          dtype=string_dtype)
        h5.create_dataset("event_concept",
                          data=np.asarray([t.class_id for t in trials], np.int32))
        h5.create_dataset("event_block",
                          data=np.asarray([t.block for t in trials], np.int32))
        h5.create_dataset("event_caption",
                          data=np.asarray([t.caption for t in trials], dtype=object),
                          dtype=string_dtype)
        h5.create_dataset("event_caption_alt",
                          data=np.asarray([t.caption_alt for t in trials], dtype=object),
                          dtype=string_dtype)
        if gt_frames is not None:
            h5.create_dataset("gt_frames", data=gt_frames, dtype=np.uint8)

        h5.attrs["dataset"] = DS005662_DATASET_ID
        h5.attrs["subject_id"] = subject
        h5.attrs["sampling_rate"] = float(branch.sampling_rate)
        h5.attrs["n_channels"] = int(n_channels)
        h5.attrs["n_epochs"] = int(n_epochs)

        h5.attrs["encoder_id"] = branch.encoder
        h5.attrs["encoder_window_specs_json"] = canonical_json(
            DS005662_ENCODER_WINDOWS)
        h5.attrs["window_samples"] = int(branch.window["window_samples"])
        h5.attrs["window_stride_samples"] = int(branch.window["stride_samples"])
        h5.attrs["n_windows"] = int(branch.window["n_windows"])
        h5.attrs["model_input_normalization"] = branch.model_input_normalization
        h5.attrs["bandpass_json"] = canonical_json(filter_record)
        h5.attrs["preprocessing_branch_json"] = canonical_json({
            "encoder": branch.encoder,
            "profile_id": branch.profile_id,
            "native_samples": DS005662_NATIVE_SAMPLES,
            "padded_to": branch.pad_to,
            "resampled_to_hz": branch.resample_hz,
            "variant": "base",
            "classes": DS005662_CLASSES,
            "model_input_normalization": branch.model_input_normalization,
            "encoder_window_specs": DS005662_ENCODER_WINDOWS,
        })

        write_common_h5_metadata(
            h5,
            dataset_id=DS005662_DATASET_ID,
            task_id=DS005662_TASK_ID,
            subject_id=subject,
            sampling_rate=branch.sampling_rate,
            source_sampling_rate=DS005662_NATIVE_RATE,
            n_channels=n_channels,
            n_epochs=n_epochs,
            split=split,
            epoch_type="video_clip",
            eeg_units=branch.eeg_units,
            profile=profile,
            preprocessing_state={
                "fitted_split": "none",
                "encoder": branch.encoder,
                "bandpass": filter_record,
                "resampled_to_hz": branch.resample_hz,
                "padded_to_samples": branch.pad_to,
                "model_input_normalization": branch.model_input_normalization,
                "seed": SEED,
            },
        )
        write_stimulus_manifest(h5, [
            StimulusRecord(
                stimulus_id=make_stimulus_id(
                    DS005662_DATASET_ID, f"class{t.class_id:03d}-{t.variant}"),
                modality="video",
                uri=str(t.stimulus_path),
                sha256=sha256_file(t.stimulus_path),
                onset_sec=0.0,
                duration_sec=0.6,
                alignment_source="events_tsv_eventnumber",
                transform={
                    "n_frames": DS005662_N_FRAMES,
                    "height": DS005662_FRAME_HEIGHT,
                    "width": DS005662_FRAME_WIDTH,
                    "source_fps": 25.0,
                    "variant": t.variant,
                },
            )
            for t in trials
        ])

    schema = get_task_config(DS005662_TASK_ID)["h5_schema"]
    require_valid_h5_file(str(out_path), schema)
    return n_epochs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def discover_subjects(data_root: Path) -> List[str]:
    subjects = sorted(
        p.name.split("_", 1)[0]
        for p in (data_root / "derivatives" / "mne").glob("sub-*_mne_epo.fif")
    )
    if not subjects:
        raise FileNotFoundError(f"no MNE derivatives under {data_root}")
    return subjects


def ds005662_process_subject(data_root: Path, out_dir: Path, subject: str,
                    branches: Sequence[Ds005662Branch]) -> Dict[str, Dict[str, int]]:
    seed_everything(SEED)
    captions = load_captions()
    trials = load_trials(data_root, subject, captions, keep_variant="base")
    epochs, code_to_index, channel_names = read_epochs(data_root, subject)
    LOG.info("%s: %d trials, %d channels, %d samples",
             subject, len(trials), len(channel_names), epochs.shape[-1])

    frames = _frame_cache([t.stimulus_path for t in trials])
    written: Dict[str, Dict[str, int]] = {b.encoder: {} for b in branches}

    for split, blocks in (("train", DS005662_TRAIN_BLOCKS), ("test", (DS005662_TEST_BLOCK,))):
        selected = [t for t in trials if t.block in blocks]
        if not selected:
            raise ValueError(f"{subject}: no trials for split {split}")
        raw = np.stack([epochs[code_to_index[t.eventnumber]] for t in selected])
        gt = (np.stack([frames[t.stimulus_path] for t in selected])
              if split == "test" else None)
        for branch in branches:
            branch_eeg, filter_record = branch.apply(raw)
            branch_dir = out_dir / branch.encoder
            branch_dir.mkdir(parents=True, exist_ok=True)
            out_path = branch_dir / f"{DS005662_DATASET_ID}_{subject}_{split}_epoched.h5"
            written[branch.encoder][split] = write_ds005662_split_h5(
                out_path=out_path, subject=subject, split=split, eeg=branch_eeg,
                channel_names=channel_names, trials=selected, gt_frames=gt,
                branch=branch, filter_record=filter_record,
            )
            LOG.info("wrote %s (%d epochs, %d samples)", out_path,
                     written[branch.encoder][split], branch_eeg.shape[-1])
    return written


def ds005662_run_loader(data_root: str, out_dir: str,
               subjects: Optional[Sequence[str]] = None,
               encoders: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, int]]:
    root = Path(data_root)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    subjects = list(subjects) if subjects else discover_subjects(root)
    branches = [ds005662_build_branch(name) for name in (encoders or ["eegnet"])]
    totals: Dict[str, Dict[str, int]] = {b.encoder: {} for b in branches}
    for subject in subjects:
        written = ds005662_process_subject(root, out, subject, branches)
        for encoder_name, splits in written.items():
            for split, count in splits.items():
                totals[encoder_name][split] = (
                    totals[encoder_name].get(split, 0) + count
                )
    return totals


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

BENCH_ROOT = Path(__file__).resolve().parent
EVA_STIMULI_PATH = BENCH_ROOT / "eeg_video_action_stimuli.json"
EVA_CAPTIONS_PATH = BENCH_ROOT / "eeg_video_action_stimulus_captions.json"
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

DS008257_DATASET_ID = "ds008257"
DS008257_TASK_ID = "video_generation"

DS008257_STIMULI_PATH = BENCH_ROOT / "ds008257_stimuli.json"
DS008257_CAPTIONS_PATH = BENCH_ROOT / "ds008257_stimulus_captions.json"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="seed_dv",
        choices=["seed_dv", "ds005662", "eeg_video_action", "ds008257"],
        help="which dataset to materialise; all four feed the "
             "video_generation task (eeg_video_action's data_root holds "
             "SingleShot/ and video/; ds008257's holds EMD/ and stimulus_set/)",
    )
    # SEED-DV selects subjects by number (1..20 -> sub1.npy .. sub20.npy);
    # ds005662 selects them by BIDS id (sub-01..sub-80).  They are different
    # keys into different files, so the flag type follows the dataset.
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument(
        "--encoder", nargs="+", default=["eegnet"],
        # ``eegvid_nice`` is EEGVid's modified-NICE encoder (arXiv:2505.21385);
        # its branch is one window spanning the epoch, so it needs its own
        # preprocessing profile like the other encoder-specific branches.
        choices=["eegnet", "labram", "reve", "luna", "eegvid_nice"],
        help="preprocessing branch(es) to materialise; one sub-directory each",
    )
    parser.add_argument("--allow_partial", action="store_true",
                        help="diagnosis only; not a releasable benchmark artefact")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    seed_everything(SEED)

    if args.dataset == "ds005662":
        subjects = args.subjects or None
        totals = ds005662_run_loader(args.data_root, args.out_dir, subjects,
                                     args.encoder)
    elif args.dataset == "ds008257":
        totals = ds008257_run_loader(args.data_root, args.out_dir,
                                     args.subjects or None, args.encoder)
    elif args.dataset == "eeg_video_action":
        # Subjects here are ``sub-01``-style ids, like ds005662's -- the
        # release's own directory names are SUB_01, and the loader maps them.
        totals = eeg_video_action_run_loader(args.data_root, args.out_dir,
                                             args.subjects or None,
                                             args.encoder)
    else:
        subjects = [int(s) for s in args.subjects] if args.subjects else None
        totals = run_loader(args.data_root, args.out_dir, subjects,
                            args.allow_partial, args.encoder)
    print(json.dumps({"dataset": args.dataset, "epochs": totals, "seed": SEED},
                     indent=2))


if __name__ == "__main__":
    main()
