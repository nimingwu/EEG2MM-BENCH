"""Smit2025（ds005662）视频预处理：600 毫秒 epoch、64 通道、200 Hz。"""
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
from .video_common import RESOURCES, _resize_rgb

# ===========================================================================
# ds005662  (OpenNeuro ds005662 v1.0.3; Smit et al., Sci Data 2026)
# ===========================================================================
#
# Second dataset for the same ``video_generation`` task.  Added here rather
# than in a separate module because the benchmark keeps one loader per task and
# routes datasets through it; see ADDING_A_DATASET.md.

#: Where the BLIP captions for the 384 stimulus videos live (produced by
#: ``scripts/caption_ds005662_stimuli.py``).
CAPTIONS_PATH = RESOURCES / "ds005662_stimulus_captions.json"
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

LOG = logging.getLogger("eeg2x.preprocessing.ds005662")


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


