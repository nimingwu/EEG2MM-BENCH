"""NOD-EEG image-generation preprocessing from the dataset's released clean data.

Single EEG authority
--------------------
``NOD-EEG/derivatives/preprocessed/raw/<sub>_ses-ImageNetNN_task-ImageNet_run-NN_eeg_clean.fif``

The dataset authors (doi:10.1038/s41597-025-05174-7) already applied the paper's
preprocessing to these 250 Hz continuous recordings: 0.1-100 Hz band-pass,
RANSAC bad-channel detection with spherical-spline interpolation, 50-Hz Zapline,
average EEG reference and ICA artifact regression.  This adapter therefore does
NOT re-filter the signal: every inherited step is declared with
``already_applied_by_dataset: True`` in the registered preprocessing profile and
recorded in the preprocessing state.  The only THINGS-EEG alignment kept for NOD
is the 0.1-100 Hz band and the 250 Hz sampling clock, both already satisfied by
these files, so a 50 Hz notch would be a no-op (measured line-noise figures are
recorded in the summary).

Benchmark steps performed here (identical for every encoder profile)
--------------------------------------------------------------------
1. keep the 62 EEG channels (-HEO/VEO)
2. epoch [-0.1, 0.8] s around each ``stim_on`` annotation: the NOD paper's
   -100..800 ms window, 226 samples at 250 Hz
3. baseline: subtract the channel mean of [-0.1, 0.0) s
4. encoder adaptation: exact rational resample to the profile's rate plus right
   zero padding to the profile's declared sample count (LUNA additionally gets
   the official per-sample, per-channel z-score over its valid samples)
5. train-only, per-session MNN whitening (Ledoit-Wolf, sigma^-1/2)
6. class-disjoint 800/200 split with the benchmark seed 2000
7. one training H5 per session plus one concept-averaged test H5
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import h5py
import mne
import numpy as np
import scipy.linalg
from sklearn.covariance import LedoitWolf

from eeg2x.protocol import (
    StimulusRecord, make_stimulus_id, resample_eeg, sha256_array, sha256_file,
    write_common_h5_metadata, write_stimulus_manifest,
)
from eeg2x.h5_schema import require_valid_h5_file
from eeg2x.profiles import get_preprocessing_profile
from eeg2x.evaluation.registry import get_task_config
from eeg2x.data import NOD_ATM_PROFILE
from eeg2x.encoders.luna.input import LUNA_INPUT_SAMPLES, LUNA_ZSCORE_EPS

log = logging.getLogger("nod_atm_preprocess")

SEED = 2000
#: The released clean files are the dataset's own paper-preprocessed product.
SOURCE_SR = 250.0
SOURCE_HIGHPASS_HZ = 0.1
SOURCE_LOWPASS_HZ = 100.0
SOURCE_FORMAT = "NOD_derivatives_preprocessed_raw_clean_fif"
DATASET_PROVIDED_STEPS = (
    "downsample_250hz",
    "ransac_bad_channel_spherical_spline_interpolation",
    "zapline_50hz_and_harmonics",
    "average_eeg_reference",
    "ica_artifact_regression_1_100hz_fit_0.1_100hz_apply",
)
#: The NOD paper's epoch: -100 ms to +800 ms relative to stimulus onset.
T_PRE_SEC = -0.1
T_POST_SEC = 0.8
EPOCH_SAMPLES = 226          # 226 samples at 250 Hz, MNE tmin/tmax inclusive
BASELINE_SAMPLES = 25        # [-0.1, 0.0) s at 250 Hz
TARGET_SR = 250.0            # ATM-S consumes the dataset clock directly
OUTPUT_SAMPLES = 226
N_CLASSES = 1000
N_TRAIN_CLASSES = 800
N_TEST_CLASSES = 200
N_EEG_CHANNELS = 62
N_TRIALS_PER_RUN = 125
RUNS_PER_SESSION = 8
EOG_CHANNELS = frozenset({"HEO", "VEO"})
STIM_ON = "stim_on"
IMAGE_ROOT = ("stimuli", "ImageNet")
FILENAME_RE = re.compile(
    r"^(?P<subject>sub-\d+)_ses-(?P<session>ImageNet\d+)_"
    r"task-ImageNet_run-(?P<run>\d+)_eeg_clean\.fif$"
)


def _session_number(value: str) -> int:
    match = re.fullmatch(r"ImageNet(\d+)", str(value))
    if match is None:
        raise ValueError(f"Unexpected NOD session label: {value!r}")
    return int(match.group(1))


def _clean_files(nod_root: Path, subject_id: str) -> List[Tuple[str, int, Path]]:
    """Return (session, run, path) for the released clean continuous files."""
    root = nod_root / "derivatives" / "preprocessed" / "raw"
    found: Dict[Tuple[str, int], Path] = {}
    for path in sorted(root.glob(f"{subject_id}_ses-ImageNet*_task-ImageNet_run-*_eeg_clean.fif")):
        match = FILENAME_RE.fullmatch(path.name)
        if match is None or match.group("subject") != subject_id:
            raise ValueError(f"Unexpected NOD clean filename: {path.name}")
        key = (match.group("session"), int(match.group("run")))
        if key in found:
            raise ValueError(f"Duplicate NOD clean run file for {key}: {path}")
        found[key] = path
    if not found:
        raise FileNotFoundError(f"No NOD clean files for {subject_id} under {root}")

    sessions = sorted({session for session, _ in found}, key=_session_number)
    result: List[Tuple[str, int, Path]] = []
    for session in sessions:
        runs = sorted(run for name, run in found if name == session)
        if runs != list(range(1, RUNS_PER_SESSION + 1)):
            raise ValueError(
                f"{subject_id} {session}: expected runs 1-{RUNS_PER_SESSION}, got {runs}"
            )
        result.extend((session, run, found[(session, run)]) for run in runs)
    log.info("%s: %d clean runs across %d sessions", subject_id, len(result), len(sessions))
    return result


def _read_detailed_events(path: Path) -> Dict[Tuple[str, int], List[Dict[str, str]]]:
    """Stimulus identity per trial; the released clean files carry no image IDs."""
    grouped: Dict[Tuple[str, int], List[Dict[str, str]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("event_name") != STIM_ON:
                continue
            key = (str(row["session"]), int(row["run"]))
            grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        if len(rows) != N_TRIALS_PER_RUN:
            raise ValueError(f"{path}: {key} has {len(rows)} trials, expected {N_TRIALS_PER_RUN}")
    return grouped


def _stim_on_onsets(raw: mne.io.BaseRaw, path: Path) -> List[float]:
    onsets = [
        float(onset) for onset, description in
        zip(raw.annotations.onset, raw.annotations.description)
        if description == STIM_ON
    ]
    if len(onsets) != N_TRIALS_PER_RUN:
        raise ValueError(f"{path}: expected {N_TRIALS_PER_RUN} stim_on events, got {len(onsets)}")
    spacing = np.diff(sorted(onsets))
    if spacing.size and not (1.0 <= float(np.median(spacing)) <= 2.0):
        raise ValueError(f"{path}: unexpected stim_on spacing {np.median(spacing):.3f}s")
    return onsets


def _adapt_epoch(
    epoch: np.ndarray, *, target_sr: float, output_samples: int, luna: bool,
) -> Tuple[np.ndarray, int]:
    """Exact rational resample plus right zero padding to the profile contract."""
    if not np.isclose(target_sr, SOURCE_SR):
        epoch = resample_eeg(epoch, SOURCE_SR, target_sr, axis=-1)
    valid = int(epoch.shape[-1])
    if valid > output_samples:
        raise ValueError(f"Adapted epoch has {valid} samples > declared {output_samples}")
    if luna:
        # Official ChannelWiseNormalize: per-sample, per-channel z-score with
        # torch.std semantics (correction=1) on the valid samples only.
        mean = epoch.mean(axis=-1, keepdims=True)
        std = epoch.std(axis=-1, ddof=1, keepdims=True)
        epoch = (epoch - mean) / (std + LUNA_ZSCORE_EPS)
    if valid < output_samples:
        epoch = np.pad(epoch, ((0, 0), (0, output_samples - valid)), constant_values=0.0)
    return epoch.astype(np.float32, copy=False), valid


def _load_trials(
    nod_root: Path, subject_id: str, *, target_sr: float = TARGET_SR,
    output_samples: int = OUTPUT_SAMPLES, luna: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str], List[Path], Dict[str, Any]]:
    detailed = _read_detailed_events(
        nod_root / "derivatives" / "detailed_events" / f"{subject_id}_events.csv"
    )
    all_data: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    source_files: List[Path] = []
    channel_names: List[str] | None = None
    valid_samples: set[int] = set()
    first_seen: Dict[str, str] = {}
    repeated_stimuli: List[Dict[str, str]] = []

    for session, run, path in _clean_files(nod_root, subject_id):
        raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        if not np.isclose(float(raw.info["sfreq"]), SOURCE_SR):
            raise ValueError(f"{path}: expected {SOURCE_SR} Hz, got {raw.info['sfreq']}")
        if not (np.isclose(float(raw.info["highpass"]), SOURCE_HIGHPASS_HZ)
                and np.isclose(float(raw.info["lowpass"]), SOURCE_LOWPASS_HZ)):
            raise ValueError(
                f"{path}: expected the dataset's {SOURCE_HIGHPASS_HZ}-{SOURCE_LOWPASS_HZ} Hz "
                f"band, got {raw.info['highpass']}-{raw.info['lowpass']} Hz"
            )
        # The protocol keeps EEG channels only.  MNE's channel types are the
        # authority: some subjects' released clean files carry extra
        # physiological channels (sub-03 adds EKG as misc and EMG), so filtering
        # by the HEO/VEO names alone is not enough.  Fall back to the documented
        # EOG names only when nothing is typed as EEG.
        channel_types = raw.get_channel_types()
        eeg_indices = [i for i, kind in enumerate(channel_types) if kind == "eeg"]
        if not eeg_indices:
            eeg_indices = [i for i, name in enumerate(raw.ch_names)
                           if str(name) not in EOG_CHANNELS]
        if len(eeg_indices) != N_EEG_CHANNELS:
            raise ValueError(
                f"{path}: expected {N_EEG_CHANNELS} EEG channels, got "
                f"{len(eeg_indices)} (types: {sorted(set(channel_types))})")
        if channel_names is None:
            channel_names = [str(raw.ch_names[i]) for i in eeg_indices]
        elif channel_names != [str(raw.ch_names[i]) for i in eeg_indices]:
            raise ValueError(f"{path}: EEG channel order differs between runs")

        onsets = _stim_on_onsets(raw, path)
        labels = detailed.get((session, run), [])
        if len(labels) != len(onsets):
            raise ValueError(f"{session} run {run}: timing/label count mismatch")
        signal = raw.get_data(picks=eeg_indices).astype(np.float32, copy=False)

        for row_index, (onset, label) in enumerate(zip(onsets, labels), start=1):
            image_id = str(label["image_id"])
            # A few released subjects repeat stimuli across sessions (sub-05
            # re-shows 239 images).  That is a dataset property, not an error:
            # record it in the preprocessing state instead of failing.
            location = f"{session}/run{run}"
            previous = first_seen.setdefault(image_id, location)
            if previous != location:
                repeated_stimuli.append(
                    {"image_id": image_id, "first": previous, "repeat": location})
            start = int(round(onset * SOURCE_SR)) - BASELINE_SAMPLES
            stop = start + EPOCH_SAMPLES
            if start < 0 or stop > signal.shape[1]:
                raise ValueError(f"{path}: trial {row_index} exceeds raw bounds")
            epoch = signal[:, start:stop].copy()
            # NOD paper baseline: the 100 ms before onset, applied on the 250 Hz
            # clock so the baseline interval is exactly [-0.1, 0.0) s.
            epoch -= epoch[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
            epoch = epoch * np.float32(1e6)          # V -> uV (numeric stability)
            epoch, valid = _adapt_epoch(
                epoch, target_sr=target_sr, output_samples=output_samples, luna=luna)
            valid_samples.add(valid)
            if epoch.shape != (N_EEG_CHANNELS, output_samples):
                raise AssertionError(epoch.shape)
            all_data.append(epoch)
            rows.append({
                "session": session,
                "run": run,
                "row_index": row_index,
                "image_id": image_id,
                "class_id": str(label["class_id"]),
                "class_name": str(label["class"]),
                "event_name": STIM_ON,
                "source_file": str(path.resolve()),
            })
        source_files.append(path.resolve())
        del raw

    data = np.stack(all_data, axis=0).astype(np.float32)
    expected = (len(source_files) * N_TRIALS_PER_RUN, N_EEG_CHANNELS, output_samples)
    if data.shape != expected:
        raise ValueError(f"Unexpected NOD shape: {data.shape}, expected {expected}")
    if channel_names is None:
        raise ValueError("No EEG channels found")
    if len(valid_samples) != 1:
        raise ValueError(f"Encoder adaptation produced inconsistent lengths: {valid_samples}")
    if repeated_stimuli:
        log.warning("%s: %d repeated stimulus presentations across sessions "
                    "(e.g. %s)", subject_id, len(repeated_stimuli),
                    repeated_stimuli[0])
    adaptation = {
        "valid_samples": int(next(iter(valid_samples))),
        "zero_padding_samples": int(output_samples - next(iter(valid_samples))),
        "repeated_stimuli_count": len(repeated_stimuli),
        "repeated_stimuli_sample": repeated_stimuli[:10],
    }
    return data, rows, channel_names, source_files, adaptation


def _compute_mnn_whitener(epochs: np.ndarray, class_keys: Sequence[str]) -> np.ndarray:
    """Ledoit-Wolf condition covariance averaged per class -> sigma^-1/2."""
    keys = np.asarray(class_keys)
    covariances = []
    for class_key in sorted(set(keys.tolist())):
        for epoch in epochs[keys == class_key]:
            covariances.append(LedoitWolf().fit(epoch.T).covariance_)
    covariance = np.mean(np.stack(covariances, axis=0), axis=0)
    whitener = np.asarray(scipy.linalg.fractional_matrix_power(covariance, -0.5)).real
    return whitener.astype(np.float32)


def _apply_mnn(data: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    return (data.swapaxes(1, 2) @ whitener).swapaxes(1, 2).astype(np.float32)


def _split_and_mnn(
    data: np.ndarray, rows: Sequence[Dict[str, Any]],
) -> Tuple[
    np.ndarray, List[Dict[str, Any]], np.ndarray, List[Dict[str, Any]],
    Dict[str, Any], np.ndarray, List[str],
]:
    class_ids = sorted({str(row["class_id"]) for row in rows})
    if len(class_ids) != N_CLASSES:
        raise ValueError(f"Expected {N_CLASSES} ImageNet classes, got {len(class_ids)}")
    rng = np.random.default_rng(SEED)
    test_classes = sorted(
        str(class_id) for class_id in rng.choice(
            np.asarray(class_ids), size=N_TEST_CLASSES, replace=False).tolist()
    )
    test_set = set(test_classes)
    train_classes = [class_id for class_id in class_ids if class_id not in test_set]
    if len(train_classes) != N_TRAIN_CLASSES:
        raise ValueError("NOD class split must be exactly 800 train / 200 test")
    train_set = set(train_classes)
    class_number = {class_id: index + 1 for index, class_id in enumerate(class_ids)}
    sessions = sorted({str(row["session"]) for row in rows}, key=_session_number)

    train_indices = [i for i, row in enumerate(rows) if str(row["class_id"]) in train_set]
    train_indices.sort(key=lambda i: (
        _session_number(rows[i]["session"]), int(rows[i]["run"]), int(rows[i]["row_index"])))
    train_rows = [dict(rows[i]) for i in train_indices]

    grouped: Dict[str, List[int]] = {class_id: [] for class_id in test_classes}
    for index, row in enumerate(rows):
        if str(row["class_id"]) in test_set:
            grouped[str(row["class_id"])].append(index)

    whitened = np.empty_like(data)
    whiteners: List[np.ndarray] = []
    for session in sessions:
        indices = [i for i, row in enumerate(rows) if row["session"] == session]
        train_session = [i for i in indices if str(rows[i]["class_id"]) in train_set]
        if not train_session:
            raise ValueError(f"Session {session} has no training-class epochs")
        whitener = _compute_mnn_whitener(
            data[train_session], [str(rows[i]["class_id"]) for i in train_session])
        whiteners.append(whitener)
        whitened[indices] = _apply_mnn(data[indices], whitener)

    for index, row in enumerate(train_rows):
        row["object_number"] = class_number[str(row["class_id"])]
        # NOD shows one unique image per class and session, so the class' four
        # sessions are kept as four training trials, numbered 1..4 like testb.
        row["stim_number"] = sum(
            1 for previous in train_rows[:index]
            if str(previous["class_id"]) == str(row["class_id"])
        ) + 1
        row["embedding_index"] = index

    test_data: List[np.ndarray] = []
    test_rows: List[Dict[str, Any]] = []
    for embedding_index, class_id in enumerate(test_classes):
        indices = sorted(grouped[class_id], key=lambda i: (
            _session_number(rows[i]["session"]), int(rows[i]["run"])))
        if not indices:
            raise ValueError(f"Test class has no epochs: {class_id}")
        source = [dict(rows[i]) for i in indices]
        aggregate = dict(source[0])
        aggregate.update({
            "session": "aggregate",
            "run": 0,
            "row_index": 0,
            "source_image_ids": [str(item["image_id"]) for item in source],
            "source_sessions": [str(item["session"]) for item in source],
            "n_repetitions": len(source),
            "object_number": class_number[class_id],
            "stim_number": 1,
            "embedding_index": embedding_index,
        })
        test_rows.append(aggregate)
        test_data.append(np.mean(whitened[indices], axis=0).astype(np.float32))

    whitener_array = np.stack(whiteners, axis=0)
    state = {
        "seed": SEED,
        "fitted_split": "train",
        "class_selection": "seeded_random_without_replacement",
        "class_split_seed": SEED,
        "n_classes": N_CLASSES,
        "n_train_classes": N_TRAIN_CLASSES,
        "n_test_classes": N_TEST_CLASSES,
        "split_unit": "class_id",
        "class_disjoint": True,
        "train_class_ids": train_classes,
        "test_class_ids": test_classes,
        "sessions": sessions,
        "session_ids": sessions,
        "test_aggregation": "mean_by_class_across_sessions_after_session_mnn",
        "mnn": "LedoitWolf_condition_covariance_train_only_per_session",
        "mnn_sessions": sessions,
        "mnn_whitener_sha256": sha256_array(whitener_array),
        "mnn_matrix_sha256": sha256_array(whitener_array),
        "mnn_matrix_sha256_by_session": {
            session: sha256_array(whitener_array[index])
            for index, session in enumerate(sessions)
        },
    }
    return (
        whitened[train_indices], train_rows, np.stack(test_data, axis=0),
        test_rows, state, whitener_array, sessions,
    )


def _string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=object)


def _write_h5(
    path: Path,
    data: np.ndarray,
    rows: Sequence[Dict[str, Any]],
    channels: Sequence[str],
    subject_id: str,
    split: str,
    state: Dict[str, Any],
    nod_root: Path,
    source_files: Sequence[Path],
    whiteners: np.ndarray,
    profile_id: str = NOD_ATM_PROFILE,
    sampling_rate: float = TARGET_SR,
) -> Path:
    profile = get_preprocessing_profile(profile_id)
    records: List[StimulusRecord] = []
    for row in rows:
        image_ids = [str(value) for value in row.get("source_image_ids", [row["image_id"]])]
        image_path = (nod_root / Path(*IMAGE_ROOT) / f"{image_ids[0]}.JPEG").resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing NOD stimulus image: {image_path}")
        records.append(StimulusRecord(
            stimulus_id=make_stimulus_id(
                "nod",
                f"class-{row['class_id']}" if split == "test" else str(row["image_id"]),
            ),
            modality="image",
            uri=str(image_path),
            sha256=sha256_file(image_path),
            # The NOD paper's epoch spans -100..800 ms, i.e. 0.9 s.
            duration_sec=T_POST_SEC - T_PRE_SEC,
            alignment_source="NOD-clean-fif-stim_on-annotations",
            transform={
                "dataset": "nod",
                "class_id": str(row["class_id"]),
                "image_id": str(row["image_id"]),
                "session": str(row["session"]),
                "run": int(row["run"]),
                "embedding_index": int(row["embedding_index"]),
                "object_number": int(row["object_number"]),
                "stim_number": int(row["stim_number"]),
                "source_image_ids": image_ids,
                "source_sessions": [
                    str(value) for value in row.get("source_sessions", [row["session"]])
                ],
                "n_repetitions": int(row.get("n_repetitions", 1)),
            },
        ))

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3 or data.shape[1] != len(channels):
        raise ValueError(f"Unexpected NOD data shape: {data.shape}")
    n_epochs, n_channels, n_samples = data.shape
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "data", data=data, compression="gzip", compression_opts=4,
            chunks=(1, n_channels, n_samples))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        h5.create_dataset("event_label", data=_string_array(
            str(row["class_id"]) for row in rows), dtype=string_dtype)
        h5.create_dataset("event_class_id", data=_string_array(
            str(row["class_id"]) for row in rows), dtype=string_dtype)
        h5.create_dataset("event_class_name", data=_string_array(
            str(row["class_name"]) for row in rows), dtype=string_dtype)
        h5.create_dataset("event_image_id", data=_string_array(
            str(row["image_id"]) for row in rows), dtype=string_dtype)
        h5.create_dataset("event_session", data=_string_array(
            str(row["session"]) for row in rows), dtype=string_dtype)
        h5.create_dataset("event_run", data=np.asarray(
            [int(row["run"]) for row in rows], dtype=np.int32))
        h5.create_dataset("event_object_number", data=np.asarray(
            [int(row["object_number"]) for row in rows], dtype=np.int32))
        h5.create_dataset("event_stim_number", data=np.asarray(
            [int(row["stim_number"]) for row in rows], dtype=np.int32))
        h5.create_dataset("event_embedding_index", data=np.asarray(
            [int(row["embedding_index"]) for row in rows], dtype=np.int32))
        h5.create_dataset("event_source_image_ids_json", data=_string_array(
            json.dumps(row.get("source_image_ids", [row["image_id"]])) for row in rows),
            dtype=string_dtype)
        h5.create_dataset("event_source_sessions_json", data=_string_array(
            json.dumps(row.get("source_sessions", [row["session"]])) for row in rows),
            dtype=string_dtype)
        h5.create_dataset("channel_names", data=_string_array(channels), dtype=string_dtype)
        # The whitened values are dimensionless.  The per-session stack is
        # written under both names so the protocol validator can match the
        # state's per-session hashes and the flat MNN hash.
        h5.create_dataset("mnn_whitener", data=np.asarray(whiteners, np.float32))
        h5.create_dataset("mnn_whiteners", data=np.asarray(whiteners, np.float32))
        write_common_h5_metadata(
            h5,
            dataset_id="nod",
            task_id="eeg_to_image",
            subject_id=subject_id,
            sampling_rate=sampling_rate,
            source_sampling_rate=SOURCE_SR,
            n_channels=n_channels,
            n_epochs=n_epochs,
            split=split,
            epoch_type="image",
            eeg_units="dimensionless_whitened",
            profile=profile,
            preprocessing_state={
                **state,
                "subject_id": subject_id,
                "split": split,
                "dataset": "nod",
                "profile_id": profile_id,
                "derived_inputs_used": False,
                "paper_doi": "10.1038/s41597-025-05174-7",
                "source_format": SOURCE_FORMAT,
                "source_files": [str(item) for item in source_files],
                "mnn_fit_source_files": [str(item) for item in source_files],
                "dataset_provided_steps": {
                    name: {"already_applied_by_dataset": True}
                    for name in DATASET_PROVIDED_STEPS
                },
                "source_filter_hz": {
                    "highpass": SOURCE_HIGHPASS_HZ, "lowpass": SOURCE_LOWPASS_HZ},
                "source_sampling_rate": SOURCE_SR,
                "output_sampling_rate": sampling_rate,
                "input_window_sec": [T_PRE_SEC, T_POST_SEC],
                "baseline_window_sec": [T_PRE_SEC, 0.0],
                "output_window_sec": [T_PRE_SEC, T_POST_SEC],
                "n_channels": n_channels,
                "n_samples": n_samples,
                "unit_conversion": "V_to_uV_before_mnn",
                "mnn_whitener_shape": list(np.asarray(whiteners).shape),
                "label_metadata": "NOD derivatives/detailed_events only for image identity",
            },
        )
        h5.attrs["derived_inputs_used"] = False
        write_stimulus_manifest(h5, records)
    require_valid_h5_file(path, get_task_config("image_generation")["h5_schema"])
    log.info("Saved %s: %s", path, data.shape)
    return path


def process_nod_atm(
    subject_id: str,
    nod_root: str | Path,
    out_dir: str | Path, *, profile_id: str = NOD_ATM_PROFILE,
    target_sr: float = TARGET_SR, output_samples: int = OUTPUT_SAMPLES,
    output_prefix: str = "nod_atm", luna: bool = False,
) -> int:
    root = Path(nod_root).resolve(strict=True)
    data, rows, channels, source_files, adaptation = _load_trials(
        root, subject_id, target_sr=target_sr, output_samples=output_samples, luna=luna)
    train_data, train_rows, test_data, test_rows, state, whiteners, sessions = (
        _split_and_mnn(data, rows))
    state = {
        **state,
        "profile_id": profile_id,
        "output_sampling_rate": target_sr,
        "n_samples": int(data.shape[-1]),
        "valid_samples_before_padding": adaptation["valid_samples"],
        "encoder_zero_padding_samples": adaptation["zero_padding_samples"],
        "encoder_adaptation": (
            "luna_official_per_sample_zscore" if luna else "identity"
        ),
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for index, session in enumerate(sessions):
        indices = [i for i, row in enumerate(train_rows) if row["session"] == session]
        _write_h5(
            output / (f"{output_prefix}_{subject_id}_"
                      f"ses{_session_number(session):02d}_training_epoched.h5"),
            train_data[indices], [train_rows[i] for i in indices], channels,
            subject_id, "train", state, root, source_files, whiteners,
            profile_id, target_sr)
    _write_h5(
        output / f"{output_prefix}_{subject_id}_test_epoched.h5",
        test_data, test_rows, channels, subject_id, "test", state, root,
        source_files, whiteners, profile_id, target_sr)
    summary = {
        "subject_id": subject_id,
        "profile_id": profile_id,
        "seed": SEED,
        "sessions": sessions,
        "n_train_classes": N_TRAIN_CLASSES,
        "n_test_classes": N_TEST_CLASSES,
        "n_train_epochs": len(train_rows),
        "n_test_epochs": len(test_rows),
        "n_channels": len(channels),
        "n_samples": int(data.shape[-1]),
        "sampling_rate": target_sr,
        "source_sampling_rate": SOURCE_SR,
        "source_format": SOURCE_FORMAT,
        "source_files": [str(item) for item in source_files],
        "dataset_provided_steps": list(DATASET_PROVIDED_STEPS),
        "input_window_sec": [T_PRE_SEC, T_POST_SEC],
        "baseline_window_sec": [T_PRE_SEC, 0.0],
        "valid_samples_before_padding": adaptation["valid_samples"],
        "encoder_zero_padding_samples": adaptation["zero_padding_samples"],
        "mnn": "LedoitWolf_condition_covariance_train_only_per_session",
        "derived_inputs_used": False,
    }
    (output / f"{output_prefix}_{subject_id}_preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("NOD %s / %s: %d train + %d test epochs",
             subject_id, profile_id, len(train_rows), len(test_rows))
    return len(train_rows) + len(test_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nod_root", type=Path,
                        default=Path("/datas/eeg_benchmark_project/NOD-EEG"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("./h5/nod_sub01_atm_seed2000"))
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--profile_id", default=NOD_ATM_PROFILE)
    parser.add_argument("--target_sr", type=float, default=TARGET_SR)
    parser.add_argument("--output_samples", type=int, default=OUTPUT_SAMPLES)
    parser.add_argument("--output_prefix", default="nod_atm")
    parser.add_argument("--luna", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    process_nod_atm(
        args.subject, args.nod_root, args.out_dir, profile_id=args.profile_id,
        target_sr=args.target_sr, output_samples=args.output_samples,
        output_prefix=args.output_prefix, luna=args.luna)


if __name__ == "__main__":
    main()
