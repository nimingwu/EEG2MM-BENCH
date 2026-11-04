"""Grootswagers2019 (OpenNeuro ds004018) preprocessing for the EEG2X benchmark.

Dataset
-------
``<root>/sub-XX/eeg/sub-XX_task-rsvp_run-0{1,2}_eeg.vhdr`` (BrainVision) plus the
BIDS ``*_events.tsv``.  Run 01 is the 5 Hz session and run 02 the 20 Hz session;
each run holds 40 sequences in which the 200 object images are presented once
per sequence (40 repetitions per image), with 1-4 target images inserted.

Paper alignment (Grootswagers, Robinson & Carlson, NeuroImage 2019, 188:668-679)
-------------------------------------------------------------------------------
The paper's own pipeline is reproduced here except where the benchmark demands an
exact rational resample: 0.1 Hz high-pass + 100 Hz low-pass Hamming-windowed FIR,
downsampled to 250 Hz, Cz reference kept (no re-referencing), no artifact
rejection, no baseline correction, epoch ``[-100, 1000] ms`` around each
non-target stimulus onset.  Epochs therefore overlap the following stimuli
exactly as in the paper's own analysis.

Benchmark steps
---------------
* **stimulus-disjoint** split with the protocol seed 2000: 20 of the 200 stimuli
  are held out and 180 are trained on, matching the THINGS-EEG2 convention the
  benchmark's other image-generation datasets follow (1654 train concepts /
  200 disjoint test concepts, a 90/10 split).  Sequences are no longer split:
  all 40 runs' sequences of a speed contribute to both sides, so every test
  stimulus is averaged over 40 sequences x 2 speeds = 80 repetitions and every
  training stimulus contributes 80 trials.
* because the held-out stimuli never appear in training, this is an
  *unseen-stimulus* reconstruction task, unlike the previous sequence split
  which left all 200 stimuli present on both sides
* train rows keep every non-target presentation of a training stimulus;
  test rows are the mean across all sequences, one row per held-out stimulus
* train-only, per-speed MNN whitening (Ledoit-Wolf, sigma^-2)
* ``--labels`` selects what ``event_label`` carries: ``stimulus`` (retrieval and
  generation; must match the candidate pool ids) or ``level_a``/``level_b``/
  ``level_c`` (concept classification)
* ``--layout`` selects the file naming: ``single`` (one training file, consumed
  by the retrieval loader) or ``per_session`` (one training file per speed,
  consumed by the ATM model's contract)
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
from eeg2x.rawdata import EXPECTED_GROOTSWAGERS_ROOT, require_grootswagers_root
from eeg2x.profiles import get_preprocessing_profile
from eeg2x.evaluation.registry import get_task_config
from eeg2x.encoders.luna.input import LUNA_ZSCORE_EPS

log = logging.getLogger("grootswagers_preprocess")

SEED = 2000
SOURCE_SR = 1000.0
BANDPASS_HZ = (0.1, 100.0)
TARGET_SR = 250.0
T_PRE_SEC = -0.1
T_POST_SEC = 1.0
PRE_SAMPLES = int(round(-T_PRE_SEC * TARGET_SR))          # 25
EPOCH_SAMPLES = int(round((T_POST_SEC - T_PRE_SEC) * TARGET_SR)) + 1   # 276
N_CHANNELS = 63
N_STIMULI = 200
N_SEQUENCES = 40
#: THINGS-EEG2's train/test concept ratio (1654/200) applied to 200 stimuli.
N_TRAIN_STIMULI = 180
N_TEST_STIMULI = 20
TARGET_MIN_NUMBER = 201        # stimulusnumber >= 201 are target images
SPEEDS = {"run-01": "5hz", "run-02": "20hz"}
DEFAULT_PROFILE = "grootswagers-image-nice-v1"
IMAGE_ROOT = "stimuli"
DATASET_PROVIDED_STEPS = ("cz_online_reference",)


def _speed_of(run: str) -> str:
    try:
        return SPEEDS[run]
    except KeyError as exc:
        raise ValueError(f"Unexpected Grootswagers run label: {run!r}") from exc


def _runs(root: Path, subject_id: str) -> List[Tuple[str, Path, Path]]:
    """Return (run, vhdr, events) for every released run of one subject."""
    eeg_dir = root / subject_id / "eeg"
    result: List[Tuple[str, Path, Path]] = []
    for vhdr in sorted(eeg_dir.glob(f"{subject_id}_task-rsvp_run-*_eeg.vhdr")):
        match = re.search(r"_run-(\d+)_eeg\.vhdr$", vhdr.name)
        if match is None:
            raise ValueError(f"Unexpected Grootswagers filename: {vhdr.name}")
        run = f"run-{int(match.group(1)):02d}"
        events = vhdr.with_name(vhdr.name.replace("_eeg.vhdr", "_events.tsv"))
        if not events.is_file():
            raise FileNotFoundError(events)
        result.append((run, vhdr, events))
    if not result:
        raise FileNotFoundError(f"No BrainVision runs for {subject_id} under {eeg_dir}")
    return result


def _read_events(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _stimulus_split(stimuli: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Hold out 20 of the 200 stimuli, without replacement, seed 2000.

    The split is a property of the *stimulus set*, so it is computed once per
    subject and reused by every speed session; a per-run split could put the
    same stimulus on both sides of the two runs.
    """
    if len(stimuli) != N_STIMULI:
        raise ValueError(f"Expected {N_STIMULI} stimuli, got {len(stimuli)}")
    rng = np.random.default_rng(SEED)
    permuted = np.asarray(sorted(stimuli))[rng.permutation(len(stimuli))]
    train = sorted(str(value) for value in permuted[:N_TRAIN_STIMULI])
    test = sorted(str(value) for value in permuted[N_TRAIN_STIMULI:])
    if len(test) != N_TEST_STIMULI:
        raise ValueError(f"Expected {N_TEST_STIMULI} test stimuli, got {len(test)}")
    return train, test


def _label_value(row: Dict[str, str], labels: str) -> str:
    if labels == "stimulus":
        return str(row["stimulusname"])
    key = {"level_a": "levelA", "level_b": "levelB", "level_c": "levelC"}[labels]
    return f"{key}={row[key]}"


def _load_speed(
    vhdr: Path, events_path: Path, *, labels: str, target_sr: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[str], Dict[str, Any]]:
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    if not np.isclose(float(raw.info["sfreq"]), SOURCE_SR):
        raise ValueError(f"{vhdr}: expected {SOURCE_SR} Hz, got {raw.info['sfreq']}")
    channel_types = raw.get_channel_types()
    eeg_indices = [i for i, kind in enumerate(channel_types) if kind == "eeg"]
    channel_names = [str(raw.ch_names[i]) for i in eeg_indices]
    if len(eeg_indices) != N_CHANNELS:
        raise ValueError(
            f"{vhdr}: expected {N_CHANNELS} EEG channels, got {len(eeg_indices)}")
    # Paper filter: Hamming-windowed FIR, 0.1 Hz high-pass, 100 Hz low-pass.
    raw.filter(l_freq=BANDPASS_HZ[0], h_freq=BANDPASS_HZ[1], method="fir",
               fir_window="hamming", verbose="ERROR")
    signal = resample_eeg(raw.get_data(picks=eeg_indices), SOURCE_SR, target_sr, axis=-1)

    rows: List[Dict[str, Any]] = []
    for row in _read_events(events_path):
        if int(row["istarget"]) != 0:
            continue
        if int(row["stimulusnumber"]) >= TARGET_MIN_NUMBER:
            continue
        onset_samples = int(row["onset"])          # this release stores samples
        start = int(round(onset_samples / SOURCE_SR * target_sr)) - PRE_SAMPLES
        stop = start + EPOCH_SAMPLES
        if start < 0 or stop > signal.shape[-1]:
            raise ValueError(f"{vhdr}: trial at {onset_samples} exceeds raw bounds")
        rows.append({
            "sequence": int(row["trialnumber"]),
            "stimulus_number": int(row["stimulusnumber"]),
            "stimulus": str(row["stimulusname"]),
            "label": _label_value(row, labels),
            "level_a": str(row["levelA"]),
            "level_b": str(row["levelB"]),
            "level_c": str(row["levelC"]),
            # The TSV header spells this "withinexemplarnumber" while the
            # shipped events.json mis-spells it "withinexamplarnumber".
            "exemplar": int(row.get("withinexemplarnumber")
                            or row.get("withinexamplarnumber") or 0),
            "start": start,
        })
    del raw
    return signal, rows, channel_names, {"n_trials": len(rows)}


def _epoch_and_split(
    signal: np.ndarray, rows: Sequence[Dict[str, Any]], *, labels: str,
    train_stimuli: Sequence[str], test_stimuli: Sequence[str],
) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    train_set, test_set = set(train_stimuli), set(test_stimuli)
    epochs = np.stack([
        signal[:, row["start"]:row["start"] + EPOCH_SAMPLES] for row in rows
    ]).astype(np.float32)
    epochs *= np.float32(1e6)                       # V -> uV

    train_rows, train_data = [], []
    for row, epoch in zip(rows, epochs):
        if row["stimulus"] in train_set:
            train_rows.append(dict(row))
            train_data.append(epoch)
    if not train_rows:
        raise ValueError("No training trials survived the stimulus split")
    leakage = sorted({row["stimulus"] for row in train_rows} & test_set)
    if leakage:
        raise ValueError(f"Test stimuli leaked into training: {leakage}")

    grouped: Dict[str, List[np.ndarray]] = {}
    for row, epoch in zip(rows, epochs):
        if row["stimulus"] in test_set:
            grouped.setdefault(row["stimulus"], []).append(epoch)
    if len(grouped) != len(test_set):
        raise ValueError(
            f"Expected {len(test_set)} held-out stimuli, got {len(grouped)}")

    test_rows, test_data = [], []
    for stimulus in sorted(grouped):
        source = [row for row in rows if row["stimulus"] == stimulus]
        aggregate = dict(source[0])
        aggregate.update({
            "sequence": -1, "start": source[0]["start"],
            "n_repetitions": len(grouped[stimulus]),
            "source_sequences": [int(row["sequence"]) for row in source],
        })
        test_rows.append(aggregate)
        test_data.append(np.mean(grouped[stimulus], axis=0).astype(np.float32))

    state = {
        "labels": labels,
        "n_sequences": len(sorted({int(row["sequence"]) for row in rows})),
        "train_stimuli": sorted(train_set),
        "test_stimuli": sorted(test_set),
        "n_train_stimuli": len(train_set),
        "n_test_stimuli": len(test_set),
        "n_train_trials": len(train_rows),
        "test_aggregation": "mean_by_stimulus_across_all_sequences",
    }
    return (np.stack(train_data, axis=0), train_rows,
            np.stack(test_data, axis=0), test_rows, state)


def _compute_mnn_whitener(epochs: np.ndarray) -> np.ndarray:
    covariances = [LedoitWolf().fit(epoch.T).covariance_ for epoch in epochs]
    covariance = np.mean(np.stack(covariances, axis=0), axis=0)
    return np.asarray(scipy.linalg.fractional_matrix_power(covariance, -0.5)).real.astype(np.float32)


def _apply_mnn(data: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    return (data.swapaxes(1, 2) @ whitener).swapaxes(1, 2).astype(np.float32)


def _adapt_epochs(
    data: np.ndarray, *, target_sr: float, output_samples: int, luna: bool,
) -> Tuple[np.ndarray, int]:
    """Encoder adaptation: crop the post-stimulus second, resample, pad.

    The paper's window is [-0.1, 1.0] s (276 samples at 250 Hz).  Encoders with
    a fixed one-second contract (LaBraM/REVE at 200 Hz, LUNA at 256 Hz) consume
    the post-stimulus [0.0, 1.0] s instead: 250 samples at 250 Hz resample
    exactly onto 200 and 256 samples, so no truncation is needed.
    """
    if np.isclose(target_sr, TARGET_SR):
        adapted, valid = data, int(data.shape[-1])
    else:
        # Encoders with a fixed one-second contract consume the half-open
        # post-stimulus second [0.0, 1.0): 250 samples at 250 Hz, which resample
        # exactly onto 200 samples at 200 Hz and 256 at 256 Hz.  (The paper's
        # inclusive tick at +1000 ms is dropped so the contract holds.)
        available = data.shape[-1] - PRE_SAMPLES
        post = data[..., PRE_SAMPLES:PRE_SAMPLES + min(available, int(round(TARGET_SR)))]
        adapted = resample_eeg(post, TARGET_SR, target_sr, axis=-1)
        valid = int(adapted.shape[-1])
        if luna:
            mean = adapted.mean(axis=-1, keepdims=True)
            std = adapted.std(axis=-1, ddof=1, keepdims=True)
            adapted = (adapted - mean) / (std + LUNA_ZSCORE_EPS)
    if valid > output_samples:
        raise ValueError(
            f"Adapted window has {valid} samples > declared {output_samples}")
    if valid < output_samples:
        # Works for both (channels, samples) and (trials, channels, samples).
        pad = ((0, 0),) * (adapted.ndim - 1) + ((0, output_samples - valid),)
        adapted = np.pad(adapted, pad, constant_values=0.0)
    return adapted.astype(np.float32, copy=False), valid


def _string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=object)


def _write_h5(
    path: Path, data: np.ndarray, rows: Sequence[Dict[str, Any]],
    channels: Sequence[str], subject_id: str, split: str, state: Dict[str, Any],
    root: Path, source_files: Sequence[Path], whiteners: np.ndarray,
    profile_id: str, sampling_rate: float, speed: str,
) -> Path:
    profile = get_preprocessing_profile(profile_id)
    records: List[StimulusRecord] = []
    for row in rows:
        image_path = (root / IMAGE_ROOT / str(row["stimulus"])).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing Grootswagers stimulus: {image_path}")
        records.append(StimulusRecord(
            stimulus_id=make_stimulus_id("grootswagers", str(row["stimulus"])),
            modality="image",
            uri=str(image_path),
            sha256=sha256_file(image_path),
            duration_sec=(0.2 if speed == "5hz" else 0.05),
            alignment_source="BIDS events.tsv onset (samples) stimulus onset",
            transform={
                "dataset": "grootswagers",
                "stimulus": str(row["stimulus"]),
                "stimulus_number": int(row["stimulus_number"]),
                "sequence": int(row["sequence"]),
                "speed": speed,
                "level_a": str(row["level_a"]),
                "level_b": str(row["level_b"]),
                "level_c": str(row["level_c"]),
                "exemplar": int(row["exemplar"]),
                "n_repetitions": int(row.get("n_repetitions", 1)),
            },
        ))

    data = np.asarray(data, dtype=np.float32)
    n_epochs, n_channels, n_samples = data.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("data", data=data, compression="gzip", compression_opts=4,
                          chunks=(1, n_channels, n_samples))
        strings = h5py.string_dtype(encoding="utf-8")
        labels = [str(row["label"]) for row in rows]
        h5.create_dataset("event_label", data=_string_array(labels), dtype=strings)
        h5.create_dataset("event_class_id", data=_string_array(labels), dtype=strings)
        h5.create_dataset("event_stimulus", data=_string_array(
            str(row["stimulus"]) for row in rows), dtype=strings)
        h5.create_dataset("event_speed", data=_string_array(
            speed for _ in rows), dtype=strings)
        h5.create_dataset("event_sequence", data=np.asarray(
            [int(row["sequence"]) for row in rows], dtype=np.int32))
        h5.create_dataset("event_level_a", data=_string_array(
            str(row["level_a"]) for row in rows), dtype=strings)
        h5.create_dataset("event_level_b", data=_string_array(
            str(row["level_b"]) for row in rows), dtype=strings)
        h5.create_dataset("event_level_c", data=_string_array(
            str(row["level_c"]) for row in rows), dtype=strings)
        # object_number identifies the presented stimulus; stim_number counts its
        # repetitions (or is 1 for an aggregated test row).
        stimulus_numbers = sorted({int(row["stimulus_number"]) for row in rows})
        number_of = {value: index + 1 for index, value in enumerate(stimulus_numbers)}
        object_numbers, stim_numbers = [], []
        counter: Dict[int, int] = {}
        for row in rows:
            key = int(row["stimulus_number"])
            object_numbers.append(number_of[key] if split == "train" else number_of[key])
            if split == "train":
                counter[key] = counter.get(key, 0) + 1
                stim_numbers.append(counter[key])
            else:
                stim_numbers.append(1)
        h5.create_dataset("event_object_number", data=np.asarray(object_numbers, dtype=np.int32))
        h5.create_dataset("event_stim_number", data=np.asarray(stim_numbers, dtype=np.int32))
        h5.create_dataset("event_embedding_index", data=np.arange(n_epochs, dtype=np.int32))
        h5.create_dataset("channel_names", data=_string_array(channels), dtype=strings)
        h5.create_dataset("mnn_whitener", data=np.asarray(whiteners, np.float32))
        h5.create_dataset("mnn_whiteners", data=np.asarray(whiteners, np.float32))
        write_common_h5_metadata(
            h5, dataset_id="grootswagers", task_id="eeg_to_image", subject_id=subject_id,
            sampling_rate=sampling_rate, source_sampling_rate=SOURCE_SR,
            n_channels=n_channels, n_epochs=n_epochs, split=split, epoch_type="image",
            eeg_units="dimensionless_whitened", profile=profile,
            preprocessing_state={
                **state, "subject_id": subject_id, "split": split,
                "dataset": "grootswagers", "profile_id": profile_id,
                "speed": speed,
                "derived_inputs_used": False,
                "paper_doi": "10.1016/j.neuroimage.2018.12.046",
                "source_format": "BIDS_BrainVision",
                "source_files": [str(item) for item in source_files],
                "mnn_fit_source_files": [str(item) for item in source_files],
                "dataset_provided_steps": {
                    name: {"already_applied_by_dataset": True}
                    for name in DATASET_PROVIDED_STEPS},
                "benchmark_filter_hz": list(BANDPASS_HZ),
                "source_sampling_rate": SOURCE_SR,
                "output_sampling_rate": sampling_rate,
                "input_window_sec": [T_PRE_SEC, T_POST_SEC],
                "output_window_sec": [T_PRE_SEC, T_POST_SEC],
                "baseline_window_sec": None,
                "n_channels": n_channels, "n_samples": n_samples,
                "unit_conversion": "V_to_uV_before_mnn",
                "mnn_whitener_shape": list(np.asarray(whiteners).shape),
            })
        h5.attrs["derived_inputs_used"] = False
        write_stimulus_manifest(h5, records)
    require_valid_h5_file(path, get_task_config("image_retrieval")["h5_schema"])
    log.info("Saved %s: %s", path, data.shape)
    return path


def process_grootswagers(
    subject_id: str, data_root: str | Path, out_dir: str | Path, *,
    profile_id: str = DEFAULT_PROFILE, labels: str = "stimulus",
    layout: str = "single", output_prefix: str = "grootswagers",
    speeds: Sequence[str] = ("5hz", "20hz"), target_sr: float = TARGET_SR,
    output_samples: int = EPOCH_SAMPLES, luna: bool = False,
) -> int:
    if layout not in ("single", "per_session"):
        raise ValueError(f"Unsupported layout: {layout!r}")
    root = require_grootswagers_root(data_root)
    runs = {_speed_of(run): (vhdr, events) for run, vhdr, events in _runs(root, subject_id)}
    unknown = sorted(set(speeds) - set(runs))
    if unknown:
        raise FileNotFoundError(f"{subject_id}: missing speed session(s) {unknown}")

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_files = [vhdr for vhdr, _ in runs.values()] + [events for _, events in runs.values()]
    per_speed = {}
    channels: List[str] | None = None
    train_stimuli: List[str] | None = None
    test_stimuli: List[str] | None = None
    for speed in speeds:
        vhdr, events_path = runs[speed]
        signal, rows, channel_names, _ = _load_speed(
            vhdr, events_path, labels=labels, target_sr=target_sr)
        channels = channels or channel_names
        if channels != channel_names:
            raise ValueError(f"{vhdr}: channel order differs between runs")
        # One split per subject: a per-run split could place the same stimulus
        # on opposite sides of the 5 Hz and 20 Hz sessions.
        stimuli = sorted({str(row["stimulus"]) for row in rows})
        if train_stimuli is None:
            train_stimuli, test_stimuli = _stimulus_split(stimuli)
            log.info("%s: held out %d stimuli: %s", subject_id,
                     len(test_stimuli), test_stimuli)
        elif sorted(set(train_stimuli) | set(test_stimuli)) != stimuli:
            raise ValueError(
                f"{subject_id}: speed sessions disagree on the stimulus set")
        train_data, train_rows, test_data, test_rows, state = _epoch_and_split(
            signal, rows, labels=labels,
            train_stimuli=train_stimuli, test_stimuli=test_stimuli)
        whitener = _compute_mnn_whitener(train_data)
        # MNN first, then the encoder adaptation (crop/resample/z-score/pad),
        # matching the ThingsEEG LUNA contract order.
        train_adapted, valid = _adapt_epochs(
            _apply_mnn(train_data, whitener), target_sr=target_sr,
            output_samples=output_samples, luna=luna)
        test_adapted, test_valid = _adapt_epochs(
            _apply_mnn(test_data, whitener), target_sr=target_sr,
            output_samples=output_samples, luna=luna)
        if valid != test_valid:
            raise ValueError("Train/test adaptation disagree")
        per_speed[speed] = {
            "train": (train_adapted, train_rows),
            "test": (test_adapted, test_rows),
            "state": {**state, "valid_samples_before_padding": valid,
                      "encoder_zero_padding_samples": output_samples - valid,
                      "encoder_adaptation": "luna_official_zscore" if luna else "identity"},
            "whitener": whitener,
        }
        del signal

    whiteners = np.stack([per_speed[speed]["whitener"] for speed in speeds])
    session_ids = [f"ses-{speed}" for speed in speeds]
    common_state = {
        "seed": SEED,
        "fitted_split": "train",
        "class_split_seed": SEED,
        "session_ids": session_ids,
        "sessions": session_ids,
        "split_unit": "stimulus",
        "stimulus_split_seed": SEED,
        "n_train_stimuli": N_TRAIN_STIMULI,
        "n_test_stimuli": N_TEST_STIMULI,
        "train_stimuli": train_stimuli,
        "test_stimuli": test_stimuli,
        "labels": labels,
        "profile_id": profile_id,
        "n_samples": output_samples,
        "output_sampling_rate": target_sr,
        "mnn": "LedoitWolf_condition_covariance_train_only_per_speed",
        "mnn_matrix_sha256": sha256_array(whiteners),
        "mnn_matrix_sha256_by_session": {
            session: sha256_array(whiteners[index])
            for index, session in enumerate(session_ids)},
    }
    if layout == "per_session":
        for index, speed in enumerate(speeds):
            train_data, train_rows = per_speed[speed]["train"]
            _write_h5(
                output / f"{output_prefix}_{subject_id}_ses-{speed}_training_epoched.h5",
                train_data, train_rows, channels or [], subject_id, "train",
                {**common_state, **per_speed[speed]["state"]}, root, source_files,
                whiteners, profile_id, target_sr, speed)
    else:
        train_data = np.concatenate([per_speed[speed]["train"][0] for speed in speeds], axis=0)
        train_rows = [row for speed in speeds for row in per_speed[speed]["train"][1]]
        test_data = np.mean([per_speed[speed]["test"][0] for speed in speeds], axis=0)
        _write_h5(
            output / f"{output_prefix}_{subject_id}_training_epoched.h5",
            train_data, train_rows, channels or [], subject_id, "train",
            {**common_state, "speed_sessions": list(speeds)}, root, source_files,
            whiteners, profile_id, target_sr, "+".join(speeds))
    test_data = np.mean([per_speed[speed]["test"][0] for speed in speeds], axis=0)
    test_rows = per_speed[speeds[0]]["test"][1]
    _write_h5(
        output / f"{output_prefix}_{subject_id}_test_epoched.h5",
        test_data, test_rows, channels or [], subject_id, "test",
        {**common_state, "test_aggregation": "mean_by_stimulus_across_speed_sessions"},
        root, source_files, whiteners, profile_id, target_sr, "+".join(speeds))
    summary = {
        "subject_id": subject_id, "profile_id": profile_id, "labels": labels,
        "layout": layout, "speeds": list(speeds), "seed": SEED,
        "split_unit": "stimulus",
        "stimulus_split_seed": SEED,
        "n_train_stimuli": N_TRAIN_STIMULI,
        "n_test_stimuli": N_TEST_STIMULI,
        "test_stimuli": test_stimuli,
        "n_train_trials": int(train_data.shape[0]),
        "n_channels": len(channels or []), "n_samples": output_samples,
        "sampling_rate": target_sr, "source_sampling_rate": SOURCE_SR,
        "benchmark_filter_hz": list(BANDPASS_HZ),
        "input_window_sec": [T_PRE_SEC, T_POST_SEC],
        "source_files": [str(item) for item in source_files],
        "derived_inputs_used": False,
    }
    (output / f"{output_prefix}_{subject_id}_preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Grootswagers %s (%s, %s): %d train + %d test rows",
             subject_id, profile_id, layout, train_data.shape[0], test_data.shape[0])
    return int(train_data.shape[0] + test_data.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=EXPECTED_GROOTSWAGERS_ROOT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--profile_id", default=DEFAULT_PROFILE)
    parser.add_argument("--labels", choices=("stimulus", "level_a", "level_b", "level_c"),
                        default="stimulus")
    parser.add_argument("--layout", choices=("single", "per_session"), default="single")
    parser.add_argument("--output_prefix", default="grootswagers")
    parser.add_argument("--speeds", default="5hz,20hz")
    parser.add_argument("--target_sr", type=float, default=TARGET_SR)
    parser.add_argument("--output_samples", type=int, default=EPOCH_SAMPLES)
    parser.add_argument("--luna", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    process_grootswagers(
        args.subject, args.data_root, args.out_dir, profile_id=args.profile_id,
        labels=args.labels, layout=args.layout, output_prefix=args.output_prefix,
        speeds=tuple(args.speeds.split(",")), target_sr=args.target_sr,
        output_samples=args.output_samples, luna=args.luna)


if __name__ == "__main__":
    main()
