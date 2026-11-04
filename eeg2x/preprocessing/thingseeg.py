"""Raw-only THINGS-EEG preprocessing for ATM image generation.

This adapter intentionally does not import or inspect any preprocessed files
from EEG_Image_decode. It starts from the dataset's native raw_eeg_*.npy files
and writes protocol-v2 HDF5 artifacts inside EEG2X-Benchmark-v2.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np

from eeg2x.protocol import (
    StimulusRecord,
    make_stimulus_id,
    normalize_channel_names,
    resample_eeg,
    sha256_array,
)
from eeg2x.h5_schema import require_valid_h5_file
from eeg2x.rawdata import (
    require_benchmark_output,
    require_raw_thingseeg_eeg,
    require_raw_thingseeg_image,
    require_thingseeg_link,
)
from eeg2x.evaluation.registry import get_task_config
from eeg2x.encoders.luna.input import luna_input_state, prepare_luna_epochs
from eeg2x.preprocessing.ops import (
    TARGET_SR, _apply_whiten, _compute_mnn_whitener,
)
from eeg2x.preprocessing.legacy_image import (
    _find_osf_sessions, _find_things_subjects, _load_eeg_and_stim,
    _load_image_metadata, _thingseeg_image_uri,
    _write_image_protocol_metadata,
)


log = logging.getLogger("atm_preprocessing")
PROFILE_ID = "thingseeg-image-atm-v1"
ATM_PROFILE_RATES = {
    PROFILE_ID: 250,
    "thingseeg-image-reve-atm-v1": 200,
    "thingseeg-image-labram-atm-v1": 200,
    "thingseeg-image-luna-atm-v1": 256,
}
LUNA_ATM_PROFILE = "thingseeg-image-luna-atm-v1"
ATM_T_PRE = -0.2
ATM_T_POST = 1.0
ATM_BASELINE_STOP = 0.0
ATM_OUTPUT_SAMPLES = 250
ATM_REPETITION_SEED = 2000
ATM_TRAIN_REPETITIONS_PER_SESSION = 2
ATM_TEST_REPETITIONS_PER_SESSION = 20

# Exact order selected by the original project's preprocessing.py.
ATM_CHANNEL_ORDER = (
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
    "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
    "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
    "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
)


def _select_atm_channels(
        eeg: np.ndarray, channel_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    normalized = normalize_channel_names(channel_names)
    index = {name: idx for idx, name in enumerate(normalized)}
    missing = [name for name in ATM_CHANNEL_ORDER if name not in index]
    if missing:
        raise ValueError(f"ATM requires all 63 paper channels; missing {missing}")
    indices = [index[name] for name in ATM_CHANNEL_ORDER]
    return eeg[indices].astype(np.float32, copy=False), list(ATM_CHANNEL_ORDER)


def _remove_marker_events(
        samples: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts = Counter(int(value) for value in values)
    markers = {value for value, count in counts.items() if count > 50 or value > 90000}
    keep = np.asarray([int(value) not in markers for value in values], dtype=bool)
    return np.asarray(samples)[keep], np.asarray(values)[keep]


def _metadata_by_training_trigger(image_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for image_file, concept in zip(
            image_meta["train_files"], image_meta["train_concepts"]):
        concept_text = str(concept)
        prefix, _, concept_name = concept_text.partition("_")
        match = re.search(r"_(\d+)[bsn]", str(image_file))
        rows.append({
            "concept_num": int(prefix) if prefix.isdigit() else -1,
            "object": concept_name or concept_text,
            "concept": concept_text,
            "stim_number": int(match.group(1)) if match else 0,
            "image_file": str(image_file),
        })
    return rows


def _build_events(
    *,
    things_root: Path,
    split_name: str,
    image_meta: Dict[str, Any],
    event_samples: np.ndarray,
    trigger_values: np.ndarray,
    source_sr: float,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if split_name == "training":
        rows = _metadata_by_training_trigger(image_meta)
        for sample, trigger in zip(event_samples, trigger_values):
            trigger = int(trigger)
            if not 1 <= trigger <= len(rows):
                continue
            row = dict(rows[trigger - 1])
            row.update({
                "image_condition": trigger,
                "onset_sample_source": int(sample),
                "onset_sec": float(sample) / float(source_sr),
            })
            row["stimulus_uri"] = _thingseeg_image_uri(
                str(things_root), split_name, row["concept"], row["image_file"])
            require_raw_thingseeg_image(row["stimulus_uri"], things_root)
            events.append(row)
        return events

    test_files = image_meta["test_files"]
    test_concepts = image_meta["test_concepts"]
    repetition_counts: Dict[int, int] = defaultdict(int)
    for sample, trigger in zip(event_samples, trigger_values):
        trigger = int(trigger)
        if not 1 <= trigger <= len(test_concepts):
            continue
        index = trigger - 1
        concept = str(test_concepts[index])
        prefix, _, concept_name = concept.partition("_")
        image_file = str(test_files[index])
        repetition = repetition_counts[trigger]
        repetition_counts[trigger] += 1
        uri = _thingseeg_image_uri(
            str(things_root), split_name, concept, image_file)
        require_raw_thingseeg_image(uri, things_root)
        events.append({
            "image_condition": trigger,
            "concept_num": int(prefix) if prefix.isdigit() else trigger,
            "object": concept_name or concept,
            "concept": concept,
            "stim_number": 0,
            "repetition": repetition,
            "image_file": image_file,
            "stimulus_uri": uri,
            "onset_sample_source": int(sample),
            "onset_sec": float(sample) / float(source_sr),
        })
    return events


def select_paper_repetitions(
    events: Sequence[Dict[str, Any]],
    split_name: str,
    *,
    seed: int = ATM_REPETITION_SEED,
) -> List[Dict[str, Any]]:
    """Apply the released preprocessing's per-session repetition contract."""
    from sklearn.utils import shuffle

    if split_name == "training":
        maximum = ATM_TRAIN_REPETITIONS_PER_SESSION
    elif split_name == "test":
        maximum = ATM_TEST_REPETITIONS_PER_SESSION
    else:
        raise ValueError(f"Unsupported ATM split: {split_name!r}")

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[int(event["image_condition"])].append(event)
    selected: List[Dict[str, Any]] = []
    for condition in sorted(grouped):
        rows = grouped[condition]
        if len(rows) < maximum:
            raise ValueError(
                f"ATM paper preprocessing requires {maximum} repetitions for "
                f"{split_name} condition {condition}, found {len(rows)}")
        indices = shuffle(
            np.arange(len(rows)), random_state=seed, n_samples=maximum)
        selected.extend(rows[int(index)] for index in indices)
    return selected


def epoch_baseline_resample_crop(
    eeg: np.ndarray,
    event_samples: np.ndarray,
    source_sr: float,
    *,
    batch_size: int = 32,
    target_sr: int = TARGET_SR,
    output_samples: int = ATM_OUTPUT_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the ATM EEG transform and return epochs plus valid-event mask."""
    n_pre = int(round(abs(ATM_T_PRE) * source_sr))
    n_post = int(round(ATM_T_POST * source_sr))
    samples = np.asarray(event_samples, dtype=np.int64)
    valid = (samples - n_pre >= 0) & (samples + n_post <= eeg.shape[-1])
    samples = samples[valid]
    output = np.empty(
        (len(samples), eeg.shape[0], output_samples), dtype=np.float32)
    offsets = np.arange(-n_pre, n_post, dtype=np.int64)
    crop_start = int(round(abs(ATM_T_PRE) * target_sr))
    crop_stop = crop_start + output_samples

    for start in range(0, len(samples), batch_size):
        stop = min(start + batch_size, len(samples))
        indices = samples[start:stop, None] + offsets[None, :]
        # np.take returns (channels, batch, time) for this index matrix.
        chunk = np.take(eeg, indices, axis=1).transpose(1, 0, 2)
        baseline = chunk[..., :n_pre].mean(axis=-1, keepdims=True, dtype=np.float64)
        chunk = chunk.astype(np.float32, copy=False) - baseline.astype(np.float32)
        resampled = resample_eeg(chunk, source_sr, target_sr, axis=-1)
        if resampled.shape[-1] < crop_stop:
            raise RuntimeError(
                f"ATM resampled epoch has {resampled.shape[-1]} samples; "
                f"requires at least {crop_stop}")
        output[start:stop] = resampled[..., crop_start:crop_stop]
    return output, valid


def _stimulus_records(
        events: Sequence[Dict[str, Any]], alignment_source: str) -> List[StimulusRecord]:
    return [StimulusRecord(
        stimulus_id=make_stimulus_id("thingseeg", event["image_file"]),
        modality="image",
        uri=str(event["stimulus_uri"]),
        onset_sec=float(event.get("onset_sec", np.nan)),
        duration_sec=1.0,
        alignment_source=alignment_source,
        transform={
            "concept": event["object"],
            "concept_num": int(event["concept_num"]),
            "image_condition": int(event["image_condition"]),
            "stim_number": int(event.get("stim_number", 0)),
            "repetition": int(event.get("repetition", 0)),
            "baseline_sec": [ATM_T_PRE, ATM_BASELINE_STOP],
        },
    ) for event in events]


def _write_atm_h5(
    path: Path,
    *,
    epochs: np.ndarray,
    events: Sequence[Dict[str, Any]],
    channels: Sequence[str],
    subject_id: str,
    split: str,
    source_sr: float,
    whitener: np.ndarray,
    session: str,
    source_file: Path,
    repetition_seed: int,
    overwrite: bool,
    target_sr: int,
    profile_id: str,
) -> None:
    path = require_benchmark_output(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to reuse existing ATM HDF5: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    n_epochs, n_channels, n_times = epochs.shape
    string_dtype = h5py.string_dtype(encoding="utf-8")
    source_stat = source_file.stat()
    state = {
        "fitted_split": "train",
        "fit_group": f"{subject_id}:{session}",
        "mnn_matrix_sha256": sha256_array(whitener),
        "source_kind": "things_eeg2_osf_raw_numpy",
        "source_file": str(source_file),
        "source_size_bytes": int(source_stat.st_size),
        "repetition_selection": {
            "seed": int(repetition_seed),
            "per_condition_per_session": (
                ATM_TRAIN_REPETITIONS_PER_SESSION
                if split == "train" else ATM_TEST_REPETITIONS_PER_SESSION),
        },
        "derived_inputs_used": False,
        **({"luna_input_adaptation": luna_input_state()}
           if profile_id == LUNA_ATM_PROFILE else {}),
    }
    try:
        with h5py.File(temporary, "w") as h5:
            h5.create_dataset(
                "data", data=epochs, dtype=np.float32, compression="gzip",
                compression_opts=4, chunks=(1, n_channels, min(n_times, target_sr)))
            h5.create_dataset(
                "event_onset_sample",
                data=np.asarray([
                    round(event["onset_sec"] * target_sr) for event in events
                ], dtype=np.int64))
            h5.create_dataset(
                "event_onset_sample_source",
                data=np.asarray([
                    event["onset_sample_source"] for event in events
                ], dtype=np.int64))
            h5.create_dataset(
                "event_onset_sec",
                data=np.asarray([event["onset_sec"] for event in events], dtype=np.float64))
            h5.create_dataset(
                "event_label",
                data=np.asarray([event["object"] for event in events], dtype=object),
                dtype=string_dtype)
            h5.create_dataset(
                "event_object_number",
                data=np.asarray([event["concept_num"] for event in events], dtype=np.int32))
            h5.create_dataset(
                "event_stim_number",
                data=np.asarray([event.get("stim_number", 0) for event in events], dtype=np.int32))
            h5.create_dataset(
                "event_image_file",
                data=np.asarray([event["image_file"] for event in events], dtype=object),
                dtype=string_dtype)
            h5.create_dataset(
                "channel_names", data=np.asarray(channels, dtype=object), dtype=string_dtype)
            h5.create_dataset("mnn_whitener", data=whitener.astype(np.float32))
            h5.attrs["t_pre"] = 0.0
            h5.attrs["t_post"] = 1.0
            h5.attrs["source_epoch_t_pre"] = ATM_T_PRE
            h5.attrs["source_epoch_t_post"] = ATM_T_POST
            h5.attrs["baseline_t_pre"] = ATM_T_PRE
            h5.attrs["baseline_t_post"] = ATM_BASELINE_STOP
            h5.attrs["session"] = session
            h5.attrs["source_eeg_file"] = str(source_file)
            h5.attrs["source_eeg_file_size"] = int(source_stat.st_size)
            h5.attrs["derived_inputs_used"] = False
            _write_image_protocol_metadata(
                h5, dataset_id="thingseeg", subject_id=subject_id,
                source_sr=source_sr, split=split, channels=list(channels),
                n_epochs=n_epochs,
                records=_stimulus_records(events, "raw_trigger+atm_epoch"),
                preprocessing_state=state, profile_id=profile_id,
                sampling_rate=target_sr)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    require_valid_h5_file(path, get_task_config("image_generation")["h5_schema"])


def _write_concept_average(
    path: Path,
    *,
    concept_epochs: Dict[int, List[np.ndarray]],
    concept_meta: Dict[int, Dict[str, Any]],
    channels: Sequence[str],
    subject_id: str,
    source_sr: float,
    session_whiteners: Dict[str, np.ndarray],
    source_files: Sequence[Path],
    repetition_seed: int,
    overwrite: bool,
    target_sr: int,
    profile_id: str,
) -> None:
    concepts = sorted(concept_epochs)
    if not concepts:
        raise ValueError(f"No ATM test epochs collected for {subject_id}")
    epochs = np.stack([
        np.mean(concept_epochs[concept], axis=0, dtype=np.float64).astype(np.float32)
        for concept in concepts
    ])
    if profile_id == LUNA_ATM_PROFILE:
        epochs = prepare_luna_epochs(epochs)
    events = [dict(concept_meta[concept]) for concept in concepts]
    path = require_benchmark_output(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to reuse existing ATM HDF5: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    string_dtype = h5py.string_dtype(encoding="utf-8")
    session_ids = sorted(session_whiteners)
    state = {
        "fitted_split": "train",
        "fit_groups": "per_session",
        "session_ids": session_ids,
        "mnn_matrix_sha256_by_session": {
            session: sha256_array(session_whiteners[session]) for session in session_ids
        },
        "aggregation": "mean_by_concept_across_session_whitened_repetitions",
        "repetition_selection": {
            "seed": int(repetition_seed),
            "train_per_image_condition_per_session": (
                ATM_TRAIN_REPETITIONS_PER_SESSION),
            "test_per_concept_per_session": ATM_TEST_REPETITIONS_PER_SESSION,
        },
        "source_kind": "things_eeg2_osf_raw_numpy",
        "source_files": [str(path) for path in source_files],
        "derived_inputs_used": False,
        **({"luna_input_adaptation": luna_input_state()}
           if profile_id == LUNA_ATM_PROFILE else {}),
    }
    try:
        with h5py.File(temporary, "w") as h5:
            h5.create_dataset("data", data=epochs, dtype=np.float32)
            h5.create_dataset(
                "event_label", data=np.asarray([
                    event["object"] for event in events], dtype=object),
                dtype=string_dtype)
            h5.create_dataset(
                "event_object_number", data=np.asarray(concepts, dtype=np.int32))
            h5.create_dataset(
                "event_stim_number", data=np.asarray([
                    event.get("stim_number", 0) for event in events], dtype=np.int32))
            h5.create_dataset(
                "event_image_file", data=np.asarray([
                    event["image_file"] for event in events], dtype=object),
                dtype=string_dtype)
            h5.create_dataset(
                "channel_names", data=np.asarray(channels, dtype=object),
                dtype=string_dtype)
            h5.create_dataset(
                "mnn_whiteners", data=np.stack([
                    session_whiteners[session] for session in session_ids]),
                dtype=np.float32)
            h5.attrs["t_pre"] = 0.0
            h5.attrs["t_post"] = 1.0
            h5.attrs["source_epoch_t_pre"] = ATM_T_PRE
            h5.attrs["source_epoch_t_post"] = ATM_T_POST
            h5.attrs["baseline_t_pre"] = ATM_T_PRE
            h5.attrs["baseline_t_post"] = ATM_BASELINE_STOP
            h5.attrs["averaged"] = True
            h5.attrs["derived_inputs_used"] = False
            records = [StimulusRecord(
                stimulus_id=make_stimulus_id("thingseeg", event["image_file"]),
                modality="image", uri=str(event["stimulus_uri"]),
                duration_sec=1.0, alignment_source="concept_average",
                transform={
                    "aggregation": "mean_by_concept_across_sessions",
                    "concept": event["object"],
                    "concept_num": int(event["concept_num"]),
                    "n_repetitions": len(concept_epochs[int(event["concept_num"])]),
                    "baseline_sec": [ATM_T_PRE, ATM_BASELINE_STOP],
                },
            ) for event in events]
            _write_image_protocol_metadata(
                h5, dataset_id="thingseeg", subject_id=subject_id,
                source_sr=source_sr, split="test", channels=list(channels),
                n_epochs=len(epochs), records=records,
                preprocessing_state=state, profile_id=profile_id,
                sampling_rate=target_sr)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    require_valid_h5_file(path, get_task_config("image_generation")["h5_schema"])


def preprocess_subject(
    subject_id: str,
    out_dir: str | Path,
    *,
    things_root: str | Path | None = None,
    sessions: Sequence[str] | None = None,
    epoch_batch_size: int = 32,
    seed: int = ATM_REPETITION_SEED,
    overwrite: bool = False,
    target_sr: int = TARGET_SR,
    profile_id: str = PROFILE_ID,
) -> Dict[str, Any]:
    expected_rate = ATM_PROFILE_RATES.get(profile_id)
    if expected_rate is None:
        raise ValueError(f"Unsupported ATM preprocessing profile: {profile_id!r}")
    if target_sr != expected_rate:
        raise ValueError(
            f"Profile {profile_id!r} requires {expected_rate} Hz, got {target_sr}")
    root = Path(things_root) if things_root is not None else require_thingseeg_link()
    root = root.absolute()
    # Formal ATM runs must use the repository link, not the resolved target path.
    if root != require_thingseeg_link():
        raise ValueError(f"ATM preprocessing must use {require_thingseeg_link()}, got {root}")
    output = require_benchmark_output(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    output_samples = int(round(target_sr * (ATM_T_POST - ATM_BASELINE_STOP)))
    if target_sr <= 0 or output_samples <= 0:
        raise ValueError(f"Invalid ATM target sampling rate: {target_sr}")
    selected_sessions = list(sessions or _find_osf_sessions(str(root), subject_id))
    if not selected_sessions:
        raise ValueError(f"No raw ThingsEEG sessions found for {subject_id}")

    image_meta = _load_image_metadata(str(root))
    concept_epochs: Dict[int, List[np.ndarray]] = defaultdict(list)
    concept_meta: Dict[int, Dict[str, Any]] = {}
    session_whiteners: Dict[str, np.ndarray] = {}
    test_source_files: List[Path] = []
    source_sr_seen: float | None = None
    channel_names_seen: List[str] | None = None
    written: List[str] = []
    split_counts: Dict[str, int] = defaultdict(int)

    for session in selected_sessions:
        whitener: np.ndarray | None = None
        for split_name in ("training", "test"):
            source_file = root / "Raw_EEG_data" / subject_id / subject_id / session / (
                f"raw_eeg_{split_name}.npy")
            require_raw_thingseeg_eeg(source_file, root)
            log.info("%s %s %s: loading %s", subject_id, session, split_name, source_file)
            eeg, source_sr, raw_channels, samples, triggers = _load_eeg_and_stim(
                str(source_file))
            eeg, channels = _select_atm_channels(eeg, raw_channels)
            if source_sr_seen is None:
                source_sr_seen = source_sr
                channel_names_seen = channels
            elif source_sr != source_sr_seen or channels != channel_names_seen:
                raise ValueError(f"Inconsistent raw EEG clocks/channels for {subject_id}")
            samples, triggers = _remove_marker_events(samples, triggers)
            events = _build_events(
                things_root=root, split_name=split_name, image_meta=image_meta,
                event_samples=samples, trigger_values=triggers, source_sr=source_sr)
            events = select_paper_repetitions(
                events, split_name, seed=seed)
            epochs, valid = epoch_baseline_resample_crop(
                eeg, np.asarray([
                    event["onset_sample_source"] for event in events], dtype=np.int64),
                source_sr, batch_size=epoch_batch_size,
                target_sr=target_sr, output_samples=output_samples)
            del eeg
            if not valid.all():
                events = [event for event, keep in zip(events, valid) if keep]
            if len(events) != len(epochs) or not events:
                raise ValueError(
                    f"ATM event/epoch mismatch for {subject_id} {session} {split_name}")

            if split_name == "training":
                # The paper estimates one covariance per native image condition.
                image_conditions = np.asarray([
                    event["image_condition"] for event in events], dtype=np.int32)
                whitener = np.asarray(
                    np.real_if_close(_compute_mnn_whitener(epochs, image_conditions)),
                    dtype=np.float32)
                session_whiteners[session] = whitener
            elif whitener is None:
                raise RuntimeError(f"Test MNN has no training state for {session}")
            epochs = _apply_whiten(epochs, whitener).astype(np.float32)
            if not np.isfinite(epochs).all():
                raise ValueError(f"Non-finite ATM epochs for {subject_id} {session}")

            aggregation_epochs = epochs
            if split_name == "test":
                test_source_files.append(source_file)
                for index, event in enumerate(events):
                    concept = int(event["concept_num"])
                    concept_epochs[concept].append(aggregation_epochs[index])
                    concept_meta.setdefault(concept, event)
            if profile_id == LUNA_ATM_PROFILE:
                epochs = prepare_luna_epochs(aggregation_epochs)

            session_label = session.replace("-", "")
            path = output / (
                f"thingseeg_atm_{subject_id}_{session_label}_{split_name}_epoched.h5")
            _write_atm_h5(
                path, epochs=epochs, events=events, channels=channels,
                subject_id=subject_id,
                split="train" if split_name == "training" else "test",
                source_sr=source_sr, whitener=whitener, session=session,
                source_file=source_file, repetition_seed=seed,
                overwrite=overwrite,
                target_sr=target_sr, profile_id=profile_id)
            written.append(str(path))
            split_counts[split_name] += len(epochs)
            log.info("Saved %s with %d epochs", path, len(epochs))
            del epochs

    final_test = output / f"thingseeg_atm_{subject_id}_test_epoched.h5"
    _write_concept_average(
        final_test, concept_epochs=concept_epochs, concept_meta=concept_meta,
        channels=channel_names_seen or [], subject_id=subject_id,
        source_sr=float(source_sr_seen), session_whiteners=session_whiteners,
        source_files=test_source_files, repetition_seed=seed,
        overwrite=overwrite,
        target_sr=target_sr, profile_id=profile_id)
    written.append(str(final_test))
    summary = {
        "subject_id": subject_id,
        "profile_id": profile_id,
        "target_sampling_rate": target_sr,
        "output_samples": 280 if profile_id == LUNA_ATM_PROFILE else output_samples,
        "sessions": selected_sessions,
        "split_epochs": dict(split_counts),
        "test_concepts": len(concept_epochs),
        "repetition_seed": int(seed),
        "files": written,
        "derived_inputs_used": False,
    }
    summary_path = output / f"thingseeg_atm_{subject_id}_preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="h5/atm")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--sessions", nargs="*", default=None)
    parser.add_argument("--epoch-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=ATM_REPETITION_SEED)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-sr", type=int, default=TARGET_SR)
    parser.add_argument("--profile-id", default=PROFILE_ID)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    root = require_thingseeg_link()
    subjects = args.subjects or _find_things_subjects(str(root))
    if not subjects:
        raise SystemExit("No ThingsEEG subjects found")
    for subject in subjects:
        preprocess_subject(
            subject, args.out_dir, things_root=root, sessions=args.sessions,
            epoch_batch_size=args.epoch_batch_size, seed=args.seed,
            overwrite=args.overwrite,
            target_sr=args.target_sr, profile_id=args.profile_id)


if __name__ == "__main__":
    main()
