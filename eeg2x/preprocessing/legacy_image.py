"""ThingsEEG 的旧版图像预处理实现（从 v2 loader 迁入）。

这里只保留 ThingsEEG 专用的 OSF 读取、通道表与图像协议元数据；通用原语在
``ops.py``，数据集入口在 ``thingseeg.py``。NOD 的旧分支与 CLI 入口已经移除，
它们不再被 v3 调用。
"""
from __future__ import annotations

import csv
import json
import logging
import os
import re
import warnings
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import scipy

from eeg2x.protocol import (
    StimulusRecord, normalize_channel_name, normalize_channel_names,
    write_common_h5_metadata, write_stimulus_manifest,
)
from eeg2x.profiles import get_preprocessing_profile
from eeg2x.rawdata import (
    require_benchmark_output, require_raw_thingseeg_eeg,
    require_raw_thingseeg_image, require_thingseeg_link,
)
from eeg2x.encoders.luna.input import luna_input_state, prepare_luna_epochs

from .ops import (
    TARGET_SR, _apply_whiten, _compute_mnn_whitener, _epoch_around_events,
    _keep_all_eeg, _norm_ch, _resample, _valid_event_mask,
)

DEFAULT_T_PRE = 0.0
DEFAULT_T_POST = 1.0
THINGSEEG_RETRIEVAL_PROFILE_RATES = {
    "thingseeg-image-atm-v1": 250,
    "thingseeg-image-labram-atm-v1": 200,
    "thingseeg-image-reve-atm-v1": 200,
    "thingseeg-image-luna-atm-v1": 256,
}
LUNA_NICE_PROFILE = "thingseeg-image-luna-nice-v1"


def _write_image_protocol_metadata(h5, *, dataset_id: str, subject_id: str,
                                   source_sr: float, split: str,
                                   channels: List[str], n_epochs: int,
                                   records: List[StimulusRecord],
                                   preprocessing_state: Optional[Dict[str, Any]] = None,
                                   profile_id: Optional[str] = None,
                                   sampling_rate: float = TARGET_SR) -> None:
    """Write the v2 contract while retaining legacy image-task fields."""
    channels = normalize_channel_names(channels)
    profile_id = profile_id or (
        "thingseeg-image-nice-v1" if dataset_id == "thingseeg"
        else "nod-image-nice-v1")
    write_common_h5_metadata(
        h5, dataset_id=dataset_id, task_id="eeg_to_image",
        subject_id=subject_id, sampling_rate=sampling_rate,
        source_sampling_rate=float(source_sr), n_channels=len(channels),
        n_epochs=n_epochs, split=split, epoch_type="image",
        eeg_units="dimensionless_whitened",
        profile=get_preprocessing_profile(profile_id),
        preprocessing_state=preprocessing_state,
    )
    write_stimulus_manifest(h5, records)

def _load_eeg_and_stim(npy_path: str) -> Tuple[np.ndarray, float, List[str], np.ndarray, np.ndarray]:
    """Load EEG data and stim channel from THINGS-EEG OSF .npy in one pass.

    Returns (eeg_data_32, sr, channel_names, stim_events, stim_values).
    """
    d = np.load(npy_path, allow_pickle=True).item()
    raw = d.pop("raw_eeg_data")       # (n_channels, n_samples) float64
    sr = float(d.pop("sfreq"))
    ch_names = d.pop("ch_names")
    ch_types = d.pop("ch_types")
    del d

    # Extract stim events and trigger values
    stim_idx = [i for i, t in enumerate(ch_types) if t == "stim"]
    if stim_idx:
        stim = raw[stim_idx[0]]
        rise = np.where(np.diff((stim > 0).astype(np.int8)) > 0)[0] + 1
        stim_events = rise
        stim_values = np.array([int(stim[e]) for e in rise])
    else:
        stim_events = np.array([], dtype=int)
        stim_values = np.array([], dtype=int)

    # Keep only EEG channels
    eeg_idx = [i for i, t in enumerate(ch_types) if t == "eeg"]
    if not eeg_idx:
        raise ValueError("No EEG channels found (only stim?)")
    data = raw[eeg_idx].astype(np.float32, copy=False)
    ch_names = [ch_names[i] for i in eeg_idx]

    log.info(f"    Loaded .npy: {len(eeg_idx)} EEG ch, {data.shape[1]} samples, "
             f"{len(stim_events)} stim events @ {sr:.0f} Hz")
    return data, sr, ch_names, stim_events, stim_values

def _thingseeg_image_uri(things_root: str, split_name: str,
                         concept: str, image_file: str) -> str:
    """Resolve a THINGS image path while preserving its native stimulus ID."""
    candidate = Path(str(image_file))
    if candidate.is_absolute():
        return str(candidate)
    split_dir = "training_images" if split_name == "training" else "test_images"
    roots = [
        Path(things_root) / "Image_set" / split_dir / split_dir,
        Path(things_root) / "Image_set" / split_dir,
        Path(things_root) / "Image_set",
    ]
    candidates = []
    for root in roots:
        candidates.extend((root / concept / candidate, root / candidate))
    for path in candidates:
        if path.is_file():
            # Preserve the lexical ThingsEEG symlink in the manifest. This keeps
            # the benchmark's declared dataset entry point machine-auditable.
            return str(path.absolute())
    raise FileNotFoundError(
        f"Cannot resolve THINGS stimulus {image_file!r}; checked: "
        + ", ".join(str(path) for path in candidates)
    )

def _load_image_metadata(things_root: str) -> Dict[str, Any]:
    """Load image_metadata.npy and return file lists."""
    meta_path = os.path.join(things_root, "Image_set", "image_metadata.npy")
    meta = np.load(meta_path, allow_pickle=True).item()
    return {
        "train_files": meta["train_img_files"],       # list of filenames
        "train_concepts": meta["train_img_concepts"], # list of concept IDs
        "test_files": meta["test_img_files"],
        "test_concepts": meta["test_img_concepts"],
    }

def _find_osf_sessions(things_root: str, sub_id: str) -> List[str]:
    """Find available session directories for a subject."""
    sub_dir = os.path.join(things_root, "Raw_EEG_data", sub_id, sub_id)
    if not os.path.isdir(sub_dir):
        return []
    sessions = sorted([
        d for d in os.listdir(sub_dir)
        if d.startswith("ses-") and os.path.isdir(os.path.join(sub_dir, d))
    ])
    return sessions

def _resolve_things_root(data_root: str) -> Optional[str]:
    """Accept either the THINGS-EEG directory or its parent data directory."""
    candidates = [Path(data_root), Path(data_root) / "THINGS-EEG"]
    for candidate in candidates:
        if (candidate / "Raw_EEG_data").is_dir() and (
                candidate / "Image_set").is_dir():
            return str(candidate)
    return None

def _find_things_subjects(things_root: str) -> List[str]:
    """Find OSF subjects with the nested Raw_EEG_data/sub-*/sub-* layout."""
    raw_root = Path(things_root) / "Raw_EEG_data"
    if not raw_root.is_dir():
        return []
    return sorted(
        path.name for path in raw_root.iterdir()
        if path.name.startswith("sub-")
        and (path / path.name).is_dir()
        and _find_osf_sessions(things_root, path.name)
    )

def _process_thingseeg_osf(
    data_root: str,
    sub_id: str,
    out_dir: str,
    sessions: List[str] = None,
    *,
    target_sr: int = TARGET_SR,
    profile_id: str = "thingseeg-image-nice-v1",
) -> int:
    """Process one subject from OSF THINGS-EEG.

    Returns number of epochs written.
    """
    expected_rate = THINGSEEG_RETRIEVAL_PROFILE_RATES.get(profile_id)
    if expected_rate is None:
        raise ValueError(f"Unsupported ThingsEEG retrieval profile: {profile_id!r}")
    if target_sr != expected_rate:
        raise ValueError(
            f"Profile {profile_id!r} requires {expected_rate} Hz, got {target_sr}")
    things_root = str(Path(data_root).absolute())
    if Path(things_root) != require_thingseeg_link():
        raise ValueError(
            f"THINGS-EEG must be read through {require_thingseeg_link()}, "
            f"got {things_root}"
        )
    out_dir = str(require_benchmark_output(out_dir))
    sub_dir = os.path.join(things_root, "Raw_EEG_data", sub_id, sub_id)

    if sessions is None:
        sessions = _find_osf_sessions(things_root, sub_id)

    # Load image metadata
    image_meta = _load_image_metadata(things_root)
    # Training images across all sessions
    all_train_files = image_meta["train_files"]
    all_train_concepts = image_meta["train_concepts"]

    # Split across sessions
    n_sessions = len(sessions)
    if n_sessions == 0:
        log.warning(f"  No sessions for {sub_id}")
        return 0

    images_per_session = len(all_train_files) // n_sessions
    log.info(f"  {sub_id}: {n_sessions} sessions, "
             f"{images_per_session} images/session")

    total_epochs = 0
    n_sessions = len(sessions)
    subject_source_sr: Optional[float] = None
    subject_channels: Optional[List[str]] = None
    session_whiteners: Dict[str, np.ndarray] = {}
    session_train_sources: Dict[str, str] = {}
    test_source_files: List[str] = []

    # Build metadata list in TRIGGER order (triggers 1..N map to metadata[trigger-1])
    all_train_files = image_meta["train_files"]
    all_train_concepts = image_meta["train_concepts"]
    meta_by_trigger = []  # index = trigger_value - 1 → (cnum, snum, fname, cstr)
    for fname, cstr in zip(all_train_files, all_train_concepts):
        parts = cstr.split("_", 1)
        cnum = int(parts[0]) if parts[0].isdigit() else -1
        stim_m = re.search(r"_(\d+)[bsn]", fname)
        snum = int(stim_m.group(1)) if stim_m else 0
        meta_by_trigger.append((cnum, snum, fname, cstr))
    # Assert: len(meta_by_trigger) should be 16540, triggers are 1..16540

    from collections import defaultdict
    concept_test_epochs = defaultdict(list)  # concept_num → [epochs]
    concept_test_meta = {}                    # concept_num → metadata

    for si, ses in enumerate(sessions):

        session_sigma_inv = None

        # Both training and test npy
        for split_name in ("training", "test"):
            npy_path = os.path.join(
                things_root, "Raw_EEG_data", sub_id, sub_id, ses,
                f"raw_eeg_{split_name}.npy"
            )
            if not os.path.exists(npy_path):
                continue
            require_raw_thingseeg_eeg(npy_path, things_root)
            source_stat = os.stat(npy_path)

            log.info(f"    {sub_id} {ses} {split_name}")

            # Load EEG and stim events
            eeg_data, orig_sr, ch_names, stim_events, trigger_vals = _load_eeg_and_stim(npy_path)
            normalized_channels = normalize_channel_names(ch_names)
            if subject_source_sr is None:
                subject_source_sr = float(orig_sr)
                subject_channels = normalized_channels
            elif not np.isclose(subject_source_sr, orig_sr):
                raise ValueError(
                    f"{sub_id} has inconsistent source sampling rates: "
                    f"{subject_source_sr} and {orig_sr}"
                )
            elif subject_channels != normalized_channels:
                raise ValueError(f"{sub_id} has inconsistent channel order across sessions")
            n_stim = len(stim_events)
            log.info(f"      {n_stim} stim events")
            # Keep all EEG channels (NICE: 63 channels)
            eeg_all = _keep_all_eeg(eeg_data, ch_names)
            eeg_resampled = _resample(eeg_all, orig_sr, target_sr)
            new_sr = target_sr
            sr_ratio = new_sr / orig_sr
            # Remove marker-only events
            val_counts = Counter(trigger_vals)
            marker_vals = {v for v, c in val_counts.items() if c > 50 or v > 90000}
            event_mask = np.array([v not in marker_vals for v in trigger_vals])
            stim_events = stim_events[event_mask]
            trigger_vals = trigger_vals[event_mask]
            n_clean = len(stim_events)
            log.info(f"      {n_clean} image events after removing markers "
                     f"({n_stim - n_clean} removed)")

            # Build event metadata — trigger value IS the image condition number
            events = []
            if split_name == "training":
                # Training: triggers 1..16540 map to meta_by_trigger[trigger-1]
                for i in range(n_clean):
                    tv = int(trigger_vals[i])
                    if 1 <= tv <= len(meta_by_trigger):
                        cnum, snum, img_file, concept_str = meta_by_trigger[tv - 1]
                        concept_name = concept_str.split("_", 1)[1] if "_" in concept_str else concept_str
                        events.append({
                            "onset_sample": int(stim_events[i]),
                            "onset_sample_250": int(round(stim_events[i] * sr_ratio)),
                            "onset_sec": stim_events[i] / orig_sr,
                            "object": concept_name,
                            "concept_num": cnum,
                            "stim_number": snum,
                            "image_file": img_file,
                            "stimulus_uri": _thingseeg_image_uri(
                                things_root, split_name, concept_str, img_file),
                        })
                log.info(f"      Training: {len(events)} epochs, max trigger={int(trigger_vals.max())}")

            else:
                # Test: triggers 1..200 cycle, map to test_concepts[trigger-1]
                test_files = image_meta.get("test_files", [])
                test_concepts = image_meta.get("test_concepts", [])
                rep_counts = {}
                for i in range(n_clean):
                    tv = int(trigger_vals[i])
                    if 1 <= tv <= len(test_concepts):
                        ci = tv - 1
                        rep_idx = rep_counts.get(tv, 0)
                        rep_counts[tv] = rep_idx + 1
                        concept_str = test_concepts[ci]
                        img_file = test_files[ci] if ci < len(test_files) else f"test_{ci:05d}.jpg"
                        concept_num = int(concept_str.split("_", 1)[0]) if concept_str.split("_", 1)[0].isdigit() else tv
                        concept_name = concept_str.split("_", 1)[1] if "_" in concept_str else concept_str
                        events.append({
                            "onset_sample": int(stim_events[i]),
                            "onset_sample_250": int(round(stim_events[i] * sr_ratio)),
                            "onset_sec": stim_events[i] / orig_sr,
                            "object": concept_name,
                            "concept_num": concept_num,
                            "stim_number": rep_idx,
                            "image_file": img_file,
                            "stimulus_uri": _thingseeg_image_uri(
                                things_root, split_name, concept_str, img_file),
                        })
                n_unique = len(set(e["concept_num"] for e in events))
                avg_rep = len(events) // max(n_unique, 1)
                log.info(f"      Test: {len(events)} epochs, {n_unique} concepts, ~{avg_rep} reps/concept")
                events.sort(key=lambda e: (e["concept_num"], e["stim_number"]))

            for event in events:
                require_raw_thingseeg_image(event["stimulus_uri"], things_root)

            # Epoch
            e_samples = np.array([e["onset_sample_250"] for e in events], dtype=int)
            valid_mask = _valid_event_mask(
                eeg_resampled.shape[1], e_samples,
                DEFAULT_T_PRE, DEFAULT_T_POST, new_sr)
            if not valid_mask.all():
                n_dropped = int((~valid_mask).sum())
                events = [event for event, keep in zip(events, valid_mask) if keep]
                e_samples = e_samples[valid_mask]
                log.warning(f"      Dropped {n_dropped} boundary events with their metadata")
            epochs = _epoch_around_events(
                eeg_resampled, e_samples, DEFAULT_T_PRE, DEFAULT_T_POST, new_sr)
            log.info(f"      {epochs.shape[0]} epochs × {epochs.shape[2]} samples")
            if not len(events):
                log.warning("      No valid image epochs remain; skip this run")
                continue

            # MNN whitening (NICE §4.1): compute from training, apply to both
            cnums = np.array([e["concept_num"] for e in events], dtype=int)
            if split_name == "training":
                session_sigma_inv = _compute_mnn_whitener(epochs, cnums)
                session_whiteners[ses] = np.asarray(session_sigma_inv, dtype=np.float32)
                session_train_sources[ses] = npy_path
                epochs = _apply_whiten(epochs, session_sigma_inv)
                log.info(f"      MNN: computed from training, applied ({session_sigma_inv.shape})")
            elif session_sigma_inv is not None:
                epochs = _apply_whiten(epochs, session_sigma_inv)
                log.info(f"      MNN: applied (from training)")
            else:
                raise ValueError(
                    f"{sub_id} {ses}: test data requires the same session's "
                    "training data to fit the declared MNN whitener"
                )

            aggregation_epochs = epochs
            if split_name == "test":
                test_source_files.append(npy_path)
                for i, ev in enumerate(events):
                    concept_test_epochs[ev["concept_num"]].append(
                        aggregation_epochs[i])
                    if ev["concept_num"] not in concept_test_meta:
                        concept_test_meta[ev["concept_num"]] = {
                            "object": ev["object"],
                            "stim_number": ev.get("stim_number", 0),
                            "image_file": ev.get("image_file", ""),
                            "stimulus_uri": ev.get("stimulus_uri", ""),
                        }
            if profile_id == LUNA_NICE_PROFILE:
                epochs = prepare_luna_epochs(aggregation_epochs)

            ses_label = ses.replace("-", "")
            out_path = os.path.join(
                out_dir,
                f"thingseeg_{sub_id}_{ses_label}_{split_name}_epoched.h5"
            )

            # Write h5
            n_ep, n_ch, n_t = epochs.shape
            with h5py.File(out_path, "w") as h5:
                h5.create_dataset("data", data=epochs, dtype=np.float32,
                                  compression="gzip", compression_opts=4,
                                  chunks=(1, n_ch, min(n_t, 256)))

                onset_samples = np.array([e["onset_sample_250"] for e in events], dtype=np.int64)
                onset_sec = np.array([e["onset_sec"] for e in events], dtype=np.float32)
                h5.create_dataset("event_onset_sample", data=onset_samples)
                h5.create_dataset("event_onset_sec", data=onset_sec)

                dt_str = h5py.string_dtype(encoding="utf-8")
                labels = np.array([e["object"] for e in events], dtype=object)
                h5.create_dataset("event_label", data=labels, dtype=dt_str)

                obj_nums = np.array([e.get("concept_num", -1) for e in events], dtype=int)
                stim_nums = np.array([e.get("stim_number", -1) for e in events], dtype=int)
                h5.create_dataset("event_object_number", data=obj_nums)
                h5.create_dataset("event_stim_number", data=stim_nums)

                img_files = np.array([e.get("image_file", "") for e in events], dtype=object)
                h5.create_dataset("event_image_file", data=img_files, dtype=dt_str)

                # Save actual channel names used
                ch_names = normalize_channel_names(ch_names)
                h5.create_dataset("channel_names",
                                  data=np.array(ch_names, dtype="S"), dtype=dt_str)
                h5.create_dataset(
                    "mnn_whitener", data=np.asarray(session_sigma_inv, dtype=np.float32))

                h5.attrs["dataset"] = "thingseeg"
                h5.attrs["subject_id"] = sub_id
                h5.attrs["sampling_rate"] = float(new_sr)
                h5.attrs["t_pre"] = DEFAULT_T_PRE
                h5.attrs["t_post"] = DEFAULT_T_POST
                h5.attrs["n_channels"] = n_ch
                h5.attrs["n_epochs"] = n_ep
                h5.attrs["channel_set"] = "all_eeg"
                h5.attrs["mnn_applied"] = (session_sigma_inv is not None)
                h5.attrs["session"] = ses
                h5.attrs["split"] = split_name
                h5.attrs["source_eeg_file"] = npy_path
                h5.attrs["source_eeg_file_size"] = int(source_stat.st_size)
                h5.attrs["mnn_fit_source_eeg_file"] = session_train_sources[ses]
                h5.attrs["derived_inputs_used"] = False
                records = [StimulusRecord(
                    stimulus_id=make_stimulus_id("thingseeg", ev.get("image_file") or f"trigger-{i}"),
                    modality="image", uri=str(ev.get("stimulus_uri", "")),
                    onset_sec=float(ev.get("onset_sec", np.nan)),
                    duration_sec=DEFAULT_T_POST - DEFAULT_T_PRE,
                    alignment_source="trigger_channel",
                    transform={"concept": ev.get("object", ""), "concept_num": ev.get("concept_num", -1),
                               "stim_number": ev.get("stim_number", -1)},
                ) for i, ev in enumerate(events)]
                _write_image_protocol_metadata(
                    h5, dataset_id="thingseeg", subject_id=sub_id,
                    source_sr=orig_sr, split=split_name, channels=ch_names,
                    n_epochs=n_ep, records=records,
                    preprocessing_state={
                        "fitted_split": "train",
                        "fit_group": f"{sub_id}:{ses}",
                        "mnn_matrix_sha256": sha256_array(
                            np.asarray(session_sigma_inv, dtype=np.float32)),
                        "source_kind": "things_eeg2_osf_raw_numpy",
                        "source_file": npy_path,
                        "source_size_bytes": int(source_stat.st_size),
                        "mnn_fit_source_file": session_train_sources[ses],
                        "derived_inputs_used": False,
                        **({"luna_input_adaptation": luna_input_state()}
                           if profile_id == LUNA_NICE_PROFILE else {}),
                    }, profile_id=profile_id, sampling_rate=new_sr)

            total_epochs += n_ep
            require_valid_h5_file(
                out_path, get_task_config("image_retrieval")["h5_schema"])
            log.info(f"      Saved → {out_path}")

    # ---- Concept-level test h5: average all reps per concept ----
    if concept_test_epochs:
        avg_concepts = sorted(concept_test_epochs.keys())
        first_epoch = concept_test_epochs[avg_concepts[0]][0]
        avg_data = np.zeros((len(avg_concepts), *first_epoch.shape), dtype=np.float32)
        avg_objnums = np.zeros(len(avg_concepts), dtype=int)
        avg_labels = np.empty(len(avg_concepts), dtype=object)
        avg_stimnums = np.zeros(len(avg_concepts), dtype=int)
        for i, cn in enumerate(avg_concepts):
            avg_data[i] = np.mean(concept_test_epochs[cn], axis=0).astype(np.float32)
            avg_objnums[i] = cn
            avg_labels[i] = concept_test_meta[cn]["object"]
            avg_stimnums[i] = concept_test_meta[cn]["stim_number"]
        if profile_id == LUNA_NICE_PROFILE:
            avg_data = prepare_luna_epochs(avg_data)

        out_path = os.path.join(out_dir, f"thingseeg_{sub_id}_test_epoched.h5")
        with h5py.File(out_path, "w") as h5:
            h5.create_dataset("data", data=avg_data, dtype=np.float32)
            h5.create_dataset("event_object_number", data=avg_objnums)
            dt_str = h5py.string_dtype(encoding="utf-8")
            h5.create_dataset("event_label", data=avg_labels, dtype=dt_str)
            h5.create_dataset("event_stim_number", data=avg_stimnums)
            avg_img_files = np.array(
                [concept_test_meta[cn].get("image_file", "") for cn in avg_concepts],
                dtype=object,
            )
            h5.create_dataset("event_image_file", data=avg_img_files, dtype=dt_str)
            h5.create_dataset("channel_names",
                              data=np.array(subject_channels, dtype=object), dtype=dt_str)
            ordered_sessions = sorted(session_whiteners)
            h5.create_dataset(
                "mnn_whiteners",
                data=np.stack([session_whiteners[ses] for ses in ordered_sessions]),
                dtype=np.float32,
            )
            h5.attrs["dataset"] = "thingseeg"
            h5.attrs["subject_id"] = sub_id
            h5.attrs["sampling_rate"] = float(target_sr)
            h5.attrs["n_channels"] = avg_data.shape[1]
            h5.attrs["n_epochs"] = len(avg_concepts)
            h5.attrs["t_pre"] = DEFAULT_T_PRE
            h5.attrs["t_post"] = DEFAULT_T_POST
            h5.attrs["channel_set"] = "all_eeg"
            h5.attrs["mnn_applied"] = True
            h5.attrs["split"] = "test"
            h5.attrs["averaged"] = True
            h5.attrs["source_eeg_files_json"] = json.dumps(test_source_files)
            h5.attrs["mnn_fit_source_eeg_files_json"] = json.dumps([
                session_train_sources[ses] for ses in ordered_sessions
            ])
            h5.attrs["derived_inputs_used"] = False
            records = [StimulusRecord(
                stimulus_id=make_stimulus_id(
                    "thingseeg", concept_test_meta[cn].get("image_file") or f"concept-{cn}"),
                modality="image",
                uri=str(concept_test_meta[cn].get("stimulus_uri", "")),
                duration_sec=DEFAULT_T_POST - DEFAULT_T_PRE,
                alignment_source="concept_average",
                transform={
                    "aggregation": "mean_over_repetitions",
                    "concept": concept_test_meta[cn]["object"],
                    "concept_num": int(cn),
                    "n_repetitions": len(concept_test_epochs[cn]),
                },
            ) for cn in avg_concepts]
            _write_image_protocol_metadata(
                h5, dataset_id="thingseeg", subject_id=sub_id,
                source_sr=float(subject_source_sr), split="test",
                channels=subject_channels, n_epochs=len(avg_concepts), records=records,
                preprocessing_state={
                    "fitted_split": "train",
                    "fit_groups": "per_session",
                    "session_ids": ordered_sessions,
                    "mnn_matrix_sha256_by_session": {
                        ses: sha256_array(session_whiteners[ses])
                        for ses in ordered_sessions
                    },
                    "aggregation": "mean_over_session_whitened_repetitions",
                    "source_kind": "things_eeg2_osf_raw_numpy",
                    "source_files": test_source_files,
                    "mnn_fit_source_files": [
                        session_train_sources[ses] for ses in ordered_sessions
                    ],
                    "derived_inputs_used": False,
                    **({"luna_input_adaptation": luna_input_state()}
                       if profile_id == LUNA_NICE_PROFILE else {}),
                }, profile_id=profile_id, sampling_rate=target_sr)
        require_valid_h5_file(
            out_path, get_task_config("image_retrieval")["h5_schema"])
        n_reps = sum(len(v) for v in concept_test_epochs.values()) // len(avg_concepts)
        log.info(f"  Concept-level test h5: {len(avg_concepts)} concepts, "
                 f"~{n_reps}× reps averaged → {out_path}")

    return total_epochs
