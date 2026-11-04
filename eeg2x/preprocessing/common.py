"""
EEG-to-Image Multimodal Benchmark — Standardized Data Loader
=============================================================

Converts raw EEG datasets (THINGS-EEG from OSF, NOD-EEG) into unified HDF5:
  - Resampled to 250 Hz
  - Common 32-channel subset (extended 10-20, present in both datasets)
  - Per-trial epoch extraction aligned to stimulus onsets
  - Consistent metadata schema

Datasets:
  THINGS-EEG — Grootswagers et al. 2022
               OSF repository (https://osf.io/hd6zk/)
               RSVP image streams @ 10 Hz, 63+1 ch, 1000 Hz, preprocessed .npy
  NOD-EEG    — Zhang et al. 2025 (OpenNeuro ds005811)
               ImageNet passive viewing, 62+2 ch, 1000 Hz, EEGLAB .set

Usage:
  python eeg2image_benchmark_loader.py --data_root /path/to/THINGS-EEG \\
                                       --out_dir ./h5_output \\
                                       --dataset thingseeg

Dependencies: numpy, scipy, h5py, mne
"""

from __future__ import annotations

import os
import re
import json
import csv
import logging
import zipfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

import numpy as np
import h5py
import scipy

from eeg2x.protocol import (normalize_channel_name, normalize_channel_names,
                                resample_eeg, seconds_to_sample,
                                StimulusRecord, make_stimulus_id,
                                sha256_array, write_common_h5_metadata,
                                write_stimulus_manifest)
from eeg2x.profiles import get_preprocessing_profile
from eeg2x.h5_schema import require_valid_h5_file
from eeg2x.evaluation.registry import get_task_config
from eeg2x.rawdata import (
    require_benchmark_output,
    require_raw_thingseeg_eeg,
    require_raw_thingseeg_image,
    require_thingseeg_link,
)
from eeg2x.encoders.luna.input import luna_input_state, prepare_luna_epochs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("eeg2image")

# ===================================================================
# TARGET_SR and baseline — aligned with NICE (Song et al., ICLR 2024)
# ===================================================================
# THINGS-EEG2 OSF: 63 EEG + 1 stim (all EEG channels kept)

TARGET_SR = 250
DEFAULT_T_PRE = 0.0           # NICE: [0, 1000] ms post-stimulus
DEFAULT_T_POST = 1.0
THINGSEEG_RETRIEVAL_PROFILE_RATES = {
    "thingseeg-image-nice-v1": 250,
    "thingseeg-image-reve-nice-v1": 200,
    "thingseeg-image-luna-nice-v1": 256,
}
LUNA_NICE_PROFILE = "thingseeg-image-luna-nice-v1"


# ===================================================================
# Channel-name normalisation
# ===================================================================

def _norm_ch(name: str) -> str:
    return normalize_channel_name(name)


def _keep_all_eeg(data: np.ndarray, src_channels: List[str]) -> np.ndarray:
    """Keep all EEG channels as-is (NICE uses all 63 channels)."""
    return data.astype(np.float32, copy=False)


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


# ===================================================================
# Resampling
# ===================================================================

def _resample(data: np.ndarray, orig_sr: float, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample on the exact declared EEG clock."""
    return resample_eeg(data, orig_sr, target_sr)


# ===================================================================
# Epoching
# ===================================================================

def _epoch_around_events(data: np.ndarray, event_samples: np.ndarray,
                         t_pre: float, t_post: float, sr: float) -> np.ndarray:
    n_pre = int(round(abs(t_pre) * sr))
    n_post = int(round(t_post * sr))
    epoch_len = n_pre + n_post
    n_chan = data.shape[0]
    epochs_list = []
    for es in event_samples:
        start = es - n_pre
        stop = es + n_post
        if start >= 0 and stop <= data.shape[1]:
            epochs_list.append(data[:, start:stop])
    if not epochs_list:
        return np.zeros((0, n_chan, 0), dtype=np.float32)
    return np.stack(epochs_list, axis=0).astype(np.float32)


def _valid_event_mask(n_samples: int, event_samples: np.ndarray,
                      t_pre: float, t_post: float, sr: float) -> np.ndarray:
    """Return the row mask used for epoching so metadata stays aligned."""
    n_pre = int(round(abs(t_pre) * sr))
    n_post = int(round(t_post * sr))
    samples = np.asarray(event_samples, dtype=np.int64)
    return (samples - n_pre >= 0) & (samples + n_post <= n_samples)


# ===================================================================
# OSF THINGS-EEG Reader (.npy format)
# ===================================================================

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


def _compute_mnn_whitener(epochs, concept_nums):
    """Compute MNN whitening matrix from data grouped by concept_num (NICE §4.1).
    Uses Ledoit-Wolf shrinkage covariance per epoch, averaged across conditions."""
    from sklearn.covariance import LedoitWolf
    n_ch = epochs.shape[1]
    unique_c = np.unique(concept_nums)
    sigma_cond = np.empty((len(unique_c), n_ch, n_ch))
    lw = LedoitWolf()
    for i, c in enumerate(unique_c):
        idx = np.where(concept_nums == c)[0]
        cond_epochs = epochs[idx]
        sigma_cond[i] = np.mean([
            lw.fit(cond_epochs[e].T).covariance_
            for e in range(len(cond_epochs))
        ], axis=0)
    sigma_tot = sigma_cond.mean(axis=0)
    return scipy.linalg.fractional_matrix_power(sigma_tot, -0.5)


def _apply_whiten(epochs, sigma_inv):
    """Apply MNN whitening: epochs (N, ch, t) → (N, ch, t)."""
    return (epochs.swapaxes(1, 2) @ sigma_inv).swapaxes(1, 2)


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


# ===================================================================
# NOD-EEG legacy adapter (disabled)
# ---------------------------------
# The former implementation below used the benchmark-era ``.set`` →
# [0,1000 ms] → MNN → alphabetic 800/200 class split path.  It is retained
# under a legacy name solely so old code references remain searchable; the
# loader no longer calls it.  New NOD retrieval preprocessing is implemented
# in preprocess_nod_nice.py from the NOD paper's released preprocessed FIF
# epochs and the ThingsEEG-aligned 800/200 class split.
# ===================================================================

def _process_nod_eeg_legacy_disabled(data_root: str, sub_id: str, out_dir: str) -> int:
    """Process one NOD-EEG subject → train/test h5 (class-split zero-shot)."""
    import mne
    nod_root = os.path.join(data_root, "NOD-EEG")
    detailed_csv = os.path.join(nod_root, "derivatives", "detailed_events",
                                f"{sub_id}_events.csv")
    if not os.path.exists(detailed_csv):
        log.warning(f"Missing detailed_events for {sub_id}, skip.")
        return 0

    # Read all events from detailed_events CSV (in run order)
    all_rows = []
    with open(detailed_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["event_name"] == "stim_on":
                all_rows.append(row)
    log.info(f"  {sub_id}: {len(all_rows)} stim_on events")

    # Build sorted class mapping (0-997 alphabetically)
    unique_classes = sorted(set(r["class"] for r in all_rows))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    n_train_classes = 800
    train_classes = set(unique_classes[:n_train_classes])
    test_classes = set(unique_classes[n_train_classes:])
    log.info(f"  Classes: {len(unique_classes)} total, "
             f"{len(train_classes)} train, {len(test_classes)} test (zero-shot)")

    # Per-class stim_number counter
    stim_counter: Dict[str, int] = {}

    # Collect all epochs across all runs
    all_epochs: List[np.ndarray] = []
    all_meta: List[Dict] = []
    embedding_idx = 0
    event_cursor = 0
    subject_source_sr: Optional[float] = None
    subject_channels: Optional[List[str]] = None

    sub_dir = os.path.join(nod_root, sub_id)
    sessions = sorted([
        d for d in os.listdir(sub_dir)
        if d.startswith("ses-") and os.path.isdir(os.path.join(sub_dir, d, "eeg"))
    ])
    session_sigma_inv = None

    for ses in sessions:
        ses_dir = os.path.join(sub_dir, ses, "eeg")
        set_files = sorted([
            f for f in os.listdir(ses_dir)
            if f.endswith(".set") and "task-ImageNet" in f
        ])
        for sf in set_files:
            run_match = re.search(r"run-(\d+)", sf)
            run_num = run_match.group(1) if run_match else "0"
            run_str = f"0{run_num}" if len(run_num) == 1 else run_num
            set_path = os.path.join(ses_dir, sf)
            events_tsv = set_path.replace("_eeg.set", "_events.tsv")
            if not os.path.exists(events_tsv):
                continue

            # Read EEG (standardize to 64 channels)
            import mne
            raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            ch_all = [raw.ch_names[i] for i in eeg_picks]
            orig_sr = int(raw.info["sfreq"])
            # Keep first 64 EEG channels (some sessions have extra 2 eye chs)
            n_keep = min(64, len(ch_all))
            # MNE exposes physical EEG values in Volt; the protocol stores uV.
            eeg_data = (raw.get_data(picks=eeg_picks[:n_keep]) * 1e6).astype(np.float32)
            ch_names = normalize_channel_names(ch_all[:n_keep])
            del raw
            if subject_source_sr is None:
                subject_source_sr = float(orig_sr)
                subject_channels = ch_names
            elif not np.isclose(subject_source_sr, orig_sr):
                raise ValueError(
                    f"{sub_id} has inconsistent source sampling rates: "
                    f"{subject_source_sr} and {orig_sr}"
                )
            elif subject_channels != ch_names:
                raise ValueError(f"{sub_id} has inconsistent channel order across runs")

            # Resample 500Hz → 250Hz
            eeg_250 = _resample(eeg_data, orig_sr)
            new_sr = TARGET_SR

            # Read events.tsv for onset times
            onsets = []
            with open(events_tsv, encoding="utf-8") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    if row.get("trial_type") == "stim_on":
                        onsets.append(float(row["onset"]))

            # Match events.tsv → detailed_events CSV by position
            # (both are in the same order)
            log.info(f"    {sub_id} {ses} run-{run_num}: {len(onsets)} onsets")

            for onset_sec in onsets:
                # Find matching detailed event
                if event_cursor >= len(all_rows):
                    raise ValueError(
                        f"{sub_id}: events.tsv has more stim_on rows than detailed_events CSV"
                    )
                row_info = all_rows[event_cursor]
                event_cursor += 1

                onset_sample_250 = seconds_to_sample(onset_sec, TARGET_SR)
                epoch = _epoch_around_events(
                    eeg_250, np.array([onset_sample_250]), 0.0, 1.0, new_sr)
                if epoch.shape[0] == 0:
                    continue

                image_path = (
                    Path(nod_root) / "stimuli" / "ImageNet" /
                    f"{row_info['image_id']}.JPEG"
                ).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"{sub_id}: stimulus image does not exist: {image_path}"
                    )

                class_name = row_info["class"]
                concept_num = class_to_idx[class_name] + 1  # 1-indexed

                # Assign stim_number within this class
                stim_key = f"{concept_num}_{row_info['image_id']}"
                if stim_key not in stim_counter:
                    stim_counter[stim_key] = 0
                stim_num = stim_counter[stim_key]
                stim_counter[stim_key] = stim_num + 1

                all_epochs.append(epoch[0])
                all_meta.append({
                    "concept_num": concept_num,
                    "stim_number": stim_num,
                    "embedding_idx": embedding_idx,
                    "class_name": class_name,
                    "class_id": row_info["class_id"],
                    "image_id": row_info["image_id"],
                    "is_animate": row_info["stim_is_animate"] == "True",
                    "onset_sec": onset_sec,
                    "stimulus_uri": str(image_path),
                })
                embedding_idx += 1

    if event_cursor != len(all_rows):
        raise ValueError(
            f"{sub_id}: consumed {event_cursor} stim_on events from runs but "
            f"detailed_events CSV contains {len(all_rows)}"
        )

    if not all_epochs:
        log.warning(f"  {sub_id}: no epochs collected")
        return 0

    epochs = np.stack(all_epochs)  # (N, 64, 250)
    n_ch, n_t = epochs.shape[1], epochs.shape[2]

    # MNN: compute from training classes, apply to all
    train_mask = np.array([m["concept_num"] <= n_train_classes for m in all_meta])
    cnums = np.array([m["concept_num"] for m in all_meta])
    if train_mask.sum() > 1:
        session_sigma_inv = _compute_mnn_whitener(epochs[train_mask], cnums[train_mask])
        epochs = _apply_whiten(epochs, session_sigma_inv)
        log.info(f"  MNN: computed from {train_mask.sum()} training epochs → applied to {len(epochs)}")
    else:
        raise ValueError(
            f"{sub_id}: at least two training epochs are required to fit MNN"
        )

    # Split train/test
    n_meta = len(all_meta)
    train_epochs = epochs[train_mask]
    train_meta = [all_meta[i] for i in range(n_meta) if train_mask[i]]
    test_epochs = epochs[~train_mask]
    test_meta = [all_meta[i] for i in range(n_meta) if not train_mask[i]]
    log.info(f"  Train epochs: {len(train_epochs)}, Test epochs: {len(test_epochs)}")

    # Write training h5
    dt_str = h5py.string_dtype(encoding="utf-8")
    _write_nod_h5(os.path.join(out_dir, f"nod_{sub_id}_training_epoched.h5"),
                  train_epochs, train_meta, subject_channels, new_sr,
                  subject_source_sr, session_sigma_inv, dt_str)
    # Write test h5
    _write_nod_h5(os.path.join(out_dir, f"nod_{sub_id}_test_epoched.h5"),
                  test_epochs, test_meta, subject_channels, new_sr,
                  subject_source_sr, session_sigma_inv, dt_str)

    return len(train_epochs) + len(test_epochs)


def _write_nod_h5(out_path, epochs, meta, ch_names, sr, source_sr,
                  mnn_matrix, dt_str):
    """Write one NOD-EEG h5 file (standardized schema)."""
    n_ep, n_ch, n_t = epochs.shape
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("data", data=epochs, dtype=np.float32,
                          compression="gzip", compression_opts=4,
                          chunks=(1, n_ch, min(n_t, 256)))

        obj_nums = np.array([m["concept_num"] for m in meta], dtype=int)
        stim_nums = np.array([m["stim_number"] for m in meta], dtype=int)
        emb_idx = np.array([m["embedding_idx"] for m in meta], dtype=int)
        labels = np.array([m["image_id"] for m in meta], dtype=object)
        class_names = np.array([m["class_name"] for m in meta], dtype=object)
        img_files = np.array([m.get("image_id", "") for m in meta], dtype=object)
        class_ids = np.array([m.get("class_id", "") for m in meta], dtype=object)
        animate = np.array([bool(m.get("is_animate", False)) for m in meta], dtype=np.bool_)

        h5.create_dataset("event_object_number", data=obj_nums)
        h5.create_dataset("event_stim_number", data=stim_nums)
        h5.create_dataset("event_embedding_index", data=emb_idx)
        h5.create_dataset("event_label", data=labels, dtype=dt_str)
        h5.create_dataset("event_class_name", data=class_names, dtype=dt_str)
        h5.create_dataset("event_image_file", data=img_files, dtype=dt_str)
        h5.create_dataset("event_class_id", data=class_ids, dtype=dt_str)
        h5.create_dataset("event_is_animate", data=animate)
        h5.create_dataset("channel_names", data=np.array(ch_names, dtype="S"), dtype=dt_str)
        h5.create_dataset(
            "mnn_whitener", data=np.asarray(mnn_matrix, dtype=np.float32))

        h5.attrs["dataset"] = "nod"
        h5.attrs["subject_id"] = Path(out_path).stem.split("_")[1]
        h5.attrs["sampling_rate"] = float(sr)
        h5.attrs["t_pre"] = 0.0
        h5.attrs["t_post"] = 1.0
        _write_image_protocol_metadata(
            h5, dataset_id="nod", subject_id=Path(out_path).stem.split("_")[1],
            source_sr=float(source_sr),
            split="train" if "training" in Path(out_path).name else "test",
            channels=ch_names, n_epochs=n_ep,
            preprocessing_state={
                "fitted_split": "train",
                "fit_group": Path(out_path).stem.split("_")[1],
                "mnn_matrix_sha256": sha256_array(
                    np.asarray(mnn_matrix, dtype=np.float32)),
            },
            records=[StimulusRecord(
                stimulus_id=make_stimulus_id("nod", m.get("image_id", f"row-{i}")),
                modality="image", uri=str(m.get("stimulus_uri", "")),
                onset_sec=float(m.get("onset_sec", np.nan)), duration_sec=1.0,
                alignment_source="events_tsv+detailed_events",
                transform={"class_name": m.get("class_name", ""), "class_id": m.get("class_id", ""),
                           "is_animate": bool(m.get("is_animate", False)),
                           "embedding_index": int(m.get("embedding_idx", i))},
            ) for i, m in enumerate(meta)],
        )
        h5.attrs["n_channels"] = n_ch
        h5.attrs["n_epochs"] = n_ep
        h5.attrs["channel_set"] = "all_eeg"
        h5.attrs["mnn_applied"] = True
    require_valid_h5_file(
        out_path, get_task_config("image_retrieval")["h5_schema"])
    log.info(f"      Saved → {out_path}")


def _find_nod_subjects(data_root: str) -> List[str]:
    nod_root = os.path.join(data_root, "NOD-EEG")
    if not os.path.isdir(nod_root):
        return []
    return sorted([
        d for d in os.listdir(nod_root)
        if d.startswith("sub-") and os.path.isdir(os.path.join(nod_root, d, "ses-ImageNet01"))
    ])


# ===================================================================
# Main entry
# ===================================================================

def run_loader(data_root: str, out_dir: str,
               datasets: List[str] = None,
               max_subjects: int = None,
               subjects: Optional[List[str]] = None,
               allow_partial: bool = False,
               thingseeg_target_sr: int = TARGET_SR,
               thingseeg_profile_id: str = "thingseeg-image-nice-v1"):
    os.makedirs(out_dir, exist_ok=True)
    datasets = datasets or ["thingseeg", "nod"]
    unknown = sorted(set(datasets) - {"thingseeg", "nod"})
    if unknown:
        raise ValueError(f"Unknown image dataset(s): {unknown}")
    failures = []

    log.info("=" * 60)
    log.info("EEG-to-Image Benchmark Data Loader")
    log.info(f"  Data root:  {data_root}")
    log.info(f"  Output:     {out_dir}")
    log.info(f"  Target SR:  {thingseeg_target_sr} Hz for ThingsEEG")
    log.info(f"  Channels:   all EEG (63 for ThingsEEG)")
    log.info(f"  Datasets:   {datasets}")
    log.info("=" * 60)

    if "thingseeg" in datasets:
        log.info("\n=== Dataset: THINGS-EEG (OSF) ===")
        things_root = _resolve_things_root(data_root)
        if things_root is not None and Path(things_root).absolute() != require_thingseeg_link():
            raise ValueError(
                f"THINGS-EEG must be read through {require_thingseeg_link()}, "
                f"got {things_root}"
            )
        subs = _find_things_subjects(things_root) if things_root else []
        if subjects:
            requested = set(subjects)
            missing = sorted(requested - set(subs))
            if missing:
                raise ValueError(f"Requested THINGS-EEG subjects not found: {missing}")
            subs = [subject for subject in subs if subject in requested]
        if max_subjects:
            subs = subs[:max_subjects]
        log.info(f"Found {len(subs)} subject(s): {subs}")
        if not subs:
            failures.append("thingseeg: no subjects found")
        n_total = 0
        for sub in subs:
            try:
                n = _process_thingseeg_osf(
                    things_root, sub, out_dir,
                    target_sr=thingseeg_target_sr,
                    profile_id=thingseeg_profile_id)
                n_total += n
                if n == 0:
                    failures.append(f"thingseeg/{sub}: no epochs written")
            except Exception as exc:
                log.error(f"Failed THINGS-EEG {sub}: {exc}", exc_info=True)
                failures.append(f"thingseeg/{sub}: {exc}")
        log.info(f"THINGS-EEG total: {n_total} epochs")

    if "nod" in datasets:
        log.info("\n=== Dataset: NOD-EEG ===")
        subs = _find_nod_subjects(data_root)
        if max_subjects:
            subs = subs[:max_subjects]
        log.info(f"Found {len(subs)} subject(s): {subs}")
        if not subs:
            failures.append("nod: no subjects found")
        n_total = 0
        for sub in subs:
            try:
                # v3: NOD 的预处理入口是 eeg2x.preprocessing.nod:process_nod_atm，
                # 不再经由这个旧 loader 分支。
                raise NotImplementedError(
                    "the legacy NOD branch of the old loader is not ported to v3; "
                    "use eeg2x.preprocessing.nod instead")
                n_total += n
                if n == 0:
                    failures.append(f"nod/{sub}: no epochs written")
            except Exception as exc:
                log.error(f"Failed NOD-EEG {sub}: {exc}", exc_info=True)
                failures.append(f"nod/{sub}: {exc}")
        log.info(f"NOD-EEG total: {n_total} epochs")

    log.info(f"\n{'='*60}")
    log.info(f"Output in: {os.path.abspath(out_dir)}")
    if failures and not allow_partial:
        raise RuntimeError(
            "Image preprocessing was incomplete:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="EEG-to-Image Benchmark Data Loader")
    ap.add_argument("--data_root", type=str, default=str(require_thingseeg_link()))
    ap.add_argument("--out_dir", type=str, default="./h5/nice")
    ap.add_argument("--dataset", type=str, nargs="+", default=["thingseeg", "nod"],
                    help="Which datasets: thingseeg nod")
    ap.add_argument("--subjects", type=str, nargs="*", default=None,
                    help="Optional explicit THINGS-EEG subject IDs; default: all")
    ap.add_argument("--max_subjects", type=int, default=None)
    ap.add_argument("--allow_partial", action="store_true",
                    help="Keep successful subjects and return success despite failures")
    ap.add_argument("--thingseeg_target_sr", type=int, default=TARGET_SR)
    ap.add_argument(
        "--thingseeg_profile_id", default="thingseeg-image-nice-v1")
    args = ap.parse_args()

    run_loader(
        data_root=args.data_root,
        out_dir=args.out_dir,
        datasets=args.dataset,
        max_subjects=args.max_subjects,
        subjects=args.subjects,
        allow_partial=args.allow_partial,
        thingseeg_target_sr=args.thingseeg_target_sr,
        thingseeg_profile_id=args.thingseeg_profile_id,
    )
