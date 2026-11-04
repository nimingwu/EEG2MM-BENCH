"""Grootswagers2025 infant dataset (OpenNeuro ds005106) preprocessing.

Dataset
-------
``<root>/sub-XXX/eeg/sub-XXX_task-fix_eeg.set`` (EEGLAB) plus the BIDS
``*_events.tsv``.  42 infants (2-12 months) viewed the same 200 object images as
the adult ds004018 release in 5 Hz RSVP streams; one block presents all 200
images once and infants completed as many blocks as they tolerated (1.5-6.5).

Papers
------
Grootswagers, Quek, Zeng & Varlet (2025), Scientific Data 12:407,
doi:10.1038/s41597-025-04744-z.  Their pipeline: EEGLAB ``clean_rawdata``
(0.5 Hz high-pass, bad-segment and bad-channel removal: ~1.5 % of segments and
~6.5 channels per infant), then a Hamming-windowed FIR 40 Hz low-pass, average
reference, and epochs [-100, 800] ms around every stimulus onset.  "No further
preprocessing steps were applied."

Benchmark steps
---------------
* two registered bands, selected with ``--band``: ``paper`` (0.5-40 Hz, the
  paper's filter) and ``wide`` (0.1-100 Hz plus a 50 Hz notch, because the wide
  band re-admits the line noise that the paper's 40 Hz low-pass removes)
* Hamming-windowed FIR filtering, average reference (as in the paper)
* reproducible bad-channel handling: flat channels and robust log-variance
  outliers are interpolated with spherical splines and recorded
* reproducible bad-segment handling: epochs whose any-channel peak-to-peak
  exceeds ``median + 5 * 1.4826 * MAD`` are rejected and recorded
* exact rational resample 500 -> 250 Hz, epoch [-0.1, 0.8] s (226 samples)
* block-disjoint split with the protocol seed 2000: only *complete* blocks
  (one presentation of every stimulus) are eligible, and
  ``max(1, round(TEST_BLOCK_FRACTION * n_complete))`` of them -- at least one,
  at most ``n_complete - 1``, so training is never empty -- form the test set;
  partial blocks contribute training trials only.  Because a complete block
  holds all 200 stimuli, the test set covers every stimulus, and every one of
  those stimuli also occurs in the training blocks (a *seen-stimulus* split).

  NOTE: a stimulus-disjoint 180/20 variant (unseen-stimulus, same 20 held-out
  images as the adult ds004018 release) was implemented and piloted on sub-001
  but is **not** active; it is parked under ``_ds005106_v2_pilot/`` together
  with its source copies and the pilot's HDF5/artifacts.
* train-only MNN whitening (Ledoit-Wolf, sigma^-1/2)
"""
from __future__ import annotations

import argparse
import collections
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
from eeg2x.rawdata import EXPECTED_GROOTSWAGERS2025_ROOT, require_grootswagers2025_root
from eeg2x.profiles import get_preprocessing_profile
from eeg2x.evaluation.registry import get_task_config

log = logging.getLogger("grootswagers2025_preprocess")

SEED = 2000
SOURCE_SR = 500.0
TARGET_SR = 250.0
T_PRE_SEC = -0.1
T_POST_SEC = 0.8
PRE_SAMPLES = int(round(-T_PRE_SEC * TARGET_SR))                      # 25
EPOCH_SAMPLES = int(round((T_POST_SEC - T_PRE_SEC) * TARGET_SR)) + 1  # 226
N_CHANNELS = 32
N_STIMULI = 200
TEST_BLOCK_FRACTION = 0.25
MIN_COMPLETE_BLOCKS = 2   # subjects below this cannot form a test set
#: 0.5-40 Hz is the paper's own band; the wide variant re-admits line noise that
#: the paper's 40 Hz low-pass removes, so it carries an explicit 50 Hz notch.
BANDS = {
    "paper": {"band_hz": (0.5, 40.0), "notch_hz": ()},
    "wide": {"band_hz": (0.1, 100.0), "notch_hz": (50.0,)},
}
BAD_CHANNEL_Z = 4.0          # robust (MAD) z-score on log channel variance
#: Infant EEG is extremely artifact laden (the 90th percentile epoch reaches
#: ~215 mV peak-to-peak), so an absolute or MAD-scaled threshold rejects a
#: quarter of the data.  The benchmark instead removes the worst 2 % of epochs
#: per subject by peak-to-peak amplitude - adaptive, reproducible, and it keeps
#: every stimulus represented in the held-out blocks.
EPOCH_REJECT_FRACTION = 0.02
STIMULUS_DIR = Path("code") / "experiment" / "stimuli"
EVENTS_NAME_RE = re.compile(r"_task-fix_events\.tsv$")


def _band_key(profile_id: str) -> str:
    for key in BANDS:
        if key in profile_id:
            return key
    raise ValueError(f"Cannot infer the band from profile {profile_id!r}")


def _subject_paths(root: Path, subject_id: str) -> Tuple[Path, Path]:
    eeg_dir = root / subject_id / "eeg"
    set_files = sorted(eeg_dir.glob(f"{subject_id}_task-*_eeg.set"))
    if len(set_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .set for {subject_id}, found {set_files}")
    events = set_files[0].with_name(set_files[0].name.replace("_eeg.set", "_events.tsv"))
    if not events.is_file():
        raise FileNotFoundError(events)
    return set_files[0], events


def _participant_metadata(root: Path, subject_id: str) -> Dict[str, Any]:
    """Age, completed blocks and the paper's exclusion flag, for later grouping."""
    path = root / "participants.tsv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("participant_id") == subject_id:
                return {
                    "age_months": float(row["age_months"]) if row.get("age_months") else None,
                    "blocks_completed_reported": (float(row["blockscompleted"])
                                                  if row.get("blockscompleted") else None),
                    "paper_exclude": int(row.get("exclude", 0) or 0),
                    "sex": row.get("sex"),
                }
    return {}


def _read_events(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle, delimiter="\t") if row.get("stim")]
    if not rows:
        raise ValueError(f"{path}: no stimulus rows")
    return rows


def _bad_channels(raw: mne.io.BaseRaw) -> Tuple[List[str], Dict[str, Any]]:
    """Flat-line and robust log-variance channel outliers (reproducible, documented)."""
    data = raw.get_data()
    std = data.std(axis=1)
    flat = [raw.ch_names[i] for i, value in enumerate(std) if value < 1e-9]
    log_std = np.log(np.maximum(std, 1e-12))
    median = float(np.median(log_std))
    mad = float(np.median(np.abs(log_std - median))) * 1.4826
    outliers = []
    if mad > 0:
        z = (log_std - median) / mad
        outliers = [raw.ch_names[i] for i, value in enumerate(z)
                    if abs(float(value)) > BAD_CHANNEL_Z]
    bad = sorted(set(flat) | set(outliers))
    return bad, {"flat": flat, "variance_outliers": outliers,
                 "threshold_z": BAD_CHANNEL_Z, "log_std_median": median,
                 "log_std_mad": mad}


def _load_raw(set_path: Path, band: str) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    if not np.isclose(float(raw.info["sfreq"]), SOURCE_SR):
        raise ValueError(f"{set_path}: expected {SOURCE_SR} Hz, got {raw.info['sfreq']}")
    types = raw.get_channel_types()
    eeg_indices = [i for i, kind in enumerate(types) if kind == "eeg"]
    if len(eeg_indices) != N_CHANNELS:
        raise ValueError(f"{set_path}: expected {N_CHANNELS} EEG channels, got {len(eeg_indices)}")
    raw.pick([raw.ch_names[i] for i in eeg_indices])
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"), on_missing="raise")

    bad, bad_info = _bad_channels(raw)
    if bad:
        raw.info["bads"] = bad
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    log.info("%s: interpolated %d channel(s): %s", set_path.name, len(bad), bad or "none")

    spec = BANDS[band]
    low, high = spec["band_hz"]
    raw.filter(l_freq=low, h_freq=high, method="fir", fir_window="hamming",
               verbose="ERROR")
    if spec["notch_hz"]:
        raw.notch_filter(freqs=list(spec["notch_hz"]), method="fir",
                         fir_window="hamming", verbose="ERROR")
    raw.set_eeg_reference("average", verbose="ERROR")
    info = {"bad_channels": bad, **bad_info, "band_hz": list(spec["band_hz"]),
            "notch_hz": list(spec["notch_hz"]), "reference": "average"}
    return raw, info


def _epochs_and_split(
    raw: mne.io.BaseRaw, events: Sequence[Dict[str, str]], *, target_sr: float,
    labels: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    signal = resample_eeg(raw.get_data(), SOURCE_SR, target_sr, axis=-1)
    onsets = np.asarray([float(row["onset"]) for row in events])       # seconds
    blocks = np.asarray([int(row["blocknr"]) for row in events])
    starts = np.round(onsets * target_sr).astype(int) - PRE_SAMPLES
    if (starts < 0).any() or ((starts + EPOCH_SAMPLES) > signal.shape[-1]).any():
        raise ValueError("Stimulus epochs exceed the recording bounds")
    epochs = np.stack([signal[:, s:s + EPOCH_SAMPLES] for s in starts]).astype(np.float32)
    epochs *= np.float32(1e6)                                          # V -> uV

    # Bad-segment handling: reject epochs that exceed a robust per-channel
    # peak-to-peak threshold.
    p2p = epochs.max(axis=-1) - epochs.min(axis=-1)                    # (n, ch)
    per_epoch = p2p.max(axis=1)
    threshold = float(np.quantile(per_epoch, 1.0 - EPOCH_REJECT_FRACTION))
    keep = per_epoch <= threshold
    n_rejected = int((~keep).sum())

    # Only complete blocks (one presentation of every stimulus) may serve as the
    # test set, so the retrieved candidate pool is always the full 200 stimuli;
    # partial blocks still contribute training trials.  Subjects with fewer than
    # MIN_COMPLETE_BLOCKS complete blocks are not usable (the paper excludes the
    # same low-block participants).
    block_counts = collections.Counter(int(b) for b in blocks)
    complete_blocks = sorted(b for b, count in block_counts.items() if count >= N_STIMULI)
    partial_blocks = sorted(b for b in block_counts if b not in set(complete_blocks))
    if len(complete_blocks) < MIN_COMPLETE_BLOCKS:
        raise ValueError(
            f"{len(complete_blocks)} complete block(s) < required "
            f"{MIN_COMPLETE_BLOCKS}; this subject is not usable")
    rng = np.random.default_rng(SEED)
    permuted = rng.permutation(len(complete_blocks))
    n_test = max(1, int(round(TEST_BLOCK_FRACTION * len(complete_blocks))))
    n_test = min(n_test, len(complete_blocks) - 1)
    test_blocks = {complete_blocks[i] for i in permuted[:n_test]}
    train_blocks = set(block_counts) - test_blocks

    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(events):
        if not keep[index]:
            continue
        block = int(row["blocknr"])
        rows.append({
            "stimulus": str(row["stim"]),
            "stimulus_number": int(row["stimnum"]),
            "block": block,
            "presentation": int(row["presentationnumber"]),
            "trial": int(row["trialnr"]),
            "block_set": "test" if block in test_blocks else "train",
            "label": (str(row["stim"]) if labels == "stimulus"
                      else f"{labels}={str(row['stim']).split('_')[1]}"),
        })
    kept = np.asarray([r["block_set"] for r in rows])

    train_data = epochs[keep][kept == "train"]
    train_rows = [row for row in rows if row["block_set"] == "train"]
    grouped: Dict[str, List[np.ndarray]] = {}
    for index, row in enumerate(rows):
        if row["block_set"] == "test":
            grouped.setdefault(row["stimulus"], []).append(epochs[keep][index])
    if not grouped:
        raise ValueError("The held-out blocks carry no usable presentations")
    test_rows, test_data = [], []
    for stimulus in sorted(grouped):
        source = [row for row in rows
                  if row["block_set"] == "test" and row["stimulus"] == stimulus]
        aggregate = dict(source[0])
        aggregate.update({"block": -1, "n_repetitions": len(grouped[stimulus]),
                          "source_blocks": sorted({row["block"] for row in source})})
        test_rows.append(aggregate)
        test_data.append(np.mean(grouped[stimulus], axis=0).astype(np.float32))

    state = {
        "labels": labels,
        "split_unit": "block",
        "blocks": sorted(block_counts),
        "complete_blocks": complete_blocks,
        "partial_blocks": partial_blocks,
        "train_blocks": sorted(train_blocks),
        "test_blocks": sorted(test_blocks),
        "test_block_rule": (f"complete blocks only, "
                            f"{TEST_BLOCK_FRACTION:.0%} held out"),
        "min_test_repetitions_per_stimulus": min(len(v) for v in grouped.values()),
        "block_split_seed": SEED,
        "n_train_trials": len(train_rows),
        "n_test_stimuli": len(test_rows),
        "epochs_rejected": n_rejected,
        "epochs_total": int(len(events)),
        "epoch_rejection_rule": ("reject the worst "
                                 f"{EPOCH_REJECT_FRACTION:.0%} of epochs by peak-to-peak "
                                 "amplitude within the subject"),
        "epoch_rejection_threshold_uv": threshold,
        "test_aggregation": "mean_by_stimulus_across_test_blocks",
    }
    return (np.stack(train_data, axis=0), train_rows,
            np.stack(test_data, axis=0), test_rows, state)


def _compute_mnn_whitener(epochs: np.ndarray) -> np.ndarray:
    covariances = [LedoitWolf().fit(epoch.T).covariance_ for epoch in epochs]
    covariance = np.mean(np.stack(covariances, axis=0), axis=0)
    return np.asarray(scipy.linalg.fractional_matrix_power(covariance, -0.5)).real.astype(np.float32)


def _apply_mnn(data: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    return (data.swapaxes(1, 2) @ whitener).swapaxes(1, 2).astype(np.float32)


def _adapt(data: np.ndarray, *, target_sr: float, output_samples: int,
           luna: bool) -> Tuple[np.ndarray, int]:
    """Encoder adaptation: crop the post-stimulus second, resample, pad."""
    if np.isclose(target_sr, TARGET_SR):
        adapted, valid = data, int(data.shape[-1])
    else:
        from eeg2x.encoders.luna.input import LUNA_ZSCORE_EPS
        post = data[..., PRE_SAMPLES:]
        adapted = resample_eeg(post, TARGET_SR, target_sr, axis=-1)
        valid = int(adapted.shape[-1])
        if luna:
            mean = adapted.mean(axis=-1, keepdims=True)
            std = adapted.std(axis=-1, ddof=1, keepdims=True)
            adapted = (adapted - mean) / (std + LUNA_ZSCORE_EPS)
    if valid > output_samples:
        raise ValueError(f"Adapted window has {valid} samples > {output_samples}")
    if valid < output_samples:
        # Works for both (channels, samples) and (trials, channels, samples).
        pad = ((0, 0),) * (adapted.ndim - 1) + ((0, output_samples - valid),)
        adapted = np.pad(adapted, pad, constant_values=0.0)
    return adapted.astype(np.float32, copy=False), valid


def _string_array(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=object)


def _write_h5(path: Path, data: np.ndarray, rows: Sequence[Dict[str, Any]],
              channels: Sequence[str], subject_id: str, split: str,
              state: Dict[str, Any], root: Path, source_files: Sequence[Path],
              whiteners: np.ndarray, profile_id: str, sampling_rate: float) -> Path:
    profile = get_preprocessing_profile(profile_id)
    image_root = root / STIMULUS_DIR
    records: List[StimulusRecord] = []
    for row in rows:
        image_path = (image_root / str(row["stimulus"])).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing infant stimulus: {image_path}")
        records.append(StimulusRecord(
            stimulus_id=make_stimulus_id("grootswagers2025", str(row["stimulus"])),
            modality="image", uri=str(image_path), sha256=sha256_file(image_path),
            duration_sec=0.2,
            alignment_source="BIDS events.tsv onset (seconds), 5 Hz RSVP",
            transform={
                "dataset": "grootswagers2025",
                "stimulus": str(row["stimulus"]),
                "stimulus_number": int(row["stimulus_number"]),
                "block": int(row["block"]),
                "presentation": int(row["presentation"]),
                "n_repetitions": int(row.get("n_repetitions", 1)),
                "source_blocks": [int(v) for v in row.get("source_blocks", [row["block"]])],
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
        h5.create_dataset("event_block", data=np.asarray(
            [int(row["block"]) for row in rows], dtype=np.int32))
        stimulus_numbers = sorted({int(row["stimulus_number"]) for row in rows})
        number_of = {value: index + 1 for index, value in enumerate(stimulus_numbers)}
        object_numbers, stim_numbers, counter = [], [], {}
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
            h5, dataset_id="grootswagers2025", task_id="eeg_to_image",
            subject_id=subject_id, sampling_rate=sampling_rate,
            source_sampling_rate=SOURCE_SR, n_channels=n_channels, n_epochs=n_epochs,
            split=split, epoch_type="image", eeg_units="dimensionless_whitened",
            profile=profile,
            preprocessing_state={
                **state, "subject_id": subject_id, "split": split,
                "dataset": "grootswagers2025", "profile_id": profile_id,
                "derived_inputs_used": False,
                "paper_doi": "10.1038/s41597-025-04744-z",
                "source_format": "BIDS_EEGLAB_set",
                "source_files": [str(item) for item in source_files],
                "mnn_fit_source_files": [str(item) for item in source_files],
                "source_sampling_rate": SOURCE_SR,
                "output_sampling_rate": sampling_rate,
                "input_window_sec": [T_PRE_SEC, T_POST_SEC],
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


def process_subject(subject_id: str, data_root: str | Path, out_dir: str | Path, *,
                    profile_id: str, labels: str = "stimulus", layout: str = "single",
                    output_prefix: str = "grootswagers2025",
                    target_sr: float = TARGET_SR, output_samples: int = EPOCH_SAMPLES,
                    luna: bool = False) -> int:
    if layout not in ("single", "per_session"):
        raise ValueError(f"Unsupported layout: {layout!r}")
    root = require_grootswagers2025_root(data_root)
    set_path, events_path = _subject_paths(root, subject_id)
    band = _band_key(profile_id)
    raw, filter_info = _load_raw(set_path, band)
    events = _read_events(events_path)
    train_data, train_rows, test_data, test_rows, state = _epochs_and_split(
        raw, events, target_sr=target_sr, labels=labels)
    whitener = _compute_mnn_whitener(train_data)
    train_adapted, valid = _adapt(_apply_mnn(train_data, whitener), target_sr=target_sr,
                                  output_samples=output_samples, luna=luna)
    test_adapted, test_valid = _adapt(_apply_mnn(test_data, whitener), target_sr=target_sr,
                                      output_samples=output_samples, luna=luna)
    if valid != test_valid:
        raise ValueError("Train/test adaptation disagree")
    whiteners = whitener[None, ...]
    session_ids = ["ses-rsvp"]
    common_state = {
        "seed": SEED, "fitted_split": "train", "class_split_seed": SEED,
        "session_ids": session_ids, "sessions": session_ids,
        "profile_id": profile_id, "n_samples": output_samples,
        "output_sampling_rate": target_sr,
        "filter_band_hz": filter_info["band_hz"], "notch_hz": filter_info["notch_hz"],
        "reference": filter_info["reference"],
        "bad_channels": filter_info["bad_channels"],
        "valid_samples_before_padding": valid,
        "encoder_zero_padding_samples": output_samples - valid,
        "encoder_adaptation": "luna_official_zscore" if luna else "identity",
        "participant": _participant_metadata(root, subject_id),
        "mnn": "LedoitWolf_condition_covariance_train_only",
        "mnn_matrix_sha256": sha256_array(whiteners),
        "mnn_matrix_sha256_by_session": {"ses-rsvp": sha256_array(whiteners[0])},
        **state,
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    source_files = [set_path, events_path]
    channels = list(raw.ch_names)
    if layout == "per_session":
        _write_h5(output / f"{output_prefix}_{subject_id}_ses-rsvp_training_epoched.h5",
                  train_adapted, train_rows, channels, subject_id, "train",
                  common_state, root, source_files, whiteners, profile_id, target_sr)
    else:
        _write_h5(output / f"{output_prefix}_{subject_id}_training_epoched.h5",
                  train_adapted, train_rows, channels, subject_id, "train",
                  common_state, root, source_files, whiteners, profile_id, target_sr)
    _write_h5(output / f"{output_prefix}_{subject_id}_test_epoched.h5",
              test_adapted, test_rows, channels, subject_id, "test", common_state,
              root, source_files, whiteners, profile_id, target_sr)
    summary = {
        "subject_id": subject_id, "profile_id": profile_id, "band": band,
        **_participant_metadata(root, subject_id),
        "filter_band_hz": filter_info["band_hz"], "notch_hz": filter_info["notch_hz"],
        "reference": filter_info["reference"], "labels": labels, "layout": layout,
        "bad_channels_interpolated": filter_info["bad_channels"],
        "epochs_total": state["epochs_total"], "epochs_rejected": state["epochs_rejected"],
        "blocks": state["blocks"], "train_blocks": state["train_blocks"],
        "test_blocks": state["test_blocks"],
        "n_train_trials": int(train_adapted.shape[0]),
        "n_test_stimuli": int(test_adapted.shape[0]),
        "n_channels": len(channels), "n_samples": output_samples,
        "sampling_rate": target_sr, "source_sampling_rate": SOURCE_SR,
        "input_window_sec": [T_PRE_SEC, T_POST_SEC],
        "source_files": [str(item) for item in source_files],
        "derived_inputs_used": False,
    }
    (output / f"{output_prefix}_{subject_id}_preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("ds005106 %s (%s): %d train + %d test rows, %d/%d epochs rejected, "
             "interpolated %d channels", subject_id, profile_id,
             train_adapted.shape[0], test_adapted.shape[0], state["epochs_rejected"],
             state["epochs_total"], len(filter_info["bad_channels"]))
    return int(train_adapted.shape[0] + test_adapted.shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, default=EXPECTED_GROOTSWAGERS2025_ROOT)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--subject", default="sub-001")
    parser.add_argument("--profile_id", required=True)
    parser.add_argument("--labels", choices=("stimulus", "level_a", "level_b", "level_c"),
                        default="stimulus")
    parser.add_argument("--layout", choices=("single", "per_session"), default="single")
    parser.add_argument("--output_prefix", default="grootswagers2025")
    parser.add_argument("--target_sr", type=float, default=TARGET_SR)
    parser.add_argument("--output_samples", type=int, default=EPOCH_SAMPLES)
    parser.add_argument("--luna", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    process_subject(args.subject, args.data_root, args.out_dir,
                    profile_id=args.profile_id, labels=args.labels,
                    layout=args.layout, output_prefix=args.output_prefix,
                    target_sr=args.target_sr, output_samples=args.output_samples,
                    luna=args.luna)


if __name__ == "__main__":
    main()
