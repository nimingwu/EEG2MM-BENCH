"""Semantic HDF5 checks for EEG2X data protocol v2."""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

from eeg2x.protocol import (
    PROTOCOL_NAME, SCHEMA_VERSION, canonical_json, sha256_array,
)


STIMULUS_COLUMNS = (
    "stimulus_id", "stimulus_modality", "stimulus_uri", "stimulus_sha256",
    "stimulus_onset_sec", "stimulus_duration_sec",
    "stimulus_alignment_source", "stimulus_transform_json",
)


def _strings(values):
    return [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def _eeg_name(h5, schema):
    requested = schema.get("eeg_field")
    if requested:
        return requested if requested in h5 else None
    return next((name for name in ("data", "eeg") if name in h5), None)


def validate_protocol_v2(h5, schema):
    errors = []
    attrs = (
        "dataset_id", "task_id", "subject_id", "sampling_rate",
        "source_sampling_rate", "n_channels", "n_epochs", "split",
        "epoch_type", "eeg_units", "schema_version", "data_protocol",
        "preprocessing_profile", "preprocessing_profile_version",
        "preprocessing_fingerprint", "preprocessing_json",
        "preprocessing_state_fingerprint", "preprocessing_state_json",
    )
    missing = [name for name in attrs if name not in h5.attrs]
    errors.extend(f"missing v2 attr '{name}'" for name in missing)
    if missing:
        return errors

    if str(h5.attrs["schema_version"]) != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if str(h5.attrs["data_protocol"]) != PROTOCOL_NAME:
        errors.append(f"data_protocol must be {PROTOCOL_NAME!r}")

    n_epochs = int(h5.attrs["n_epochs"])
    n_channels = int(h5.attrs["n_channels"])
    if n_epochs < 0 or n_channels <= 0:
        errors.append("n_epochs must be non-negative and n_channels positive")
    for name in ("sampling_rate", "source_sampling_rate"):
        value = float(h5.attrs[name])
        if not math.isfinite(value) or value <= 0:
            errors.append(f"{name} must be finite and positive")
    if not str(h5.attrs["eeg_units"]).strip():
        errors.append("eeg_units must be explicit")
    if str(h5.attrs["split"]) not in {"train", "validation", "test", "unspecified"}:
        errors.append("invalid split")

    eeg_name = _eeg_name(h5, schema)
    if eeg_name is None:
        errors.append("missing root EEG dataset ('data' or 'eeg')")
    else:
        eeg = h5[eeg_name]
        if eeg.ndim < 2 or eeg.shape[0] != n_epochs:
            errors.append(f"invalid {eeg_name} epoch dimension")
        if eeg.ndim >= 3 and eeg.shape[1] != n_channels:
            errors.append(f"{eeg_name}.shape[1] != n_channels")
        if not np.issubdtype(eeg.dtype, np.number):
            errors.append(f"{eeg_name} must be numeric")
        else:
            for start in range(0, eeg.shape[0], 64):
                if not np.isfinite(eeg[start:start + 64]).all():
                    errors.append(f"{eeg_name} contains non-finite values")
                    break

    if "channel_names" not in h5:
        errors.append("missing channel_names")
    else:
        channels = _strings(h5["channel_names"][:])
        if len(channels) != n_channels:
            errors.append("channel_names length != n_channels")
        if len(set(channels)) != len(channels) or any(not name for name in channels):
            errors.append("channel_names must be non-empty and unique")
    for name in ("valid_samples", "valid_eeg_samples"):
        if name in h5 and eeg_name is not None and h5[eeg_name].ndim >= 3:
            valid = np.asarray(h5[name][:])
            if valid.shape != (n_epochs,):
                errors.append(f"{name} must have shape (n_epochs,)")
            elif np.any(valid < 0) or np.any(valid > h5[eeg_name].shape[-1]):
                errors.append(f"{name} is outside the EEG time axis")

    profile_steps = []
    valid_profile_steps = []
    try:
        profile = json.loads(str(h5.attrs["preprocessing_json"]))
    except (TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid preprocessing_json: {exc}")
    else:
        digest = hashlib.sha256(canonical_json(profile).encode("utf-8")).hexdigest()
        if digest != str(h5.attrs["preprocessing_fingerprint"]):
            errors.append("preprocessing fingerprint mismatch")
        if not isinstance(profile, dict):
            errors.append("preprocessing_json must contain a JSON object")
        elif profile.get("profile_id") != str(h5.attrs["preprocessing_profile"]):
            errors.append("preprocessing profile id mismatch")
        profile_steps = profile.get("steps", []) if isinstance(profile, dict) else []
        if not isinstance(profile_steps, list):
            errors.append("preprocessing steps must be a JSON array")
            profile_steps = []
        for index, step in enumerate(profile_steps):
            if not isinstance(step, dict):
                errors.append(f"preprocessing step {index} must be a JSON object")
                continue
            valid_profile_steps.append(step)
            scope = step.get("fit_scope", "none")
            if scope not in {"none", "train", "source"}:
                errors.append(f"preprocessing step {index} has invalid fit_scope")
            if str(h5.attrs["split"]) == "test" and scope == "source":
                errors.append(f"preprocessing step {index} fits on a test source")

    try:
        state = json.loads(str(h5.attrs["preprocessing_state_json"]))
    except (TypeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid preprocessing_state_json: {exc}")
    else:
        digest = hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()
        if digest != str(h5.attrs["preprocessing_state_fingerprint"]):
            errors.append("preprocessing state fingerprint mismatch")
        if not isinstance(state, dict):
            errors.append("preprocessing_state_json must contain a JSON object")
        fitted_split = state.get("fitted_split") if isinstance(state, dict) else None
        if any(step.get("fit_scope") == "train" for step in valid_profile_steps):
            if fitted_split != "train":
                errors.append(
                    "train-fitted preprocessing requires state.fitted_split='train'"
                )
        if str(h5.attrs["split"]) == "test" and fitted_split == "test":
            errors.append("test preprocessing state was fitted on test")
        if isinstance(state, dict):
            array_hashes = {
                "mnn_matrix_sha256": "mnn_whitener",
                "scaler_median_sha256": "scaler_median",
                "scaler_iqr_sha256": "scaler_iqr",
            }
            for state_key, dataset_name in array_hashes.items():
                if state_key not in state:
                    continue
                if dataset_name not in h5:
                    errors.append(
                        f"preprocessing state references missing {dataset_name}"
                    )
                elif str(state[state_key]) != sha256_array(h5[dataset_name][:]):
                    errors.append(f"preprocessing state hash mismatch for {dataset_name}")
            session_hashes = state.get("mnn_matrix_sha256_by_session")
            session_ids = state.get("session_ids")
            if session_hashes is not None or session_ids is not None:
                if not isinstance(session_hashes, dict) or not isinstance(session_ids, list):
                    errors.append("invalid per-session MNN preprocessing state")
                elif "mnn_whiteners" not in h5:
                    errors.append("preprocessing state references missing mnn_whiteners")
                elif h5["mnn_whiteners"].shape[0] != len(session_ids):
                    errors.append("mnn_whiteners length does not match session_ids")
                else:
                    for index, session_id in enumerate(session_ids):
                        if session_id not in session_hashes:
                            errors.append(f"missing MNN hash for session {session_id}")
                        elif str(session_hashes[session_id]) != sha256_array(
                                h5["mnn_whiteners"][index]):
                            errors.append(
                                f"preprocessing state hash mismatch for session {session_id}"
                            )

    present = [name for name in STIMULUS_COLUMNS if name in h5]
    if len(present) != len(STIMULUS_COLUMNS):
        errors.append(
            f"incomplete stimulus manifest: {sorted(set(STIMULUS_COLUMNS) - set(present))}"
        )
    if len(present) == len(STIMULUS_COLUMNS):
        for name in STIMULUS_COLUMNS:
            if h5[name].shape != (n_epochs,):
                errors.append(f"{name} must have shape (n_epochs,)")
        if any(not value for value in _strings(h5["stimulus_id"][:])):
            errors.append("stimulus_id contains an empty value")
        if any(not value for value in _strings(h5["stimulus_modality"][:])):
            errors.append("stimulus_modality contains an empty value")
        if any(not value for value in _strings(
                h5["stimulus_alignment_source"][:])):
            errors.append("stimulus_alignment_source contains an empty value")
        for index, value in enumerate(_strings(h5["stimulus_transform_json"][:])):
            try:
                transform = json.loads(value)
            except (TypeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid stimulus_transform_json row {index}: {exc}")
                continue
            if not isinstance(transform, dict):
                errors.append(
                    f"stimulus_transform_json row {index} must be a JSON object")
        for digest in _strings(h5["stimulus_sha256"][:]):
            if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
                errors.append("invalid stimulus_sha256")
                break
        onsets = np.asarray(h5["stimulus_onset_sec"][:], dtype=float)
        durations = np.asarray(h5["stimulus_duration_sec"][:], dtype=float)
        if np.any(np.isfinite(onsets) & (onsets < 0)):
            errors.append("negative stimulus_onset_sec")
        if np.any(np.isfinite(durations) & (durations < 0)):
            errors.append("negative stimulus_duration_sec")

    exceptions = {
        "channel_names", "scaler_median", "scaler_iqr",
        "mnn_whitener", "mnn_whiteners", eeg_name,
    }
    for name, dataset in h5.items():
        if name in exceptions or dataset.ndim == 0:
            continue
        if name.startswith(("event_", "word_", "sentence_", "stimulus_")):
            if dataset.shape[0] != n_epochs:
                errors.append(f"row metadata '{name}' length != n_epochs")
    return errors
