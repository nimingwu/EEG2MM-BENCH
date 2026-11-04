"""Shared, versioned data protocol for EEG2X-Benchmark v2.

The benchmark keeps the legacy root datasets consumed by existing models, but
all loaders also write the metadata defined here.  The additional metadata is
what makes preprocessing and stimulus alignment reproducible without tying the
contract to a dataset-specific filename.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "2.0.0"
PROTOCOL_NAME = "eeg2x-data-protocol"
VALID_SPLITS = {"train", "validation", "test", "unspecified"}
VALID_FIT_SCOPES = {"none", "train", "source"}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return stable JSON used for profile fingerprints and HDF5 metadata."""
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


@dataclass(frozen=True)
class ProcessingStep:
    """One ordered preprocessing operation.

    ``fit_scope`` is explicit because normalization/whitening fitted on test
    samples is a materially different benchmark protocol.
    """

    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    fit_scope: str = "none"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Processing step name must not be empty")
        if self.fit_scope not in VALID_FIT_SCOPES:
            raise ValueError(
                f"Invalid fit_scope {self.fit_scope!r}; expected one of "
                f"{sorted(VALID_FIT_SCOPES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "fit_scope": self.fit_scope,
        }


@dataclass(frozen=True)
class PreprocessingProfile:
    """Versioned, ordered EEG/stimulus preprocessing definition."""

    profile_id: str
    task_id: str
    steps: Sequence[ProcessingStep]
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol": PROTOCOL_NAME,
            "schema_version": SCHEMA_VERSION,
            "profile_id": self.profile_id,
            "profile_version": self.version,
            "task_id": self.task_id,
            "steps": [step.to_dict() for step in self.steps],
        }

    @property
    def fingerprint(self) -> str:
        payload = canonical_json(self.to_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def step_names(self) -> List[str]:
        return [step.name for step in self.steps]


@dataclass(frozen=True)
class StimulusRecord:
    """Canonical stimulus identity and EEG-clock alignment for one sample."""

    stimulus_id: str
    modality: str
    uri: str = ""
    sha256: str = ""
    onset_sec: float = math.nan
    duration_sec: float = math.nan
    alignment_source: str = ""
    transform: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stimulus_id:
            raise ValueError("stimulus_id must not be empty")
        if not self.modality:
            raise ValueError("stimulus modality must not be empty")
        if self.sha256 and not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise ValueError("stimulus sha256 must contain exactly 64 hex digits")


def make_stimulus_id(dataset_id: str, native_id: Any) -> str:
    """Build a stable ID without relying on an absolute filesystem path."""
    dataset = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(dataset_id)).strip("-")
    native = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(native_id)).strip("-")
    if not dataset or not native:
        raise ValueError("dataset_id and native stimulus id must be non-empty")
    return f"{dataset}:{native}"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a stimulus without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash array shape, dtype and C-order bytes for fitted-state provenance."""
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(canonical_json({
        "shape": list(array.shape), "dtype": array.dtype.str,
    }).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def normalize_channel_name(name: str) -> str:
    """Normalize common EEG channel spellings while preserving unknown names."""
    compact = re.sub(r"[\s_-]+", "", str(name).strip())
    upper = compact.upper()
    special = {
        "FPZ": "Fpz", "FP1": "Fp1", "FP2": "Fp2",
        "AFZ": "AFz", "FZ": "Fz", "FCZ": "FCz", "CZ": "Cz",
        "CPZ": "CPz", "PZ": "Pz", "POZ": "POz", "OZ": "Oz",
    }
    if upper in special:
        return special[upper]
    match = re.fullmatch(r"([A-Z]+)(\d+|Z)", upper)
    if match:
        prefix, suffix = match.groups()
        return prefix + ("z" if suffix == "Z" else suffix)
    return compact


def normalize_channel_names(names: Iterable[str]) -> List[str]:
    normalized = [normalize_channel_name(name) for name in names]
    duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate channels after normalization: {duplicates}")
    if any(not name for name in normalized):
        raise ValueError("Channel names must not be empty")
    return normalized


def _rate_fraction(rate: float) -> Fraction:
    value = float(rate)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Sampling rate must be finite and positive, got {rate!r}")
    return Fraction(str(value)).limit_denominator(1_000_000)


def expected_resampled_length(n_samples: int, orig_sr: float, target_sr: float) -> int:
    """Length used by scipy.signal.resample_poly for a finite input."""
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    ratio = _rate_fraction(target_sr) / _rate_fraction(orig_sr)
    return (n_samples * ratio.numerator + ratio.denominator - 1) // ratio.denominator


def resample_eeg(
    data: np.ndarray,
    orig_sr: float,
    target_sr: float,
    *,
    axis: int = -1,
) -> np.ndarray:
    """Resample EEG at an exact rational rate, including 256/512 -> 250 Hz.

    The old loaders rounded the ratio and silently treated 256 Hz as 250 Hz.
    This function uses polyphase anti-alias filtering and verifies output length.
    """
    array = np.asarray(data)
    if array.ndim == 0:
        raise ValueError("EEG data must have at least one dimension")
    orig = _rate_fraction(orig_sr)
    target = _rate_fraction(target_sr)
    if orig == target:
        return array.astype(np.float32, copy=True)

    from scipy.signal import resample_poly

    ratio = target / orig
    output = resample_poly(
        array.astype(np.float64, copy=False),
        ratio.numerator,
        ratio.denominator,
        axis=axis,
        padtype="line",
    )
    expected = expected_resampled_length(array.shape[axis], float(orig), float(target))
    if output.shape[axis] != expected:
        raise RuntimeError(
            f"Resampling length mismatch: expected {expected}, got {output.shape[axis]}"
        )
    return output.astype(np.float32, copy=False)


def seconds_to_sample(seconds: float, sampling_rate: float) -> int:
    """Map a timestamp to the nearest sample on the declared output clock."""
    if not math.isfinite(float(seconds)):
        raise ValueError(f"Timestamp must be finite, got {seconds!r}")
    _rate_fraction(sampling_rate)
    return int(round(float(seconds) * float(sampling_rate)))


def write_common_h5_metadata(
    h5: Any,
    *,
    dataset_id: str,
    task_id: str,
    subject_id: str,
    sampling_rate: float,
    source_sampling_rate: float,
    n_channels: int,
    n_epochs: int,
    split: str,
    epoch_type: str,
    eeg_units: str,
    profile: PreprocessingProfile,
    preprocessing_state: Mapping[str, Any] | None = None,
    adapter_version: str = "1.0.0",
) -> None:
    """Write common v2 attrs while retaining legacy attrs used by models."""
    normalized_split = str(split).lower()
    if normalized_split == "training":
        normalized_split = "train"
    if normalized_split not in VALID_SPLITS:
        raise ValueError(
            f"Invalid split {split!r}; expected one of {sorted(VALID_SPLITS)}"
        )
    if profile.task_id != task_id:
        raise ValueError(
            f"Profile task {profile.task_id!r} does not match HDF5 task {task_id!r}"
        )
    if n_channels <= 0 or n_epochs < 0:
        raise ValueError("n_channels must be positive and n_epochs non-negative")

    h5.attrs["dataset"] = str(dataset_id)
    h5.attrs["dataset_id"] = str(dataset_id)
    h5.attrs["task_id"] = str(task_id)
    h5.attrs["subject_id"] = str(subject_id)
    h5.attrs["sampling_rate"] = float(sampling_rate)
    h5.attrs["source_sampling_rate"] = float(source_sampling_rate)
    h5.attrs["n_channels"] = int(n_channels)
    h5.attrs["n_epochs"] = int(n_epochs)
    h5.attrs["split"] = normalized_split
    h5.attrs["epoch_type"] = str(epoch_type)
    h5.attrs["eeg_units"] = str(eeg_units)
    h5.attrs["schema_version"] = SCHEMA_VERSION
    h5.attrs["data_protocol"] = PROTOCOL_NAME
    # 种子协议：视频侧 v2 的每个 H5 都写了这四个属性，v3 的 protocol 从图像侧
    # 迁入时漏掉了；补上后 validate_protocol_v2 才能在 v3 产物上通过。
    h5.attrs["seed_protocol"] = SEED_PROTOCOL_NAME
    h5.attrs["seed_value"] = int(SEED)
    h5.attrs["seed_state_json"] = canonical_json(seed_state())
    h5.attrs["seed_fingerprint"] = seed_fingerprint()
    h5.attrs["adapter_version"] = str(adapter_version)
    h5.attrs["preprocessing_profile"] = profile.profile_id
    h5.attrs["preprocessing_profile_version"] = profile.version
    h5.attrs["preprocessing_fingerprint"] = profile.fingerprint
    h5.attrs["preprocessing_json"] = profile.json
    state_json = canonical_json(dict(preprocessing_state or {}))
    h5.attrs["preprocessing_state_json"] = state_json
    h5.attrs["preprocessing_state_fingerprint"] = hashlib.sha256(
        state_json.encode("utf-8")).hexdigest()


def write_stimulus_manifest(h5: Any, records: Sequence[StimulusRecord]) -> None:
    """Write row-aligned stimulus identity, transformation and clock metadata."""
    import h5py

    n_epochs = int(h5.attrs.get("n_epochs", -1))
    if n_epochs >= 0 and len(records) != n_epochs:
        raise ValueError(
            f"Stimulus manifest has {len(records)} rows but HDF5 has {n_epochs} epochs"
        )
    string_dtype = h5py.string_dtype(encoding="utf-8")

    def strings(values: Iterable[str]) -> np.ndarray:
        return np.asarray(list(values), dtype=object)

    columns = {
        "stimulus_id": strings(record.stimulus_id for record in records),
        "stimulus_modality": strings(record.modality for record in records),
        "stimulus_uri": strings(record.uri for record in records),
        "stimulus_sha256": strings(record.sha256.lower() for record in records),
        "stimulus_alignment_source": strings(
            record.alignment_source for record in records
        ),
        "stimulus_transform_json": strings(
            canonical_json(dict(record.transform)) for record in records
        ),
    }
    for name, values in columns.items():
        if name in h5:
            del h5[name]
        h5.create_dataset(name, data=values, dtype=string_dtype)

    numeric_columns = {
        "stimulus_onset_sec": np.asarray(
            [record.onset_sec for record in records], dtype=np.float64
        ),
        "stimulus_duration_sec": np.asarray(
            [record.duration_sec for record in records], dtype=np.float64
        ),
    }
    for name, values in numeric_columns.items():
        if name in h5:
            del h5[name]
        h5.create_dataset(name, data=values)


# ---------------------------------------------------------------------------
# 随机种子协议
# ---------------------------------------------------------------------------

#: 整个 benchmark 唯一的随机种子。
SEED = 2000
SEED_PROTOCOL_NAME = "eeg2x-seed-protocol"
SEED_PROTOCOL_VERSION = "1.0.0"

#: 允许从主种子派生的独立随机流。显式列出，避免某个组件悄悄发明新的常量。
DERIVABLE_COMPONENTS = (
    "train_shuffle",
    "validation_split",
    "weight_init",
    "generation_noise",
    "metric_trials",
)


def derive_seed(component: str) -> int:
    """从主种子派生一个稳定的、按组件名区分的种子。

    两个随机流在同一进程里必须互不相同时使用（例如训练打乱与验证切分）。
    返回值只由组件名与协议版本决定，因此跨机器可复现。
    """
    if component not in DERIVABLE_COMPONENTS:
        raise KeyError(
            f"Unknown seed component {component!r}; "
            f"expected one of {list(DERIVABLE_COMPONENTS)}"
        )
    payload = canonical_json(
        {"protocol": SEED_PROTOCOL_NAME, "version": SEED_PROTOCOL_VERSION,
         "seed": SEED, "component": component}
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def seed_state() -> Dict[str, Any]:
    """写进 HDF5 属性、并被校验器哈希的种子协议状态。"""
    return {
        "protocol": SEED_PROTOCOL_NAME,
        "version": SEED_PROTOCOL_VERSION,
        "seed": SEED,
        "derivable_components": list(DERIVABLE_COMPONENTS),
    }


def seed_fingerprint() -> str:
    """种子协议定义的指纹。"""
    return hashlib.sha256(
        canonical_json(seed_state()).encode("utf-8")
    ).hexdigest()
