"""HDF5 and stimulus-feature datasets for the ATM benchmark pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Dict, Iterable, Iterator, List, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from eeg2x.rawdata import (
    EXPECTED_GROOTSWAGERS_ROOT, EXPECTED_GROOTSWAGERS2025_ROOT,
    EXPECTED_NOD_ROOT, require_benchmark_output, require_raw_grootswagers_image,
    require_raw_grootswagers2025_image, require_raw_nod_image,
    require_raw_thingseeg_image, require_thingseeg_link,
)


NICE_PROFILE = "thingseeg-image-nice-v1"
ATM_PROFILE = "thingseeg-image-atm-v1"
REVE_NICE_PROFILE = "thingseeg-image-reve-nice-v1"
REVE_ATM_PROFILE = "thingseeg-image-reve-atm-v1"
LABRAM_ATM_PROFILE = "thingseeg-image-labram-atm-v1"
LUNA_NICE_PROFILE = "thingseeg-image-luna-nice-v1"
LUNA_ATM_PROFILE = "thingseeg-image-luna-atm-v1"
NOD_ATM_PROFILE = "nod-image-atm-v1"
NOD_REVE_ATM_PROFILE = "nod-image-reve-atm-v1"
NOD_LUNA_ATM_PROFILE = "nod-image-luna-atm-v1"
NOD_LABRAM_ATM_PROFILE = "nod-image-labram-atm-v1"
# NOD retrieval profiles: identical preprocessing to the NOD reconstruction
# profiles above, registered separately for the retrieval task.
NOD_NICE_CLEAN_PROFILE = "nod-image-nice-clean-v1"
NOD_LABRAM_NICE_CLEAN_PROFILE = "nod-image-labram-nice-clean-v1"
NOD_REVE_NICE_CLEAN_PROFILE = "nod-image-reve-nice-clean-v1"
NOD_LUNA_NICE_CLEAN_PROFILE = "nod-image-luna-nice-clean-v1"
# Grootswagers2019 (ds004018) image-generation profiles.
GROOTSWAGERS_ATM_PROFILE = "grootswagers-image-atm-v1"
GROOTSWAGERS_LABRAM_ATM_PROFILE = "grootswagers-image-labram-atm-v1"
GROOTSWAGERS_REVE_ATM_PROFILE = "grootswagers-image-reve-atm-v1"
GROOTSWAGERS_LUNA_ATM_PROFILE = "grootswagers-image-luna-atm-v1"

PROFILE_CONTRACTS = {
    NICE_PROFILE: {"prefix": "thingseeg", "n_samples": 250, "sampling_rate": 250},
    ATM_PROFILE: {"prefix": "thingseeg_atm", "n_samples": 250, "sampling_rate": 250},
    REVE_NICE_PROFILE: {
        "prefix": "thingseeg", "n_samples": 200, "sampling_rate": 200},
    REVE_ATM_PROFILE: {
        "prefix": "thingseeg_atm", "n_samples": 200, "sampling_rate": 200},
    LABRAM_ATM_PROFILE: {
        "prefix": "thingseeg_atm", "n_samples": 200, "sampling_rate": 200},
    LUNA_NICE_PROFILE: {
        "prefix": "thingseeg", "n_samples": 280, "sampling_rate": 256},
    LUNA_ATM_PROFILE: {
        "prefix": "thingseeg_atm", "n_samples": 280, "sampling_rate": 256},
    NOD_ATM_PROFILE: {
        # The NOD paper's epoch: -100..800 ms inclusive at the dataset's own
        # 250 Hz clock, i.e. 226 samples.
        "prefix": "nod_atm", "n_samples": 226, "sampling_rate": 250,
        "n_channels": 62, "dataset": "nod",
    },
    NOD_REVE_ATM_PROFILE: {
        "prefix": "nod_reve_atm", "n_samples": 200, "sampling_rate": 200,
        "n_channels": 62, "dataset": "nod",
    },
    NOD_LABRAM_ATM_PROFILE: {
        "prefix": "nod_labram_atm", "n_samples": 200, "sampling_rate": 200,
        "n_channels": 62, "dataset": "nod",
    },
    NOD_LUNA_ATM_PROFILE: {
        "prefix": "nod_luna_atm", "n_samples": 280, "sampling_rate": 256,
        "n_channels": 62, "dataset": "nod",
    },
    # NOD retrieval shares the reconstruction preprocessing; the retrieval
    # namespace keeps the benchmark's retrieval filename convention.
    NOD_NICE_CLEAN_PROFILE: {
        "prefix": "nod", "n_samples": 226, "sampling_rate": 250,
        "n_channels": 62, "dataset": "nod",
    },
    NOD_LABRAM_NICE_CLEAN_PROFILE: {
        "prefix": "nod", "n_samples": 200, "sampling_rate": 200,
        "n_channels": 62, "dataset": "nod",
    },
    NOD_REVE_NICE_CLEAN_PROFILE: {
        "prefix": "nod", "n_samples": 200, "sampling_rate": 200,
        "n_channels": 62, "dataset": "nod",
    },
    NOD_LUNA_NICE_CLEAN_PROFILE: {
        "prefix": "nod", "n_samples": 280, "sampling_rate": 256,
        "n_channels": 62, "dataset": "nod",
    },
    # Grootswagers2025 infant (ds005106): 32 channels, paper epoch [-0.1, 0.8] s.
    "grootswagers2025-image-atm-paper-v1": {
        "prefix": "grootswagers2025_atm", "n_samples": 226,
        "sampling_rate": 250, "n_channels": 32, "dataset": "grootswagers2025",
    },
    "grootswagers2025-image-labram-atm-paper-v1": {
        "prefix": "grootswagers2025_labram_atm", "n_samples": 200,
        "sampling_rate": 200, "n_channels": 32, "dataset": "grootswagers2025",
    },
    "grootswagers2025-image-reve-atm-paper-v1": {
        "prefix": "grootswagers2025_reve_atm", "n_samples": 200,
        "sampling_rate": 200, "n_channels": 32, "dataset": "grootswagers2025",
    },
    "grootswagers2025-image-luna-atm-paper-v1": {
        "prefix": "grootswagers2025_luna_atm", "n_samples": 280,
        "sampling_rate": 256, "n_channels": 32, "dataset": "grootswagers2025",
    },
    # Grootswagers2019 (ds004018): 63 channels, paper epoch [-0.1, 1.0] s.
    GROOTSWAGERS_ATM_PROFILE: {
        "prefix": "grootswagers_atm", "n_samples": 276, "sampling_rate": 250,
        "n_channels": 63, "dataset": "grootswagers",
    },
    GROOTSWAGERS_LABRAM_ATM_PROFILE: {
        "prefix": "grootswagers_labram_atm", "n_samples": 200, "sampling_rate": 200,
        "n_channels": 63, "dataset": "grootswagers",
    },
    GROOTSWAGERS_REVE_ATM_PROFILE: {
        "prefix": "grootswagers_reve_atm", "n_samples": 200, "sampling_rate": 200,
        "n_channels": 63, "dataset": "grootswagers",
    },
    GROOTSWAGERS_LUNA_ATM_PROFILE: {
        "prefix": "grootswagers_luna_atm", "n_samples": 280, "sampling_rate": 256,
        "n_channels": 63, "dataset": "grootswagers",
    },
}


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def subject_number(subject_id: str) -> int:
    match = re.fullmatch(r"sub-(\d+)", subject_id)
    if match is None:
        raise ValueError(f"Invalid ThingsEEG subject ID: {subject_id!r}")
    return int(match.group(1))


def sample_id_for_h5_row(path: str | Path, row: int) -> str:
    """Mirror benchmark_eval's single-authority image sample-ID convention."""
    if row < 0:
        raise ValueError("HDF5 row must be non-negative")
    return f"{Path(path).stem}_trial-{row:06d}"


def h5_fingerprint(path: str | Path) -> Dict[str, object]:
    """Return the protocol identity used in downstream artifact manifests."""
    candidate = require_benchmark_output(path)
    with h5py.File(candidate, "r") as h5:
        return {
            "path": str(candidate),
            "size_bytes": int(candidate.stat().st_size),
            "subject_id": _text(h5.attrs["subject_id"]),
            "split": _text(h5.attrs["split"]),
            "profile_id": _text(h5.attrs["preprocessing_profile"]),
            "preprocessing_fingerprint": _text(h5.attrs["preprocessing_fingerprint"]),
            "preprocessing_state_fingerprint": _text(
                h5.attrs["preprocessing_state_fingerprint"]),
            "n_epochs": int(h5["data"].shape[0]),
            "derived_inputs_used": bool(h5.attrs.get("derived_inputs_used", True)),
        }


def validate_atm_h5(path: str | Path, *, profile_id: str, split: str | None = None) -> None:
    candidate = require_benchmark_output(path)
    with h5py.File(candidate, "r") as h5:
        actual_profile = _text(h5.attrs.get("preprocessing_profile", ""))
        actual_split = _text(h5.attrs.get("split", ""))
        if actual_profile != profile_id:
            raise ValueError(
                f"{candidate} uses {actual_profile!r}, expected {profile_id!r}")
        if split is not None and actual_split != split:
            raise ValueError(f"{candidate} is split={actual_split!r}, expected {split!r}")
        if bool(h5.attrs.get("derived_inputs_used", True)):
            raise ValueError(f"Derived EEG input is forbidden: {candidate}")
        state = json.loads(_text(h5.attrs.get("preprocessing_state_json", "{}")))
        if bool(state.get("derived_inputs_used", True)):
            raise ValueError(
                f"Preprocessing provenance does not prove raw-only EEG input: {candidate}")
        contract = PROFILE_CONTRACTS.get(profile_id)
        if contract is None:
            raise ValueError(f"Unsupported ATM profile: {profile_id}")
        dataset = str(contract.get("dataset", "thingseeg"))
        source_root = {
            "nod": str(EXPECTED_NOD_ROOT.resolve()),
            "grootswagers": str(EXPECTED_GROOTSWAGERS_ROOT.resolve()),
            "grootswagers2025": str(EXPECTED_GROOTSWAGERS2025_ROOT.resolve()),
        }.get(dataset, str(require_thingseeg_link()))
        source_values = []
        for key in ("source_file", "mnn_fit_source_file", "source_epoch_file"):
            if state.get(key):
                source_values.append(str(state[key]))
        for key in ("source_files", "mnn_fit_source_files"):
            source_values.extend(str(value) for value in state.get(key, []))
        # 按**解析后**的路径比较，不用字面前缀：迁移阶段读的 v2 HDF5 记录的是
        # v2 仓库里的 ThingsEEG 链接（``EEG2X-Benchmark-v2-test/ThingsEEG``），
        # v3 的链接是另一个字面路径，但两者解析到同一个数据集根。按字面比较会把
        # 合法的 v2 产物误判为「越界」。
        resolved_root = Path(source_root).resolve()

        def _under_source_root(value: str) -> bool:
            return resolved_root in Path(value).resolve().parents

        if not source_values or not all(_under_source_root(v) for v in source_values):
            raise ValueError(
                f"EEG sources must be raw files below the declared dataset root "
                f"({resolved_root}): {candidate}")
        try:
            n_samples = int(contract["n_samples"])
            expected_rate = float(contract["sampling_rate"])
        except KeyError as exc:
            raise ValueError(f"Unsupported ATM profile: {profile_id}") from exc
        n_channels = int(contract.get("n_channels", 63))
        if h5["data"].shape[1:] != (n_channels, n_samples):
            raise ValueError(
                f"{profile_id} requires [N,{n_channels},{n_samples}], got "
                f"{tuple(h5['data'].shape)} in {candidate}")
        sampling_rate = float(h5.attrs.get("sampling_rate", 0.0))
        if sampling_rate != expected_rate:
            raise ValueError(
                f"{profile_id} requires {expected_rate:g} Hz, got {sampling_rate} in "
                f"{candidate}")
        required = {
            "stimulus_id", "stimulus_uri", "stimulus_sha256",
            "event_label", "event_object_number", "event_stim_number",
        }
        missing = sorted(required - set(h5.keys()))
        if missing:
            raise ValueError(f"{candidate} is missing manifest fields: {missing}")


def discover_subject_h5(
    directory: str | Path,
    subject_id: str,
    *,
    profile_id: str,
    split: str,
) -> List[Path]:
    """Discover only the formal train or concept-averaged test inputs."""
    root = require_benchmark_output(directory)
    try:
        namespace = str(PROFILE_CONTRACTS[profile_id]["prefix"])
    except KeyError as exc:
        raise ValueError(f"Unsupported ATM profile: {profile_id}") from exc
    prefix = f"{namespace}_{subject_id}_"

    if split == "train":
        paths = sorted(root.glob(f"{prefix}ses*_training_epoched.h5"))
    elif split == "test":
        paths = [root / f"{prefix}test_epoched.h5"]
        paths = [path for path in paths if path.is_file()]
    else:
        raise ValueError(f"Unsupported split: {split!r}")
    if not paths:
        raise FileNotFoundError(
            f"No {profile_id} {split} HDF5 files for {subject_id} below {root}")
    for path in paths:
        validate_atm_h5(path, profile_id=profile_id, split=split)
    return paths


@dataclass(frozen=True)
class StimulusRow:
    stimulus_id: str
    uri: str
    sha256: str
    label: str
    concept_number: int
    stimulus_number: int
    source_h5: str
    source_row: int
    source_uris: tuple[str, ...] = ()


def iter_stimulus_rows(h5_paths: Iterable[str | Path]) -> Iterator[StimulusRow]:
    for path_value in h5_paths:
        path = require_benchmark_output(path_value)
        with h5py.File(path, "r") as h5:
            profile_id = _text(h5.attrs.get("preprocessing_profile", ""))
            contract = PROFILE_CONTRACTS.get(profile_id, {})
            dataset = str(contract.get("dataset", h5.attrs.get("dataset", "thingseeg")))
            things_root = require_thingseeg_link() if dataset != "nod" else EXPECTED_NOD_ROOT
            n_rows = int(h5["data"].shape[0])
            for row in range(n_rows):
                uri = _text(h5["stimulus_uri"][row])
                if dataset == "nod":
                    require_raw_nod_image(uri, things_root)
                elif dataset == "grootswagers":
                    require_raw_grootswagers_image(uri)
                elif dataset == "grootswagers2025":
                    require_raw_grootswagers2025_image(uri)
                else:
                    require_raw_thingseeg_image(uri, things_root)
                source_uris: tuple[str, ...] = ()
                if "event_source_image_ids_json" in h5 and dataset == "nod":
                    try:
                        ids = json.loads(_text(h5["event_source_image_ids_json"][row]))
                        source_uris = tuple(str((EXPECTED_NOD_ROOT / "stimuli" / "ImageNet" /
                                                 f"{image_id}.JPEG").resolve()) for image_id in ids)
                        for source_uri in source_uris:
                            require_raw_nod_image(source_uri, things_root)
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise ValueError(f"Invalid NOD source image manifest in {path} row {row}") from exc
                yield StimulusRow(
                    stimulus_id=_text(h5["stimulus_id"][row]),
                    uri=uri,
                    sha256=_text(h5["stimulus_sha256"][row]),
                    label=_text(h5["event_label"][row]),
                    concept_number=int(h5["event_object_number"][row]),
                    stimulus_number=int(h5["event_stim_number"][row]),
                    source_h5=str(path),
                    source_row=row,
                    source_uris=source_uris,
                )


def unique_stimuli(h5_paths: Iterable[str | Path]) -> List[StimulusRow]:
    """Deduplicate repeated EEG trials while rejecting inconsistent manifests."""
    rows: Dict[str, StimulusRow] = {}
    for row in iter_stimulus_rows(h5_paths):
        previous = rows.get(row.stimulus_id)
        if previous is not None:
            if (previous.uri, previous.sha256, previous.label) != (
                    row.uri, row.sha256, row.label):
                raise ValueError(
                    f"Stimulus {row.stimulus_id!r} has inconsistent HDF5 metadata")
            continue
        rows[row.stimulus_id] = row
    return sorted(
        rows.values(), key=lambda item: (
            item.concept_number, item.stimulus_number, item.stimulus_id))


class FeatureStore:
    """Read-only feature store with a stable stimulus-ID row map."""

    def __init__(self, path: str | Path, required_kinds: Sequence[str] = ("clip_vit_h",)):
        self.path = require_benchmark_output(path)
        self.required_kinds = tuple(required_kinds)
        with h5py.File(self.path, "r") as h5:
            if bool(h5.attrs.get("derived_inputs_used", True)):
                raise ValueError(f"Feature store is not raw-image-derived: {self.path}")
            ids = [_text(value) for value in h5["stimulus_id"][:]]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate stimulus IDs in feature store: {self.path}")
            for kind in self.required_kinds:
                if kind not in h5:
                    raise ValueError(f"Feature store {self.path} has no {kind!r} dataset")
                if not bool(h5[kind].attrs.get("complete", False)):
                    raise ValueError(f"Feature dataset {kind!r} is incomplete in {self.path}")
                if h5[kind].shape[0] != len(ids):
                    raise ValueError(f"Feature row mismatch for {kind!r} in {self.path}")
                dataset = h5[kind]
                row_bytes = max(
                    int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize, 1)
                rows_per_chunk = max(1, min(256, 16 * 1024 * 1024 // row_bytes))
                for first in range(0, dataset.shape[0], rows_per_chunk):
                    stop = min(first + rows_per_chunk, dataset.shape[0])
                    if not np.isfinite(dataset[first:stop]).all():
                        raise ValueError(
                            f"Feature dataset {kind!r} contains non-finite values "
                            f"in rows [{first}, {stop}) of {self.path}")
        self.row_by_id = {value: index for index, value in enumerate(ids)}
        self._handle: h5py.File | None = None
        self._pid: int | None = None

    def _h5(self) -> h5py.File:
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            if self._handle is not None:
                self._handle.close()
            self._handle = h5py.File(self.path, "r")
            self._pid = pid
        return self._handle

    def row(self, stimulus_id: str) -> int:
        try:
            return self.row_by_id[stimulus_id]
        except KeyError as exc:
            raise KeyError(
                f"Stimulus {stimulus_id!r} is absent from {self.path}") from exc

    def tensor(self, kind: str, row: int) -> torch.Tensor:
        if kind not in self.required_kinds:
            raise ValueError(f"Feature kind {kind!r} was not validated")
        return torch.from_numpy(np.asarray(self._h5()[kind][row], dtype=np.float32))

    def all(self, kind: str) -> torch.Tensor:
        if kind not in self.required_kinds:
            raise ValueError(f"Feature kind {kind!r} was not validated")
        return torch.from_numpy(np.asarray(self._h5()[kind][:], dtype=np.float32))

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._pid = None


class ATMH5Dataset(Dataset):
    """Stream or memory-cache EEG rows and align them through stimulus IDs."""

    def __init__(
        self,
        h5_paths: Sequence[str | Path],
        feature_store: str | Path,
        *,
        profile_id: str,
        feature_kind: str = "clip_vit_h",
        cache_eeg: bool = False,
        cache_features: bool = False,
    ):
        if not h5_paths:
            raise ValueError("ATMH5Dataset needs at least one HDF5 file")
        self.paths = [require_benchmark_output(path) for path in h5_paths]
        self.feature_kind = feature_kind
        self.features = FeatureStore(feature_store, (feature_kind,))
        self.feature_cache = (
            self.features.all(feature_kind) if cache_features else None)
        self.index: List[tuple[int, int, int, str, int, str]] = []
        self.fingerprints = []
        self._handles: Dict[int, h5py.File] = {}
        self._pid: int | None = None
        cached: List[np.ndarray] = []
        for file_index, path in enumerate(self.paths):
            validate_atm_h5(path, profile_id=profile_id)
            self.fingerprints.append(h5_fingerprint(path))
            with h5py.File(path, "r") as h5:
                subject_id = _text(h5.attrs["subject_id"])
                token = subject_number(subject_id)
                n_rows = int(h5["data"].shape[0])
                if cache_eeg:
                    cached.append(np.asarray(h5["data"][:], dtype=np.float32))
                for row in range(n_rows):
                    stimulus_id = _text(h5["stimulus_id"][row])
                    feature_row = self.features.row(stimulus_id)
                    label = _text(h5["event_label"][row])
                    self.index.append((file_index, row, token, stimulus_id, feature_row, label))
        self.eeg_cache = np.concatenate(cached, axis=0) if cached else None

    def __len__(self) -> int:
        return len(self.index)

    def _h5(self, file_index: int) -> h5py.File:
        pid = os.getpid()
        if self._pid != pid:
            for handle in self._handles.values():
                handle.close()
            self._handles = {}
            self._pid = pid
        if file_index not in self._handles:
            self._handles[file_index] = h5py.File(self.paths[file_index], "r")
        return self._handles[file_index]

    def __getitem__(self, item: int) -> Dict[str, object]:
        file_index, row, token, stimulus_id, feature_row, label = self.index[item]
        if self.eeg_cache is None:
            eeg = np.asarray(self._h5(file_index)["data"][row], dtype=np.float32)
        else:
            eeg = self.eeg_cache[item]
        return {
            "eeg": torch.from_numpy(eeg),
            "target": (
                self.feature_cache[feature_row]
                if self.feature_cache is not None
                else self.features.tensor(self.feature_kind, feature_row)
            ),
            "subject_id": torch.tensor(token, dtype=torch.long),
            "stimulus_id": stimulus_id,
            "label": label,
            "feature_row": torch.tensor(feature_row, dtype=torch.long),
        }

    def labels(self) -> List[str]:
        return [item[-1] for item in self.index]

    def stimulus_ids(self) -> List[str]:
        return [item[3] for item in self.index]

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles = {}
        self.features.close()


class EmbeddingH5Dataset(Dataset):
    """已编码的 EEG 嵌入与 CLIP 目标，供投影头训练/评估。

    逐字迁自 v2 ``model/atm/training.py:EmbeddingH5Dataset``。读取 ``embed_split``
    写出的 ``<split>_embeddings.h5``（``eeg_embedding`` / ``clip_target``）。
    """

    def __init__(self, path: str | Path, *, cache: bool = True):
        self.path = require_benchmark_output(path)
        with h5py.File(self.path, "r") as h5:
            if bool(h5.attrs.get("derived_inputs_used", True)):
                raise ValueError(f"Invalid ATM embedding provenance: {self.path}")
            self.source_h5 = json.loads(h5.attrs["source_h5_json"])
            self.length = len(h5["eeg_embedding"])
            if cache:
                self.conditions = np.asarray(h5["eeg_embedding"][:], dtype=np.float32)
                self.targets = np.asarray(h5["clip_target"][:], dtype=np.float32)
            else:
                self.conditions = None
                self.targets = None
        self.handle: h5py.File | None = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.conditions is not None:
            condition = self.conditions[index]
            target = self.targets[index]
        else:
            if self.handle is None:
                self.handle = h5py.File(self.path, "r")
            condition = np.asarray(self.handle["eeg_embedding"][index], dtype=np.float32)
            target = np.asarray(self.handle["clip_target"][index], dtype=np.float32)
        return {
            "condition": torch.from_numpy(condition),
            "target": torch.from_numpy(target),
        }


def artifact_manifest(
    *,
    artifact_type: str,
    source_h5: Sequence[Dict[str, object]],
    config: Dict[str, object],
    extra: Dict[str, object] | None = None,
) -> Dict[str, object]:
    value: Dict[str, object] = {
        "artifact_type": artifact_type,
        "source_h5": list(source_h5),
        "config": dict(config),
        "derived_inputs_used": False,
    }
    if extra:
        value.update(extra)
    return value


def write_json(path: str | Path, value: object) -> Path:
    output = require_benchmark_output(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, output)
    return output
