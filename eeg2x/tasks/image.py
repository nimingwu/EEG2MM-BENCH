"""
EEG-to-Image Task Interface
============================
Standardised task definitions, I/O JSON schemas, and validation for the
EEG → Image downstream benchmark.

Subtasks defined (per EEG多模态下游任务Benchmark构建.docx §6):
  1. image_retrieval    — EEG-conditioned image retrieval (zero-shot / few-shot)
  2. image_generation    — EEG-to-Image reconstruction / generation
  3. concept_classification — EEG-based object-concept classification

Datasets supported:
  - ThingsEEG  (Grootswagers et al., 2022, OpenNeuro ds003825)
               THINGS concepts (1854), 22K images, RSVP @ 10 Hz
  - NOD-EEG    (Zhang et al., 2025, OpenNeuro ds005811)
               ImageNet categories (~200), 3.2K images/subject, passive viewing

Usage
-----
    from eeg2image_task_interface import (
        TaskSpec, Sample, RetrievalOutput, validate_output, load_schema_registry,
    )
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

import numpy as np


# ============================================================================
# Constants
# ============================================================================

# 采样率与通道数**不在这里硬编码**：它们随 (数据集 x 编码器 profile) 变化。
# 实测值：ThingsEEG-ATM 是 250 Hz / 63 通道，LaBraM 与 REVE 是 200 Hz，
# LUNA 是 256 Hz（右补零到 280），ds004018 与 ds005106 各有自己的几何，
# NOD 是 250 Hz / 64 通道。权威来源是产物 HDF5 本身，见 ``EegSpec.from_h5``。
# 旧的 ``SAMPLING_RATE = 250 / N_CHANNELS = 32`` 只对某个早期草案成立，
# 用它描述这些数据集是错的（例如 ThingsEEG 实际是 63 通道而非 32）。
EEG_UNIT = "µV"

BENCHMARK_VERSION = "0.1.0"

# ---- subtask registry ------------------------------------------------------

class Subtask(Enum):
    IMAGE_RETRIEVAL       = "image_retrieval"
    IMAGE_GENERATION      = "image_generation"
    CONCEPT_CLASSIFICATION = "concept_classification"

    @classmethod
    def all_values(cls) -> List[str]:
        return [e.value for e in cls]


# ============================================================================
# Task specification  (static metadata per subtask)
# ============================================================================

@dataclass
class TaskSpec:
    """Static specification of a single EEG→Image subtask.

    Defines what the model receives, what it must produce, and how it is
    evaluated.  One instance per Subtask enum value.
    """
    task_name: str
    task_id: str                    # e.g. "eeg2img-retrieval-v1"
    description: str

    # ---- I/O shapes & types ----
    #: (n_channels, n_times)，``None`` = 随数据集与编码器 profile 变化。
    #: 实例化时的具体几何由 :meth:`EegSpec.from_h5` 从产物 HDF5 读出。
    eeg_shape: Optional[Tuple[int, int]]
    eeg_dtype: str                  # "float32"
    output_type: str                # "ranking" | "image" | "class_probs"
    output_spec: Dict[str, Any]     # detailed output schema (see below)

    # ---- evaluation ----
    primary_metric: str             # e.g. "top1_acc" "fid" "clip_similarity"
    secondary_metrics: List[str]

    # ---- dataset constraints ----
    min_trials_per_concept: int
    # Concept counts vary by dataset — see per-dataset config.
    # ThingsEEG: 1854 concepts (1654 train / 200 test)
    # NOD-EEG:   ~200 ImageNet categories (split TBD)
    dataset_num_classes: Dict[str, Any] = field(default_factory=dict)

    #: 指标方向：列在这里的越低越好，其余越高越好。
    lower_is_better: List[str] = field(default_factory=list)

    # ---- metadata ----
    benchmark_version: str = BENCHMARK_VERSION
    references: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pre-defined task specs  (aligned with doc §6 Table 0)
# ---------------------------------------------------------------------------

TASK_IMAGE_RETRIEVAL = TaskSpec(
    task_name="EEG-Conditioned Image Retrieval",
    task_id="eeg2img-retrieval-v2",
    description=(
        "Given an EEG epoch recorded while a subject viewed image I, "
        "retrieve I from a candidate pool of N images.  "
        "The model outputs a ranking over the candidate set.  "
        "Supports ThingsEEG (1854 THINGS concepts, zero-shot) and "
        "NOD-EEG (~200 ImageNet categories)."
    ),
    eeg_shape=None,          # 随数据集/profile 变化，见 EegSpec.from_h5
    eeg_dtype="float32",
    output_type="ranking",
    output_spec={
        "candidate_pool_size": "int  (≥1)",
        "ranked_indices":       "int[N]  (0-indexed into candidate pool, best first)",
        "similarity_scores":    "float[N]  (optional)",
    },
    min_trials_per_concept=1,
    dataset_num_classes={"thingseeg": 1854, "nod": None},  # NOD: varies by split
    primary_metric="top1_accuracy",
    secondary_metrics=["top5_accuracy", "mAP", "recall@k"],
    references=[
        "ThingsEEG: Grootswagers et al. (2022) Sci Data 9, 3",
        "NOD-EEG: Zhang et al. (2025) Sci Data 12, 1598",
        "NeuroCLIP (Du et al., 2024) — zero-shot retrieval on THINGS-EEG2",
    ],
)

TASK_IMAGE_GENERATION = TaskSpec(
    task_name="EEG-to-Image Generation",
    task_id="eeg2img-generation-v2",
    description=(
        "Given an EEG epoch recorded during visual stimulation, generate "
        "the viewed image.  Output is an RGB image tensor.  "
        "Quality measured against the ground-truth stimulus image.  "
        "Supports both ThingsEEG (THINGS images) and NOD-EEG (ImageNet)."
    ),
    eeg_shape=None,          # 随数据集/profile 变化，见 EegSpec.from_h5
    eeg_dtype="float32",
    output_type="image",
    output_spec={
        "image_shape":   "[3, H, W]  (C=RGB, H/W task-defined)",
        "image_dtype":   "float32  or  uint8",
        "value_range":   "[vmin, vmax]  e.g. [-1, 1] or [0, 255]",
        "pixel_format":  '"RGB" | "BGR"',
    },
    min_trials_per_concept=1,
    dataset_num_classes={"thingseeg": 1854, "nod": 200,
                         "grootswagers": 200, "grootswagers2025": 200},
    # 与 ``eeg2x.evaluation.registry`` 实际评测的 8 项一致（单一事实来源）。
    primary_metric="clip_similarity",
    secondary_metrics=["pixcorr", "ssim", "alexnet_2", "alexnet_5",
                       "inception", "swav", "fid"],
    lower_is_better=["swav", "fid"],
    references=[
        "DynaMind (Chang et al., 2025) — EEG → dynamic scene reconstruction",
        "NeuroCLIP (Du et al., 2024)",
    ],
)

TASK_CONCEPT_CLASSIFICATION = TaskSpec(
    task_name="EEG Object-Concept Classification",
    task_id="eeg2img-classification-v2",
    description=(
        "Given an EEG epoch, predict which object category was being viewed.  "
        "For ThingsEEG: 1854-way classification over THINGS concepts.  "
        "For NOD-EEG: N-way classification over ImageNet categories "
        "(N depends on the dataset split, typically ~200)."
    ),
    eeg_shape=None,          # 随数据集/profile 变化，见 EegSpec.from_h5
    eeg_dtype="float32",
    output_type="class_probs",
    output_spec={
        "num_classes":      "int  (dataset-dependent; see dataset_num_classes)",
        "predicted_class":  "int  (0-indexed category id)",
        "class_probs":      "float[num_classes]  (softmax, optional)",
        "class_names":      "str[num_classes]  (human-readable, optional)",
    },
    min_trials_per_concept=1,
    dataset_num_classes={"thingseeg": 1854, "nod": None},
    primary_metric="top1_accuracy",
    secondary_metrics=["top5_accuracy", "mean_per_class_accuracy"],
    references=[
        "Grootswagers et al. (2022)",
        "Zhang et al. (2025) — NOD-EEG animate/inanimate classification",
    ],
)

# Registry
TASK_REGISTRY: Dict[str, TaskSpec] = {
    Subtask.IMAGE_RETRIEVAL.value:       TASK_IMAGE_RETRIEVAL,
    Subtask.IMAGE_GENERATION.value:      TASK_IMAGE_GENERATION,
    Subtask.CONCEPT_CLASSIFICATION.value: TASK_CONCEPT_CLASSIFICATION,
}


# ============================================================================
# Sample JSON schema  (input side — one sample = one EEG epoch + metadata)
# ============================================================================

@dataclass
class EegSpec:
    """In-line or referenced EEG data descriptor.

    默认值**不表示任何具体数据集**：采样率与通道列表由 ``from_h5`` 从产物
    HDF5 读取。手工构造时必须显式给出这两个字段。
    """
    shape: List[int]                        # [n_channels, n_times]
    sampling_rate_hz: Optional[int] = None
    channels: List[str] = field(default_factory=list)
    units: str = EEG_UNIT
    t_pre_s: float = -0.2
    t_post_s: float = 0.8
    data: Optional[List[List[float]]] = None   # in-line array OR …
    data_ref: Optional[str] = None             # reference: "h5://path#dataset[row]"
    preprocessing: List[str] = field(default_factory=list)
    # e.g. ["resampled_250hz", "baseline_corrected_-200_0ms"]

    @classmethod
    def from_h5(cls, path: str | Path, source: str = "data") -> "EegSpec":
        """从 benchmark HDF5 读出真实的 EEG 契约。

        ``source`` 是根 EEG 数据集名（``data`` 或 ``eeg``）。目前 v3 的产物
        用 ``data``；文本任务用 ``eeg``。
        """
        import h5py

        with h5py.File(path, "r") as h5:
            if source not in h5:
                raise KeyError(f"{path} has no {source!r} dataset")
            data = h5[source]
            n_channels, n_times = int(data.shape[1]), int(data.shape[2])
            rate = h5.attrs.get("sampling_rate")
            if rate is None:
                raise ValueError(f"{path} has no sampling_rate attribute")
            channels: List[str] = []
            if "channel_names" in h5:
                channels = [v.decode("utf-8") if isinstance(v, bytes) else str(v)
                            for v in h5["channel_names"][:]]
            if len(channels) != n_channels:
                raise ValueError(
                    f"{path}: {len(channels)} channel_names for {n_channels} channels")
            t_pre = h5.attrs.get("t_pre", h5.attrs.get("t_pre_s", -0.2))
            t_post = h5.attrs.get("t_post", h5.attrs.get("t_post_s", 0.8))
            # 单位随 profile 变化：ThingsEEG-NOD 的 MNN 分支是 dimensionless_whitened，
            # 其余是 µV。硬编码 µV 会把白化后的产物描述错。
            units = str(h5.attrs.get("eeg_units", EEG_UNIT))
            steps: List[str] = []
            try:
                profile = json.loads(str(h5.attrs.get("preprocessing_json", "{}")))
                steps = [step["name"] for step in profile.get("steps", [])
                         if isinstance(step, dict) and step.get("name")]
            except (TypeError, json.JSONDecodeError):
                steps = []
        return cls(
            shape=[n_channels, n_times],
            sampling_rate_hz=int(round(float(rate))),
            channels=channels,
            units=units,
            t_pre_s=float(t_pre),
            t_post_s=float(t_post),
            data_ref=f"h5://{Path(path).name}#{source}[row]",
            preprocessing=steps,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def eeg_spec_from_h5_files(paths: List[str], source: str = "data") -> EegSpec:
    """把一个评测目录里的 HDF5 归约成**一个** EegSpec，并校验彼此一致。

    评测按 (数据集, 被试) 进行，同一目录下的文件必须共享同一套几何；不一致
    说明混进了别的 profile，应当报错而不是挑第一个继续算。
    """
    if not paths:
        raise ValueError("no HDF5 files to derive the EEG contract from")
    specs = [EegSpec.from_h5(path, source=source) for path in paths]
    first = specs[0]
    for spec, path in zip(specs[1:], paths[1:]):
        if spec.shape != first.shape:
            raise ValueError(
                f"inconsistent EEG shape in {path}: {spec.shape} vs {first.shape}")
        if spec.sampling_rate_hz != first.sampling_rate_hz:
            raise ValueError(
                f"inconsistent sampling rate in {path}: "
                f"{spec.sampling_rate_hz} vs {first.sampling_rate_hz}")
        if spec.channels != first.channels:
            raise ValueError(f"inconsistent channel list in {path}")
    return first


@dataclass
class StimulusSpec:
    """What was shown to the subject during this epoch.

    Fields vary by dataset:
      ThingsEEG: concept, concept_id, instance_id, image_path
      NOD-EEG:   concept (= class_name), concept_id (mapped), image_id (= synset_imgid),
                 class_id (ImageNet synset), is_animate (bool)
    """
    concept: str                            # label string: "carousel" or "hand-held computer"
    concept_id: int                         # mapped int index (0-based); -1 if not mapped
    image_path: str                         # relative / reconstructable path
    image_format: str = "JPEG"
    # ThingsEEG-specific
    instance_id: int = -1                   # within-concept image index (0–11)
    # NOD-EEG-specific
    class_id: str = ""                      # ImageNet synset ID e.g. "n03485407"
    is_animate: Optional[bool] = None       # animate/inanimate label (NOD-EEG)
    # For generation tasks
    image_ref: Optional[str] = None         # e.g. "file://.../n03485407_9884.JPEG"


@dataclass
class SubjectMeta:
    """Per-subject metadata (from participants.tsv)."""
    subject_id: str                         # "sub-01"
    dataset: str                            # "thingseeg"
    task: str                               # "rsvp"
    age: Optional[int] = None
    gender: Optional[str] = None
    exclude: bool = False                   # whether original authors flagged for exclusion


@dataclass
class Sample:
    """A single EEG→Image sample in its canonical JSON representation.

    This is the *input* side: what a model receives.

    Serialisation helpers
    ---------------------
    >>> s = Sample(...)
    >>> s.to_json()            # compact dict (data_ref preferred)
    >>> s.to_json(inline_eeg=True)   # embed EEG data (small test sets)
    >>> Sample.from_json(d)    # parse from dict
    """
    sample_id: str                          # globally unique: "thingseeg_sub-01_ses-01_training_epoched_trial-000042"
    subtask: str                            # one of Subtask.values
    eeg: EegSpec
    stimulus: StimulusSpec
    subject: SubjectMeta

    # reserved for future splits
    split: str = "train"                    # "train" | "val" | "test"

    def to_dict(self, inline_eeg: bool = False) -> Dict[str, Any]:
        """Convert to plain dict suitable for json.dump.

        Parameters
        ----------
        inline_eeg : bool
            If True, embed raw EEG values under ``eeg.data``.
            If False (default), use ``eeg.data_ref`` only (much smaller).
        """
        d: Dict[str, Any] = {
            "benchmark_version": BENCHMARK_VERSION,
            "sample_id": self.sample_id,
            "subtask": self.subtask,
            "split": self.split,
            "subject": {
                "subject_id": self.subject.subject_id,
                "dataset": self.subject.dataset,
                "task": self.subject.task,
            },
            "eeg": {
                "shape": self.eeg.shape,
                "sampling_rate_hz": self.eeg.sampling_rate_hz,
                "channels": self.eeg.channels,
                "units": self.eeg.units,
                "t_pre_s": self.eeg.t_pre_s,
                "t_post_s": self.eeg.t_post_s,
                "preprocessing": self.eeg.preprocessing,
            },
            "stimulus": {
                "concept": self.stimulus.concept,
                "concept_id": self.stimulus.concept_id,
                "image_path": self.stimulus.image_path,
                "image_format": self.stimulus.image_format,
            },
        }
        # EEG data: prefer ref or inline
        if inline_eeg and self.eeg.data is not None:
            d["eeg"]["data"] = self.eeg.data
        elif self.eeg.data_ref:
            d["eeg"]["data_ref"] = self.eeg.data_ref

        # Optional fields
        if self.subject.age is not None:
            d["subject"]["age"] = self.subject.age
        if self.subject.gender:
            d["subject"]["gender"] = self.subject.gender
        if self.subject.exclude:
            d["subject"]["exclude"] = True
        if self.eeg.data_ref:
            d["eeg"]["data_ref"] = self.eeg.data_ref
        # Dataset-specific stimulus fields
        if self.subject.dataset == "thingseeg" and self.stimulus.instance_id >= 0:
            d["stimulus"]["instance_id"] = self.stimulus.instance_id
        if self.subject.dataset == "nod":
            if self.stimulus.class_id:
                d["stimulus"]["class_id"] = self.stimulus.class_id
            if self.stimulus.is_animate is not None:
                d["stimulus"]["is_animate"] = self.stimulus.is_animate
        if self.stimulus.image_ref:
            d["stimulus"]["image_ref"] = self.stimulus.image_ref

        return d

    def to_json(self, inline_eeg: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(inline_eeg=inline_eeg), indent=indent,
                          ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sample":
        return cls(
            sample_id=d["sample_id"],
            subtask=d["subtask"],
            split=d.get("split", "train"),
            eeg=EegSpec(
                shape=d["eeg"]["shape"],
                sampling_rate_hz=d["eeg"].get("sampling_rate_hz"),
                channels=d["eeg"].get("channels", []),
                units=d["eeg"].get("units", EEG_UNIT),
                t_pre_s=d["eeg"].get("t_pre_s", -0.2),
                t_post_s=d["eeg"].get("t_post_s", 0.8),
                data=d["eeg"].get("data"),
                data_ref=d["eeg"].get("data_ref"),
                preprocessing=d["eeg"].get("preprocessing", []),
            ),
            stimulus=StimulusSpec(
                concept=d["stimulus"]["concept"],
                concept_id=d["stimulus"]["concept_id"],
                image_path=d["stimulus"]["image_path"],
                image_format=d["stimulus"].get("image_format", "JPEG"),
                instance_id=d["stimulus"].get("instance_id", -1),
                class_id=d["stimulus"].get("class_id", ""),
                is_animate=d["stimulus"].get("is_animate"),
                image_ref=d["stimulus"].get("image_ref"),
            ),
            subject=SubjectMeta(
                subject_id=d["subject"]["subject_id"],
                dataset=d["subject"]["dataset"],
                task=d["subject"].get("task", d["subject"]["dataset"]),
                age=d["subject"].get("age"),
                gender=d["subject"].get("gender"),
                exclude=d["subject"].get("exclude", False),
            ),
        )

    @classmethod
    def from_h5_row(cls, h5_path: str, row_idx: int,
                    subtask: str = "image_retrieval",
                    split: str = "train",
                    t_pre: float = -0.2, t_post: float = 0.8,
                    ) -> "Sample":
        """Build a Sample from a row of a .h5 file (ThingsEEG or NOD-EEG).

        This is the primary factory for converting the HDF5 output of
        ``eeg2image_benchmark_loader.py`` into benchmark-ready samples.
        """
        import h5py

        with h5py.File(h5_path, "r") as f:
            n_epochs = f["data"].shape[0]
            if row_idx < 0 or row_idx >= n_epochs:
                raise IndexError(
                    f"row_idx {row_idx} out of range [0, {n_epochs}) for {h5_path}"
                )

            n_chan = f["data"].shape[1]
            n_time = f["data"].shape[2]
            sr = float(f.attrs["sampling_rate"])
            subj = str(f.attrs["subject_id"])
            dataset = str(f.attrs["dataset"])
            split_name = str(f.attrs.get("split", split))

            def _decode(value):
                return value.decode("utf-8") if isinstance(value, bytes) else str(value)

            channels = (
                [_decode(value) for value in f["channel_names"][:]]
                if "channel_names" in f else []
            )
            units = str(f.attrs.get("eeg_units", EEG_UNIT))
            h5_t_pre = float(f.attrs.get("t_pre", f.attrs.get("t_pre_s", t_pre)))
            h5_t_post = float(f.attrs.get("t_post", f.attrs.get("t_post_max_s", t_post)))
            try:
                profile = json.loads(str(f.attrs.get("preprocessing_json", "{}")))
                preprocessing = [
                    step["name"] for step in profile.get("steps", [])
                    if isinstance(step, dict) and step.get("name")
                ]
            except (TypeError, json.JSONDecodeError):
                preprocessing = []
            if not preprocessing:
                preprocessing = [f"resampled_to_{int(sr)}hz"]

            # event label
            label_bytes = f["event_label"][row_idx]
            concept = label_bytes.decode("utf-8") if isinstance(label_bytes, bytes) else str(label_bytes)

            # --- ThingsEEG branch ---
            if dataset == "thingseeg":
                concept_id = int(f["event_object_number"][row_idx])
                instance_id = int(f["event_stim_number"][row_idx])
                img_file_bytes = f["event_image_file"][row_idx]
                img_file = img_file_bytes.decode("utf-8") if isinstance(img_file_bytes, bytes) else str(img_file_bytes)
                concept_dir = f"{concept_id:05d}_{concept}"
                image_path = (_decode(f["stimulus_uri"][row_idx]) if "stimulus_uri" in f else img_file)
                class_id = ""
                is_animate = None
                task_name = "rsvp"

            # --- NOD-EEG branch ---
            elif dataset == "nod":
                # concept = image_id ("n03485407_9884")
                # class_name stored in event_class_name
                class_bytes = f["event_class_name"][row_idx]
                class_name = class_bytes.decode("utf-8") if isinstance(class_bytes, bytes) else str(class_bytes)
                concept = class_name    # use human-readable class name as concept
                concept_id = -1         # no global concept index in NOD-EEG
                instance_id = -1        # N/A for NOD-EEG

                class_id_bytes = f["event_class_id"][row_idx]
                class_id = class_id_bytes.decode("utf-8") if isinstance(class_id_bytes, bytes) else str(class_id_bytes)

                image_id = label_bytes.decode("utf-8") if isinstance(label_bytes, bytes) else str(label_bytes)
                image_path = (_decode(f["stimulus_uri"][row_idx]) if "stimulus_uri" in f else image_id)
                is_animate = bool(f["event_is_animate"][row_idx])
                task_name = "ImageNet"

            else:
                concept_id = -1
                instance_id = -1
                image_path = ""
                class_id = ""
                is_animate = None
                task_name = "unknown"

        # sample_id 唯一权威 = benchmark_eval.extract_ground_truths
        # （原则 3：{h5_stem}_trial-{i:06d}，禁止另造格式）
        sample_id = f"{Path(h5_path).stem}_trial-{row_idx:06d}"
        data_ref = f"h5://{h5_path}#data[{row_idx}]"

        return cls(
            sample_id=sample_id,
            subtask=subtask,
            split=split_name,
            eeg=EegSpec(
                shape=[n_chan, n_time],
                sampling_rate_hz=int(sr),
                channels=channels,
                units=units,
                t_pre_s=h5_t_pre,
                t_post_s=h5_t_post,
                data_ref=data_ref,
                preprocessing=preprocessing,
            ),
            stimulus=StimulusSpec(
                concept=concept,
                concept_id=concept_id,
                instance_id=instance_id,
                image_path=image_path,
                image_format="JPEG",
                class_id=class_id,
                is_animate=is_animate,
            ),
            subject=SubjectMeta(
                subject_id=subj,
                dataset=dataset,
                task=task_name,
            ),
        )

    def concept_label(self) -> str:
        """Human-readable concept name (canonical for classification tasks)."""
        return self.stimulus.concept

    def concept_index(self) -> int:
        """0-based concept class index."""
        return self.stimulus.concept_id


# ============================================================================
# Model-output JSON schema  (output side — what a model returns per sample)
# ============================================================================

@dataclass
class RetrievalOutput:
    """Model output for ``image_retrieval``.

    JSON example
    ------------
    {
      "sample_id": "thingseeg_sub-01_ses-01_training_epoched_trial-000042",
      "model_name": "NeuroCLIP-v1",
      "task": "image_retrieval",
      "candidate_pool_ids": ["carousel_11s", "carousel_04n", ...],
      "ranked_indices": [0, 7, 3, 12, ...],
      "similarity_scores": [0.87, 0.73, 0.65, 0.42, ...]
    }
    """
    sample_id: str
    model_name: str
    task: str = "image_retrieval"
    candidate_pool_ids: List[str] = field(default_factory=list)
    ranked_indices: List[int] = field(default_factory=list)
    similarity_scores: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "task": self.task,
            "candidate_pool_ids": self.candidate_pool_ids,
            "ranked_indices": self.ranked_indices,
        }
        if self.similarity_scores is not None:
            d["similarity_scores"] = self.similarity_scores
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RetrievalOutput":
        return cls(
            sample_id=d["sample_id"],
            model_name=d["model_name"],
            task=d.get("task", "image_retrieval"),
            candidate_pool_ids=d.get("candidate_pool_ids", []),
            ranked_indices=d.get("ranked_indices", []),
            similarity_scores=d.get("similarity_scores"),
        )


@dataclass
class GenerationOutput:
    """Model output for ``image_generation``.

    JSON example
    ------------
    {
      "sample_id": "thingseeg_sub-01_trial-0042",
      "model_name": "EEG-Diffusion-v2",
      "task": "image_generation",
      "generated_image": {
        "shape": [3, 256, 256],
        "dtype": "float32",
        "value_range": [-1.0, 1.0],
        "pixel_format": "RGB",
        "data_ref": "h5://outputs/gen.h5#img_0042"
      }
    }
    """
    sample_id: str
    model_name: str
    task: str = "image_generation"
    image_shape: List[int] = field(default_factory=lambda: [3, 256, 256])
    image_dtype: str = "float32"
    value_range: Tuple[float, float] = (-1.0, 1.0)
    pixel_format: str = "RGB"
    data_ref: Optional[str] = None           # reference to the generated image blob
    data: Optional[List[List[List[float]]]] = None   # in-line (small images only)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "task": self.task,
            "generated_image": {
                "shape": self.image_shape,
                "dtype": self.image_dtype,
                "value_range": list(self.value_range),
                "pixel_format": self.pixel_format,
            },
        }
        if self.data is not None:
            d["generated_image"]["data"] = self.data
        if self.data_ref is not None:
            d["generated_image"]["data_ref"] = self.data_ref
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GenerationOutput":
        g = d["generated_image"]
        return cls(
            sample_id=d["sample_id"],
            model_name=d["model_name"],
            task=d.get("task", "image_generation"),
            image_shape=g["shape"],
            image_dtype=g.get("dtype", "float32"),
            value_range=tuple(g.get("value_range", [-1.0, 1.0])),
            pixel_format=g.get("pixel_format", "RGB"),
            data_ref=g.get("data_ref"),
            data=g.get("data"),
        )


@dataclass
class ClassificationOutput:
    """Model output for ``concept_classification``.

    JSON example
    ------------
    {
      "sample_id": "thingseeg_sub-01_trial-0042",
      "model_name": "EEG-Classifier-v1",
      "task": "concept_classification",
      "predicted_class": 275,
      "predicted_concept": "carousel",
      "class_probs": [0.001, 0.002, ..., 0.87, ...]
    }
    """
    sample_id: str
    model_name: str
    task: str = "concept_classification"
    predicted_class: int = -1
    predicted_concept: str = ""
    class_probs: Optional[List[float]] = None
    num_classes: int = 1854

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "task": self.task,
            "predicted_class": self.predicted_class,
            "predicted_concept": self.predicted_concept,
            "num_classes": self.num_classes,
        }
        if self.class_probs is not None:
            d["class_probs"] = self.class_probs
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ClassificationOutput":
        return cls(
            sample_id=d["sample_id"],
            model_name=d["model_name"],
            task=d.get("task", "concept_classification"),
            predicted_class=d.get("predicted_class", -1),
            predicted_concept=d.get("predicted_concept", ""),
            class_probs=d.get("class_probs"),
            num_classes=d.get("num_classes", 1854),
        )


# Unified output type
ModelOutput = Union[RetrievalOutput, GenerationOutput, ClassificationOutput]


# ============================================================================
# Output validation
# ============================================================================

_OUTPUT_CLASS_MAP: Dict[str, type] = {
    "image_retrieval":        RetrievalOutput,
    "image_generation":       GenerationOutput,
    "concept_classification": ClassificationOutput,
}


def validate_output(d: Dict[str, Any]) -> ModelOutput:
    """Parse and lightly validate a model-output dict.

    Returns the appropriate dataclass instance.

    Raises
    ------
    KeyError if ``task`` is missing / unknown.
    ValueError on schema violations.
    """
    task = d.get("task")
    if task not in _OUTPUT_CLASS_MAP:
        raise KeyError(
            f"Unknown task '{task}'. Expected one of {list(_OUTPUT_CLASS_MAP)}."
        )
    cls = _OUTPUT_CLASS_MAP[task]

    instance = cls.from_dict(d)

    # Light validation
    if task == "image_retrieval":
        inst = instance  # type: RetrievalOutput
        if len(inst.ranked_indices) == 0:
            raise ValueError("ranked_indices must be non-empty for retrieval output.")
        if inst.similarity_scores is not None and \
           len(inst.similarity_scores) != len(inst.ranked_indices):
            raise ValueError(
                f"similarity_scores length {len(inst.similarity_scores)} != "
                f"ranked_indices length {len(inst.ranked_indices)}"
            )

    elif task == "image_generation":
        inst = instance  # type: GenerationOutput
        if len(inst.image_shape) != 3:
            raise ValueError(
                f"image_shape must be [C, H, W], got {inst.image_shape}"
            )
        if inst.data is None and inst.data_ref is None:
            raise ValueError(
                "Generation output must provide 'data' or 'data_ref'."
            )

    elif task == "concept_classification":
        inst = instance  # type: ClassificationOutput
        if inst.predicted_class < 0:
            raise ValueError("predicted_class must be ≥ 0.")
        if inst.class_probs is not None and len(inst.class_probs) != inst.num_classes:
            raise ValueError(
                f"class_probs length {len(inst.class_probs)} != num_classes {inst.num_classes}"
            )

    return instance


# ============================================================================
# Schema registry (exportable as JSON Schema draft-07 for tooling)
# ============================================================================

def _build_json_schema_for_sample() -> Dict[str, Any]:
    """Auto-generate a JSON Schema for Sample."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://eeg2image-benchmark/schemas/sample-v1.json",
        "title": "EEG-to-Image Benchmark — Sample (Input)",
        "type": "object",
        "required": ["sample_id", "subtask", "split", "subject", "eeg", "stimulus"],
        "properties": {
            "benchmark_version": {"type": "string", "const": BENCHMARK_VERSION},
            "sample_id": {"type": "string",
                          "pattern": "^[a-z]+_[a-z0-9-]+_trial-[0-9]{6}$"},
            "subtask": {"type": "string", "enum": Subtask.all_values()},
            "split": {"type": "string", "enum": ["train", "val", "test"]},
            "subject": {
                "type": "object",
                "required": ["subject_id", "dataset"],
                "properties": {
                    "subject_id": {"type": "string"},
                    "dataset": {"type": "string",
                                "enum": ["thingseeg", "nod",
                                         "grootswagers", "grootswagers2025"]},
                    "task": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                    "gender": {"type": "string", "enum": ["M", "F", ""]},
                    "exclude": {"type": "boolean"},
                },
            },
            "eeg": {
                "type": "object",
                "required": ["shape", "sampling_rate_hz", "channels"],
                "properties": {
                    "shape": {"type": "array", "minItems": 2, "maxItems": 2,
                              "items": {"type": "integer"}},
                    # 具体值随 (数据集 x 编码器 profile) 变化，由 from_h5 读出。
                    "sampling_rate_hz": {"type": "integer", "minimum": 1},
                    "channels": {"type": "array", "minItems": 1,
                                 "items": {"type": "string"}},
                    "units": {"type": "string", "const": EEG_UNIT},
                    "t_pre_s": {"type": "number"},
                    "t_post_s": {"type": "number"},
                    "data": {"type": "array"},
                    "data_ref": {"type": "string"},
                    "preprocessing": {"type": "array", "items": {"type": "string"}},
                },
            },
            "stimulus": {
                "type": "object",
                "required": ["concept", "concept_id", "image_path"],
                "properties": {
                    "concept": {"type": "string"},
                    "concept_id": {"type": "integer", "minimum": -1},
                    "image_path": {"type": "string"},
                    "image_format": {"type": "string", "enum": ["JPEG", "PNG", "WEBP"]},
                    "instance_id": {"type": "integer", "minimum": -1, "maximum": 11},
                    "class_id": {"type": "string"},
                    "is_animate": {"type": "boolean"},
                    "image_ref": {"type": "string"},
                },
            },
        },
    }


def load_schema_registry() -> Dict[str, Any]:
    """Return all schemas as a dict, suitable for writing to a .json file."""
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": datetime.now().isoformat(),
        "tasks": {k: asdict(v) for k, v in TASK_REGISTRY.items()},
        "sample_schema": _build_json_schema_for_sample(),
        "output_schemas": {
            "image_retrieval": {
                "class": "RetrievalOutput",
                "example": RetrievalOutput(
                    sample_id="thingseeg_sub-01_trial-000042",
                    model_name="ExampleModel-v0",
                    candidate_pool_ids=["carousel_11s", "carousel_04n"],
                    ranked_indices=[0, 1],
                    similarity_scores=[0.87, 0.73],
                ).to_dict(),
            },
            "image_generation": {
                "class": "GenerationOutput",
                "example": GenerationOutput(
                    sample_id="thingseeg_sub-01_trial-000042",
                    model_name="ExampleModel-v0",
                    data_ref="h5://outputs/gen.h5#img_0042",
                ).to_dict(),
            },
            "concept_classification": {
                "class": "ClassificationOutput",
                "example": ClassificationOutput(
                    sample_id="thingseeg_sub-01_trial-000042",
                    model_name="ExampleModel-v0",
                    predicted_class=275,
                    predicted_concept="carousel",
                ).to_dict(),
            },
        },
    }


# ============================================================================
# Utility: generate a manifest JSON for an entire .h5 file
# ============================================================================

def generate_manifest(h5_path: str,
                      subtask: str = "image_retrieval",
                      split: str = "train",
                      output_json: Optional[str] = None,
                      max_samples: Optional[int] = None,
                      ) -> List[Dict[str, Any]]:
    """Produce a sample-manifest JSON file for one .h5 subject file.

    Each entry is a lightweight Sample dict (data_ref, no inline EEG),
    suitable for indexing a benchmark split.

    Parameters
    ----------
    h5_path : str
    subtask : str
    split : str
    output_json : str, optional
        If given, write JSON to this path.
    max_samples : int, optional
        Cap number of samples (for testing).

    Returns
    -------
    samples : list of dict
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        n_epochs = f["data"].shape[0]
        if max_samples:
            n_epochs = min(n_epochs, max_samples)

    samples = []
    for i in range(n_epochs):
        s = Sample.from_h5_row(h5_path, i, subtask=subtask, split=split)
        samples.append(s.to_dict(inline_eeg=False))

    manifest = {
        "manifest_version": BENCHMARK_VERSION,
        "created_at": datetime.now().isoformat(),
        "source_h5": h5_path,
        "subtask": subtask,
        "split": split,
        "n_samples": len(samples),
        "samples": samples,
    }

    if output_json:
        with open(output_json, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

    return samples


# ============================================================================
# CLI quick-test
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="EEG-to-Image Task Interface utilities")
    ap.add_argument("--export_schemas", type=str, default=None,
                    help="Write all schemas to a JSON file")
    ap.add_argument("--manifest", type=str, default=None,
                    help="Generate a sample manifest from an .h5 file")
    ap.add_argument("--subtask", type=str, default="image_retrieval",
                    choices=Subtask.all_values())
    ap.add_argument("--max_samples", type=int, default=5,
                    help="Max samples for manifest (default 5, for preview)")
    ap.add_argument("--example", action="store_true",
                    help="Print an example Sample JSON to stdout")
    args = ap.parse_args()

    if args.export_schemas:
        registry = load_schema_registry()
        with open(args.export_schemas, "w", encoding="utf-8") as fh:
            json.dump(registry, fh, indent=2, ensure_ascii=False)
        print(f"Schemas written to {args.export_schemas}")
        sys.exit(0)

    if args.manifest:
        samples = generate_manifest(
            args.manifest,
            subtask=args.subtask,
            output_json=None,       # print to stdout
            max_samples=args.max_samples,
        )
        print(json.dumps(samples, indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.example:
        # ThingsEEG example
        s = Sample(
            sample_id="thingseeg_sub-01_trial-000042",
            subtask="image_retrieval",
            split="train",
            eeg=EegSpec(
                shape=[32, 250],
                data_ref="h5://thingseeg_sub-01_epoched.h5#data[42]",
                preprocessing=["resampled_to_250hz", "baseline_corrected"],
            ),
            stimulus=StimulusSpec(
                concept="carousel",
                concept_id=275,
                instance_id=10,
                image_path="stimuli/carousel/carousel_11s.jpg",
                image_ref="/data/THINGS/images/carousel_11s.jpg",
            ),
            subject=SubjectMeta(
                subject_id="sub-01", dataset="thingseeg", task="rsvp",
                age=30, gender="F",
            ),
        )
        print("=== ThingsEEG Sample ===")
        print(s.to_json())
        print()

        # NOD-EEG example
        s2 = Sample(
            sample_id="nod_sub-29_trial-000003",
            subtask="image_retrieval",
            split="train",
            eeg=EegSpec(
                shape=[32, 250],
                data_ref="h5://nod_sub-29_sesImageNet01_run-01_epoched.h5#data[3]",
                preprocessing=["resampled_to_250hz"],
            ),
            stimulus=StimulusSpec(
                concept="hand-held computer",
                concept_id=-1,
                image_path="ImageNet/n03485407_9884.JPEG",
                class_id="n03485407",
                is_animate=False,
            ),
            subject=SubjectMeta(
                subject_id="sub-29", dataset="nod", task="ImageNet",
            ),
        )
        print("=== NOD-EEG Sample ===")
        print(s2.to_json())
        print()

        print("=== Retrieval output example ===")
        r_out = RetrievalOutput(
            sample_id="thingseeg_sub-01_trial-000042",
            model_name="ExampleModel-v0",
            candidate_pool_ids=[f"candidate_{i}" for i in range(200)],
            ranked_indices=[0, 7, 3],
            similarity_scores=[0.87, 0.73, 0.65],
        )
        print(r_out.to_json())
        print()

        print("=== Generation output example ===")
        g_out = GenerationOutput(
            sample_id="thingseeg_sub-01_trial-000042",
            model_name="ExampleModel-v0",
            data_ref="h5://outputs/gen.h5#img_0042",
        )
        print(g_out.to_json())

        sys.exit(0)

    # Default: print task specs
    print("=== EEG-to-Image Task Registry ===\n")
    for tid, spec in TASK_REGISTRY.items():
        print(f"[{tid}]")
        print(f"  Name:        {spec.task_name}")
        print(f"  Description: {spec.description[:120]}...")
        print(f"  Output:      {spec.output_type}")
        print(f"  Primary:     {spec.primary_metric}")
        print(f"  Secondary:   {', '.join(spec.secondary_metrics)}")
        print(f"  Num classes: {spec.dataset_num_classes}")
        print()
