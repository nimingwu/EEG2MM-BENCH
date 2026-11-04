"""
EEG-to-Speech Task Interface
==============================
Standardised task definitions, I/O JSON schemas, and validation for the
EEG to Speech downstream benchmark.

Subtasks defined:
  1. speech_reconstruction  — EEG → audio waveform (mel spectrogram or raw wav)
  2. speech_detection       — EEG → binary speaking / silent label
  3. speech_retrieval       — EEG → word-level audio retrieval (ranking over a
                              candidate pool of test words, wav2vec2 space)

Datasets covered:
  - EEG-Speech Brain Decoding  (Japanese overt speech, g.tec Pangolin 128ch, 1200 Hz)
  - KARA ONE Database          (English phoneme production, Neuroscan 64ch, 1000 Hz)
  - Brennan2019                (naturalistic listening to "Alice in Wonderland",
                                actiCHamp 61ch, 500 Hz; word-level retrieval)

Key distinction from EEG-to-Text:
  Output is audio waveform, not text. Text labels (transcriptions) serve as
  auxiliary ground truth for ASR-based evaluation.

Usage:
    from eeg2speech_task_interface import (
        SpeechTaskSpec, SpeechSample, SpeechModelOutput,
        TASK_REGISTRY,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

import numpy as np

BENCHMARK_VERSION = "0.1.0"
TARGET_EEG_SR = 250
TARGET_AUDIO_SR = 16000


class SpeechSubtask(Enum):
    SPEECH_RECONSTRUCTION = "speech_reconstruction"
    SPEECH_DETECTION      = "speech_detection"
    SPEECH_RETRIEVAL      = "speech_retrieval"

    @classmethod
    def all_values(cls) -> List[str]:
        return [e.value for e in cls]


# ===================================================================
# Task specification
# ===================================================================

@dataclass
class SpeechTaskSpec:
    task_name: str
    task_id: str
    description: str

    output_type: str              # "audio_waveform" | "mel_spectrogram" | "binary_label"
    output_spec: Dict[str, Any]

    applicable_datasets: List[str]

    primary_metric: str
    secondary_metrics: List[str]

    benchmark_version: str = BENCHMARK_VERSION
    references: List[str] = field(default_factory=list)


TASK_SPEECH_RECONSTRUCTION = SpeechTaskSpec(
    task_name="EEG-to-Speech Reconstruction",
    task_id="eeg2speech-reconstruction-v1",
    description=(
        "Given an EEG segment recorded during overt or covert speech production, "
        "reconstruct the corresponding audio waveform or mel spectrogram. "
        "Quality is evaluated against the ground-truth microphone recording "
        "using speech-specific metrics."
    ),
    output_type="audio_waveform",
    output_spec={
        "sample_rate":     "int  (default 16000)",
        "waveform_shape":  "[n_samples]  (variable length)",
        "waveform_dtype":  "float32",
        "representation":  '"waveform" | "mel_spectrogram"',
        "mel_config":      "{n_mels: 80, hop_length: 256}  (if mel)",
    },
    applicable_datasets=["speech_brain", "karaone"],
    primary_metric="MCD",          # Mel Cepstral Distortion
    secondary_metrics=["STOI", "PESQ", "WER_via_ASR", "character_error_rate"],
    references=[
        "MiSTR (Lee et al., 2024) — non-invasive EEG to speech synthesis",
        "EEG-to-Speech Brain Decoding (Sato et al., 2025)",
    ],
)

TASK_SPEECH_DETECTION = SpeechTaskSpec(
    task_name="EEG Speech State Detection",
    task_id="eeg2speech-detection-v1",
    description=(
        "Given an EEG segment, classify whether the subject is actively "
        "speaking, preparing to speak, or silent.  Binary or 3-class "
        "classification using KARA ONE's speaking/thinking/clearing labels."
    ),
    output_type="binary_label",
    output_spec={
        "predicted_state": '"speaking" | "thinking" | "silent"',
        "num_classes":     "int  (2 or 3)",
        "confidence":      "float  (optional)",
    },
    applicable_datasets=["karaone"],
    primary_metric="balanced_accuracy",
    secondary_metrics=["f1", "auc_roc"],
    references=["KARA ONE (Zhao & Rudzicz, 2015)"],
)

TASK_SPEECH_RETRIEVAL = SpeechTaskSpec(
    task_name="EEG-to-Speech Retrieval",
    task_id="eeg2speech-retrieval-v1",
    description=(
        "Given an EEG epoch recorded while a subject listened to word W, "
        "retrieve W from a candidate pool of test words.  "
        "The model outputs a ranking over the candidate audio segments "
        "(wav2vec2 embedding space, contrastive alignment).  "
        "Supports Brennan2019 (naturalistic listening, word-level)."
    ),
    output_type="ranking",
    output_spec={
        "candidate_pool_size": "int  (≥1)",
        "ranked_indices":       "int[N]  (0-indexed into candidate pool, best first)",
        "similarity_scores":    "float[N]  (optional)",
    },
    applicable_datasets=["brennan2019"],
    primary_metric="speech_top1_accuracy",
    secondary_metrics=[
        "speech_top5_accuracy", "speech_top10_accuracy", "speech_map",
        "speech_median_retrieval_rank", "speech_mean_retrieval_rank",
        "speech_std_retrieval_rank",
        "speech_top1_accuracy_subject-agg", "speech_top5_accuracy_subject-agg",
        "speech_top10_accuracy_subject-agg",
        "speech_median_retrieval_rank_subject-agg",
    ],
    references=[
        "Brennan, J. R., & Hale, J. T. (2019). Hierarchical structure guides "
        "rapid linguistic predictions during naturalistic listening. "
        "PLoS ONE 14(1): e0207741",
        "NeuralBench speech task (Brennan2019Hierarchical) — word-level "
        "EEG→audio retrieval with wav2vec2-large-xlsr-53 targets and "
        "top-k accuracy",
    ],
)

TASK_REGISTRY: Dict[str, SpeechTaskSpec] = {
    SpeechSubtask.SPEECH_RECONSTRUCTION.value: TASK_SPEECH_RECONSTRUCTION,
    SpeechSubtask.SPEECH_DETECTION.value:      TASK_SPEECH_DETECTION,
    SpeechSubtask.SPEECH_RETRIEVAL.value:      TASK_SPEECH_RETRIEVAL,
}


# ===================================================================
# Sample schema
# ===================================================================

@dataclass
class EegSpec:
    shape: List[int]
    sampling_rate_hz: int = TARGET_EEG_SR
    channels: List[str] = field(default_factory=list)
    units: str = "microV"
    t_pre_s: float = 0.5
    t_post_s: float = 3.0
    data_ref: Optional[str] = None
    preprocessing: List[str] = field(default_factory=list)


@dataclass
class AudioSpec:
    """Ground-truth audio for one utterance."""
    sample_rate: int = TARGET_AUDIO_SR
    duration_s: float = 0.0
    n_samples: int = 0
    data_ref: Optional[str] = None           # "h5://path#audio[row]"
    text_transcription: str = ""             # what was actually spoken


@dataclass
class SpeechSample:
    """A single EEG-to-Speech sample.

    JSON example (speech_brain dataset)
    -----------------------------------
    {
      "sample_id": "speechbrain_sub-01_run-01_utt-0003",
      "subtask": "speech_reconstruction",
      "dataset": "speech_brain",
      "subject_id": "sub-01",
      "split": "train",
      "eeg": {
        "shape": [139, 875],
        "sampling_rate_hz": 250,
        "channels": ["EEG001", "EEG002", ...],
        "units": "microV",
        "t_pre_s": 0.5,
        "t_post_s": 3.0,
        "data_ref": "h5://speechbrain_sub-01_run-01_epoched.h5#eeg[3]",
        "preprocessing": ["resampled_to_250hz"]
      },
      "audio": {
        "sample_rate": 16000,
        "duration_s": 3.5,
        "n_samples": 56000,
        "data_ref": "h5://speechbrain_sub-01_run-01_epoched.h5#audio[3]",
        "text_transcription": "ルーシーは伝統のスイッチをつけた"
      }
    }
    """
    sample_id: str
    subtask: str
    dataset: str
    subject_id: str
    split: str = "train"
    eeg: EegSpec = field(default_factory=EegSpec)
    audio: AudioSpec = field(default_factory=AudioSpec)
    # KARA ONE specific: speech state label for detection task
    speech_state: str = ""       # "speaking" | "thinking" | "clearing"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "benchmark_version": BENCHMARK_VERSION,
            "sample_id": self.sample_id,
            "subtask": self.subtask,
            "dataset": self.dataset,
            "subject_id": self.subject_id,
            "split": self.split,
            "eeg": {
                "shape": self.eeg.shape,
                "sampling_rate_hz": self.eeg.sampling_rate_hz,
                "channels": self.eeg.channels,
                "units": self.eeg.units,
                "t_pre_s": self.eeg.t_pre_s,
                "t_post_s": self.eeg.t_post_s,
                "preprocessing": self.eeg.preprocessing,
            },
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "duration_s": self.audio.duration_s,
                "n_samples": self.audio.n_samples,
                "text_transcription": self.audio.text_transcription,
            },
        }
        if self.eeg.data_ref:
            d["eeg"]["data_ref"] = self.eeg.data_ref
        if self.audio.data_ref:
            d["audio"]["data_ref"] = self.audio.data_ref
        if self.speech_state:
            d["speech_state"] = self.speech_state

        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeechSample":
        eeg_d = d["eeg"]
        aud_d = d.get("audio", {})
        return cls(
            sample_id=d["sample_id"],
            subtask=d["subtask"],
            dataset=d["dataset"],
            subject_id=d["subject_id"],
            split=d.get("split", "train"),
            eeg=EegSpec(
                shape=eeg_d["shape"],
                sampling_rate_hz=eeg_d.get("sampling_rate_hz", TARGET_EEG_SR),
                channels=eeg_d.get("channels", []),
                units=eeg_d.get("units", "microV"),
                t_pre_s=eeg_d.get("t_pre_s", 0.5),
                t_post_s=eeg_d.get("t_post_s", 3.0),
                data_ref=eeg_d.get("data_ref"),
                preprocessing=eeg_d.get("preprocessing", []),
            ),
            audio=AudioSpec(
                sample_rate=aud_d.get("sample_rate", TARGET_AUDIO_SR),
                duration_s=aud_d.get("duration_s", 0.0),
                n_samples=aud_d.get("n_samples", 0),
                data_ref=aud_d.get("data_ref"),
                text_transcription=aud_d.get("text_transcription", ""),
            ),
            speech_state=d.get("speech_state", ""),
        )

    @classmethod
    def from_h5_row(cls, h5_path: str, row_idx: int,
                    subtask: str = "speech_reconstruction",
                    split: str = "train") -> "SpeechSample":
        """Build a SpeechSample from a row of the speech loader's HDF5 output."""
        import h5py

        with h5py.File(h5_path, "r") as f:
            n_epochs = f["eeg"].shape[0]
            if row_idx < 0 or row_idx >= n_epochs:
                raise IndexError(f"row_idx {row_idx} out of range [0, {n_epochs})")

            n_chan = f["eeg"].shape[1]
            n_eeg_time = int(f["valid_eeg_samples"][row_idx])
            eeg_sr = float(f.attrs["eeg_sampling_rate"])
            subj = str(f.attrs["subject_id"])
            dataset = str(f.attrs["dataset"])
            channels_raw = f["channel_names"][:]
            channels = [c.decode() if isinstance(c, bytes) else str(c) for c in channels_raw]

            text_bytes = f["text"][row_idx]
            text = text_bytes.decode("utf-8") if isinstance(text_bytes, bytes) else str(text_bytes)

            has_audio = "audio" in f
            audio_n = 0
            audio_dur = 0.0
            audio_sr = TARGET_AUDIO_SR
            if has_audio:
                audio_n = int(f["valid_audio_samples"][row_idx])
                audio_sr = float(f.attrs["audio_sampling_rate"])
                audio_dur = audio_n / audio_sr

        # sample_id 唯一权威 = benchmark_eval.extract_ground_truths
        # （原则 3：{h5_stem}_utt-{i:04d}，禁止另造格式）
        sample_id = f"{Path(h5_path).stem}_utt-{row_idx:04d}"

        return cls(
            sample_id=sample_id,
            subtask=subtask,
            dataset=dataset,
            subject_id=subj,
            split=split,
            eeg=EegSpec(
                shape=[n_chan, n_eeg_time],
                sampling_rate_hz=int(eeg_sr),
                channels=channels,
                data_ref=f"h5://{h5_path}#eeg[{row_idx}]",
                preprocessing=[f"resampled_to_{int(eeg_sr)}hz"],
            ),
            audio=AudioSpec(
                sample_rate=int(audio_sr),
                duration_s=audio_dur,
                n_samples=audio_n,
                data_ref=f"h5://{h5_path}#audio[{row_idx}]" if has_audio else None,
                text_transcription=text,
            ),
        )


# ===================================================================
# Model-output schemas
# ===================================================================

@dataclass
class SpeechReconstructionOutput:
    """Model output for speech_reconstruction.

    JSON example
    ------------
    {
      "sample_id": "speechbrain_sub-01_run-01_utt-0003",
      "model_name": "EEG-Vocoder-v1",
      "task": "speech_reconstruction",
      "generated_audio": {
        "sample_rate": 16000,
        "n_samples": 56000,
        "representation": "waveform",
        "data_ref": "h5://outputs/recon.h5#audio_0003"
      }
    }
    """
    sample_id: str
    model_name: str
    task: str = "speech_reconstruction"
    sample_rate: int = TARGET_AUDIO_SR
    representation: str = "waveform"     # "waveform" | "mel_spectrogram"
    data_ref: Optional[str] = None
    n_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "task": self.task,
            "generated_audio": {
                "sample_rate": self.sample_rate,
                "n_samples": self.n_samples,
                "representation": self.representation,
            },
        }
        if self.data_ref:
            d["generated_audio"]["data_ref"] = self.data_ref
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeechReconstructionOutput":
        g = d["generated_audio"]
        return cls(
            sample_id=d["sample_id"],
            model_name=d["model_name"],
            task=d.get("task", "speech_reconstruction"),
            sample_rate=g.get("sample_rate", TARGET_AUDIO_SR),
            representation=g.get("representation", "waveform"),
            data_ref=g.get("data_ref"),
            n_samples=g.get("n_samples", 0),
        )


@dataclass
class SpeechDetectionOutput:
    """Model output for speech_detection."""

    sample_id: str
    model_name: str
    task: str = "speech_detection"
    predicted_state: str = ""          # "speaking" | "thinking" | "silent"
    ground_truth_state: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "task": self.task,
            "predicted_state": self.predicted_state,
            "ground_truth_state": self.ground_truth_state,
            "confidence": self.confidence,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeechDetectionOutput":
        return cls(
            sample_id=d["sample_id"],
            model_name=d["model_name"],
            task=d.get("task", "speech_detection"),
            predicted_state=d.get("predicted_state", ""),
            ground_truth_state=d.get("ground_truth_state", ""),
            confidence=d.get("confidence", 0.0),
        )


@dataclass
class SpeechRetrievalOutput:
    """Model output for speech_retrieval.

    JSON example
    ------------
    {
      "sample_id": "brennan2019_S01_test_epoched_word-0003",
      "model_name": "EEGNet-Speech-v1",
      "task": "speech_retrieval",
      "candidate_pool_ids": ["for", "cake", "said", ...],
      "ranked_indices": [0, 7, 3, 12, ...],
      "similarity_scores": [0.87, 0.73, 0.65, 0.42, ...]
    }
    """
    sample_id: str
    model_name: str
    task: str = "speech_retrieval"
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
    def from_dict(cls, d: Dict[str, Any]) -> "SpeechRetrievalOutput":
        return cls(
            sample_id=d["sample_id"],
            model_name=d["model_name"],
            task=d.get("task", "speech_retrieval"),
            candidate_pool_ids=d.get("candidate_pool_ids", []),
            ranked_indices=d.get("ranked_indices", []),
            similarity_scores=d.get("similarity_scores"),
        )


SpeechModelOutput = Union[SpeechReconstructionOutput,
                          SpeechDetectionOutput, SpeechRetrievalOutput]

_OUTPUT_CLASS_MAP: Dict[str, type] = {
    "speech_reconstruction": SpeechReconstructionOutput,
    "speech_detection":      SpeechDetectionOutput,
    "speech_retrieval":      SpeechRetrievalOutput,
}


def validate_speech_output(d: Dict[str, Any]) -> SpeechModelOutput:
    task = d.get("task")
    if task not in _OUTPUT_CLASS_MAP:
        raise KeyError(f"Unknown task '{task}'. Expected {list(_OUTPUT_CLASS_MAP)}.")
    cls = _OUTPUT_CLASS_MAP[task]
    instance = cls.from_dict(d)

    if task == "speech_reconstruction":
        inst = instance  # type: SpeechReconstructionOutput
        if inst.n_samples <= 0 and inst.data_ref is None:
            raise ValueError("Speech reconstruction output must provide n_samples or data_ref.")

    elif task == "speech_detection":
        inst = instance  # type: SpeechDetectionOutput
        valid = {"speaking", "thinking", "clearing", "silent", ""}
        if inst.predicted_state not in valid:
            raise ValueError(f"predicted_state must be one of {valid}")

    elif task == "speech_retrieval":
        inst = instance  # type: SpeechRetrievalOutput
        if not inst.ranked_indices:
            raise ValueError("Speech retrieval output must provide ranked_indices")
        if inst.candidate_pool_ids:
            pool_size = len(inst.candidate_pool_ids)
            if any(idx >= pool_size or idx < 0 for idx in inst.ranked_indices):
                raise ValueError("ranked_indices out of candidate_pool range")

    return instance


# ====================================================================
# Schema registry
# ====================================================================

def load_speech_schema_registry() -> Dict[str, Any]:
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": datetime.now().isoformat(),
        "tasks": {k: asdict(v) for k, v in TASK_REGISTRY.items()},
        "output_schemas": {
            "speech_reconstruction": {
                "class": "SpeechReconstructionOutput",
                "example": SpeechReconstructionOutput(
                    sample_id="speechbrain_sub-01_run-01_utt-0003",
                    model_name="ExampleModel-v0",
                    n_samples=56000,
                    data_ref="h5://outputs/recon.h5#audio_0003",
                ).to_dict(),
            },
            "speech_detection": {
                "class": "SpeechDetectionOutput",
                "example": SpeechDetectionOutput(
                    sample_id="karaone_MM05_utt-0042",
                    model_name="ExampleModel-v0",
                    predicted_state="speaking",
                    ground_truth_state="speaking",
                    confidence=0.94,
                ).to_dict(),
            },
            "speech_retrieval": {
                "class": "SpeechRetrievalOutput",
                "example": SpeechRetrievalOutput(
                    sample_id="brennan2019_S01_test_epoched_word-0003",
                    model_name="ExampleModel-v0",
                    candidate_pool_ids=["for", "cake", "said"],
                    ranked_indices=[0, 2, 1],
                    similarity_scores=[0.87, 0.73, 0.65],
                ).to_dict(),
            },
        },
    }


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="EEG-to-Speech Task Interface")
    ap.add_argument("--example", action="store_true")
    args = ap.parse_args()

    if args.example:
        s = SpeechSample(
            sample_id="speechbrain_sub-01_run-01_utt-0003",
            subtask="speech_reconstruction",
            dataset="speech_brain",
            subject_id="sub-01",
            split="train",
            eeg=EegSpec(
                shape=[139, 875],
                channels=["EEG001", "EEG002", "... (139 ch)"],
                data_ref="h5://speechbrain_sub-01_run-01_epoched.h5#eeg[3]",
                preprocessing=["resampled_to_250hz"],
            ),
            audio=AudioSpec(
                sample_rate=16000,
                duration_s=3.5,
                n_samples=56000,
                data_ref="h5://speechbrain_sub-01_run-01_epoched.h5#audio[3]",
                text_transcription="ルーシーは伝統のスイッチをつけた",
            ),
        )
        print("=== SpeechSample (speech_brain) ===")
        print(s.to_json())
        print()

        print("=== Reconstruction output ===")
        print(SpeechReconstructionOutput(
            sample_id="speechbrain_sub-01_run-01_utt-0003",
            model_name="ExampleModel-v0",
            n_samples=56000,
            data_ref="h5://outputs/recon.h5#audio_0003",
        ).to_json())
        print()

        print("=== Detection output ===")
        print(SpeechDetectionOutput(
            sample_id="karaone_MM05_utt-0042",
            model_name="ExampleModel-v0",
            predicted_state="speaking",
            ground_truth_state="speaking",
            confidence=0.94,
        ).to_json())
        print()

        print("=== Retrieval output ===")
        print(SpeechRetrievalOutput(
            sample_id="brennan2019_S01_test_epoched_word-0003",
            model_name="ExampleModel-v0",
            candidate_pool_ids=["for", "cake", "said"],
            ranked_indices=[0, 2, 1],
            similarity_scores=[0.87, 0.73, 0.65],
        ).to_json())

        sys.exit(0)

    # Default
    print("=== EEG-to-Speech Task Registry ===\n")
    for tid, spec in TASK_REGISTRY.items():
        print(f"[{tid}]")
        print(f"  Name:        {spec.task_name}")
        print(f"  Output:      {spec.output_type}")
        print(f"  Primary:     {spec.primary_metric}")
        print(f"  Secondary:   {', '.join(spec.secondary_metrics)}")
        print(f"  Datasets:    {', '.join(spec.applicable_datasets)}")
        print()
