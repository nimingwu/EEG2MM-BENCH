"""
EEG-to-Text Task Interface
===========================
Standardised task definitions, I/O JSON schemas, and validation for the
EEG → Text downstream benchmark.

Subtasks defined (per EEG多模态下游任务Benchmark构建.docx §6 Table 0):
  1. sentence_generation   — EEG → free-form sentence / passage
  2. word_decoding         — EEG aligned to a single word → word text
  3. mind_wandering_detect — EEG → attention-state label  (ROAMM-specific)

Datasets covered
----------------
  ROAMM  (Sun et al., OpenNeuro ds007629)
    - 44 subjects, 64-channel BioSemi ActiveTwo, 256 Hz
    - Task: naturalistic reading of wiki articles
    - Text: word-level coordinates, sentence-aligned wiki stories
    - Extra: eye-tracking fixations, span-level mind-wandering labels

  EEG responses to continuous naturalistic speech  (Di Liberto et al.,
    Broderick et al., OpenNeuro ds004408)
    - 19 subjects, 128-channel BioSemi ActiveTwo, 512 Hz
    - Task: passive listening to "The Old Man and the Sea" audiobook
    - Text: phoneme + word alignment via TextGrid (Praat forced-alignment)
    - Extra: .wav audio stimuli

Usage
-----
    from eeg2text_task_interface import (
        TextTaskSpec, TextSample, TextModelOutput,
        TASK_REGISTRY, build_sample_from_roamm,
    )
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime

import numpy as np


# ============================================================================
# Constants
# ============================================================================

BENCHMARK_VERSION = "0.1.0"

# These datasets have different channel counts — the interface is agnostic.
# Common *possible* target for cross-dataset standardisation (not enforced):
# ROAMM: 64ch BioSemi → maps well to extended 10-20
# Speech: 128ch BioSemi → A1-D32 naming (equidistant layout)

class TextSubtask(Enum):
    SENTENCE_GENERATION    = "sentence_generation"
    WORD_DECODING          = "word_decoding"
    MIND_WANDERING_DETECT  = "mind_wandering_detect"

    @classmethod
    def all_values(cls) -> List[str]:
        return [e.value for e in cls]


# ============================================================================
# Task specification
# ============================================================================

@dataclass
class TextTaskSpec:
    """Static specification of a single EEG→Text subtask."""
    task_name: str
    task_id: str
    description: str

    # I/O
    output_type: str              # "free_text" | "word_label" | "binary_label"
    output_spec: Dict[str, Any]

    # Dataset applicability
    applicable_datasets: List[str]   # ["ROAMM", "speech_continuous"]

    # Metrics
    primary_metric: str
    secondary_metrics: List[str]

    # Metadata
    benchmark_version: str = BENCHMARK_VERSION
    references: List[str] = field(default_factory=list)


TASK_SENTENCE_GENERATION = TextTaskSpec(
    task_name="EEG-to-Text Sentence Generation",
    task_id="eeg2text-sentence-v1",
    description=(
        "Given a multi-second EEG segment recorded while a subject reads "
        "(ROAMM) or listens to (continuous speech) a sentence, generate "
        "the corresponding natural-language text.  "
        "Output is free-form text; evaluated against the ground-truth "
        "sentence using NLG metrics."
    ),
    output_type="free_text",
    output_spec={
        "generated_text": "str  (free-form sentence / passage)",
        "max_length":      "int  (optional cap)",
        "language":        '"en"',
    },
    applicable_datasets=["ROAMM", "speech_continuous"],
    primary_metric="BLEU-1",
    secondary_metrics=["BLEU-4", "ROUGE-L", "METEOR", "CIDEr",
                       "word_error_rate", "BERTScore"],
    references=[
        "Thought2Text (Chen et al., 2024)",
        "EEG2TEXT (Wang et al., 2023)",
    ],
)

TASK_WORD_DECODING = TextTaskSpec(
    task_name="EEG Word-Level Decoding",
    task_id="eeg2text-word-v1",
    description=(
        "Given a short EEG epoch time-locked to a single word onset "
        "(fixation onset in ROAMM, word audio onset in speech), "
        "predict which word was being processed.  "
        "Closed-vocabulary classification over the union of words "
        "appearing in the dataset."
    ),
    output_type="word_label",
    output_spec={
        "predicted_word":    "str",
        "vocab_size":        "int  (depends on dataset split)",
        "word_probs":        "float[vocab_size]  (optional full softmax)",
    },
    applicable_datasets=["ROAMM", "speech_continuous"],
    primary_metric="top1_accuracy",
    secondary_metrics=["top5_accuracy", "mean_per_class_accuracy"],
    references=[],
)

TASK_MIND_WANDERING = TextTaskSpec(
    task_name="Mind-Wandering Detection from EEG",
    task_id="eeg2text-mw-v1",
    description=(
        "Given an EEG segment during naturalistic reading, classify "
        "whether the subject was attentively reading or mind-wandering.  "
        "Binary classification with span-level ground-truth annotations "
        "(ROAMM-specific)."
    ),
    output_type="binary_label",
    output_spec={
        "predicted_label":  '0 (attentive) | 1 (mind-wandering)',
        "confidence":       "float  (optional)",
    },
    applicable_datasets=["ROAMM"],
    primary_metric="balanced_accuracy",
    secondary_metrics=["f1", "auc_roc", "precision", "recall"],
    references=[
        "ROAMM (Sun et al.) — span-level MW annotations during reading",
    ],
)

TASK_REGISTRY: Dict[str, TextTaskSpec] = {
    TextSubtask.SENTENCE_GENERATION.value:   TASK_SENTENCE_GENERATION,
    TextSubtask.WORD_DECODING.value:         TASK_WORD_DECODING,
    TextSubtask.MIND_WANDERING_DETECT.value: TASK_MIND_WANDERING,
}


# ============================================================================
# Sample input schema  (one EEG epoch + aligned text ground truth)
# ============================================================================

@dataclass
class EegSpec:
    """EEG data descriptor (same as eeg2image, channel-agnostic)."""
    shape: List[int]
    sampling_rate_hz: int
    channels: List[str]
    units: str = "µV"
    t_pre_s: float = 0.0
    t_post_s: float = 0.0
    data: Optional[List[List[float]]] = None
    data_ref: Optional[str] = None
    preprocessing: List[str] = field(default_factory=list)


@dataclass
class TextAlignment:
    """Precise temporal alignment of text tokens to the EEG timeseries.

    This is the critical metadata that distinguishes EEG-to-Text from
    simpler classification tasks — it maps every word (and optionally
    phoneme) to a time window within the EEG recording.
    """
    # Word-level alignment
    words: List[str]                           # ordered word tokens
    word_onsets_sec: List[float]               # start of each word (seconds)
    word_offsets_sec: List[float]              # end of each word (seconds)

    # Sentence-level grouping
    sentence_ids: Optional[List[int]] = None   # which sentence each word belongs to
    sentences: Optional[List[str]] = None      # full sentence strings (dedup'd)

    # Phoneme-level (speech dataset only)
    phonemes: Optional[List[str]] = None
    phoneme_onsets_sec: Optional[List[float]] = None
    phoneme_offsets_sec: Optional[List[float]] = None

    # Source
    alignment_source: str = ""                 # "TextGrid" | "coordinates_csv" | "manual"

    def n_words(self) -> int:
        return len(self.words)

    def n_sentences(self) -> int:
        return len(self.sentences) if self.sentences else 0

    def get_sentence_words(self, sent_idx: int) -> List[str]:
        """Return word tokens belonging to sentence *sent_idx*."""
        if self.sentence_ids is None:
            raise ValueError("sentence_ids not set — cannot group by sentence.")
        return [w for w, s in zip(self.words, self.sentence_ids) if s == sent_idx]

    def get_sentence_text(self, sent_idx: int) -> str:
        """Reconstructed sentence string."""
        if self.sentences and sent_idx < len(self.sentences):
            return self.sentences[sent_idx]
        return " ".join(self.get_sentence_words(sent_idx))

    def get_word_window(self, word_idx: int) -> Tuple[float, float]:
        """Return (onset, offset) in seconds for word *word_idx*."""
        return (self.word_onsets_sec[word_idx], self.word_offsets_sec[word_idx])


@dataclass
class EyetrackingData:
    """Eye-tracking data for a reading trial (ROAMM-specific)."""
    onset_sec: List[float]
    duration_ms: List[float]
    event_type: List[str]          # "fixation_L", "saccade_R", etc.
    data_ref: Optional[str] = None


@dataclass
class TextSample:
    """A single EEG→Text sample in canonical JSON representation.

    Serialisation
    -------------
    >>> s = TextSample(...)
    >>> s.to_json()
    >>> TextSample.from_dict(d)
    """
    sample_id: str                   # globally unique
    subtask: str                     # one of TextSubtask values
    dataset: str                     # "ROAMM" | "speech_continuous"

    # Core
    eeg: EegSpec
    text_alignment: TextAlignment

    # Metadata
    subject_id: str
    split: str = "train"

    # Dataset-specific extras (nullable)
    eyetracking: Optional[EyetrackingData] = None      # ROAMM
    audio_ref: Optional[str] = None                    # speech dataset: .wav path
    mind_wandering_label: Optional[int] = None         # ROAMM: 0/1 span label
    comprehension_score: Optional[float] = None        # ROAMM: page-level MCQ score

    def to_dict(self, inline_eeg: bool = False) -> Dict[str, Any]:
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
            "text_alignment": {
                "words": self.text_alignment.words,
                "word_onsets_sec": self.text_alignment.word_onsets_sec,
                "word_offsets_sec": self.text_alignment.word_offsets_sec,
                "n_words": self.text_alignment.n_words(),
                "alignment_source": self.text_alignment.alignment_source,
            },
        }
        if inline_eeg and self.eeg.data is not None:
            d["eeg"]["data"] = self.eeg.data
        elif self.eeg.data_ref:
            d["eeg"]["data_ref"] = self.eeg.data_ref

        # Sentence grouping
        if self.text_alignment.sentences:
            d["text_alignment"]["sentences"] = self.text_alignment.sentences
            d["text_alignment"]["n_sentences"] = self.text_alignment.n_sentences()
        if self.text_alignment.sentence_ids:
            d["text_alignment"]["sentence_ids"] = self.text_alignment.sentence_ids

        # Phoneme-level
        if self.text_alignment.phonemes:
            d["text_alignment"]["phonemes"] = self.text_alignment.phonemes
            d["text_alignment"]["phoneme_onsets_sec"] = self.text_alignment.phoneme_onsets_sec
            d["text_alignment"]["phoneme_offsets_sec"] = self.text_alignment.phoneme_offsets_sec

        # Extras
        if self.eyetracking:
            d["eyetracking"] = {
                "n_events": len(self.eyetracking.event_type),
                "data_ref": self.eyetracking.data_ref,
            }
        if self.audio_ref:
            d["audio_ref"] = self.audio_ref
        if self.mind_wandering_label is not None:
            d["mind_wandering_label"] = self.mind_wandering_label
        if self.comprehension_score is not None:
            d["comprehension_score"] = self.comprehension_score

        return d

    def to_json(self, inline_eeg: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(inline_eeg=inline_eeg), indent=indent,
                          ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TextSample":
        ta = d["text_alignment"]
        et = d.get("eyetracking")
        return cls(
            sample_id=d["sample_id"],
            subtask=d["subtask"],
            dataset=d["dataset"],
            subject_id=d["subject_id"],
            split=d.get("split", "train"),
            eeg=EegSpec(
                shape=d["eeg"]["shape"],
                sampling_rate_hz=d["eeg"]["sampling_rate_hz"],
                channels=d["eeg"].get("channels", []),
                units=d["eeg"].get("units", "µV"),
                t_pre_s=d["eeg"].get("t_pre_s", 0.0),
                t_post_s=d["eeg"].get("t_post_s", 0.0),
                data=d["eeg"].get("data"),
                data_ref=d["eeg"].get("data_ref"),
                preprocessing=d["eeg"].get("preprocessing", []),
            ),
            text_alignment=TextAlignment(
                words=ta["words"],
                word_onsets_sec=ta["word_onsets_sec"],
                word_offsets_sec=ta["word_offsets_sec"],
                sentence_ids=ta.get("sentence_ids"),
                sentences=ta.get("sentences"),
                phonemes=ta.get("phonemes"),
                phoneme_onsets_sec=ta.get("phoneme_onsets_sec"),
                phoneme_offsets_sec=ta.get("phoneme_offsets_sec"),
                alignment_source=ta.get("alignment_source", ""),
            ),
            eyetracking=EyetrackingData(
                onset_sec=et.get("onset_sec", []),
                duration_ms=et.get("duration_ms", []),
                event_type=et.get("event_type", []),
                data_ref=et.get("data_ref"),
            ) if et else None,
            audio_ref=d.get("audio_ref"),
            mind_wandering_label=d.get("mind_wandering_label"),
            comprehension_score=d.get("comprehension_score"),
        )


# ============================================================================
# Model-output schemas
# ============================================================================

@dataclass
class SentenceGenerationOutput:
    """Model output for ``sentence_generation``.

    JSON example
    ------------
    {
      "sample_id": "roamm_sub-10014_sentence-03",
      "model_name": "Thought2Text-v1",
      "task": "sentence_generation",
      "generated_text": "Pluto is a dwarf planet in the Kuiper belt.",
      "ground_truth_text": "Pluto is a dwarf planet in the Kuiper belt...",
      "token_probs": [[...], ...]
    }
    """
    sample_id: str
    model_name: str
    task: str = "sentence_generation"
    generated_text: str = ""
    ground_truth_text: str = ""
    token_probs: Optional[List[List[float]]] = None   # per-token distributions
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "task": self.task,
            "generated_text": self.generated_text,
            "ground_truth_text": self.ground_truth_text,
            "language": self.language,
        }
        if self.token_probs is not None:
            d["token_probs"] = self.token_probs
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SentenceGenerationOutput":
        return cls(
            sample_id=d["sample_id"],
            model_name=d["model_name"],
            task=d.get("task", "sentence_generation"),
            generated_text=d.get("generated_text", ""),
            ground_truth_text=d.get("ground_truth_text", ""),
            token_probs=d.get("token_probs"),
            language=d.get("language", "en"),
        )


@dataclass
class WordDecodingOutput:
    """Model output for ``word_decoding``.

    JSON example
    ------------
    {
      "sample_id": "speech_sub-001_word-0042",
      "model_name": "EEG-WordDecoder-v1",
      "task": "word_decoding",
      "predicted_word": "man",
      "ground_truth_word": "MAN",
      "confidence": 0.87,
      "vocab_size": 512,
      "word_probs": null
    }
    """
    sample_id: str
    model_name: str
    task: str = "word_decoding"
    predicted_word: str = ""
    ground_truth_word: str = ""
    confidence: float = 0.0
    vocab_size: int = 0
    word_probs: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "task": self.task,
            "predicted_word": self.predicted_word,
            "ground_truth_word": self.ground_truth_word,
            "confidence": self.confidence,
            "vocab_size": self.vocab_size,
        }
        if self.word_probs is not None:
            d["word_probs"] = self.word_probs
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WordDecodingOutput":
        return cls(
            sample_id=d["sample_id"],
            model_name=d["model_name"],
            task=d.get("task", "word_decoding"),
            predicted_word=d.get("predicted_word", ""),
            ground_truth_word=d.get("ground_truth_word", ""),
            confidence=d.get("confidence", 0.0),
            vocab_size=d.get("vocab_size", 0),
            word_probs=d.get("word_probs"),
        )


@dataclass
class MWDetectionOutput:
    """Model output for ``mind_wandering_detect``.

    JSON example
    ------------
    {
      "sample_id": "roamm_sub-10014_span-07",
      "model_name": "MW-Detector-v1",
      "task": "mind_wandering_detect",
      "predicted_label": 0,
      "ground_truth_label": 0,
      "confidence": 0.92
    }
    """
    sample_id: str
    model_name: str
    task: str = "mind_wandering_detect"
    predicted_label: int = -1          # 0 = attentive, 1 = mind-wandering
    ground_truth_label: int = -1
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model_name": self.model_name,
            "task": self.task,
            "predicted_label": self.predicted_label,
            "ground_truth_label": self.ground_truth_label,
            "confidence": self.confidence,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MWDetectionOutput":
        return cls(
            sample_id=d["sample_id"],
            model_name=d["model_name"],
            task=d.get("task", "mind_wandering_detect"),
            predicted_label=d.get("predicted_label", -1),
            ground_truth_label=d.get("ground_truth_label", -1),
            confidence=d.get("confidence", 0.0),
        )


TextModelOutput = Union[SentenceGenerationOutput, WordDecodingOutput, MWDetectionOutput]


# ============================================================================
# Validation
# ============================================================================

_OUTPUT_CLASS_MAP: Dict[str, type] = {
    "sentence_generation":   SentenceGenerationOutput,
    "word_decoding":         WordDecodingOutput,
    "mind_wandering_detect": MWDetectionOutput,
}


def validate_text_output(d: Dict[str, Any]) -> TextModelOutput:
    """Parse and validate a model-output dict for EEG→Text."""
    task = d.get("task")
    if task not in _OUTPUT_CLASS_MAP:
        raise KeyError(f"Unknown task '{task}'. Expected {list(_OUTPUT_CLASS_MAP)}.")
    cls = _OUTPUT_CLASS_MAP[task]
    instance = cls.from_dict(d)

    if task == "sentence_generation":
        inst = instance  # type: SentenceGenerationOutput
        if not inst.generated_text.strip():
            raise ValueError("generated_text must be non-empty.")

    elif task == "word_decoding":
        inst = instance  # type: WordDecodingOutput
        if not inst.predicted_word.strip():
            raise ValueError("predicted_word must be non-empty.")

    elif task == "mind_wandering_detect":
        inst = instance  # type: MWDetectionOutput
        if inst.predicted_label not in (0, 1):
            raise ValueError("predicted_label must be 0 or 1.")

    return instance


# ============================================================================
# Factory: build a TextSample from ROAMM data
# ============================================================================

def build_sample_from_roamm(
    eeg_h5_path: str,
    row_idx: int,
    subtask: str = "sentence_generation",
    split: str = "train",
) -> TextSample:
    """Construct a TextSample from pre-processed ROAMM .h5 data.

    Parameters
    ----------
    eeg_h5_path : str
        Path to an .h5 file produced by the ROAMM loader.
    row_idx : int
        Epoch (sentence) index.
    subtask : str
    split : str
    """
    import h5py

    def _decode(value: Any) -> Any:
        return value.decode("utf-8") if isinstance(value, bytes) else value

    def _json_list(value: Any) -> List[Any]:
        value = _decode(value)
        if isinstance(value, str):
            parsed = json.loads(value)
            return list(parsed) if isinstance(parsed, list) else []
        if value is None:
            return []
        return np.asarray(value).reshape(-1).tolist()

    with h5py.File(eeg_h5_path, "r") as f:
        n_chan = f["data"].shape[1]
        n_time = (
            int(f["valid_samples"][row_idx])
            if "valid_samples" in f else f["data"].shape[2]
        )
        sr = int(f.attrs["sampling_rate"])
        subj = str(f.attrs["subject_id"])
        dataset = str(f.attrs.get("dataset_id", f.attrs.get("dataset", "ROAMM")))
        actual_split = str(f.attrs.get("split", split))
        units = str(f.attrs.get("eeg_units", "uV"))
        channels = [
            str(_decode(value)) for value in f["channel_names"][:]
        ] if "channel_names" in f else []
        try:
            profile = json.loads(str(f.attrs.get("preprocessing_json", "{}")))
            preprocessing = [
                str(step["name"]) for step in profile.get("steps", [])
            ]
        except (TypeError, json.JSONDecodeError, KeyError):
            preprocessing = []

        # Text
        words = [str(_decode(word)) for word in _json_list(f["word_lists"][row_idx])] if "word_lists" in f else []

        sentence_value = f["sentence_text"][row_idx] if "sentence_text" in f else ""
        sentence = str(_decode(sentence_value))

        # Alignment
        w_onsets = _json_list(f["word_onsets_sec"][row_idx]) if "word_onsets_sec" in f else []
        w_offsets = _json_list(f["word_offsets_sec"][row_idx]) if "word_offsets_sec" in f else []
        alignment_source = (
            str(_decode(f["stimulus_alignment_source"][row_idx]))
            if "stimulus_alignment_source" in f else "roamm_coordinates"
        )

        # MW label
        mw = int(f["mind_wandering_label"][row_idx]) if "mind_wandering_label" in f else None

    # sample_id 唯一权威 = benchmark_eval.extract_ground_truths
    # （原则 3：{h5_stem}_sent-{i:04d}，禁止另造格式）
    sample_id = f"{Path(eeg_h5_path).stem}_sent-{row_idx:04d}"

    return TextSample(
        sample_id=sample_id,
        subtask=subtask,
        dataset=dataset,
        subject_id=subj,
        split=actual_split,
        eeg=EegSpec(
            shape=[n_chan, n_time],
            sampling_rate_hz=sr,
            channels=channels,
            units=units,
            data_ref=f"h5://{eeg_h5_path}#data[{row_idx}]",
            preprocessing=preprocessing,
        ),
        text_alignment=TextAlignment(
            words=words,
            word_onsets_sec=w_onsets,
            word_offsets_sec=w_offsets,
            sentences=[sentence] if sentence else None,
            alignment_source=alignment_source,
        ),
        mind_wandering_label=mw,
    )


# ============================================================================
# Schema registry
# ============================================================================

def load_text_schema_registry() -> Dict[str, Any]:
    """Export all EEG→Text schemas."""
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": datetime.now().isoformat(),
        "tasks": {k: asdict(v) for k, v in TASK_REGISTRY.items()},
        "output_schemas": {
            "sentence_generation": {
                "class": "SentenceGenerationOutput",
                "example": SentenceGenerationOutput(
                    sample_id="roamm_sub-10014_sentence-0003",
                    model_name="ExampleModel-v0",
                    generated_text="Pluto is a dwarf planet in the Kuiper belt.",
                    ground_truth_text="Pluto (minor-planet designation: 134340 Pluto) is a dwarf planet in the Kuiper belt, a ring of bodies beyond Neptune.",
                ).to_dict(),
            },
            "word_decoding": {
                "class": "WordDecodingOutput",
                "example": WordDecodingOutput(
                    sample_id="speech_sub-001_word-0042",
                    model_name="ExampleModel-v0",
                    predicted_word="man",
                    ground_truth_word="MAN",
                    confidence=0.87,
                    vocab_size=512,
                ).to_dict(),
            },
            "mind_wandering_detect": {
                "class": "MWDetectionOutput",
                "example": MWDetectionOutput(
                    sample_id="roamm_sub-10014_span-0007",
                    model_name="ExampleModel-v0",
                    predicted_label=0,
                    ground_truth_label=0,
                    confidence=0.92,
                ).to_dict(),
            },
        },
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="EEG-to-Text Task Interface")
    ap.add_argument("--export_schemas", type=str, default=None,
                    help="Write all schemas to JSON file")
    ap.add_argument("--example", action="store_true",
                    help="Print example sample + outputs")
    args = ap.parse_args()

    if args.export_schemas:
        registry = load_text_schema_registry()
        with open(args.export_schemas, "w", encoding="utf-8") as fh:
            json.dump(registry, fh, indent=2, ensure_ascii=False)
        print(f"Schemas written to {args.export_schemas}")
        sys.exit(0)

    if args.example:
        # ---- Build an example ROAMM sample ----
        sample = TextSample(
            sample_id="roamm_sub-10014_sentence-0003",
            subtask="sentence_generation",
            dataset="ROAMM",
            subject_id="sub-10014",
            split="train",
            eeg=EegSpec(
                shape=[64, 2048],
                sampling_rate_hz=256,
                channels=["Fp1", "AF7", "AF3", "... (64ch BioSemi)"],
                units="µV",
                t_pre_s=0.0,
                t_post_s=8.0,
                data_ref="h5://roamm_sub-10014_run-1.h5#data[3]",
                preprocessing=["resampled_to_256hz", "baseline_corrected"],
            ),
            text_alignment=TextAlignment(
                words=["Pluto", "is", "a", "dwarf", "planet", "in", "the",
                       "Kuiper", "belt"],
                word_onsets_sec=[0.0, 0.42, 0.61, 0.79, 1.15, 1.37, 1.51, 1.71, 2.05],
                word_offsets_sec=[0.40, 0.59, 0.77, 1.13, 1.35, 1.49, 1.69, 2.03, 2.39],
                sentence_ids=[0]*9,
                sentences=["Pluto is a dwarf planet in the Kuiper belt."],
                alignment_source="roamm_coordinates",
            ),
        )
        print("=== TextSample (input) — ROAMM ===")
        print(sample.to_json())
        print()

        # ---- Example speech dataset sample ----
        sample2 = TextSample(
            sample_id="speech_sub-001_run-01_sentence-0000",
            subtask="sentence_generation",
            dataset="speech_continuous",
            subject_id="sub-001",
            split="train",
            eeg=EegSpec(
                shape=[128, 3584],
                sampling_rate_hz=512,
                channels=[f"{b}{n}" for b in "ABCD" for n in range(1, 33)],
                units="µV",
                t_pre_s=0.0,
                t_post_s=7.0,
                data_ref="h5://speech_sub-001_run-01.h5#data[0]",
                preprocessing=["resampled_to_512hz"],
            ),
            text_alignment=TextAlignment(
                words=["HE", "WAS", "AN", "OLD", "MAN", "WHO", "FISHED",
                       "ALONE", "IN", "A", "SKIFF"],
                word_onsets_sec=[0.76, 0.88, 1.04, 1.12, 1.41, 1.87, 2.08,
                                 2.41, 2.77, 2.88, 2.95],
                word_offsets_sec=[0.88, 1.04, 1.12, 1.41, 1.79, 2.08, 2.41,
                                  2.77, 2.88, 2.95, 3.41],
                phonemes=["HH", "IY1", "W", "AH0", "Z", "AH0", "N", "OW1",
                          "L", "D", "M", "AE1", "N", "EH1", "L", "D"],
                phoneme_onsets_sec=[0.76, 0.81, 0.88, 0.96, 0.99, 1.04, 1.09,
                                    1.12, 1.18, 1.21, 1.41, 1.44, 1.47, 1.50, 1.53, 1.56],
                phoneme_offsets_sec=[0.81, 0.88, 0.96, 0.99, 1.04, 1.09, 1.12,
                                     1.18, 1.21, 1.25, 1.44, 1.47, 1.50, 1.53, 1.56, 1.60],
                alignment_source="TextGrid",
            ),
            audio_ref="h5://speech_stimuli.h5#audio01",
        )
        print("=== TextSample (input) — Continuous Speech ===")
        print(sample2.to_json())
        print()

        # ---- Output examples ----
        print("=== Sentence Generation output ===")
        print(SentenceGenerationOutput(
            sample_id="roamm_sub-10014_sentence-0003",
            model_name="ExampleModel-v0",
            generated_text="Pluto is a dwarf planet in the Kuiper belt.",
            ground_truth_text=(
                "Pluto (minor-planet designation: 134340 Pluto) is a "
                "dwarf planet in the Kuiper belt, a ring of bodies beyond Neptune."
            ),
        ).to_json())
        print()

        print("=== Word Decoding output ===")
        print(WordDecodingOutput(
            sample_id="speech_sub-001_word-0042",
            model_name="ExampleModel-v0",
            predicted_word="man",
            ground_truth_word="MAN",
            confidence=0.87,
            vocab_size=512,
        ).to_json())
        print()

        print("=== Mind-Wandering Detection output ===")
        print(MWDetectionOutput(
            sample_id="roamm_sub-10014_span-0007",
            model_name="ExampleModel-v0",
            predicted_label=0,
            ground_truth_label=0,
            confidence=0.92,
        ).to_json())

        sys.exit(0)

    # Default: print task registry
    print("=== EEG-to-Text Task Registry ===\n")
    for tid, spec in TASK_REGISTRY.items():
        print(f"[{tid}]")
        print(f"  Name:        {spec.task_name}")
        print(f"  Description: {spec.description[:120]}...")
        print(f"  Output:      {spec.output_type}")
        print(f"  Datasets:    {', '.join(spec.applicable_datasets)}")
        print(f"  Primary:     {spec.primary_metric}")
        print(f"  Secondary:   {', '.join(spec.secondary_metrics)}")
        print()
