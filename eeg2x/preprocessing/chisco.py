#!/usr/bin/env python3
"""
Chisco (ds005170) dataframe -> protocol-v2 HDF5.

Second half of the Chisco branch: consumes the dataframe produced by
``preprocess_chisco_seed2000.py`` (500 Hz, 122 EEG channels, imagine phase) and
writes the four model-rate HDF5 branches, mirroring ``prepare_zuco1_glim.py``:

* ``chisco_glim_*``   / ``chisco_luna_*``   128 Hz, 1280 samples
* ``chisco_labram_*`` / ``chisco_reve_*``   200 Hz, 2000 samples

Those are exactly ZuCo's shapes, so the existing runners and encoders work with
no change.

Bandwidth: the dataset ships a **1 Hz high-pass with no low-pass** (verified in
the fif metadata, `highpass=1.0 / lowpass=250.0`).  A high-pass cannot be undone
from a derivative, so the upstream step supplies only the missing side, giving an
effective **1-100 Hz**.  At the 128 Hz output rate the Nyquist limit is 64 Hz, so
on the glim/luna branches the low-pass is bounded by the output rate rather than
by the filter -- the accepted cost of matching ZuCo's shape contract.

Two protocol points worth stating explicitly, because both are judgement calls:

* **Channel padding.**  Chisco's `preprocessing.py` drops 8 of the 133 EDF
  signals and keeps 125 channels (122 EEG + VEO + HEO + Trigger).
  DATA_PROTOCOL.md requires EEG channels only, so VEO/HEO/Trigger are dropped
  here, leaving **122**.  The models' contract is 128 channels (the LaBraM
  adapter hard-rejects anything else), so 6 explicit zero channels are appended,
  the same shape-preserving move ZuCo makes when padding 104 -> 128.  Real
  montage names are kept for the 122; the pad channels are named ``PAD000..005``
  so they are visibly not electrodes.  The mapping is recorded in
  ``preprocessing_state``.

* **No ``target_texts_json``.**  Chisco has one sentence per trial -- there is no
  multi-variant reference set, so no such field is written.  SemKey's MTV BLEU
  consequently raises a clear "dataset does not provide it" error for Chisco
  instead of silently scoring against a fabricated reference; Standard BLEU and
  every other metric are unaffected.

Splitting is text-based (8:1:1 over unique sentences, ``seed=2000``) so a
sentence can never appear in two splits.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from eeg2x.protocol import (
    StimulusRecord,
    make_stimulus_id,
    normalize_channel_names,
    resample_eeg,
    write_common_h5_metadata,
    write_stimulus_manifest,
)
from eeg2x.h5_schema import require_valid_h5_file
from eeg2x.profiles import get_preprocessing_profile
from eeg2x.evaluation.registry import get_task_config

log = logging.getLogger("prepare_chisco_glim")

SEED = 2000
TASK_ID = "sentence_generation"
# Channels written to the HDF5, PER BRANCH.  ds005170 has 122 real EEG
# channels.
#
#   labram / luna / reve : 122  -- the real channels, no padding.  Their
#                          adapters read the channel count off the tensor
#                          (LUNA/REVE build positions from
#                          `_fixed_positions(c, ...)`, LaBraM selects `c` of its
#                          128 pretrained position embeddings via
#                          `input_chans`), and GLIM's throwaway EEGEncoder is
#                          widened by `encoder_width()` to satisfy num_heads.
#   glim                 : 128  -- 122 + 6 declared `PAD0xx` zeros.  glim is
#                          the one branch that does NOT replace the encoder, so
#                          its `input_dim` IS the EEGEncoder's hidden_dim and
#                          must satisfy `n_channels % num_heads == 0`.
#                          `encoder_width(122, 8)` is 128, so padding here and
#                          rounding there are the same number -- but the h5 has
#                          to carry it, because input_dim is read from the file.
#
# The padding is declared in `channel_names`, never disguised as electrodes.
N_CHANNELS_OUT = {"glim": 128, "labram": 122, "luna": 122, "reve": 122}
SOURCE_SAMPLING_RATE = 500.0
# 8:1:1 on unique sentences; k mirrors prepare_zuco1_glim's text-uid split.
TRAIN_FRACTION = 0.8


def _as_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def assign_splits(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Text-based 8:1:1 split over unique sentences.

    Every reading of a given sentence lands in the same split, so no sentence
    can straddle the boundary -- the same rule ZuCo's preprocessing uses.
    """
    uid_of = {t: i for i, t in enumerate(sorted(df["text"].astype(str).unique()))}
    uids = np.array([uid_of[t] for t in df["text"].astype(str)], dtype=np.int64)
    unique = np.arange(len(uid_of), dtype=np.int64)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_train = int(round(TRAIN_FRACTION * len(unique)))
    n_val = int(round(0.1 * len(unique)))
    train = set(shuffled[:n_train].tolist())
    val = set(shuffled[n_train:n_train + n_val].tolist())
    test = set(shuffled[n_train + n_val:].tolist())

    phase = []
    for u in uids:
        phase.append("train" if u in train else "val" if u in val else "test")
    out = df.copy()
    out["text_uid"] = uids
    out["phase"] = phase
    return out


def pad_channels(eeg: np.ndarray, n_out: int) -> np.ndarray:
    """(T, C) -> (T, n_out), zero-filling extra channels on the right."""
    t, c = eeg.shape
    if c > n_out:
        raise ValueError(f"cannot pad {c} channels down to {n_out}")
    if c == n_out:
        return eeg.astype(np.float32, copy=False)
    pad = np.zeros((t, n_out - c), dtype=np.float32)
    return np.concatenate([eeg.astype(np.float32, copy=False), pad], axis=1)


# Column order of the variant table, matching ZuCo's `target_texts_json` layout
# so downstream consumers (SemKey MTV BLEU, glim_text's target keys) are
# unchanged.  Chisco's table is produced by ../chisco_variants/.
TARGET_KEYS = (
    "lexical simplification (v0)", "lexical simplification (v1)",
    "semantic clarity (v0)", "semantic clarity (v1)",
    "syntax simplification (v0)", "syntax simplification (v1)",
    "naive rewritten", "naive simplified",
)


def load_variants(path: str | None) -> dict:
    """sentence -> list of 8 variant strings (see TARGET_KEYS).

    Returns an empty dict when no table is supplied, in which case no
    `target_texts_json` is written and SemKey's MTV BLEU reports the dataset as
    not providing one -- a visible failure rather than a fabricated reference.
    """
    if not path:
        return {}
    df = pd.read_pickle(path)
    key = "sentence" if "sentence" in df.columns else df.columns[0]
    missing = [k for k in TARGET_KEYS if k not in df.columns]
    if missing:
        raise ValueError(f"variant table {path} missing columns: {missing}")
    out = {}
    for _, r in df.iterrows():
        vals = [str(r[k]) if isinstance(r[k], str) else "" for k in TARGET_KEYS]
        if all(vals):
            out[str(r[key])] = vals
    log.info("loaded %d sentences with a complete 8-variant set from %s",
             len(out), path)
    return out


def _write_group(rows: pd.DataFrame, out_path: Path, *, branch: str,
                 source_path: Path, channel_names: list, eeg_units: str,
                 variants: dict | None = None) -> None:
    if rows.empty:
        return
    variants = variants or {}
    is_200hz = branch in {"labram", "reve"}
    target_hz = 200.0 if is_200hz else 128.0
    n_times = 2000 if is_200hz else 1280
    profile_id = (f"chisco-reve-raw-v1" if branch == "reve" else
                  f"chisco-labram-raw-v1" if branch == "labram" else
                  f"chisco-luna-raw-v1" if branch == "luna" else
                  f"chisco-glim-raw-v1")
    dataset_id = f"chisco_{branch}"
    subject = _as_text(rows.iloc[0]["subject"])
    split = _as_text(rows.iloc[0]["phase"])
    split = "validation" if split == "val" else split
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_epochs = len(rows)
    n_channels = N_CHANNELS_OUT[branch]
    names = list(channel_names) + [f"PAD{i:03d}" for i in range(n_channels - len(channel_names))]
    names = normalize_channel_names(names)

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("data", shape=(n_epochs, n_channels, n_times), dtype="f4",
                          chunks=(1, n_channels, n_times), compression="gzip",
                          compression_opts=1)
        for name in ("sentence_text", "input_text", "raw_input_text",
                     "subject", "task"):
            h5.create_dataset(name, shape=(n_epochs,), dtype=str_dt)
        if variants:
            h5.create_dataset("target_texts_json", shape=(n_epochs,), dtype=str_dt)
        h5.create_dataset("text_uid", shape=(n_epochs,), dtype="i8")
        h5.create_dataset("valid_samples", shape=(n_epochs,), dtype="i4")
        h5.create_dataset("channel_names", data=np.asarray(names, dtype=object),
                          dtype=str_dt)

        valid, records = [], []
        for i, (_, row) in enumerate(rows.iterrows()):
            eeg = np.asarray(row["eeg"], dtype=np.float32)          # (T, 122)

            # exact rational 500 -> 128 / 200, with anti-aliasing
            eeg = resample_eeg(eeg, SOURCE_SAMPLING_RATE, target_hz, axis=0)
            n_valid = int(eeg.shape[0])
            eeg = pad_channels(eeg, n_channels)                      # (T, 128)

            if eeg.shape[0] >= n_times:
                eeg = eeg[:n_times]
            else:
                eeg = np.pad(eeg, ((0, n_times - eeg.shape[0]), (0, 0)), "constant")
            valid.append(min(n_valid, n_times))
            h5["data"][i] = eeg.T                                    # (C, T)

            sentence = _as_text(row.get("text", ""))
            h5["sentence_text"][i] = sentence
            h5["input_text"][i] = sentence          # model input side
            h5["raw_input_text"][i] = sentence      # stimulus side
            if variants:
                refs = variants.get(sentence)
                # Fail visible rather than emit a partial reference set: a
                # sentence missing from the variant table is a real gap.
                h5["target_texts_json"][i] = json.dumps(refs or [], ensure_ascii=False)
            h5["subject"][i] = subject
            h5["task"][i] = _as_text(row.get("task", "task1"))
            h5["text_uid"][i] = int(row.get("text_uid", i))

            records.append(StimulusRecord(
                stimulus_id=make_stimulus_id(
                    dataset_id,
                    f"{subject}-run{int(row.get('run_number', 0)):02d}-"
                    f"imagine-{int(row.get('event_index', i)):04d}"),
                modality="text",
                uri=f"textdataset/split_data_{int(row.get('run_number', 0))}.xlsx",
                onset_sec=float(row.get("onset_sec", np.nan)),
                duration_sec=float(8.3 - 5.0),
                alignment_source="chisco_trigger_65380:imagine_5.0-8.3s",
                transform={
                    "branch": branch,
                    "source_edf": _as_text(row.get("source_edf", "")),
                    "run_number": int(row.get("run_number", 0)),
                    "event_index": int(row.get("event_index", i)),
                    "source_sampling_rate": SOURCE_SAMPLING_RATE,
                    "output_sampling_rate": target_hz,
                    "n_eeg_channels": len(channel_names),
                    "n_channels_out": n_channels,
                },
            ))

        h5.create_dataset("valid_eeg_samples", data=np.asarray(valid, dtype="i4"))
        write_common_h5_metadata(
            h5, dataset_id=dataset_id, task_id=TASK_ID,
            subject_id=subject, sampling_rate=target_hz,
            source_sampling_rate=SOURCE_SAMPLING_RATE,
            n_channels=n_channels, n_epochs=n_epochs, split=split,
            epoch_type="sentence", eeg_units=eeg_units,
            profile=get_preprocessing_profile(profile_id),
            preprocessing_state={
                "fitted_split": None,
                "source_dataframe": source_path.name,
                "source_sampling_rate": SOURCE_SAMPLING_RATE,
                "source_signals": 133,
                "n_eeg_channels": len(channel_names),
                "n_channels_out": n_channels,
                "channel_padding": f"{len(channel_names)} EEG + "
                                   f"{n_channels - len(channel_names)} zero",
                "channel_names": list(channel_names),
                "resampling": "polyphase",
                "bandpass_hz": [1.0, 100.0],
                "bandpass_note": (
                    "1 Hz high-pass is inherited from the dataset's own "
                    "preprocessing and cannot be undone from a derivative; the "
                    "100 Hz low-pass is applied by this benchmark. At 128 Hz the "
                    "output is Nyquist-limited to 64 Hz."),
                "phase": "imagine",
                "source": "derivatives/preprocessed_pkl (dataset's own output)",
                "fit_scope_note": (
                    "Nothing is fitted here. The dataset's PREP/AutoReject/ICA ran "
                    "per recording inside its own preprocessing, so no fitted "
                    "artifact crosses the train/test boundary"),
                "random_seed": SEED,
                "split_rule": "text-based 8:1:1 over unique sentences",
            },
            adapter_version="chisco-glim-benchmark-v1",
        )
        write_stimulus_manifest(h5, records)

    require_valid_h5_file(out_path, get_task_config(TASK_ID)["h5_schema"])
    log.info("wrote %s (%d epochs, %d ch, %.0f Hz)", out_path, n_epochs,
             n_channels, target_hz)


def prepare(source: str, out_dir: str, branches: list,
            variants_path: str | None = None,
            subjects: list | None = None,
            name_suffix: str = "") -> None:
    source_path = Path(source).resolve()
    df = pd.read_pickle(source_path)
    variants = load_variants(variants_path)
    if variants:
        present = set(df["text"].astype(str).unique())
        missing = present - set(variants)
        if missing:
            log.warning("%d of %d sentences have no complete variant set; their "
                        "target_texts_json will be empty", len(missing), len(present))
    required = {"eeg", "text", "subject", "task"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"source dataframe missing columns: {missing}")

    channel_names = list(df.attrs.get("channel_names", []))
    if not channel_names:
        raise ValueError("source dataframe has no channel_names in .attrs")
    eeg_units = str(df.attrs.get("eeg_units", "volts"))

    # 切分必须先在全体被试上分配，再筛被试；否则子集运行会改变全局
    # text_uid 与 phase，产物与全量运行不一致（sub-01 的 test 会从 632 变成 625）。
    df = assign_splits(df)
    if subjects:
        keep = set(subjects)
        present = set(df["subject"].astype(str))
        missing = keep - present
        if missing:
            raise ValueError(f"requested subjects not in the source dataframe: "
                             f"{sorted(missing)}; available: {sorted(present)}")
        df = df[df["subject"].astype(str).isin(keep)].reset_index(drop=True)
        log.info("subject filter -> %d subjects, %d rows", len(keep), len(df))

    log.info("rows=%d subjects=%s splits=%s", len(df),
             sorted(df["subject"].unique()), df["phase"].value_counts().to_dict())

    out_root = Path(out_dir).resolve()
    for branch in branches:
        # One directory PER BRANCH: the loaders scan a directory flat
        # (`sorted(h5_dir.glob("*.h5"))`), so a directory holding several
        # branches would make generate_sentence_predictions.py feed every
        # branch's files to one checkpoint.  `name_suffix` keeps a subject
        # subset (e.g. "sub2") from overwriting the full-subject directory that
        # a running job may still be reading.
        branch_root = out_root / f"{branch}{name_suffix}"
        for (subject, phase), rows in df.groupby(["subject", "phase"], sort=True):
            name = f"chisco_{branch}_{subject}_{phase}_sentences.h5"
            _write_group(rows, branch_root / name, branch=branch,
                         source_path=source_path, channel_names=channel_names,
                         eeg_units=eeg_units, variants=variants)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="chisco_eeg_500hz_imagine.df")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--branches", nargs="+",
                    default=["glim", "labram", "luna", "reve"])
    ap.add_argument("--variants", default=None,
                    help="variant table (chisco_label_8variants.df) produced by "
                         "../chisco_variants/gen_variants_deepseek.py; when given, "
                         "target_texts_json is written so SemKey's MTV BLEU applies")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="keep only these subjects, e.g. --subjects sub-01 sub-02")
    ap.add_argument("--name_suffix", default="",
                    help="inserted into the output directory name, e.g. "
                         "--name_suffix _sub2 gives chisco_<branch>_sub2_seed2000_h5. "
                         "Keeps a subset run from overwriting the full-subject \n"
                         "directory")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    prepare(args.source, args.out_dir, args.branches, args.variants,
            args.subjects, args.name_suffix)


if __name__ == "__main__":
    main()
