#!/usr/bin/env python3
"""TMNRED (ds005383) — dataframe -> protocol-v2 HDF5.

Same task and h5 layout as the other Chinese sentence datasets, so
`benchmark_eval.py` and `metrics/` need no changes.

The one structural difference: TMNRED has **30 EEG channels**, not 128
---------------------------------------------------------------
DATA_PROTOCOL.md:31 is explicit that the protocol does not fix the channel
count -- each adapter declares its own -- so 30 is compliant on its own terms.
The per-branch counts here are:

    labram / reve / luna : 30   the dataset's real channels, no padding
    glim                 : 32   30 real + 2 declared `PAD00x` zero channels

`glim` pads for a reason that is architectural, not contractual: GLIM's
`EEGEncoder` uses `in_dim` (the channel count) as its input blocks'
`hidden_dim`, and `num_heads` is shared with the 256-wide decoder blocks, so
`num_heads` must divide `gcd(n_channels, 256)`.  For 30 that is 2, which would
force `num_heads=2` on a 256-wide decoder; 32 keeps the shared `num_heads=8`.
This is also why Chisco's 122 channels were padded to 128 -- 122 has the same
problem.  The padding is declared in `channel_names` (`PAD000`, `PAD001`), never
disguised as real electrodes, and the zero channels stay exactly zero through
the model (the input-block LayerNorms are `elementwise_affine=False`, so nothing
can activate them).

LUNA and REVE derive their electrode positions from `_fixed_positions(c, ...)`
and are channel-count agnostic; LaBraM selects a subset of its 128 pretrained
channel position embeddings via `input_chans`.  All three were verified to run
with 30 channels directly, so padding them would be a needless deviation.

Split is the same text-based 8:1:1 used everywhere else (seed 2000, `text_uid` =
the ordinal of the sorted unique sentence).  With only 258 unique sentences the
test split is unavoidably small and highly repeated -- recorded per file, and a
reason the dataset's N-way retrieval numbers must be read against chance rather
than against 1.0.

Run with the `glim_2` env:

    python prepare_tmnred_glim.py \\
        --source seed2000_source/tmnred_eeg_200hz.df \\
        --out_dir tmnred_glim_seed2000_h5
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

log = logging.getLogger("prepare_tmnred_glim")

SEED = 2000
TASK_ID = "sentence_generation"
SOURCE_SAMPLING_RATE = 200.0
TRAIN_FRACTION = 0.8

# Per branch: (profile_id, target_hz, n_samples, n_channels_out)
BRANCHES = {
    "glim":   ("tmnred-glim-raw-v1",   128.0, 1280, 32),
    "luna":   ("tmnred-luna-raw-v1",   128.0, 1280, 30),
    "labram": ("tmnred-labram-raw-v1", 200.0, 2000, 30),
    "reve":   ("tmnred-reve-raw-v1",   200.0, 2000, 30),
}

ALIGNMENT_SOURCE = "tmnred_events_tsv:sentence_display_2.2s"


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
    can straddle the boundary -- the same rule ZuCo, Chisco and Chisco-2.0 use,
    and the same code, so a TMNRED run is comparable to them.
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


def _write_group(rows: pd.DataFrame, out_path: Path, *, branch: str,
                 source_path: Path, channel_names: list, eeg_units: str,
                 variants: dict | None = None) -> None:
    if rows.empty:
        return
    variants = variants or {}
    profile_id, target_hz, n_times, n_channels = BRANCHES[branch]
    dataset_id = f"tmnred_{branch}"
    subject = _as_text(rows.iloc[0]["subject"])
    split = _as_text(rows.iloc[0]["phase"])
    split = "validation" if split == "val" else split
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_epochs = len(rows)
    n_pad = n_channels - len(channel_names)
    names = list(channel_names) + [f"PAD{i:03d}" for i in range(n_pad)]
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
            eeg = np.asarray(row["eeg"], dtype=np.float32)      # (T, 30)

            # 200 -> 128 / 200.  On the 200 Hz branches this is identical to the
            # source rate, i.e. a no-op, which `resample_eeg` handles exactly.
            eeg = resample_eeg(eeg, SOURCE_SAMPLING_RATE, target_hz, axis=0)
            n_valid = int(eeg.shape[0])
            eeg = pad_channels(eeg, n_channels)                  # (T, 30 or 32)

            if eeg.shape[0] >= n_times:
                eeg = eeg[:n_times]
            else:
                eeg = np.pad(eeg, ((0, n_times - eeg.shape[0]), (0, 0)), "constant")
            valid.append(min(n_valid, n_times))
            h5["data"][i] = eeg.T                                # (C, T)

            sentence = _as_text(row.get("text", ""))
            h5["sentence_text"][i] = sentence
            h5["input_text"][i] = sentence
            h5["raw_input_text"][i] = sentence
            if variants:
                # Fail visible rather than emit a partial reference set: a
                # sentence missing from the variant table is a real gap, and
                # SemKey's MTV BLEU would otherwise score against a short
                # reference list without saying so.
                h5["target_texts_json"][i] = json.dumps(
                    variants.get(sentence) or [], ensure_ascii=False)
            h5["subject"][i] = subject
            h5["task"][i] = _as_text(row.get("task", "task1"))
            h5["text_uid"][i] = int(row.get("text_uid", i))

            session = _as_text(row.get("session", ""))
            event_index = int(row.get("event_index", i))
            records.append(StimulusRecord(
                stimulus_id=make_stimulus_id(
                    dataset_id, f"{subject}-{session}-{event_index:04d}"),
                modality="text",
                uri="derivatives/source material/source material.xlsx",
                onset_sec=float(row.get("onset_sec", np.nan)),
                duration_sec=float(row.get("duration_sec", np.nan)),
                alignment_source=ALIGNMENT_SOURCE,
                transform={
                    "branch": branch,
                    "source_edf": _as_text(row.get("source_edf", "")),
                    "session": session,
                    "block": _as_text(row.get("block", "")),
                    "condition": _as_text(row.get("condition", "")),
                    "label_value": int(row.get("label_value", -1)),
                    "event_index": event_index,
                    "source_sampling_rate": SOURCE_SAMPLING_RATE,
                    "output_sampling_rate": target_hz,
                    "n_eeg_channels": len(channel_names),
                    "n_channels_out": n_channels,
                    "channel_padding": n_pad,
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
                "source_channels": len(channel_names),
                "n_eeg_channels": len(channel_names),
                "n_channels_out": n_channels,
                "channel_padding": (f"{len(channel_names)} EEG + {n_pad} zero"
                                    if n_pad else "none (real channels only)"),
                "channel_names": list(channel_names),
                "padded_channel_names": names[len(channel_names):],
                "resampling": ("polyphase 200->128" if target_hz == 128.0
                               else "none (already 200 Hz)"),
                "bandpass_hz": [0.5, 80.0],
                "notch_hz": 50.0,
                "bandpass_note": (
                    "0.5-80 Hz FIR + 50 Hz notch, the dataset's own published "
                    "scheme (paper s41597-025-05036-2 and pre.m). The BIDS EDFs "
                    "are already 200 Hz, so the paper's downsample-to-200 step "
                    "is a no-op and is not re-applied."),
                "channel_padding_note": (
                    "glim pads 30 -> 32 because GLIM's EEGEncoder uses the "
                    "channel count as its input blocks' hidden_dim and shares "
                    "num_heads with the 256-wide decoder, so num_heads must "
                    "divide gcd(n_channels, 256); 30 admits only {1,2}. "
                    "labram/reve/luna use the real 30 channels: LUNA and REVE "
                    "derive positions from _fixed_positions(c, ...) and LaBraM "
                    "selects a subset of its 128 pretrained position embeddings "
                    "via input_chans, so all three were verified to run at 30."
                    if n_pad else
                    "None: these encoders are channel-count agnostic and take "
                    "the dataset's real 30 channels."),
                "segment": "sentence display, 2.2 s from each event onset",
                "source": ("ds005383 raw BIDS EDF, cleaned by "
                           "preprocess_tmnred_seed2000.py"),
                "fit_scope_note": (
                    "Nothing in THIS profile fits: read -> select -> resample -> "
                    "pad -> prompt are all deterministic. That is not a claim "
                    "that the data is unfitted. Bad-channel interpolation "
                    "(spherical, from the dataset's CapTrak positions), ICA "
                    "(30 components, extended Infomax, MNE-ICALabel) and the "
                    "bad-trial test all fit on the recording, upstream in "
                    "preprocess_tmnred_seed2000.py. Under a text-level split a "
                    "recording contributes rows to more than one split, so those "
                    "fitted artifacts cross the train/test boundary. "
                    "DATA_PROTOCOL offers no honest fit_scope for that "
                    "('source' is rejected on split=test, 'train' would require "
                    "fitting on the training rows only, 'none' would be false), "
                    "so it is declared here instead."),
                "upstream_pipeline": [
                    "drop mastoid channels A1/A2",
                    "bad channels: PSD > 2 SD -> spherical interpolation",
                    "average reference",
                    "0.5-80 Hz FIR band-pass + 50 Hz notch",
                    "downsample to 200 Hz (no-op: BIDS EDFs are already 200 Hz)",
                    "ICA 30 components + MNE-ICALabel, pop_icflag thresholds",
                    "bad-trial rejection (median variance / median deviation)",
                    "segment: 2.2 s from each event onset",
                ],
                "ica_flag_thresholds": {
                    "muscle artifact": 0.8, "eye blink": 0.6,
                    "channel noise": 0.6, "other": 0.7,
                    "brain": None, "heart beat": None, "line noise": None,
                },
                "bad_channel_rule": (
                    "EEGLAB pop_rejchan(measure='spec', norm='on', "
                    "threshold=2), transcribed from functions/popfunc/"
                    "pop_rejchan.m: per-channel spectrum in dB, per-frequency "
                    "z-scores across channels, max over frequencies, flag > 2. "
                    "Measured 13.89% on six pilot recordings vs the paper's "
                    "10.10%."),
                "known_deviations": [
                    "bad-trial rule exists only as prose (no code released; "
                    "pre.m/Checkbad.m do channel-level checks only and allGo.m "
                    "is 27 lines of orchestration). Measured ~6% rejected vs "
                    "the paper's 2.17%. Not tuned to match -- a threshold "
                    "fitted to a number is not a reproduced method.",
                ],
                "random_seed": SEED,
                "split_rule": "text-based 8:1:1 over unique sentences",
            },
            adapter_version="tmnred-glim-benchmark-v1",
        )
        write_stimulus_manifest(h5, records)

    require_valid_h5_file(out_path, get_task_config(TASK_ID)["h5_schema"])
    log.info("wrote %s (%d epochs, %d ch, %.0f Hz)", out_path, n_epochs,
             n_channels, target_hz)


# Column order of the variant table, matching ZuCo's `target_texts_json`
# layout so downstream consumers (SemKey MTV BLEU, glim_text's target keys) are
# unchanged.  TMNRED's table is produced by ../tmnred_variants/, which uses the
# same generator and prompts as ds005170's -- the two sentence sets do not
# overlap at all, so nothing could be reused.
TARGET_KEYS = (
    "lexical simplification (v0)", "lexical simplification (v1)",
    "semantic clarity (v0)", "semantic clarity (v1)",
    "syntax simplification (v0)", "syntax simplification (v1)",
    "naive rewritten", "naive simplified",
)


def load_variants(path: str | None) -> dict:
    """sentence -> list of 8 variant strings (see TARGET_KEYS)."""
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


def prepare(source: str, out_dir: str, branches: list,
            variants_path: str | None = None,
            subjects: list | None = None,
            name_suffix: str = "",
            keep_full_split: bool = False) -> None:
    source_path = Path(source).resolve()
    df = pd.read_pickle(source_path)
    required = {"eeg", "text", "subject", "task", "session", "event_index",
                "onset_sec", "duration_sec", "source_edf"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"source dataframe missing columns: {missing}")
    channel_names = list(df.attrs.get("channel_names", []))
    if not channel_names:
        raise ValueError("source dataframe has no channel_names in .attrs")
    eeg_units = str(df.attrs.get("eeg_units", "volts"))
    if float(df.attrs.get("source_sampling_rate", 0)) != SOURCE_SAMPLING_RATE:
        raise ValueError(f"expected {SOURCE_SAMPLING_RATE} Hz source, got "
                         f"{df.attrs.get('source_sampling_rate')}")

    variants = load_variants(variants_path)
    if variants:
        present = set(df["text"].astype(str))
        missing_v = present - set(variants)
        if missing_v:
            log.warning("%d of %d sentences have no complete variant set; their "
                        "target_texts_json will be empty",
                        len(missing_v), len(present))

    # Subject subset, filtered BEFORE the split assignment so the result is a
    # self-contained dataset with its own 8:1:1 -- the same order Chisco's
    # `--subjects sub-01 sub-02` uses.  The mapping is re-derived from the
    # surviving sentences, so it CAN differ from the full run's; for TMNRED it
    # does not (sub-01..sub-12 still cover all 258 sentences -- checked), which
    # makes the subset's test rows a subset of the full run's.
    pending = set(subjects) if subjects else None
    if pending and not keep_full_split:
        present = set(df["subject"].astype(str))
        missing_s = pending - present
        if missing_s:
            raise ValueError(f"requested subjects not in the source dataframe: "
                             f"{sorted(missing_s)}; available: {sorted(present)}")
        df = df[df["subject"].astype(str).isin(pending)].reset_index(drop=True)
        log.info("subject filter -> %d subjects, %d rows", len(pending), len(df))

    df = assign_splits(df)
    if pending and keep_full_split:
        df = df[df["subject"].astype(str).isin(pending)].reset_index(drop=True)
        log.info("subject filter after split -> %d subjects, %d rows",
                 len(pending), len(df))
    log.info("rows=%d subjects=%d unique_sentences=%d splits=%s", len(df),
             df["subject"].nunique(), df["text"].nunique(),
             df["phase"].value_counts().to_dict())
    log.warning("only %d unique sentences for %d rows (~%.0fx repetition); the "
                "test split will be small and highly repeated",
                df["text"].nunique(), len(df), len(df) / df["text"].nunique())

    out_root = Path(out_dir).resolve()
    for branch in branches:
        if branch not in BRANCHES:
            raise ValueError(f"unknown branch {branch!r}")
        # One directory PER BRANCH, matching the chisco_*_seed2000_h5 layout.
        # This is load-bearing, not cosmetic: the loaders scan a directory flat
        # (`sorted(h5_dir.glob("*.h5"))`), so a directory holding several
        # branches would make generate_sentence_predictions.py feed every
        # branch's files to one checkpoint.  `name_suffix` keeps a subject
        # subset (e.g. "sub12") from overwriting the full-subject directory a
        # running job may still be reading.
        branch_root = out_root / f"{branch}{name_suffix}"
        for (subject, phase), rows in df.groupby(["subject", "phase"], sort=True):
            name = f"tmnred_{branch}_{subject}_{phase}_sentences.h5"
            _write_group(rows, branch_root / name, branch=branch,
                         source_path=source_path, channel_names=channel_names,
                         eeg_units=eeg_units, variants=variants)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out_dir", required=True,
                    help="root directory; one tmnred_<branch>_seed2000_h5/ is "
                         "created inside it per branch, because the loaders "
                         "scan a directory flat")
    ap.add_argument("--branches", nargs="+", default=list(BRANCHES))
    ap.add_argument("--variants", default=None,
                    help="variant table from ../tmnred_variants/"
                         "gen_variants_deepseek.py; when given, "
                         "target_texts_json is written so SemKey's MTV BLEU "
                         "applies instead of reporting the dataset as not "
                         "providing references")
    ap.add_argument("--subjects", nargs="+", default=None,
                    help="keep only these subjects, e.g. --subjects sub-01 ... "
                         "sub-12; the 8:1:1 split is then re-derived from the "
                         "surviving sentences")
    ap.add_argument("--name_suffix", default="",
                    help="appended to the dataset part of the directory name; "
                         "--name_suffix _sub12 gives "
                         "tmnred_<branch>_sub12_seed2000_h5, so a subject "
                         "subset cannot overwrite the full-subject directory")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    prepare(args.source, args.out_dir, args.branches, args.variants,
            args.subjects, args.name_suffix)


if __name__ == "__main__":
    main()
