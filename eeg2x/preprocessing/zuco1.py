#!/usr/bin/env python3
"""Create protocol-v2 ZuCo1 raw-EEG files for the GLIM experiments.

The source is the already versioned GLIM ZuCo1 dataframe (128 Hz, 1280
samples, 128 channels).  Two explicit branches are written:

* ``zuco1_glim_*``   keeps the 128-Hz GLIM input contract;
* ``zuco1_labram_*`` applies exact 128 -> 200 Hz polyphase resampling and
  writes 2000 samples, which is the LaBraM downstream contract.

No target, split, or preprocessing statistic is fitted from test data.  Each
subject/split is a separate HDF5 file so the protocol ``subject_id`` remains
unambiguous.  The script deliberately keeps the source dataframe outside the
benchmark artifact tree; only protocol HDF5 files are consumed by models.
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
    resample_eeg,
    write_common_h5_metadata,
    write_stimulus_manifest,
)
from eeg2x.h5_schema import require_valid_h5_file
from eeg2x.profiles import get_preprocessing_profile
from eeg2x.evaluation.registry import get_task_config

log = logging.getLogger("prepare_zuco1_glim")

TARGETS = (
    "lexical simplification (v0)", "lexical simplification (v1)",
    "semantic clarity (v0)", "semantic clarity (v1)",
    "syntax simplification (v0)", "syntax simplification (v1)",
    "naive rewritten", "naive simplified",
)


def _as_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _write_group(rows: pd.DataFrame, out_path: Path, *, branch: str,
                 source_path: Path) -> None:
    if rows.empty:
        return
    is_200hz = branch in {"labram", "reve"}
    n_epochs = len(rows)
    n_channels = 128
    n_times = 2000 if is_200hz else 1280
    sampling_rate = 200.0 if is_200hz else 128.0
    profile_id = ("zuco1-reve-raw-v1" if branch == "reve" else
                  "zuco1-labram-raw-v1" if branch == "labram" else
                  "zuco1-luna-raw-v1" if branch == "luna" else
                  "zuco1-glim-raw-v1")
    dataset_id = f"zuco1_{branch}"
    subject = _as_text(rows.iloc[0]["subject"])
    split = _as_text(rows.iloc[0]["phase"])
    # benchmark uses validation, while the original GLIM dataframe calls it
    # val.  Keep a single canonical spelling in the HDF5 protocol.
    split = "validation" if split == "val" else split
    out_path.parent.mkdir(parents=True, exist_ok=True)

    str_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset(
            "data", shape=(n_epochs, n_channels, n_times), dtype="f4",
            chunks=(1, n_channels, n_times), compression="gzip",
            compression_opts=1,
        )
        for name in ("sentence_text", "input_text", "raw_input_text",
                     "target_texts_json", "subject", "task"):
            h5.create_dataset(name, shape=(n_epochs,), dtype=str_dt)
        h5.create_dataset("text_uid", shape=(n_epochs,), dtype="i8")
        h5.create_dataset("valid_samples", shape=(n_epochs,), dtype="i4")
        h5.create_dataset("channel_names", data=np.asarray(
            [f"ZuCoEEG{i:03d}" for i in range(n_channels)], dtype=object),
            dtype=str_dt)

        valid = []
        records = []
        for i, (_, row) in enumerate(rows.iterrows()):
            eeg = np.asarray(row["eeg"], dtype=np.float32)
            if eeg.shape != (1280, 128):
                raise ValueError(f"Unexpected EEG shape {eeg.shape} in row {i}")
            mask = np.asarray(row.get("mask", np.ones(1280)), dtype=np.int8)
            n_valid_src = int(np.count_nonzero(mask))
            if is_200hz:
                # resample_eeg follows the protocol's exact rational ratio;
                # the result is then represented in (channels, time) order.
                eeg = resample_eeg(eeg, 128.0, 200.0, axis=0)
                if eeg.shape[0] != 2000:
                    raise RuntimeError(f"128->200 resampling yielded {eeg.shape}")
                n_valid = int(round(n_valid_src * 200.0 / 128.0))
            else:
                n_valid = n_valid_src
            valid.append(min(n_valid, n_times))
            h5["data"][i] = np.asarray(eeg[:n_times, :].T, dtype=np.float32)
            sentence = _as_text(row.get("input text", row.get("raw text", "")))
            raw = _as_text(row.get("raw text", sentence))
            targets = [_as_text(row.get(k, "")) for k in TARGETS]
            h5["sentence_text"][i] = sentence
            h5["input_text"][i] = sentence
            h5["raw_input_text"][i] = raw
            h5["target_texts_json"][i] = json.dumps(targets, ensure_ascii=False)
            h5["subject"][i] = _as_text(row.get("subject", subject))
            h5["task"][i] = _as_text(row.get("task", "task1"))
            h5["text_uid"][i] = int(row.get("text uid", i))
            native = f"{subject}-{_as_text(row.get('task', 'task1'))}-" \
                     f"row-{int(row.name):06d}"
            records.append(StimulusRecord(
                stimulus_id=make_stimulus_id(dataset_id, native),
                modality="text",
                uri=str(source_path),
                alignment_source="zuco1_glim_dataframe:sentence_eeg_text_match",
                transform={
                    "branch": branch, "source_row": int(row.name),
                    "text_uid": int(row.get("text uid", i)),
                    "target_variants": len(targets),
                    "source_sampling_rate": 128.0,
                    "output_sampling_rate": sampling_rate,
                },
            ))

        h5.create_dataset("valid_eeg_samples", data=np.asarray(valid, dtype="i4"))
        write_common_h5_metadata(
            h5, dataset_id=dataset_id, task_id="sentence_generation",
            subject_id=subject, sampling_rate=sampling_rate,
            source_sampling_rate=128.0, n_channels=n_channels,
            n_epochs=n_epochs, split=split, epoch_type="sentence",
            eeg_units="dataset_native",
            profile=get_preprocessing_profile(profile_id),
            preprocessing_state={
                "fitted_split": None,
                "source_dataframe": str(source_path.name),
                "source_sampling_rate": 128.0,
                "resampling": "polyphase" if is_200hz else "none",
                "random_seed": 2000,
            },
            adapter_version="zuco1-glim-benchmark-v1",
        )
        write_stimulus_manifest(h5, records)

    ok_path = out_path
    require_valid_h5_file(ok_path, get_task_config("sentence_generation")["h5_schema"])
    log.info("wrote %s (%d epochs)", out_path, n_epochs)


def prepare(source: str, out_dir: str, *, subjects=None, encoders=None) -> None:
    source_path = Path(source).resolve()
    out_root = Path(out_dir).resolve()
    log.info("loading source dataframe %s", source_path)
    df = pd.read_pickle(source_path)
    if subjects:
        df = df[df["subject"].astype(str).isin([str(s) for s in subjects])]
        if df.empty:
            raise ValueError(f"no rows for subjects {list(subjects)}")
    required = {"eeg", "mask", "subject", "phase", "input text", "text uid"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"source dataframe missing columns: {missing}")
    if set(df["phase"].astype(str)) - {"train", "val", "test"}:
        raise ValueError("unexpected phase in source dataframe")
    branches = tuple(encoders) if encoders else ("glim", "labram", "luna", "reve")
    for branch in branches:
        branch_root = out_root / branch
        for (subject, phase), rows in df.groupby(["subject", "phase"], sort=True):
            slug = str(phase)
            name = f"zuco1_{branch}_{subject}_{slug}_sentences.h5"
            _write_group(rows, branch_root / name, branch=branch, source_path=source_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                    help="GLIM v2 zuco1_eeg_label_8variants.df")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s | %(levelname)-7s | %(message)s")
    prepare(args.source, args.out_dir)


if __name__ == "__main__":
    main()
