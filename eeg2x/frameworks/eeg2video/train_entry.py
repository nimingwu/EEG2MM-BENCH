"""Subprocess entry point for one EEG-to-Video training track.

``eeg2video_core.config`` snapshots ``EEG2VIDEO_RUN_ROOT`` and
``EEG2VIDEO_CHECKPOINT_ROOT`` when it is imported.  Training therefore has to
start in a fresh process with those variables already set; the benchmark CLI
spawns this module once per (encoder, subject) pair.

The HDF5 branch's ``model_input_normalization`` is exported before importing
the model as well.  The training loop receives the mode explicitly, but the
checkpoint writer records the module-level value, so importing first would
stamp ``zscore_train_split`` onto LaBraM/LUNA checkpoints that were actually
trained without standardisation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--h5-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage-one-epochs", type=int, default=200)
    parser.add_argument("--stage-two-epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--result-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    train_h5 = (
        Path(args.h5_dir)
        / f"{args.dataset_id}_{args.subject}_train_epoched.h5"
    )
    import h5py

    with h5py.File(train_h5, "r") as h5:
        normalization = str(
            h5.attrs.get("model_input_normalization", "zscore_train_split")
        )

    number = str(args.subject).split("-")[-1].lstrip("0") or "1"
    os.environ.update({
        "EEG2VIDEO_SUBJECT": number,
        "EEG2VIDEO_RUN_ROOT": str(work_dir / args.subject),
        "EEG2VIDEO_CHECKPOINT_ROOT": str(checkpoint_dir),
        "EEG2VIDEO_SHARED_RUN_ROOT": str(work_dir / "shared"),
        "EEG2VIDEO_SHARED_TUNE_ROOT": str(checkpoint_dir / "shared"),
        "EEG2VIDEO_SEMANTIC_TARGET_MODE": "per_video_caption",
        "EEG2VIDEO_MODEL_INPUT_NORMALIZATION": normalization,
        "PYTHONHASHSEED": "2000",
        "HF_HOME": "/datas/hf-cache",
    })

    from eeg2x.frameworks.eeg2video.core import scheme_d_training as trainer
    from eeg2x.frameworks.eeg2video.model import EEG2VideoModel

    # Stage 2 checks Stage-1 completeness against the module default.  Keep
    # that default equal to this run's Stage-1 epochs, otherwise a shortened
    # run would silently retrain Stage 1 for 200 epochs inside Stage 2.
    trainer.SETTINGS = dataclasses.replace(
        trainer.SETTINGS,
        scheme_d_stage_one_epochs=args.stage_one_epochs,
    )

    model = EEG2VideoModel({
        "h5_dir": args.h5_dir,
        "subject_id": args.subject,
        "encoder": args.encoder,
        "dataset_id": args.dataset_id,
        "checkpoint_dir": str(checkpoint_dir),
        "work_dir": str(work_dir),
        "device": args.device,
        "stage_one_epochs": args.stage_one_epochs,
        "stage_two_epochs": args.stage_two_epochs,
        "batch_size": args.batch_size,
    })
    result = model.train()
    payload = {
        "result": result,
        "encoder": args.encoder,
        "subject": args.subject,
        "checkpoint_dir": str(checkpoint_dir),
        "stage_one_epochs": args.stage_one_epochs,
        "stage_two_epochs": args.stage_two_epochs,
    }
    if args.result_json:
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("TRAIN_RESULT " + json.dumps(payload, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
