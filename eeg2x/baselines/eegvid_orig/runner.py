"""EEGVid（Orig.）基线：从视频仓库的干净轨道重跑推理与评测。

EEGVid（arXiv:2505.21385）用自己的 triplet 编码器把整段 EEG 变成 psi，
再交给一个时序条件 GAN 生成帧。它是 baseline：编码器固定，不经过
``eeg2x.encoders``，也不接受 ``--encoder``。

**只做推理**：编码器 checkpoint 与 GAN 快照都从视频仓库的干净轨道读，
共享缓存用软链接接进来，不复制那十几 GB 的帧与 GAN 中间产物。

EMD 只保留 sub-02_k1250 这一条轨道（产物路径文件里的最终集合）。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import h5py

from eeg2x.data import write_json
from eeg2x.datasets import DatasetSpec
from eeg2x.evaluation.api import evaluate_predictions
from eeg2x.frameworks.eeg2video import noise as video_noise
from eeg2x.paths import (
    V2_VIDEO_ROOT, V3_ROOT, require_v3_output, video_dataset_id, video_h5_dir,
)
from eeg2x.protocol import SEED
from eeg2x.tasks.video import VideoSample

log = logging.getLogger("eeg2x.baselines.eegvid_orig")

_SOURCE_DIRS = {
    "seed_dv": "eegvid",
    "ds005662": "eegvid_ds005662",
    "ds008257": "eegvid_ds008257",
}

#: 最终纳入的轨道。EMD 只保留 sub-02_k1250；其余数据集只有 sub-01。
FINAL_SUBJECTS = {
    "seed_dv": ("sub-01",),
    "ds005662": ("sub-01",),
    "ds008257": ("sub-02_k1250",),
}


def available_subjects(dataset: str) -> tuple[str, ...]:
    if dataset not in FINAL_SUBJECTS:
        raise KeyError(
            f"EEGVid Orig. has no track for {dataset!r}; known: "
            f"{sorted(FINAL_SUBJECTS)}")
    return FINAL_SUBJECTS[dataset]


def source_dir(dataset: str, subject: str) -> Path:
    return V2_VIDEO_ROOT / "runs" / _SOURCE_DIRS[dataset] / subject


def output_dir(dataset: str, subject: str) -> Path:
    return V3_ROOT / "artifacts" / dataset / "eegvid_orig" / subject


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve())


def _stage(source: Path, track: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"missing EEGVid Orig. track {source}")
    track.mkdir(parents=True, exist_ok=True)
    _link(source / "checkpoints", track / "checkpoints")
    _link(source / "shared", track / "shared")


def _set_env(dataset: str, subject: str, track: Path) -> None:
    os.environ.update({
        "EEG2VIDEO_SUBJECT": str(subject.split("_")[0]).split("-")[-1].lstrip("0") or "1",
        "EEG2VIDEO_DATASET": dataset,
        "EEG2VIDEO_RUN_ROOT": str(track.resolve()),
        "EEG2VIDEO_CHECKPOINT_ROOT": str((track / "checkpoints").resolve()),
        "EEG2VIDEO_SHARED_RUN_ROOT": str((track / "shared").resolve()),
    })


def _load_samples(h5_path: Path) -> List[VideoSample]:
    with h5py.File(h5_path, "r") as h5:
        n = int(h5.attrs["n_epochs"])
        return [VideoSample.from_h5_row(h5, row, h5_path=h5_path)
                for row in range(n)]


def _scoped_h5(test_h5: Path, track: Path) -> Path:
    scope = track / "eval_h5"
    scope.mkdir(parents=True, exist_ok=True)
    link = scope / test_h5.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(test_h5.resolve())
    return scope


def run_inference(dataset: DatasetSpec, subject: str, *,
                  device: str = "cuda:0", noise: str = "none",
                  noise_seed: int = SEED, overwrite: bool = False) -> Path:
    """从视频仓库的干净轨道重跑 EEGVid Orig. 的推理与 7 指标评测。"""
    if subject not in available_subjects(dataset.name):
        raise ValueError(
            f"{dataset.name}: EEGVid Orig. only retains "
            f"{list(available_subjects(dataset.name))}; got {subject!r}")
    dataset_id = video_dataset_id(dataset.name)
    source = source_dir(dataset.name, subject)
    clean_track = output_dir(dataset.name, subject)
    _stage(source, clean_track)

    output_root = require_v3_output(clean_track)
    if noise != "none":
        output_root = require_v3_output(output_root / "noise_control")
    output_root.mkdir(parents=True, exist_ok=True)
    video_noise.configure(noise, noise_seed)
    _set_env(dataset.name, subject, clean_track)

    from .model import EEGVidModel

    # H5 用被试本名（sub-02），轨道目录用 k1250 后缀（sub-02_k1250）。
    h5_subject = subject.split("_")[0]
    h5_dir = video_h5_dir(dataset.name, "eegvid_nice")
    test_h5 = h5_dir / f"{dataset_id}_{h5_subject}_test_epoched.h5"
    if not test_h5.is_file():
        raise FileNotFoundError(f"missing video test HDF5 {test_h5}")

    predictions_path = output_root / "predictions.json"
    if overwrite or not predictions_path.is_file():
        samples = _load_samples(test_h5)
        model = EEGVidModel({
            "subject_id": h5_subject,
            "h5_dir": str(h5_dir),
            "dataset_id": dataset_id,
            "work_dir": str(clean_track),
            "device": device,
        })
        generated = model.generate(samples)
        predictions = {item["sample_id"]: {"path": item["path"]}
                       for item in generated}
        if len(predictions) != len(samples):
            raise RuntimeError(
                f"EEGVid returned {len(predictions)} predictions for "
                f"{len(samples)} samples")
        write_json(predictions_path, {"predictions": predictions})
    else:
        log.info("reusing predictions: %s", predictions_path)

    evaluation = evaluate_predictions(
        task="video_generation",
        h5_dir=_scoped_h5(test_h5, output_root),
        predictions_path=predictions_path,
        output_path=output_root / "evaluation" / "generation_eegvid.json",
        metrics=None)

    write_json(output_root / "run_config.json", {
        "experiment_id": output_root.name,
        "dataset": dataset.name, "subject_id": subject,
        "task": "video_generation", "model": "eegvid_orig",
        "pipeline": ("EEG -> triplet encoder -> psi -> temporally "
                     "conditioned GAN -> frames"),
        "noise": noise, "noise_seed": int(noise_seed),
        "metrics": ["video_2way_accuracy", "video_40way_accuracy",
                    "frame_2way_accuracy", "frame_40way_accuracy",
                    "video_ssim", "video_psnr", "video_hue_pcc"],
        "source_track": str(source),
        "seed": SEED,
        "derived_inputs_used": False,
    })
    return evaluation


__all__ = ["available_subjects", "output_dir", "run_inference", "source_dir"]
