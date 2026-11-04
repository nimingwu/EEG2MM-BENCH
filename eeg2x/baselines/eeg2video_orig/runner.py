"""EEG2Video（Orig.）基线：从视频仓库的干净轨道重跑推理与评测。

EEG2Video 原论文的链路不是逐窗口方案：它先做 EEG 预处理与冻结扩散目标的
缓存，再训练三个预测器、微调一个 Tune-A-Video UNet，最后对留出的整个 block
一次性重建。本 runner **只做推理**：把视频仓库里已训练好的 checkpoint 与
缓存接到 v3 的产物目录，绝不重训。

与统一框架的区别（这也是它作为 baselines 的原因）：它用自己的
``eeg2x.baselines.eeg2video_orig.paper_core``，不经过 ``eeg2x.encoders``，
也不接受 ``--encoder``。

产物布局：v2 的共享缓存有十几 GB，不复制；用软链接接进来。checkpoint 同样
软链接，因此 v3 目录只增加几 KB 的目标文件与预测/报告。
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, List

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

log = logging.getLogger("eeg2x.baselines.eeg2video_orig")

#: v2 的 Orig 轨道目录名。EEG2Video Orig. 没有 SEED-DV 的产物。
_SOURCE_DIRS = {
    "ds005662": "eeg2video_paper_ds005662",
    "ds008257": "eeg2video_paper_ds008257",
}


def source_dir(dataset: str, subject: str) -> Path:
    if dataset not in _SOURCE_DIRS:
        raise KeyError(
            f"EEG2Video Orig. has no track for {dataset!r}; known: "
            f"{sorted(_SOURCE_DIRS)} (SEED-DV was never run for this chain)")
    return V2_VIDEO_ROOT / "runs" / _SOURCE_DIRS[dataset] / subject


def output_dir(dataset: str, subject: str) -> Path:
    return V3_ROOT / "artifacts" / dataset / "eeg2video_orig" / subject


def _link(target: Path, link: Path) -> None:
    if link.is_symlink() or link.exists():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target.resolve())


def _stage(source: Path, track: Path) -> None:
    """把 v2 轨道的缓存与 checkpoint 以软链接形式接进 v3 轨道。"""
    if not source.is_dir():
        raise FileNotFoundError(f"missing EEG2Video Orig. track {source}")
    track.mkdir(parents=True, exist_ok=True)
    _link(source / "checkpoints", track / "checkpoints")
    _link(source / "shared", track / "shared")
    for path in source.iterdir():
        if path.is_file() and (
                path.suffix in (".npy", ".jsonl")
                or path.name == "prepare.complete.json"):
            _link(path, track / path.name)
    # Tune-A-Video 的 UNet 是整个数据集共享的，按模型默认布局放在轨道父目录。
    _link(source.parent / "shared" / "tune_a_video",
          track.parent / "shared" / "tune_a_video")


def _set_env(dataset: str, subject: str, track: Path) -> None:
    os.environ.update({
        "EEG2VIDEO_SUBJECT": str(subject).split("-")[-1].lstrip("0") or "1",
        "EEG2VIDEO_DATASET": dataset,
        "EEG2VIDEO_RUN_ROOT": str(track.resolve()),
        "EEG2VIDEO_CHECKPOINT_ROOT": str((track / "checkpoints").resolve()),
        "EEG2VIDEO_SHARED_RUN_ROOT": str((track / "shared").resolve()),
        "EEG2VIDEO_SHARED_TUNE_ROOT": str(
            (track.parent / "shared" / "tune_a_video").resolve()),
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
    """从视频仓库的干净轨道重跑 EEG2Video Orig. 的推理与 7 指标评测。"""
    if subject not in dataset.subjects:
        raise ValueError(f"{dataset.name} has no subject {subject!r}")
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

    # 必须在环境变量就位后再导入：paper_core.config 在 import 时快照根目录。
    from .model import EEG2VideoPaperModel

    h5_dir = video_h5_dir(dataset.name, "eegnet")
    test_h5 = h5_dir / f"{dataset_id}_{subject}_test_epoched.h5"
    if not test_h5.is_file():
        raise FileNotFoundError(f"missing video test HDF5 {test_h5}")

    predictions_path = output_root / "predictions.json"
    if overwrite or not predictions_path.is_file():
        samples = _load_samples(test_h5)
        model = EEG2VideoPaperModel({
            "subject_id": subject,
            "h5_dir": str(h5_dir),
            "work_dir": str(clean_track),
        })
        generated = model.generate(samples)
        predictions = {item["sample_id"]: {"path": item["path"]}
                       for item in generated}
        if len(predictions) != len(samples):
            raise RuntimeError(
                f"paper chain returned {len(predictions)} predictions for "
                f"{len(samples)} samples")
        write_json(predictions_path, {"predictions": predictions})
    else:
        log.info("reusing predictions: %s", predictions_path)

    evaluation = evaluate_predictions(
        task="video_generation",
        h5_dir=_scoped_h5(test_h5, output_root),
        predictions_path=predictions_path,
        output_path=output_root / "evaluation" / "generation_eeg2video.json",
        metrics=None)

    write_json(output_root / "run_config.json", {
        "experiment_id": output_root.name,
        "dataset": dataset.name, "subject_id": subject,
        "task": "video_generation", "model": "eeg2video_orig",
        "pipeline": ("EEG -> seq2seq + semantic + dynamic predictors -> "
                     "Tune-A-Video UNet -> reconstructed clip"),
        "noise": noise, "noise_seed": int(noise_seed),
        "metrics": ["video_2way_accuracy", "video_40way_accuracy",
                    "frame_2way_accuracy", "frame_40way_accuracy",
                    "video_ssim", "video_psnr", "video_hue_pcc"],
        "source_track": str(source),
        "seed": SEED,
        "derived_inputs_used": False,
    })
    return evaluation


__all__ = ["output_dir", "run_inference", "source_dir"]
