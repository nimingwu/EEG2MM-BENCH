"""EEG-to-Video 框架的编排：train -> generate -> evaluate。

迁自视频仓库的 ``scripts/run_eeg2video_track.py``。训练逻辑仍在
``eeg2x.frameworks.eeg2video.core.scheme_d_training``；因为视频的配置模块在
import 时快照目录环境变量，训练由 :mod:`eeg2x.frameworks.eeg2video.train_entry`
在独立子进程里跑，CLI 通过 :func:`run_training` 调度。推理优先使用 v3 训练
出的 checkpoint，没有时回落到视频仓库的干净轨道。

与图像框架的差别：

* 目标空间是 CLIP 文本上下文 ``[B, 77, 768]``，不是 1024 维图像嵌入；
* 生成器是 AnimateDiff，不是 SDXL；
* 噪声对照的语义不同：视频分支存的是原始幅度 EEG，噪声按每个试次自身的
  均值与方差抽样，而不是标准正态。共享实现见 ``eeg2x.frameworks.eeg2video.noise``。
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import h5py

from eeg2x.data import write_json
from eeg2x.datasets import DatasetSpec
from eeg2x.evaluation.api import evaluate_predictions
from eeg2x.paths import (
    V3_ROOT, require_v3_output, video_artifact_dir, video_dataset_id,
    video_h5_dir,
)
from eeg2x.protocol import SEED

from . import noise as video_noise

log = logging.getLogger("eeg2x.eeg2video.pipeline")

STAGES = ("generate", "evaluate")
INFERENCE_STAGES = STAGES


@dataclass
class RunPaths:
    """一次视频运行涉及的输入（v2 视频仓库）与输出（v3）。"""

    dataset: str
    encoder: str
    subject: str
    dataset_id: str
    source: Path          # v2 轨道目录（checkpoint 在这里）
    output: Path          # v3 产物目录
    h5: Path              # v2 HDF5 分支目录（含全部被试）

    @property
    def checkpoint_dir(self) -> Path:
        return self.source / "checkpoints"

    @property
    def trained_checkpoint_dir(self) -> Path:
        return self.output / "checkpoints"

    @property
    def active_checkpoint_dir(self) -> Path:
        """Prefer a complete v3-trained track, else the v2 source track."""
        trained = self.trained_checkpoint_dir
        required = (
            "stage_one.pt", "stage_one.complete.json",
            "stage_two_linear.pt", "stage_two_linear.complete.json",
        )
        if all((trained / name).is_file() for name in required):
            return trained
        return self.checkpoint_dir

    @property
    def test_h5(self) -> Path:
        return self.h5 / f"{self.dataset_id}_{self.subject}_test_epoched.h5"


def resolve_paths(dataset: DatasetSpec, encoder: str, subject: str) -> RunPaths:
    if subject not in dataset.subjects:
        raise ValueError(
            f"{dataset.name} has no subject {subject!r}; "
            f"known ({len(dataset.subjects)}): {list(dataset.subjects)[:5]}...")
    source = video_artifact_dir(dataset.name, encoder, subject)
    output = V3_ROOT / "artifacts" / dataset.name / "eeg2video" / subject / encoder
    return RunPaths(dataset=dataset.name, encoder=encoder, subject=subject,
                    dataset_id=video_dataset_id(dataset.name),
                    source=source, output=output,
                    h5=video_h5_dir(dataset.name, encoder))


def _set_track_env(paths: RunPaths, output_root: Path,
                   *, checkpoint_dir: Path | None = None) -> None:
    """在导入模型之前固定视频框架的目录环境变量。

    ``eeg2x.frameworks.eeg2video.core.config`` 在 import 时快照这些变量；
    显式的 ``checkpoint_dir`` 与 ``output_dir`` 仍然是权威，这里只是让共享
    缓存目录不串到别的被试。
    """
    number = str(paths.subject).split("-")[-1].lstrip("0") or "1"
    os.environ["EEG2VIDEO_SUBJECT"] = number
    os.environ["EEG2VIDEO_RUN_ROOT"] = str((output_root / "work" / paths.subject).resolve())
    os.environ["EEG2VIDEO_CHECKPOINT_ROOT"] = str(
        (checkpoint_dir or paths.active_checkpoint_dir).resolve())
    os.environ["EEG2VIDEO_SHARED_RUN_ROOT"] = str((output_root / "work" / "shared").resolve())


def load_test_samples(paths: RunPaths, limit: int | None = None) -> List[Any]:
    """测试集的每一行，转成 ``eeg2x.tasks.video.VideoSample``。

    ``limit`` 只用于冒烟测试：它让生成端只跑前若干条，评测仍会对全量真值
    计分并报告匹配数，因此冒烟结果不能当成正式指标。
    """
    from eeg2x.tasks.video import VideoSample

    h5_path = paths.test_h5
    if not h5_path.is_file():
        raise FileNotFoundError(f"missing video test HDF5 {h5_path}")
    with h5py.File(h5_path, "r") as h5:
        n = int(h5.attrs["n_epochs"])
        if limit is not None:
            n = min(n, int(limit))
        return [VideoSample.from_h5_row(h5, row, h5_path=h5_path)
                for row in range(n)]


def _scoped_h5_dir(paths: RunPaths, output_root: Path) -> Path:
    """只含本被试测试 H5 的目录。

    ``evaluate_predictions`` 接收目录而不是文件；若直接传编码器分支目录，
    会把同数据集其他被试的真值也算进来。软链接保持解析后的路径不变，因此
    ``h5://`` 引用仍能对上。
    """
    scope = output_root / "eval_h5"
    scope.mkdir(parents=True, exist_ok=True)
    link = scope / paths.test_h5.name
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(paths.test_h5.resolve())
    return scope


def _eeg_contract(h5_path: Path) -> Dict[str, Any]:
    """从测试 H5 读出这次运行实际使用的 EEG 几何。"""
    with h5py.File(h5_path, "r") as h5:
        data = h5["data"] if "data" in h5 else h5["eeg"]
        return {
            "shape": [int(data.shape[1]), int(data.shape[2])],
            "sampling_rate_hz": float(h5.attrs["sampling_rate"]),
            "n_channels": int(h5.attrs["n_channels"]),
            "channels": [c.decode() if isinstance(c, bytes) else str(c)
                         for c in h5["channel_names"][:]],
            "n_epochs": int(h5.attrs["n_epochs"]),
            "data_ref": f"h5://{h5_path.name}#data[row]",
        }


def run_inference(paths: RunPaths, *, device: str = "cuda:0",
                  generator_dtype: str = "float16",
                  noise: str = "none", noise_seed: int = SEED,
                  overwrite: bool = False,
                  stages: Sequence[str] = INFERENCE_STAGES,
                  export_videos: bool = False,
                  limit: int | None = None) -> Path:
    """跑 generate -> evaluate，返回评测 JSON 路径。

    ``noise="gaussian"`` 时产物落到 ``<干净输出>/noise_control/``，权重仍从
    视频仓库的干净轨道读，绝不重训。
    """
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"unsupported video stage(s) {unknown}; known: {list(STAGES)}")
    checkpoint_dir = paths.active_checkpoint_dir
    for required in (checkpoint_dir / "stage_one.pt",
                     checkpoint_dir / "stage_two_linear.pt"):
        if not required.is_file():
            raise FileNotFoundError(
                f"missing video checkpoint {required}; run the training stage "
                "or place a complete v2 track under EEG2X_V2_VIDEO_ROOT")

    output_root = require_v3_output(paths.output)
    if noise != "none":
        # 统一的噪声开关：这里只接受 none / gaussian，具体抽样在框架内部完成。
        output_root = require_v3_output(output_root / "noise_control")
    output_root.mkdir(parents=True, exist_ok=True)
    video_noise.configure(noise, noise_seed)
    _set_track_env(paths, output_root, checkpoint_dir=checkpoint_dir)

    generation_dir = output_root / "generation"
    predictions_path = generation_dir / "predictions.json"
    frames_path = generation_dir / "generated_frames.uint8.npy"

    if "generate" in stages:
        if not (predictions_path.is_file() and frames_path.is_file() and not overwrite):
            samples = load_test_samples(paths, limit=limit)
            from .model import EEG2VideoModel

            model = EEG2VideoModel({
                "h5_dir": str(paths.h5),
                "subject_id": paths.subject,
                "encoder": paths.encoder,
                "dataset_id": paths.dataset_id,
                "checkpoint_dir": str(checkpoint_dir),
                "output_dir": str(generation_dir),
                "generator_dtype": generator_dtype,
                "export_videos": export_videos,
            })
            generated = model.generate(samples)
            predictions = {item["sample_id"]: {"path": item["path"]}
                           for item in generated}
            if len(predictions) != len(samples):
                raise RuntimeError(
                    f"model returned {len(predictions)} predictions for "
                    f"{len(samples)} samples")
            write_json(predictions_path, {"predictions": predictions})
            if video_noise.enabled():
                video_noise.write_meta(predictions_path, encoder=paths.encoder,
                                       subject=paths.subject,
                                       dataset_id=paths.dataset_id,
                                       generator_dtype=generator_dtype)
        else:
            log.info("reusing generated clips: %s", predictions_path)

    if "evaluate" not in stages:
        return predictions_path

    if not predictions_path.is_file():
        raise FileNotFoundError(
            f"no predictions to evaluate at {predictions_path}; run the "
            "'generate' stage first")

    evaluation = evaluate_predictions(
        task="video_generation",
        h5_dir=_scoped_h5_dir(paths, output_root),
        predictions_path=predictions_path,
        output_path=output_root / "evaluation" / "generation_video.json",
        metrics=None)

    contract = _eeg_contract(paths.test_h5)
    write_json(output_root / "run_config.json", {
        "experiment_id": output_root.name,
        "dataset": paths.dataset, "subject_id": paths.subject,
        "task": "video_generation", "model": "eeg2video",
        "encoder": paths.encoder,
        "pipeline": ("EEG -> encoder over windows -> mean over windows -> "
                     "projection to CLIP text context [77,768] -> "
                     "AnimateDiff"),
        "generator_dtype": generator_dtype,
        "metrics": ["video_2way_accuracy", "video_40way_accuracy",
                    "frame_2way_accuracy", "frame_40way_accuracy",
                    "video_ssim", "video_psnr", "video_hue_pcc"],
        "noise": noise, "noise_seed": int(noise_seed),
        "eeg_contract": contract,
        "seed": SEED,
        "checkpoint_dir": str(checkpoint_dir),
        "source_h5": [str(paths.test_h5)],
        "derived_inputs_used": False,
    })
    return evaluation


def run_training(paths: RunPaths, *, device: str = "cuda:0",
                 stage_one_epochs: int = 200, stage_two_epochs: int = 150,
                 batch_size: int = 32) -> Path:
    """Train both Scheme D stages and return ``training_result.json``.

    The actual training runs in a fresh ``train_entry`` subprocess because the
    video configuration snapshots its checkpoint/run roots at import time.
    The parent process therefore never imports the video model, so consecutive
    subjects or encoders cannot share one checkpoint root.
    """
    output_root = require_v3_output(paths.output)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_root / "checkpoints"
    work_dir = output_root / "work"
    result_json = output_root / "training_result.json"

    command = [
        sys.executable, "-m", "eeg2x.frameworks.eeg2video.train_entry",
        "--encoder", paths.encoder,
        "--subject", paths.subject,
        "--dataset-id", paths.dataset_id,
        "--h5-dir", str(paths.h5),
        "--checkpoint-dir", str(checkpoint_dir),
        "--work-dir", str(work_dir),
        "--device", device,
        "--stage-one-epochs", str(int(stage_one_epochs)),
        "--stage-two-epochs", str(int(stage_two_epochs)),
        "--batch-size", str(int(batch_size)),
        "--result-json", str(result_json),
    ]
    log.info("training %s/%s/%s -> %s", paths.dataset, paths.encoder,
             paths.subject, checkpoint_dir)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "2000"
    completed = subprocess.run(command, cwd=str(V3_ROOT), env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"video training failed with exit code {completed.returncode}: "
            + " ".join(command))
    if not result_json.is_file():
        raise FileNotFoundError(
            f"training subprocess exited successfully but wrote no {result_json}")

    write_json(output_root / "train_config.json", {
        "experiment_id": output_root.name,
        "dataset": paths.dataset, "subject_id": paths.subject,
        "task": "video_generation", "model": "eeg2video",
        "encoder": paths.encoder,
        "h5_dir": str(paths.h5),
        "checkpoint_dir": str(checkpoint_dir),
        "work_dir": str(work_dir),
        "stage_one_epochs": int(stage_one_epochs),
        "stage_two_epochs": int(stage_two_epochs),
        "batch_size": int(batch_size),
        "device": device, "seed": SEED,
        "derived_inputs_used": False,
    })
    return result_json


__all__ = [
    "INFERENCE_STAGES", "RunPaths", "STAGES", "load_test_samples",
    "resolve_paths", "run_inference", "run_training",
]
