"""Only-CLIP 框架：唯一的链路实现。

阶段表（``STAGES``）就是 v2 那 7 个 runner 里反复复制的那串调用，只是**写一遍**：

    preprocess -> [train] -> embed -> project -> generate -> evaluate

``train`` 由 ``train.py`` 实现：先训练编码器 + ATM 头，再在训练集嵌入上训练
替换扩散先验的 Linear 投影头。不请求 ``train`` 时沿用已有检查点（优先 v3 自己
训练的，其次回退到 v2 产物）。换编码器只改 ``encoder`` 一个参数；框架代码不动。
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from eeg2x import encoders as encoder_registry
from eeg2x import noise_control
from eeg2x.data import discover_subject_h5, write_json
from eeg2x.datasets import DatasetSpec
from eeg2x.evaluation.api import evaluate_predictions
from eeg2x.paths import (
    V2_ROOT, V3_ROOT, artifact_dir, h5_dir, require_v3_output,
)
from eeg2x.tasks.image import TASK_IMAGE_GENERATION, eeg_spec_from_h5_files

from .checkpoint import load_only_clip_checkpoint
from .embed import embed_split
from .inference import generate_projected_embeddings, generate_stage

log = logging.getLogger("eeg2x.only_clip.pipeline")

#: 框架的完整阶段表。``train`` 只有显式请求时才跑；不请求就复用已有检查点。
STAGES = ("preprocess", "train", "embed", "project", "generate", "evaluate")

#: 只推理时跑的阶段。
INFERENCE_STAGES = ("embed", "project", "generate", "evaluate")

PROJECTION_TYPE = "atm_linear_projection"
GENERATION_TYPE = "atm_generation"

SEED = 2000


@dataclass
class RunPaths:
    """一次运行涉及的输入（v2）与输出（v3）路径。"""

    dataset: str
    encoder: str
    subject: str
    source: Path          # v2 产物目录（训练好的 checkpoint 在这里）
    output: Path          # v3 产物目录（embedding / 生成 / 评测写这里）
    h5: Path              # HDF5 目录（v2）
    feature_store: Path
    profile_id: str

    @property
    def generation_checkpoint(self) -> Path:
        """优先使用 v3 自己训练出的权重，没有再回退到 v2 的产物。

        ``verify`` 要对照 v2 的参考结果，所以它显式使用下面的 ``source_*``
        版本，避免被本地的训练产物干扰。
        """
        trained = self.output / "atm_generation" / "model" / "final.pt"
        if trained.is_file():
            return trained
        return self.source / "atm_generation" / "model" / "final.pt"

    @property
    def projection_checkpoint(self) -> Path:
        trained = self.output / "projection" / "model" / "final.pt"
        if trained.is_file():
            return trained
        return self.source / "projection" / "model" / "final.pt"

    @property
    def source_generation_checkpoint(self) -> Path:
        return self.source / "atm_generation" / "model" / "final.pt"

    @property
    def source_projection_checkpoint(self) -> Path:
        return self.source / "projection" / "model" / "final.pt"


def resolve_paths(dataset: DatasetSpec, encoder: str, subject: str) -> RunPaths:
    """把 (数据集, 编码器, 被试) 解析成输入/输出路径。"""
    if subject not in dataset.subjects:
        raise ValueError(
            f"{dataset.name} has no subject {subject!r}; "
            f"known ({len(dataset.subjects)}): {list(dataset.subjects)}")
    profile = dataset.profile_for(encoder)
    source = artifact_dir(dataset.name, "only_clip", subject, encoder=encoder)
    output = artifact_dir(dataset.name, "only_clip", subject,
                          encoder=encoder, layout="v3")
    store = _feature_store(dataset, source)
    # 输入契约与别的编码器相同的（如 ENIGMA 之于 ATM）复用后者的 HDF5。
    h5_owner = encoder_registry.resolve(encoder).h5_encoder or encoder
    return RunPaths(dataset=dataset.name, encoder=encoder, subject=subject,
                    source=source, output=output,
                    h5=h5_dir(dataset.name, h5_owner, subject),
                    feature_store=store, profile_id=profile)


def _feature_store(dataset: DatasetSpec, source: Path) -> Path:
    """定位刺激侧的 CLIP 特征库。

    ThingsEEG 的框架运行共用一份（``only_clip_linear_seed2000/shared_features``）；
    其余数据集把它放在各自的产物目录里。
    """
    shared = (V2_ROOT / "artifacts" / "only_clip_linear_seed2000"
              / "shared_features" / "clip_vit_h_seed2000.h5")
    if dataset.name == "thingseeg" and shared.is_file():
        return shared
    local = source / "features" / "clip_vit_h_seed2000.h5"
    if local.is_file():
        return local
    raise FileNotFoundError(
        f"no CLIP feature store for {dataset.name}/{source.name}; looked at "
        f"{shared} and {local}")


def run_inference(paths: RunPaths, *, device: str = "cuda:0",
                  overwrite: bool = False,
                  stages: Sequence[str] = INFERENCE_STAGES,
                  noise: str = "none", noise_seed: int = SEED,
                  distinct_images: int | None = None,
                  epochs: int = 40, batch_size: int = 64,
                  projection_epochs: int = 150,
                  use_source_checkpoints: bool = False) -> Path:
    """跑 train（可选）-> embed -> project -> generate -> evaluate。

    ``noise="gaussian"`` 是噪声对照：所有产物落到 ``<干净输出>/noise_control/``，
    干净结果不可能被覆盖；噪声条件**禁止训练**。

    ``use_source_checkpoints=True`` 强制读 v2 的权重，供 ``verify`` 使用——
    否则本地训练出的 v3 权重会覆盖掉待验证的参照。
    """
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(
            f"unsupported stage(s) {unknown}; known: {list(STAGES)}")
    if "preprocess" in stages:
        raise ValueError(
            "preprocess is a separate subcommand: python benchmark.py preprocess ...")
    if noise_control.noise_enabled(noise) and "train" in stages:
        raise ValueError(
            "the noise control never trains: weights must be frozen and come "
            "from a clean run")

    # 噪声条件的输出根：约定 1——绝不覆盖干净结果。
    output_root = noise_control.resolve_noise_dir(paths.output, noise)
    if noise_control.noise_enabled(noise):
        output_root = require_v3_output(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

    if "train" in stages:
        from .train import train_framework
        projection_checkpoint = train_framework(
            paths, epochs=epochs, batch_size=batch_size,
            projection_epochs=projection_epochs, device=device, overwrite=overwrite)
        if not any(s in stages for s in ("embed", "project", "generate", "evaluate")):
            return projection_checkpoint

    generation_checkpoint = (paths.source_generation_checkpoint
                             if use_source_checkpoints else paths.generation_checkpoint)
    projection_checkpoint = (paths.source_projection_checkpoint
                             if use_source_checkpoints else paths.projection_checkpoint)
    if any(s in stages for s in ("embed", "project", "generate", "evaluate")):
        if not generation_checkpoint.is_file():
            raise FileNotFoundError(f"missing framework checkpoint: {generation_checkpoint}")
    if any(s in stages for s in ("project", "generate", "evaluate")):
        if not projection_checkpoint.is_file():
            raise FileNotFoundError(f"missing projection checkpoint: {projection_checkpoint}")

    model, _ = load_only_clip_checkpoint(generation_checkpoint,
                                         subject_id=paths.subject)
    embeddings: Path | None = None
    predictions: Path | None = None
    images: Path | None = None

    if "embed" in stages:
        embeddings = embed_split(
            model, subject_id=paths.subject, h5_dir=paths.h5,
            feature_store=paths.feature_store, checkpoint=generation_checkpoint,
            output_dir=output_root / "embeddings", split="test",
            profile_id=paths.profile_id, device=device, overwrite=overwrite,
            noise=noise, noise_seed=noise_seed)
    if "project" in stages:
        if embeddings is None:
            embeddings = output_root / "embeddings" / "test_embeddings.h5"
        predictions = generate_projected_embeddings(
            test_embeddings=embeddings,
            projection_checkpoint=projection_checkpoint,
            output_h5=output_root / "projection" / "test_clip_predictions.h5",
            device=device, overwrite=overwrite)
    if "generate" in stages:
        if predictions is None:
            predictions = output_root / "projection" / "test_clip_predictions.h5"
        images = generate_stage(
            stage="clip", subject_id=paths.subject, h5_dir=paths.h5,
            output_dir=output_root / "generation" / "clip",
            image_embedding_predictions=predictions,
            num_inference_steps=4, seed=SEED, device=device,
            overwrite=overwrite, profile_id=paths.profile_id,
            distinct_images=distinct_images)
    if "evaluate" in stages:
        if images is None:
            images = output_root / "generation" / "clip" / "predictions.json"
        evaluation = evaluate_predictions(
            task="image_generation", h5_dir=paths.h5, predictions_path=images,
            output_path=output_root / "evaluation" / "generation_clip.json")
        _write_run_config(paths, output_root, stages=stages, noise=noise,
                          noise_seed=noise_seed,
                          distinct_images=distinct_images)
        if noise_control.noise_enabled(noise):
            _finalize_noise_condition(
                paths, output_root, noise, noise_seed, evaluation,
                distinct_images=distinct_images,
                generation_checkpoint=generation_checkpoint,
                projection_checkpoint=projection_checkpoint)
        return evaluation
    # 只跑了部分阶段（例如 ``--stages embed`` 用于冒烟测试）：返回最后一个产物。
    last = images or predictions or embeddings
    if last is None:
        raise ValueError("no stage was run")
    return last


def _write_run_config(paths: RunPaths, output_root: Path, *, stages: Sequence[str],
                      noise: str, noise_seed: int,
                      distinct_images: int | None = None) -> None:
    """把任务契约与运行条件写进 ``run_config.json``。

    v2 的框架产物没有任何 run_config；baseline 有。v3 统一补上，并让
    ``eeg2x.tasks`` 的契约成为记录的一部分——报告里的指标、方向、EEG 几何
    都能从这一个文件复现。
    """
    test_h5 = discover_subject_h5(paths.h5, paths.subject,
                                  profile_id=paths.profile_id, split="test")
    spec = TASK_IMAGE_GENERATION
    eeg = eeg_spec_from_h5_files([str(path) for path in test_h5])
    write_json(output_root / "run_config.json", {
        "experiment_id": output_root.name,
        "dataset": paths.dataset, "subject_id": paths.subject,
        "task": "image_generation", "task_id": spec.task_id,
        "output_type": spec.output_type,
        "model": "only_clip", "encoder": paths.encoder,
        "preprocessing_profile": paths.profile_id,
        "pipeline": ("EEG -> encoder -> ATM head (1024-d CLIP) -> "
                     "Linear(1024,1024) replacing the Diffusion Prior -> "
                     "SDXL-Turbo + IP-Adapter (4 steps)"),
        "stages": list(stages),
        "metrics": [spec.primary_metric, *spec.secondary_metrics],
        "lower_is_better": list(spec.lower_is_better),
        "eeg_contract": eeg.to_dict(),
        "noise": noise, "noise_seed": int(noise_seed),
        "distinct_images": distinct_images,
        "seed": SEED,
        "pythonhashseed": int(os.environ.get("PYTHONHASHSEED", SEED)),
        "source_h5": [str(path) for path in test_h5],
        "derived_inputs_used": False,
    })


def _finalize_noise_condition(paths: RunPaths, output_root: Path, noise: str,
                              noise_seed: int, evaluation: Path,
                              distinct_images: int | None = None,
                              generation_checkpoint: Path | None = None,
                              projection_checkpoint: Path | None = None) -> None:
    """噪声条件收尾：注入生效断言 + 旁挂 meta（约定 2/3）。"""
    clean_images = paths.output / "generation" / "clip" / "images"
    noise_images = output_root / "generation" / "clip" / "images"
    if clean_images.is_dir():
        noise_control.validate_noise_outputs(clean_images, noise_images)
        log.info("noise injection verified: none of the %d images is byte-identical "
                 "to the clean condition", len(list(noise_images.glob("sample-*.png"))))
    else:
        log.warning("no clean image dir to compare against (%s); skipping the "
                    "identical-output assertion", clean_images)
    # 条件元数据写旁挂 meta：predictions.json 的顶层 key 是样本 id。
    noise_control.write_noise_meta(
        output_root / "noise_meta.json",
        noise_control.noise_meta(
            noise, noise_seed, model=f"only_clip/{paths.encoder}",
            dataset=paths.dataset, subject=paths.subject,
            profile_id=paths.profile_id, h5_dir=str(paths.h5),
            atm_checkpoint=str(generation_checkpoint or paths.generation_checkpoint),
            projection_checkpoint=str(projection_checkpoint
                                      or paths.projection_checkpoint),
            clean_root=str(paths.output), evaluation=str(evaluation),
            distinct_images=distinct_images))


__all__ = [
    "INFERENCE_STAGES", "PROJECTION_TYPE", "GENERATION_TYPE", "RunPaths",
    "STAGES", "resolve_paths", "run_inference",
]
