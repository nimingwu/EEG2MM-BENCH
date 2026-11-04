"""train 阶段：编码器 + ATM 头，以及替换扩散先验的 Linear 投影头。

v2 把这两段训练写在 ``model/atm/training.py`` 的 ``train_atm`` 与
``train_projection`` 里，并由 runner 串起来。v3 把编排收进框架层，**编码器仍只
从注册表构造**——换编码器不需要改这里。

训练器只认 :class:`~eeg2x.encoders.base.EncoderSpec`：

* ``spec.optimizer is None`` 的编码器用框架统一配方（AdamW，平坦 lr=3e-4，
  无 warmup、无调度），与 v2 一致；
* 声明了 ``OptimizerSpec`` 的编码器（目前只有 LUNA）走它自己的配方，
  ``layer_decay`` 非空时启用 BEiT 式逐层学习率衰减。训练循环仍然只有一条代码
  路径，差异是数据而不是 ``if encoder == "luna"``。

**写侧守卫**：所有训练产物都先过 :func:`eeg2x.paths.require_v3_output`，因此
产物不可能落到 ``EEG2X-Benchmark-v3`` 之外。读 v2 的 HDF5 与特征库仍走
``require_benchmark_output``。

checkpoint 沿用 v2 的 ``artifact_type`` 约定（``atm_generation`` /
``atm_linear_projection``），并额外写 ``state_layout="v3"``：v2 的 ``atm_s``
state 需要一次键名迁移才能灌进 v3 模型，而 v3 自己训出来的键已经是目标布局，
加载器据这个标记跳过迁移（见 ``checkpoint.py``）。
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from eeg2x import encoders as encoder_registry
from eeg2x.checkpoints import load_state_checkpoint
from eeg2x.data import (
    ATMH5Dataset, EmbeddingH5Dataset, artifact_manifest, discover_subject_h5,
    write_json,
)
from eeg2x.paths import require_v3_output
from eeg2x.protocol import sha256_file
from eeg2x.rawdata import require_benchmark_output

from .checkpoint import load_only_clip_checkpoint
from .embed import embed_split
from .model import OnlyClipModel, generation_alignment_loss, retrieval_alignment_loss
from .projection import EEGToImageProjection

if TYPE_CHECKING:                       # pragma: no cover
    from .pipeline import RunPaths

log = logging.getLogger("eeg2x.only_clip.train")

SEED = 2000
FRAMEWORK_LEARNING_RATE = 3e-4
DEFAULT_EPOCHS = 40
DEFAULT_BATCH_SIZE = 64
DEFAULT_PROJECTION_EPOCHS = 150
DEFAULT_PROJECTION_BATCH_SIZE = 1024
DEFAULT_PROJECTION_LR = 1e-3

#: LUNA 主干里属于「输入侧」的参数名（BEiT 层号 0）。逐层衰减只对声明了
#: ``OptimizerSpec.layer_decay`` 的编码器生效；这套名字来自 v2 的实测层表。
_INPUT_SIDE = {"patch_embed", "freq_embed", "channel_location_embedder",
               "cross_attn", "mask_token"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but CUDA is unavailable")
    return device


def h5_encoder_contract(paths: Sequence[str | Path]) -> Dict[str, object]:
    """从 HDF5 读出编码器输入契约，并校验所有文件一致。

    逐字迁自 v2 ``model/atm/training.py:_h5_encoder_contract``。
    """
    channel_names: List[str] | None = None
    n_samples: int | None = None
    sampling_rate: float | None = None
    for path in paths:
        import h5py

        with h5py.File(path, "r") as h5:
            names = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in h5["channel_names"][:]
            ]
            current_samples = int(h5["data"].shape[-1])
            current_rate = float(h5.attrs["sampling_rate"])
        if channel_names is None:
            channel_names = names
            n_samples = current_samples
            sampling_rate = current_rate
        elif (names != channel_names or current_samples != n_samples or
              current_rate != sampling_rate):
            raise ValueError("ATM HDF5 files disagree on encoder input contract")
    if channel_names is None or n_samples is None or sampling_rate is None:
        raise ValueError("ATM encoder contract needs at least one HDF5 file")
    return {
        "channel_names": channel_names,
        "n_chans": len(channel_names),
        "n_samples": n_samples,
        "sampling_rate": sampling_rate,
    }


def build_model(encoder_name: str, contract: Dict[str, object],
                subject_id: str) -> tuple[OnlyClipModel, Dict[str, object]]:
    """按注册表构造 ``OnlyClipModel``，返回模型与 checkpoint 用的 model_config。"""
    spec = encoder_registry.resolve(encoder_name)
    spec.require_packages()
    encoder_config = dict(contract)
    encoder_config.setdefault("subject_id", subject_id)
    encoder = encoder_registry.build(
        spec.name, n_chans=int(encoder_config["n_chans"]),
        n_samples=int(encoder_config["n_samples"]), config=encoder_config)
    model = OnlyClipModel(encoder)
    return model, {"encoder": spec.name, "encoder_config": encoder_config}


def _framework_loss(model: OnlyClipModel, predicted: Tensor, target: Tensor,
                    mode: str) -> Tensor:
    if mode == "retrieval":
        return retrieval_alignment_loss(
            predicted, F.normalize(target, dim=-1), model.logit_scale,
            model.loss_func)
    if mode == "generation":
        return generation_alignment_loss(
            predicted, target, model.logit_scale, model.loss_func)
    raise ValueError(f"Unsupported training mode: {mode!r}")


@torch.inference_mode()
def evaluate_model(model: OnlyClipModel, dataset: ATMH5Dataset, *,
                   device: torch.device, batch_size: int = 64,
                   mode: str = "generation") -> Dict[str, float]:
    """测试集上的损失与 top-1/top-5（与 v2 ``evaluate_atm`` 同口径）。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predicted_batches: List[Tensor] = []
    target_batches: List[Tensor] = []
    losses: List[float] = []
    for batch in loader:
        predicted = model(batch["eeg"].to(device)).float()
        target = batch["target"].to(device).float()
        losses.append(float(_framework_loss(model, predicted, target, mode).item()))
        predicted_batches.append(predicted.cpu())
        target_batches.append(target.cpu())
    predicted_all = torch.cat(predicted_batches)
    target_all = torch.cat(target_batches)
    if mode == "retrieval":
        target_all = F.normalize(target_all, dim=-1)
    logits = model.logit_scale.detach().cpu() * predicted_all @ target_all.T
    labels = torch.arange(len(dataset))
    top1 = (logits.argmax(dim=1) == labels).float().mean()
    top5 = (logits.topk(min(5, logits.shape[1]), dim=1).indices ==
            labels[:, None]).any(dim=1).float().mean()
    return {"loss": float(np.mean(losses)),
            "top1": float(top1.item()), "top5": float(top5.item())}


def _backbone_blocks(model: nn.Module) -> int:
    """主干 block 数，从参数名实际数出来，而不是假设固定深度。"""
    highest = -1
    for name, _ in model.named_parameters():
        parts = name.split(".")
        if "blocks" in parts:
            index = parts.index("blocks")
            if index + 1 < len(parts) and parts[index + 1].isdigit():
                highest = max(highest, int(parts[index + 1]))
    if highest < 0:
        raise RuntimeError(
            "layer-wise LR decay needs numbered backbone blocks, but no "
            "'blocks.<N>' parameter names were found")
    return highest + 1


def _layer_id(name: str, *, backbone_blocks: int) -> int:
    """参数在逐层学习率衰减里的层号（BEiT 约定：越大学习率越高）。"""
    parts = name.split(".")
    if "blocks" in parts:
        index = parts.index("blocks")
        if index + 1 < len(parts) and parts[index + 1].isdigit():
            return int(parts[index + 1]) + 1
    if "backbone" in parts:
        index = parts.index("backbone")
        if index + 1 < len(parts) and parts[index + 1] in _INPUT_SIDE:
            return 0
    return backbone_blocks + 1


def build_optimizer(model: nn.Module, spec,
                    *, steps_per_epoch: int, epochs: int):
    """按 ``EncoderSpec.optimizer`` 构造优化器与调度器。

    ``spec is None`` 时用框架统一配方。否则按声明执行：weight decay 对
    ``ndim <= 1`` 的参数（bias、LayerNorm 权重）关闭；每层学习率为
    ``peak * layer_decay ** (layer_max - layer_id)``，再由 warmup/cosine 的
    ``LambdaLR`` 统一缩放。
    """
    if spec is None:
        return torch.optim.AdamW(model.parameters(), lr=FRAMEWORK_LEARNING_RATE), None

    peak = float(spec.peak_lr)
    floor = float(spec.min_lr if spec.min_lr is not None else 0.0)
    weight_decay = float(spec.weight_decay)
    layer_decay = float(spec.layer_decay if spec.layer_decay is not None else 1.0)
    if not 0.0 <= floor < peak:
        raise ValueError(f"min_lr must be in [0, {peak}), got {floor}")

    backbone_blocks = _backbone_blocks(model)
    layer_max = backbone_blocks + 1
    buckets: Dict[tuple, Dict[str, object]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer_id = _layer_id(name, backbone_blocks=backbone_blocks)
        if not 0 <= layer_id <= layer_max:
            raise RuntimeError(f"layer id {layer_id} out of range for {name!r}")
        decayed = parameter.ndim > 1
        bucket = buckets.setdefault((layer_id, decayed), {
            "lr": peak * layer_decay ** (layer_max - layer_id),
            "weight_decay": weight_decay if decayed else 0.0,
            "params": [],
        })
        bucket["params"].append(parameter)          # type: ignore[union-attr]
    if not buckets:
        raise RuntimeError("optimizer received no trainable parameter")
    if len({layer for layer, _ in buckets}) < 2 and any(
            "blocks." in name for name, parameter in model.named_parameters()
            if parameter.requires_grad):
        raise RuntimeError(
            "layer-wise LR decay collapsed to a single layer even though the "
            "backbone is trainable; the layer map is not matching parameter names")
    groups = [buckets[key] for key in sorted(buckets)]
    optimizer = torch.optim.AdamW(groups, lr=peak)

    warmup_steps = int(spec.warmup_epochs) * int(steps_per_epoch)
    total_steps = int(epochs) * int(steps_per_epoch)
    if total_steps <= 0:
        raise ValueError("the schedule needs at least one training step")
    floor_ratio = floor / peak

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return floor_ratio + (1.0 - floor_ratio) * cosine

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def _atomic_torch_save(value: object, path: str | Path) -> Path:
    output = require_v3_output(Path(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)
    return output


def _write_manifest(checkpoint: Path, *, artifact_type: str,
                    source_h5: Sequence[Dict[str, object]],
                    config: Dict[str, object],
                    extra: Dict[str, object] | None = None) -> Path:
    value = artifact_manifest(
        artifact_type=artifact_type, source_h5=source_h5, config=config,
        extra={"checkpoint": str(checkpoint),
               "checkpoint_sha256": sha256_file(checkpoint), **(extra or {})})
    return write_json(checkpoint.with_suffix(checkpoint.suffix + ".manifest.json"),
                      value)


def train_encoder(paths: "RunPaths", *, epochs: int = DEFAULT_EPOCHS,
                  batch_size: int = DEFAULT_BATCH_SIZE, device: str = "cuda:0",
                  seed: int = SEED, overwrite: bool = False,
                  cache_eeg: bool = True,
                  ) -> tuple[OnlyClipModel, Path, Dict[str, object]]:
    """训练编码器 + ATM 头，返回 ``(model, checkpoint, model_config)``。

    已有 ``atm_generation/model/final.pt`` 且 ``overwrite=False`` 时直接加载，
    这是 v2 runner 的「有 checkpoint 就跳过」约定。
    """
    output = require_v3_output(Path(paths.output))
    checkpoint = output / "atm_generation" / "model" / "final.pt"
    if checkpoint.is_file() and not overwrite:
        model, payload = load_only_clip_checkpoint(
            checkpoint, subject_id=paths.subject)
        log.info("reusing trained encoder checkpoint: %s", checkpoint)
        return model, checkpoint, dict(payload["model_config"])

    train_paths = discover_subject_h5(
        paths.h5, paths.subject, profile_id=paths.profile_id, split="train")
    test_paths = discover_subject_h5(
        paths.h5, paths.subject, profile_id=paths.profile_id, split="test")
    contract = h5_encoder_contract([*train_paths, *test_paths])
    seed_everything(seed)
    model, model_config = build_model(paths.encoder, contract, paths.subject)
    spec = encoder_registry.resolve(paths.encoder)
    target_device = _device(device)
    model = model.to(target_device)

    train_data = ATMH5Dataset(
        train_paths, paths.feature_store, profile_id=paths.profile_id,
        cache_eeg=cache_eeg, cache_features=True)
    test_data = ATMH5Dataset(
        test_paths, paths.feature_store, profile_id=paths.profile_id,
        cache_eeg=True, cache_features=True)
    try:
        loader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=0,
            drop_last=True, generator=torch.Generator().manual_seed(seed))
        if len(loader) == 0:
            raise ValueError(
                f"batch_size={batch_size} exceeds the {len(train_data)} training rows")
        # 例外按 (编码器 x 数据集) 判定：LUNA 的配方只在 ThingsEEG 上生效，
        # 其余数据集回落到框架统一配方（见 EncoderSpec.optimizer_datasets）。
        optimizer_spec = spec.optimizer_for(paths.dataset)
        optimizer, scheduler = build_optimizer(
            model, optimizer_spec, steps_per_epoch=len(loader), epochs=epochs)
        effective_lr = (float(optimizer_spec.peak_lr) if optimizer_spec is not None
                        else FRAMEWORK_LEARNING_RATE)
        if optimizer_spec is not None:
            log.info("%s on %s uses its own fine-tuning recipe (peak_lr=%g, "
                     "warmup=%s epochs, wd=%g, layer_decay=%s); the framework "
                     "default lr=%g does not apply. See "
                     "docs/LUNA_TRAINING_DEVIATION.md.",
                     spec.name, paths.dataset, optimizer_spec.peak_lr,
                     optimizer_spec.warmup_epochs, optimizer_spec.weight_decay,
                     optimizer_spec.layer_decay, FRAMEWORK_LEARNING_RATE)
        elif spec.optimizer is not None:
            log.info("%s declares its own optimizer but not for dataset %r; "
                     "using the framework default lr=%g.",
                     spec.name, paths.dataset, FRAMEWORK_LEARNING_RATE)
        config: Dict[str, object] = {
            "subject_id": paths.subject, "mode": "generation",
            "profile_id": paths.profile_id, "epochs": epochs,
            "batch_size": batch_size, "learning_rate": effective_lr,
            "seed": seed, "encoder": spec.name,
            "optimizer_recipe": ("encoder_spec" if optimizer_spec is not None
                                 else "framework_default"),
            "optimizer_override": (asdict(optimizer_spec)
                                   if optimizer_spec is not None else None),
        }
        sources = train_data.fingerprints + test_data.fingerprints
        history: List[Dict[str, float | int]] = []
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            n_batches = 0
            for batch in loader:
                eeg = batch["eeg"].to(target_device, non_blocking=True)
                target = batch["target"].to(target_device, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                predicted = model(eeg).float()
                loss = _framework_loss(model, predicted, target, "generation")
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                loss_sum += float(loss.item())
                n_batches += 1
            metrics = evaluate_model(model, test_data, device=target_device,
                                     batch_size=batch_size)
            row: Dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": loss_sum / max(n_batches, 1),
                "test_loss": metrics["loss"],
                "test_top1": metrics["top1"],
                "test_top5": metrics["top5"],
            }
            history.append(row)
            write_json(output / "atm_generation" / "training_history.json", history)
            log.info("%s epoch %d/%d train=%.5f test=%.5f top1=%.4f top5=%.4f",
                     spec.name, epoch, epochs, row["train_loss"],
                     row["test_loss"], row["test_top1"], row["test_top5"])
            if epoch % 5 == 0:
                _atomic_torch_save({
                    "artifact_type": "atm_generation",
                    "state_layout": "v3",
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": config,
                    "model_config": model_config,
                    "source_h5": sources,
                    "feature_store": str(require_benchmark_output(paths.feature_store)),
                    "derived_inputs_used": False,
                }, output / "atm_generation" / "checkpoints" / f"epoch-{epoch:03d}.pt")
        checkpoint = _atomic_torch_save({
            "artifact_type": "atm_generation",
            "state_layout": "v3",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epochs,
            "config": config,
            "model_config": model_config,
            "source_h5": sources,
            "feature_store": str(require_benchmark_output(paths.feature_store)),
            "metrics": history[-1],
            "derived_inputs_used": False,
        }, checkpoint)
        _write_manifest(
            checkpoint, artifact_type="atm_generation", source_h5=sources,
            config=config,
            extra={"feature_store": str(require_benchmark_output(paths.feature_store)),
                   "final_metrics": history[-1]})
        return model, checkpoint, model_config
    finally:
        train_data.close()
        test_data.close()


@torch.inference_mode()
def evaluate_projection(model: EEGToImageProjection, data: EmbeddingH5Dataset,
                        *, device: torch.device, batch_size: int) -> float:
    model.eval()
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0)
    squared_error = 0.0
    element_count = 0
    for batch in loader:
        condition = batch["condition"].to(device)
        target = batch["target"].to(device)
        predicted = model(condition)
        squared_error += float(F.mse_loss(predicted, target, reduction="sum").item())
        element_count += target.numel()
    if element_count == 0:
        raise ValueError("projection evaluation needs at least one embedding")
    return squared_error / element_count


def train_projection(*, train_embeddings: str | Path,
                     test_embeddings: str | Path,
                     output_dir: str | Path,
                     epochs: int = DEFAULT_PROJECTION_EPOCHS,
                     batch_size: int = DEFAULT_PROJECTION_BATCH_SIZE,
                     learning_rate: float = DEFAULT_PROJECTION_LR,
                     device: str = "cuda:0", seed: int = SEED,
                     overwrite: bool = False) -> Path:
    """训练替换扩散先验的单层 ``Linear(1024,1024)``。

    逐条沿用 v2 ``model/atm/training.py:train_projection``：Adam + cosine
    （500 步 warmup）、MSE、梯度裁剪 1.0、150 epoch。
    """
    from diffusers.optimization import get_cosine_schedule_with_warmup

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("Invalid projection training configuration")
    output = require_v3_output(Path(output_dir))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "final.pt"
    train_sha = sha256_file(train_embeddings)
    test_sha = sha256_file(test_embeddings)
    if checkpoint.is_file() and not overwrite:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if (payload.get("train_embedding_sha256") == train_sha and
                payload.get("test_embedding_sha256") == test_sha):
            log.info("reusing trained projection checkpoint: %s", checkpoint)
            return checkpoint
        log.info("projection checkpoint was trained on other embeddings; retraining")

    train_data = EmbeddingH5Dataset(train_embeddings, cache=True)
    test_data = EmbeddingH5Dataset(test_embeddings, cache=True)
    seed_everything(seed)
    target_device = _device(device)
    model = EEGToImageProjection().to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True, generator=torch.Generator().manual_seed(seed))
    if len(loader) == 0:
        raise ValueError(
            "projection batch size exceeds the number of training embeddings")
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500, num_training_steps=len(loader) * epochs)
    config = {
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate,
        "architecture": "Linear(1024,1024,bias=True)",
        "loss": "MSE", "optimizer": "Adam",
        "lr_scheduler": "cosine_with_500_warmup_steps", "seed": seed,
    }
    history: List[Dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            condition = batch["condition"].to(target_device)
            target = batch["target"].to(target_device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(condition)
            loss = F.mse_loss(predicted, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            loss_sum += float(loss.item())
        row: Dict[str, float | int] = {
            "epoch": epoch, "train_loss": loss_sum / max(len(loader), 1)}
        history.append(row)
        write_json(output / "training_history.json", history)
        log.info("Linear projection epoch %d/%d loss=%.6f",
                 epoch, epochs, row["train_loss"])
        if epoch % 25 == 0:
            _atomic_torch_save({
                "artifact_type": "atm_linear_projection",
                "model_state": model.state_dict(), "epoch": epoch,
                "config": config, "source_h5": train_data.source_h5,
                "train_embedding_artifact": str(train_data.path),
                "train_embedding_sha256": train_sha,
                "test_embedding_sha256": test_sha,
                "derived_inputs_used": False,
            }, output / "checkpoints" / f"epoch-{epoch:03d}.pt")
    test_mse = evaluate_projection(
        model, test_data, device=target_device, batch_size=batch_size)
    final_metrics = {**history[-1], "test_mse": test_mse}
    checkpoint = _atomic_torch_save({
        "artifact_type": "atm_linear_projection",
        "model_state": model.state_dict(), "epoch": epochs,
        "config": config, "source_h5": train_data.source_h5,
        "train_embedding_artifact": str(train_data.path),
        "test_embedding_artifact": str(test_data.path),
        "train_embedding_sha256": train_sha,
        "test_embedding_sha256": test_sha,
        "metrics": final_metrics, "derived_inputs_used": False,
    }, checkpoint)
    _write_manifest(
        checkpoint, artifact_type="atm_linear_projection",
        source_h5=train_data.source_h5, config=config,
        extra={"train_embedding_artifact": str(train_data.path),
               "test_embedding_artifact": str(test_data.path),
               "final_metrics": final_metrics})
    return checkpoint


def train_framework(paths: "RunPaths", *, epochs: int = DEFAULT_EPOCHS,
                    batch_size: int = DEFAULT_BATCH_SIZE,
                    projection_epochs: int = DEFAULT_PROJECTION_EPOCHS,
                    projection_batch_size: int = DEFAULT_PROJECTION_BATCH_SIZE,
                    device: str = "cuda:0", seed: int = SEED,
                    overwrite: bool = False, cache_eeg: bool = True) -> Path:
    """完整训练链路：编码器 + ATM 头，训练集嵌入，Linear 投影头。

    返回投影头 checkpoint；之后 ``project`` 阶段直接用它。
    """
    output = require_v3_output(Path(paths.output))
    (output / "atm_generation" / "model").mkdir(parents=True, exist_ok=True)
    (output / "projection" / "model").mkdir(parents=True, exist_ok=True)

    model, encoder_checkpoint, _ = train_encoder(
        paths, epochs=epochs, batch_size=batch_size, device=device, seed=seed,
        overwrite=overwrite, cache_eeg=cache_eeg)

    # 投影头要在训练集的嵌入上拟合。嵌入与 checkpoint 绑定：embed_split 会校验
    # 已有文件的 checkpoint 哈希，换了编码器权重就重算（见 embed.py）。
    train_embeddings = embed_split(
        model, subject_id=paths.subject, h5_dir=paths.h5,
        feature_store=paths.feature_store, checkpoint=encoder_checkpoint,
        output_dir=output / "embeddings", split="train",
        profile_id=paths.profile_id, device=device, overwrite=overwrite)
    test_embeddings = embed_split(
        model, subject_id=paths.subject, h5_dir=paths.h5,
        feature_store=paths.feature_store, checkpoint=encoder_checkpoint,
        output_dir=output / "embeddings", split="test",
        profile_id=paths.profile_id, device=device, overwrite=overwrite)

    return train_projection(
        train_embeddings=train_embeddings, test_embeddings=test_embeddings,
        output_dir=output / "projection" / "model",
        epochs=projection_epochs, batch_size=projection_batch_size,
        device=device, seed=seed, overwrite=overwrite)


__all__ = [
    "DEFAULT_BATCH_SIZE", "DEFAULT_EPOCHS", "DEFAULT_PROJECTION_EPOCHS",
    "FRAMEWORK_LEARNING_RATE", "build_model", "build_optimizer",
    "h5_encoder_contract", "train_encoder", "train_framework", "train_projection",
]
