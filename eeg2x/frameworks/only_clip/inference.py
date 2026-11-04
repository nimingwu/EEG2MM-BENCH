"""仅推理链路：投影 -> SDXL 生成。

两个函数逐字来自 v2 ``model/atm/inference.py``（``generate_projected_embeddings`` 与
``generate_stage``），数学未改。**只保留 Only-CLIP 需要的 ``clip`` 阶段**；
``low_level`` / ``semantic`` 属于完整 ATM baseline，在 v3 里由对应 baseline 负责，
这里遇到会明确报错而不是静默跑错路径。

v2 的 ``generate_stage`` 用 ``prior_predictions`` 作为 ``image_embedding_predictions``
的旧别名；v3 只接受后者，旧名字仍然可解析但会警告。
"""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
from PIL import Image
import torch

from eeg2x.checkpoints import load_state_checkpoint
from eeg2x.data import (
    ATM_PROFILE, artifact_manifest, discover_subject_h5, sample_id_for_h5_row,
    write_json,
)
from eeg2x.protocol import sha256_file
from eeg2x.rawdata import require_benchmark_output

from .projection import EEGToImageProjection
from .sdxl import Generator4Embeds as ClipGenerator


class _StageNotWired:
    """`only_clip` 框架只跑 `clip` 阶段。

    `low_level` 与 `semantic` 属于**完整 ATM baseline**（三分支），由
    `eeg2x.baselines.atm_full` 负责。这里让它们明确报错，而不是因缺 import 抛出
    难懂的 NameError，也不是静默走到错的分支。
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "the low-level / semantic stages belong to the atm_full baseline; "
            "v3's only_clip framework supports stage='clip' only")


LowLevelGenerator = LowLevelEEGEncoder = PixelProjector = GitForCausalLMClipEmb = _StageNotWired

log = logging.getLogger("eeg2x.only_clip.inference")

#: 只有 clip 阶段在本框架里实现；其余属于 atm_full baseline（见 _StageNotWired）。
STAGES = ("clip", "low_level", "semantic")
HF_CACHE = "/datas/hf-cache/hub"


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {value!r} requested but CUDA is unavailable")
    return device


def _clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_embedding_source(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    candidate = require_benchmark_output(path)
    with h5py.File(candidate, "r") as h5:
        if bool(h5.attrs.get("derived_inputs_used", True)):
            raise ValueError(f"Embedding file used forbidden external inputs: {candidate}")
        values = np.asarray(h5["eeg_embedding"][:], dtype=np.float32)
        targets = np.asarray(h5["clip_target"][:], dtype=np.float32)
        source_h5 = json.loads(h5.attrs["source_h5_json"])
    return values, targets, source_h5

def generate_projected_embeddings(
    *,
    test_embeddings: str | Path,
    projection_checkpoint: str | Path,
    output_h5: str | Path,
    batch_size: int = 64,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> Path:
    """Apply the frozen affine map to every test EEG embedding once."""
    output = require_benchmark_output(output_h5)
    if output.exists() and not overwrite:
        with h5py.File(output, "r") as h5:
            if (h5.attrs.get("projection_checkpoint_sha256", "") !=
                    sha256_file(projection_checkpoint)):
                raise ValueError(
                    f"Existing projection output uses another checkpoint: {output}")
            if not bool(h5.attrs.get("complete", False)):
                raise ValueError(
                    f"Existing projection output is incomplete: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    conditions, targets, source_h5 = _read_embedding_source(test_embeddings)
    target_device = _device(device)
    model = EEGToImageProjection()
    load_state_checkpoint(
        model, projection_checkpoint, expected_type="atm_linear_projection")
    model.to(target_device).eval()
    predictions: List[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(conditions), batch_size):
            stop = min(start + batch_size, len(conditions))
            values = model(
                torch.from_numpy(conditions[start:stop]).to(target_device))
            predictions.append(values.float().cpu().numpy())
            log.info("Linear projection inference %d/%d", stop, len(conditions))
    predicted = np.concatenate(predictions)
    if not np.isfinite(predicted).all():
        raise RuntimeError("Linear projection produced non-finite embeddings")
    test_mse = float(np.mean((predicted.astype(np.float64) - targets) ** 2))
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        with h5py.File(temporary, "w") as h5:
            h5.create_dataset(
                "clip_prediction", data=predicted.astype(np.float16),
                chunks=(min(batch_size, len(predicted)), 1024))
            h5.attrs["schema"] = "eeg2x-atm-linear-projection-predictions-v1"
            h5.attrs["complete"] = True
            h5.attrs["test_mse"] = test_mse
            h5.attrs["source_h5_json"] = json.dumps(source_h5, sort_keys=True)
            h5.attrs["test_embedding_artifact"] = str(
                require_benchmark_output(test_embeddings))
            h5.attrs["projection_checkpoint"] = str(
                require_benchmark_output(projection_checkpoint))
            h5.attrs["projection_checkpoint_sha256"] = sha256_file(
                projection_checkpoint)
            h5.attrs["derived_inputs_used"] = False
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    write_json(output.with_suffix(output.suffix + ".manifest.json"), artifact_manifest(
        artifact_type="atm_linear_projection_predictions", source_h5=source_h5,
        config={
            "batch_size": batch_size,
            "architecture": "Linear(1024,1024,bias=True)",
        },
        extra={
            "artifact": str(output), "artifact_sha256": sha256_file(output),
            "test_embeddings": str(require_benchmark_output(test_embeddings)),
            "projection_checkpoint": str(
                require_benchmark_output(projection_checkpoint)),
            "test_mse": test_mse,
        }))
    del model
    _clear_cuda()
    return output


def generate_prior_embeddings(*args, **kwargs):
    """Reject accidental use of the removed Diffusion Prior path."""
    raise RuntimeError(
        "Diffusion Prior is disabled in this copy; use "
        "generate_projected_embeddings")


def decode_low_level_conditions(
    *,
    subject_id: str,
    h5_dir: str | Path,
    low_level_checkpoint: str | Path,
    output_dir: str | Path,
    batch_size: int = 20,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> Path:
    """Predict unscaled latents and decode them before SDXL img2img."""
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor

    output = require_benchmark_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    test_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=ATM_PROFILE, split="test")
    source = test_paths[0]
    checkpoint_hash = sha256_file(low_level_checkpoint)
    if manifest_path.is_file() and not overwrite:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"Existing Low-Level conditions use another checkpoint: {output}")
        images = sorted(output.glob("condition-*.png"))
        if len(images) == int(manifest.get("n_images", -1)):
            return manifest_path
        raise ValueError(f"Existing Low-Level conditions are incomplete: {output}")

    target_device = _device(device)
    model = LowLevelEEGEncoder()
    load_state_checkpoint(model, low_level_checkpoint, expected_type="atm_low_level")
    model.to(target_device).eval()
    with h5py.File(source, "r") as h5:
        eeg = h5["data"]
        subject_number = int(subject_id.split("-")[-1])
        predicted_batches: List[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(eeg), batch_size):
                stop = min(start + batch_size, len(eeg))
                values = torch.from_numpy(np.asarray(eeg[start:stop], dtype=np.float32))
                predicted = model(values.to(target_device)).float().cpu().numpy()
                predicted_batches.append(predicted)
                log.info("low-level latent inference %d/%d", stop, len(eeg))
    predicted_latents = np.concatenate(predicted_batches)
    if not np.isfinite(predicted_latents).all():
        raise RuntimeError("Low-Level inference produced non-finite VAE latents")
    stored_latents = predicted_latents.astype(np.float16)
    if not np.isfinite(stored_latents).all():
        raise RuntimeError(
            "Low-Level VAE latents overflowed during float16 storage conversion")
    latent_path = output / "predicted_latents.h5"
    with h5py.File(latent_path, "w") as h5:
        h5.create_dataset(
            "latent", data=stored_latents,
            chunks=(1, 4, 64, 64))
        h5.attrs["latent_scaling"] = "unscaled"
        h5.attrs["source_h5"] = str(source)
        h5.attrs["low_level_checkpoint"] = str(
            require_benchmark_output(low_level_checkpoint))
        h5.attrs["low_level_checkpoint_sha256"] = checkpoint_hash
        h5.attrs["subject_token"] = subject_number
        h5.attrs["derived_inputs_used"] = False
    del model
    _clear_cuda()

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae",
        variant="fp16", cache_dir=HF_CACHE, local_files_only=True,
        torch_dtype=torch.float32).to(target_device).eval()
    processor = VaeImageProcessor(vae_scale_factor=8)
    with torch.inference_mode():
        for start in range(0, len(predicted_latents), batch_size):
            stop = min(start + batch_size, len(predicted_latents))
            latent = torch.from_numpy(predicted_latents[start:stop]).to(
                target_device, dtype=torch.float32)
            decoded = vae.decode(latent).sample
            if not torch.isfinite(decoded).all():
                raise RuntimeError(
                    f"SDXL VAE decode produced non-finite values at row {start}")
            images = processor.postprocess(decoded, output_type="pil")
            for row, image in enumerate(images, start=start):
                image.save(output / f"condition-{row:03d}.png")
            log.info("low-level VAE decode %d/%d", stop, len(predicted_latents))
    del vae, processor
    _clear_cuda()
    value = artifact_manifest(
        artifact_type="atm_low_level_conditions",
        source_h5=[{"path": str(source)}],
        config={
            "subject_id": subject_id, "latent_scaling": "unscaled",
            "decode_model": "stabilityai/stable-diffusion-xl-base-1.0:vae",
        },
        extra={
            "checkpoint": str(require_benchmark_output(low_level_checkpoint)),
            "checkpoint_sha256": checkpoint_hash,
            "latent_artifact": str(latent_path),
            "latent_artifact_sha256": sha256_file(latent_path),
            "n_images": len(predicted_latents),
        })
    return write_json(manifest_path, value)


def generate_captions(
    *,
    prior_predictions: str | Path,
    semantic_checkpoint: str | Path,
    output_json: str | Path,
    batch_size: int = 8,
    max_new_tokens: int = 25,
    device: str = "cuda:0",
    overwrite: bool = False,
) -> Path:
    """Prior CLIP -> PixelProjector -> frozen GIT caption, as released."""
    from transformers import AutoProcessor

    output = require_benchmark_output(output_json)
    if output.is_file() and not overwrite:
        value = json.loads(output.read_text(encoding="utf-8"))
        if value.get("semantic_checkpoint_sha256") != sha256_file(semantic_checkpoint):
            raise ValueError(f"Existing captions use another semantic checkpoint: {output}")
        return output
    with h5py.File(require_benchmark_output(prior_predictions), "r") as h5:
        clip_predictions = np.asarray(h5["clip_prediction"][:], dtype=np.float32)
        source_h5 = json.loads(h5.attrs["source_h5_json"])
    target_device = _device(device)
    dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    projector = PixelProjector()
    load_state_checkpoint(
        projector, semantic_checkpoint, expected_type="atm_semantic_projector")
    projector.to(device=target_device, dtype=dtype).eval()
    processor = AutoProcessor.from_pretrained(
        "microsoft/git-large-coco", cache_dir=HF_CACHE, local_files_only=True)
    captioner = GitForCausalLMClipEmb.from_pretrained(
        "microsoft/git-large-coco", cache_dir=HF_CACHE,
        local_files_only=True, torch_dtype=torch.float32)
    captioner.to(target_device).eval().requires_grad_(False)
    captions: List[str] = []
    with torch.inference_mode():
        for start in range(0, len(clip_predictions), batch_size):
            stop = min(start + batch_size, len(clip_predictions))
            values = torch.from_numpy(clip_predictions[start:stop]).unsqueeze(1).to(
                target_device, dtype=dtype)
            tokens = projector(values).float()
            generated = captioner.generate(
                pixel_values=tokens, max_new_tokens=max_new_tokens,
                # The released GIT implementation predates Transformers'
                # DynamicCache. Cache-free greedy decoding is equivalent in
                # eval mode and keeps the copied model compatible.
                use_cache=False)
            captions.extend(processor.batch_decode(
                generated, skip_special_tokens=True))
            log.info("semantic captions %d/%d", stop, len(clip_predictions))
    value = artifact_manifest(
        artifact_type="atm_semantic_captions", source_h5=source_h5,
        config={
            "caption_model": "microsoft/git-large-coco",
            "max_new_tokens": max_new_tokens, "batch_size": batch_size,
        },
        extra={
            "semantic_checkpoint": str(require_benchmark_output(semantic_checkpoint)),
            "semantic_checkpoint_sha256": sha256_file(semantic_checkpoint),
            "prior_predictions": str(require_benchmark_output(prior_predictions)),
            "captions": captions,
        })
    result = write_json(output, value)
    del projector, captioner, processor
    _clear_cuda()
    return result


def _validate_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(f"Unknown ATM generation stage {stage!r}; choose from {STAGES}")


def generate_stage(
    *,
    stage: str,
    subject_id: str,
    h5_dir: str | Path,
    output_dir: str | Path,
    image_embedding_predictions: str | Path | None = None,
    prior_predictions: str | Path | None = None,
    low_level_conditions: str | Path | None = None,
    captions_json: str | Path | None = None,
    num_inference_steps: int = 4,
    img2img_strength: float = 0.5,
    seed: int = 2000,
    device: str = "cuda:0",
    overwrite: bool = False,
    profile_id: str = ATM_PROFILE,
    distinct_images: int | None = None,
) -> Path:
    """Generate one benchmark stage and write sample-ID keyed predictions.

    ``distinct_images=N`` is the **noise-control time saver**: only the first N
    rows are actually generated, and row ``r >= N`` reuses the file produced by
    row ``r % N`` (copied, so every sample still has its own ``sample-rrr.png``
    and every downstream name/count check keeps working).  This is only sound
    for the noise condition, where the predictions are meaningless by
    construction and replicating them changes no per-row metric's expectation;
    it must never be used for a clean run, whose rows are stimulus-specific.
    Note it does make `fid` meaningless (N distinct images replicated).
    """
    _validate_stage(stage)
    if ((image_embedding_predictions is None) == (prior_predictions is None)):
        raise ValueError(
            "Provide exactly one image embedding prediction artifact")
    embedding_predictions = (
        image_embedding_predictions
        if image_embedding_predictions is not None else prior_predictions)
    if embedding_predictions is None:
        raise RuntimeError("Image embedding prediction path resolution failed")
    output = require_benchmark_output(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    test_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=profile_id, split="test")
    with h5py.File(require_benchmark_output(embedding_predictions), "r") as h5:
        clip_predictions = np.asarray(h5["clip_prediction"][:], dtype=np.float32)
        source_h5 = json.loads(h5.attrs["source_h5_json"])
    with h5py.File(test_paths[0], "r") as h5:
        n_test_rows = int(h5["data"].shape[0])
    if len(clip_predictions) != n_test_rows:
        raise ValueError(
            "Prior prediction count does not match the formal ATM test HDF5: "
            f"{len(clip_predictions)} != {n_test_rows}")
    low_paths: List[Path] = []
    captions = ["" for _ in range(len(clip_predictions))]
    if stage in ("low_level", "semantic"):
        if low_level_conditions is None:
            raise ValueError(f"Stage {stage} requires decoded Low-Level conditions")
        low_root = require_benchmark_output(low_level_conditions)
        low_paths = [low_root / f"condition-{row:03d}.png"
                     for row in range(len(clip_predictions))]
        missing = [str(path) for path in low_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing Low-Level condition images: {missing[:3]}")
    if stage == "semantic":
        if captions_json is None:
            raise ValueError("Semantic stage requires GIT captions")
        caption_value = json.loads(
            require_benchmark_output(captions_json).read_text(encoding="utf-8"))
        captions = caption_value["captions"]
        if len(captions) != len(clip_predictions):
            raise ValueError("Semantic caption count does not match ATM test rows")

    target_device = _device(device)
    if stage == "clip":
        generator = ClipGenerator(
            num_inference_steps=num_inference_steps, device=str(target_device),
            cache_dir=HF_CACHE, local_files_only=True)
    else:
        generator = LowLevelGenerator(
            num_inference_steps=num_inference_steps, device=str(target_device),
            img2img_strength=img2img_strength,
            cache_dir=HF_CACHE, local_files_only=True)
    if distinct_images is not None and distinct_images < 1:
        raise ValueError("distinct_images must be >= 1 when set")
    predictions: Dict[str, Dict[str, str]] = {}
    for row, embedding in enumerate(clip_predictions):
        destination = images_dir / f"sample-{row:03d}.png"
        if distinct_images is not None and row >= distinct_images:
            # 噪声对照省时档：复用第 (row % N) 张图，不再过模型。
            source = images_dir / f"sample-{row % distinct_images:03d}.png"
            if overwrite or not destination.is_file():
                shutil.copyfile(source, destination)
            sample_id = sample_id_for_h5_row(test_paths[0], row)
            predictions[sample_id] = {"path": str(destination)}
            continue
        if overwrite or not destination.is_file():
            torch_generator = torch.Generator(device=target_device).manual_seed(seed)
            if stage != "clip":
                with Image.open(low_paths[row]) as condition:
                    generator.low_level_image = condition.convert("RGB").copy()
            image = generator.generate(
                torch.from_numpy(embedding).unsqueeze(0),
                text_prompt=captions[row], generator=torch_generator)
            image.save(destination)
        sample_id = sample_id_for_h5_row(test_paths[0], row)
        predictions[sample_id] = {"path": str(destination)}
        if (row + 1) % 20 == 0:
            log.info("%s generation %d/%d", stage, row + 1, len(clip_predictions))
    del generator
    _clear_cuda()
    predictions_path = write_json(output / "predictions.json", {
        "predictions": predictions,
        "provenance": {
            "artifact_type": "atm_generation_predictions",
            "stage": stage, "subject_id": subject_id,
            "source_h5": source_h5,
            "image_embedding_predictions": str(
                require_benchmark_output(embedding_predictions)),
            "low_level_conditions": (
                str(require_benchmark_output(low_level_conditions))
                if low_level_conditions is not None else None),
            "captions": (
                str(require_benchmark_output(captions_json))
                if captions_json is not None else None),
            "config": {
                "sdxl_model": "stabilityai/sdxl-turbo",
                "ip_adapter": "h94/IP-Adapter:ip-adapter_sdxl_vit-h",
                "num_inference_steps": num_inference_steps,
                "guidance_scale": 0.0,
                "ip_adapter_scale": 1.0,
                "img2img_strength": img2img_strength if stage != "clip" else None,
                "seed": seed,
                "seed_policy": "fixed_per_sample",
                "profile_id": profile_id,
            },
            "derived_inputs_used": False,
        },
    })
    write_json(output / "manifest.json", artifact_manifest(
        artifact_type="atm_generation_stage", source_h5=source_h5,
        config={
            "stage": stage, "subject_id": subject_id,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": 0.0, "ip_adapter_scale": 1.0,
            "img2img_strength": img2img_strength if stage != "clip" else None,
            "seed": seed,
            "seed_policy": "fixed_per_sample",
        },
        extra={
            "predictions": str(predictions_path),
            "n_images": len(predictions),
            "image_sha256": {
                path.name: sha256_file(path) for path in sorted(images_dir.glob("*.png"))
            },
        }))
    return predictions_path
