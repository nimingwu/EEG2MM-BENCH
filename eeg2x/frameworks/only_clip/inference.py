"""Only-CLIP 推理链路：投影 -> SDXL 生成。

只保留 Only-CLIP 需要的 ``clip`` 阶段。``low_level`` 与 ``semantic`` 属于完整
ATM baseline，代码在 ``eeg2x.baselines.atm_full``；这里遇到非 clip 阶段会明确
报错，而不是走到不存在的分支。
"""
from __future__ import annotations

import gc
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
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

log = logging.getLogger("eeg2x.only_clip.inference")

STAGES = ("clip",)
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


def generate_stage(
    *,
    stage: str = "clip",
    subject_id: str,
    h5_dir: str | Path,
    output_dir: str | Path,
    image_embedding_predictions: str | Path,
    num_inference_steps: int = 4,
    seed: int = 2000,
    device: str = "cuda:0",
    overwrite: bool = False,
    profile_id: str = ATM_PROFILE,
    distinct_images: int | None = None,
) -> Path:
    """Generate the Only-CLIP (clip) stage and write sample-ID predictions.

    ``distinct_images=N`` is the noise-control time saver: only the first N rows
    are generated and later rows reuse row ``r % N``.  It is only sound for the
    noise condition and must not be used for a clean run.
    """
    if stage != "clip":
        raise ValueError(
            f"only_clip supports stage='clip' only; got {stage!r}. "
            "low_level / semantic belong to eeg2x.baselines.atm_full")
    output = require_benchmark_output(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    test_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=profile_id, split="test")
    with h5py.File(require_benchmark_output(image_embedding_predictions), "r") as h5:
        clip_predictions = np.asarray(h5["clip_prediction"][:], dtype=np.float32)
        source_h5 = json.loads(h5.attrs["source_h5_json"])
    with h5py.File(test_paths[0], "r") as h5:
        n_test_rows = int(h5["data"].shape[0])
    if len(clip_predictions) != n_test_rows:
        raise ValueError(
            "Prediction count does not match the formal ATM test HDF5: "
            f"{len(clip_predictions)} != {n_test_rows}")
    if distinct_images is not None and distinct_images < 1:
        raise ValueError("distinct_images must be >= 1 when set")

    target_device = _device(device)
    generator = ClipGenerator(
        num_inference_steps=num_inference_steps, device=str(target_device),
        cache_dir=HF_CACHE, local_files_only=True)
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
            image = generator.generate(
                torch.from_numpy(embedding).unsqueeze(0),
                text_prompt="", generator=torch_generator)
            image.save(destination)
        sample_id = sample_id_for_h5_row(test_paths[0], row)
        predictions[sample_id] = {"path": str(destination)}
        if (row + 1) % 20 == 0:
            log.info("clip generation %d/%d", row + 1, len(clip_predictions))
    del generator
    _clear_cuda()
    predictions_path = write_json(output / "predictions.json", {
        "predictions": predictions,
        "provenance": {
            "artifact_type": "atm_generation_predictions",
            "stage": "clip", "subject_id": subject_id,
            "source_h5": source_h5,
            "image_embedding_predictions": str(
                require_benchmark_output(image_embedding_predictions)),
            "low_level_conditions": None,
            "captions": None,
            "config": {
                "sdxl_model": "stabilityai/sdxl-turbo",
                "ip_adapter": "h94/IP-Adapter:ip-adapter_sdxl_vit-h",
                "num_inference_steps": num_inference_steps,
                "guidance_scale": 0.0,
                "ip_adapter_scale": 1.0,
                "img2img_strength": None,
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
            "stage": "clip", "subject_id": subject_id,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": 0.0, "ip_adapter_scale": 1.0,
            "img2img_strength": None,
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
