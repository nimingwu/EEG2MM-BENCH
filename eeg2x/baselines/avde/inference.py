"""AVDE reconstruction inference for the EEG2X benchmark.

Upstream ships no inference script; this mirrors the generation half of the
paper's pipeline (Section 3.1) and writes the benchmark's sample-ID keyed
prediction artifact so the unchanged 8-metric evaluator can consume it:

    EEG -> fine-tuned LaBraM embedding -> VAR next-scale prediction
        -> multi-scale VQ-VAE decoder -> 256x256 RGB reconstruction

Generation settings are the paper's: 10 scales ``(1,2,3,4,5,6,8,10,13,16)``,
classifier-free guidance ``cfg = 4.0`` and top-k sampling with ``k = 900``.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
from PIL import Image

from eeg2x.rawdata import require_benchmark_output
from eeg2x import noise_control
from eeg2x.data import (
    artifact_manifest, sample_id_for_h5_row, write_json,
)
from eeg2x.baselines.avde import bootstrap

SEED = 2000


log = logging.getLogger(__name__)

PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
DEPTH = 16


def build_models(device: str):
    """Build the frozen VQ-VAE, the fine-tuned LaBraM encoder and the VAR."""
    bootstrap.install()
    bootstrap.require_pretrained()

    from models import build_vae_var_eeg
    from models.labram import load_labram_ckpt

    vae_local, labram_local, var_wo_ddp = build_vae_var_eeg(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=PATCH_NUMS,
        depth=DEPTH, shared_aln=False, attn_l2_norm=True,
        flash_if_available=True, fused_if_available=True,
    )
    vae_local.load_state_dict(
        torch.load(bootstrap.VAE_CKPT, map_location="cpu"), strict=True)
    vae_local.eval()
    for parameter in vae_local.parameters():
        parameter.requires_grad = False
    var_wo_ddp.load_state_dict(
        torch.load(bootstrap.VAR_CKPT, map_location="cpu"), strict=False)
    return vae_local, labram_local, var_wo_ddp, load_labram_ckpt


def generate_stage(
    *,
    subject_id: str,
    h5_dir: str | Path,
    test_h5: str | Path,
    profile_id: str,
    feature_store: str | Path,
    labram_checkpoint: str | Path,
    var_checkpoint: str | Path,
    output_dir: str | Path,
    dataset: str,
    cfg_scale: float = 4.0,
    top_k: int = 900,
    seed: int = SEED,
    device: str = "cuda:0",
    overwrite: bool = False,
    noise: str = "none",
    noise_seed: int = SEED,
    distinct_images: int | None = None,
) -> Path:
    """Reconstruct every formal test row and write ``predictions.json``.

    ``noise="gaussian"`` is the noise control (``model/noise_control.py``): the
    test EEG row is replaced by N(0,1) with both frozen checkpoints untouched.
    The caller points ``output_dir`` at the noise namespace, so the clean run is
    never touched.
    """
    output = require_benchmark_output(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    test_h5 = require_benchmark_output(test_h5)
    labram_checkpoint = require_benchmark_output(labram_checkpoint)
    var_checkpoint = require_benchmark_output(var_checkpoint)

    vae_local, labram_local, var_wo_ddp, load_labram_ckpt = build_models(device)
    state = torch.load(var_checkpoint, map_location="cpu")
    trainer_state = state["trainer"]
    var_wo_ddp.load_state_dict(trainer_state["var_wo_ddp"], strict=False)
    var_wo_ddp.eval()
    for parameter in var_wo_ddp.parameters():
        parameter.requires_grad = False
    load_labram_ckpt(labram_local, torch.load(labram_checkpoint, map_location="cpu"))
    labram_local.eval()
    for parameter in labram_local.parameters():
        parameter.requires_grad = False

    # The dataset supplies the benchmark HDF5 rows, the montage-derived LaBraM
    # input_chans and the whitened EEG in upstream's (channels, 1, time) layout.
    from eeg_data.dataset import (
        EEGDataset, configure as configure_avde_dataset, current_input_chans)

    configure_avde_dataset(
        subject=subject_id, h5_dir=h5_dir, profile_id=profile_id,
        feature_store=feature_store)
    dataset_test = EEGDataset(subjects=[subject_id], train=False)
    input_chans = current_input_chans()
    n_rows = len(dataset_test)
    log.info("AVDE generation: %d test rows, cfg=%s top_k=%s", n_rows, cfg_scale, top_k)

    torch.manual_seed(seed)
    predictions: Dict[str, Dict[str, str]] = {}
    started = time.time()
    # 噪声对照：生成器在循环外建一次，每张图消耗一次状态——同一份噪声由
    # (种子, 行顺序) 决定，与干净条件逐行对齐。
    generator = (noise_control.noise_generator(noise_seed)
                 if noise_control.noise_enabled(noise) else None)
    for row in range(n_rows):
        destination = images_dir / f"sample-{row:03d}.png"
        if distinct_images is not None and row >= distinct_images:
            # 噪声对照省时档：只真出前 N 张，其余复用第 (row % N) 张。
            # 只对噪声条件成立——干净条件下每行是刺激特异的，不能复制。
            import shutil
            source = images_dir / f"sample-{row % distinct_images:03d}.png"
            if overwrite or not destination.is_file():
                shutil.copyfile(source, destination)
            sample_id = sample_id_for_h5_row(test_h5, row)
            predictions[sample_id] = {"path": str(destination)}
            continue
        eeg, _image, _label, _text, _text_features, _image_features = dataset_test[row]
        if overwrite or not destination.is_file():
            inp_eeg = noise_control.inject_noise(eeg.unsqueeze(0).to(device), generator)
            with torch.no_grad():
                eeg_emb, _, _ = labram_local.forward(inp_eeg, input_chans=input_chans)
                recon = var_wo_ddp.autoregressive_infer_cfg(
                    B=1, eeg_emb=eeg_emb, cfg=cfg_scale, top_k=top_k, top_p=0.0)
            array = (recon[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0)
            Image.fromarray(array.round().astype(np.uint8)).save(destination)
        sample_id = sample_id_for_h5_row(test_h5, row)
        predictions[sample_id] = {"path": str(destination)}
        if (row + 1) % 20 == 0:
            elapsed = time.time() - started
            log.info("AVDE generation %d/%d (%.0fs)", row + 1, n_rows, elapsed)

    source_h5 = [dataset_test.fingerprints[0]]
    predictions_path = write_json(output / "predictions.json", {
        "predictions": predictions,
        "provenance": {
            "artifact_type": "avde_generation_predictions",
            "dataset": dataset, "subject_id": subject_id,
            "source_h5": source_h5,
            "labram_checkpoint": str(labram_checkpoint),
            "var_checkpoint": str(var_checkpoint),
            "config": {
                "cfg": cfg_scale, "top_k": top_k, "steps": len(PATCH_NUMS),
                "patch_nums": list(PATCH_NUMS), "depth": DEPTH,
                "image_size": 256, "vqvae_vocab": 4096,
                "seed": seed, "seed_policy": "fixed_for_the_whole_split",
            },
            "derived_inputs_used": False,
            **noise_control.noise_meta(noise, noise_seed, model="avde",
                                       distinct_images=distinct_images),
        },
    })
    if noise_control.noise_enabled(noise):
        # 约定 2：条件溯源写旁挂 meta（predictions.json 的顶层 key 是样本 id）。
        noise_control.write_noise_meta(
            output / "noise_meta.json",
            noise_control.noise_meta(
                noise, noise_seed, model="avde", dataset=dataset,
                subject=subject_id, profile_id=profile_id,
                labram_checkpoint=str(labram_checkpoint),
                var_checkpoint=str(var_checkpoint), n_rows=n_rows,
                distinct_images=distinct_images))
    write_json(output / "manifest.json", artifact_manifest(
        artifact_type="avde_generation_stage", source_h5=source_h5,
        config={"dataset": dataset, "subject_id": subject_id, "cfg": cfg_scale,
                "top_k": top_k, "seed": seed, "patch_nums": list(PATCH_NUMS)},
        extra={"predictions": str(predictions_path), "n_images": len(predictions)}))
    del vae_local, labram_local, var_wo_ddp
    torch.cuda.empty_cache()
    log.info("AVDE generation wrote %d images to %s", len(predictions), images_dir)
    return predictions_path
