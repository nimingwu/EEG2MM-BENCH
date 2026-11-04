"""ENIGMA baseline orchestration for the EEG2X benchmark.

ENIGMA (Kneeland et al., 2026, arXiv:2602.10361) is a *baseline*: it keeps its
own spatio-temporal CNN encoder and never swaps encoders with the Only-CLIP
framework.

Two properties make this a thin adapter rather than a port:

1. ENIGMA's own preprocessing (paper Appendix A.1) is the contract the
   benchmark already publishes for the ATM image-generation profiles — MNE,
   epoch ``[-0.2, 1.0]`` s, pre-stimulus baseline, 250 Hz, train-only MNN.
   No new profile is registered.
2. Its generation stage is SDXL-Turbo + IP-Adapter at 4 steps, i.e. exactly the
   benchmark's ``generate_stage(stage="clip")``.  ENIGMA only has to produce the
   predicted 1024-d CLIP ViT-H/14 embedding.

Training reproduces upstream: AdamW at ``3e-4`` with OneCycleLR, batch 512,
150 epochs, gradient clipping at 1.0, and the loss
``MSE + 0.5 * InfoNCE(image) + 0.05 * InfoNCE(text)``.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from eeg2x.protocol import sha256_file
from eeg2x.rawdata import require_benchmark_output
from eeg2x import noise_control
from eeg2x.data import artifact_manifest, discover_subject_h5, write_json
from eeg2x.targets.clip import extract_features

log = logging.getLogger(__name__)

SEED = 2000
EPOCHS = 150
BATCH_SIZE = 512
MAX_LR = 3e-4
PCT_START = 0.1
FINAL_DIV_FACTOR = 1000.0
MSE_LOSS_SCALE = 1.0
RETRIEVAL_IMG_LOSS_SCALE = 0.5
RETRIEVAL_TXT_LOSS_SCALE = 0.05
SDXL_STEPS = 4


def _predict_embeddings(model, dataset, device, batch_size: int,
                        noise: str = "none", noise_seed: int = SEED) -> np.ndarray:
    """ENIGMA forward over the formal test rows, mirroring upstream inference.

    ``noise="gaussian"`` is the noise control (``model/noise_control.py``): the
    test EEG is replaced by N(0,1) with the trained weights frozen.
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    outputs = []
    model.eval()
    generator = (noise_control.noise_generator(noise_seed)
                 if noise_control.noise_enabled(noise) else None)
    with torch.no_grad():
        for batch in loader:
            eeg = noise_control.inject_noise(batch.eeg.to(device), generator)
            subjects = batch.subject
            projected = model(eeg, subjects)
            outputs.append(projected.float().cpu().numpy())
    predicted = np.concatenate(outputs, axis=0)
    if not np.isfinite(predicted).all():
        raise RuntimeError("ENIGMA produced non-finite CLIP embeddings")
    return predicted


def run_enigma(
    *,
    dataset: str,
    subject: str,
    profile_id: str,
    h5_dir: str | Path,
    artifact_root: str | Path,
    feature_store: str | Path,
    device: str = "cuda:0",
    seed: int = SEED,
    overwrite: bool = False,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    max_lr: float = MAX_LR,
    test_batch_size: int = 1000,
    noise: str = "none",
    noise_seed: int = SEED,
    distinct_images: int | None = None,
) -> Path:
    """Train ENIGMA, predict CLIP embeddings, generate and score images.

    ``noise="gaussian"`` switches to the noise control: the checkpoint is read
    from the clean ``artifact_root`` (and must already exist -- the control
    never trains), the test EEG is replaced by N(0,1), and every output goes to
    ``artifact_root/noise_control/`` so the clean run is untouched.
    """
    if seed != SEED:
        raise ValueError(
            f"EEG2X-Benchmark fixes the global seed at {SEED}, got {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    root = require_benchmark_output(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    # 噪声条件：权重仍从干净根目录读，产物写到独立子目录（约定 1）。
    output_root = noise_control.resolve_noise_dir(root, noise)
    if noise_control.noise_enabled(noise):
        output_root = require_benchmark_output(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    h5_dir = require_benchmark_output(h5_dir)
    store = require_benchmark_output(feature_store)

    train_h5 = discover_subject_h5(h5_dir, subject, profile_id=profile_id, split="train")
    test_h5 = discover_subject_h5(h5_dir, subject, profile_id=profile_id, split="test")
    extract_features([*train_h5, *test_h5], store, kinds=("clip_vit_h",),
                     batch_size=32, device=device, overwrite=overwrite, seed=seed)

    from eeg2x.baselines.enigma.source.dataset import EEGDataset
    from eeg2x.baselines.enigma.source.models import ENIGMA
    from eeg2x.baselines.enigma.source.training import (
        get_dataloaders, prepare_optimizer, train_epoch,
    )

    text_cache: dict = {}
    train_loader, test_loader = get_dataloaders(
        None, [subject], batch_size, h5_dir=str(h5_dir), profile_id=profile_id,
        feature_store=str(store), text_features=text_cache,
        test_batch_size=test_batch_size)
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    n_channels, n_timepoints = train_dataset.eeg_data.shape[-2:]
    log.info("ENIGMA %s/%s: %d train samples, %d test rows, %d channels x %d timepoints",
             dataset, subject, len(train_dataset), len(test_dataset),
             n_channels, n_timepoints)

    # EEG2X-Benchmark: upstream hardcodes hidden_dim=184, which is only correct
    # for THINGS-EEG2's 250 timepoints (the backbone flattens to
    # 4 * (floor((N_T - 21) / 5) + 1)).  The other benchmark datasets have
    # 276 (ds004018) and 226 (NOD, ds005106) timepoints, so the projector's
    # input width is derived from the actual backbone output instead.
    hidden_dim = 4 * (((n_timepoints - 21) // 5) + 1)
    checkpoint = root / "model" / "final.pt"
    model = ENIGMA(n_channels, n_timepoints, subjects=[subject],
                   hidden_dim=hidden_dim, embed_dim=1024)
    model = model.to(device)
    if overwrite or not checkpoint.is_file():
        optimizer, scheduler = prepare_optimizer(
            model=model, max_lr=max_lr, num_epochs=epochs,
            train_dataloader=train_loader, pct_start=PCT_START,
            final_div_factor=FINAL_DIV_FACTOR)
        for epoch in range(epochs):
            train_epoch(model, train_loader, optimizer, scheduler, device,
                        MSE_LOSS_SCALE, RETRIEVAL_IMG_LOSS_SCALE,
                        RETRIEVAL_TXT_LOSS_SCALE, retrieval_only=False)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log.info("ENIGMA epoch %d/%d (lr %.3g)", epoch + 1, epochs,
                         scheduler.get_last_lr()[-1])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "epochs": epochs,
                    "n_channels": n_channels, "n_timepoints": n_timepoints,
                    "hidden_dim": hidden_dim,
                    "subject": subject, "profile_id": profile_id,
                    "seed": seed}, checkpoint)
    else:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
    model.eval()

    predicted = _predict_embeddings(model, test_dataset, device, 256,
                                    noise=noise, noise_seed=noise_seed)
    targets = test_dataset.img_vectors.numpy()
    test_mse = float(np.mean((predicted.astype(np.float64) - targets) ** 2))

    predictions_h5 = output_root / "projection" / "test_clip_predictions.h5"
    predictions_h5 = require_benchmark_output(predictions_h5)
    predictions_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(predictions_h5, "w") as h5:
        h5.create_dataset("clip_prediction", data=predicted.astype(np.float16),
                          chunks=(min(256, len(predicted)), 1024))
        h5.attrs["schema"] = "eeg2x-enigma-predictions-v1"
        h5.attrs["complete"] = True
        h5.attrs["test_mse"] = test_mse
        h5.attrs["source_h5_json"] = json.dumps(
            [str(path) for path in (*train_h5, *test_h5)], sort_keys=True)
        h5.attrs["checkpoint"] = str(checkpoint)
        h5.attrs["checkpoint_sha256"] = sha256_file(checkpoint)
        h5.attrs["derived_inputs_used"] = False

    from eeg2x.frameworks.only_clip.inference import generate_stage
    from eeg2x.evaluation.api import evaluate_predictions

    images = generate_stage(
        stage="clip", subject_id=subject, h5_dir=h5_dir,
        image_embedding_predictions=predictions_h5,
        output_dir=output_root / "generation" / "clip",
        num_inference_steps=SDXL_STEPS, seed=seed, device=device,
        overwrite=overwrite, profile_id=profile_id,
        distinct_images=distinct_images)
    evaluation = evaluate_predictions(
        task="image_generation", h5_dir=h5_dir, predictions_path=images,
        output_path=output_root / "evaluation" / "generation_enigma.json",
        metrics=None)
    if noise_control.noise_enabled(noise):
        # 约定 2/3：条件溯源写旁挂 meta，不混进 predictions.json。
        noise_control.write_noise_meta(
            output_root / "noise_meta.json",
            noise_control.noise_meta(
                noise, noise_seed, model="enigma", dataset=dataset,
                subject=subject, profile_id=profile_id,
                checkpoint=str(checkpoint), clean_root=str(root),
                test_mse_noise=test_mse, evaluation=str(evaluation),
                distinct_images=distinct_images))

    write_json(output_root / "run_config.json", {
        "experiment_id": root.name, "dataset": dataset, "subject_id": subject,
        "task": "image_generation", "encoder": "enigma_spatiotemporal_cnn",
        "model": "enigma", "preprocessing_profile": profile_id,
        "pipeline": ("EEG -> spatio-temporal CNN -> per-subject linear alignment "
                     "-> MLP projector -> predicted CLIP ViT-H/14 embedding -> "
                     "SDXL-Turbo + IP-Adapter (4 steps)"),
        "epochs": epochs, "batch_size": batch_size, "max_lr": max_lr,
        "pct_start": PCT_START, "final_div_factor": FINAL_DIV_FACTOR,
        "mse_loss_scale": MSE_LOSS_SCALE,
        "retrieval_img_loss_scale": RETRIEVAL_IMG_LOSS_SCALE,
        "retrieval_txt_loss_scale": RETRIEVAL_TXT_LOSS_SCALE,
        "n_channels": int(n_channels), "n_timepoints": int(n_timepoints),
        "test_mse": test_mse, "sdxl_steps": SDXL_STEPS,
        "seed": seed, "pythonhashseed": seed,
        "source_h5": [str(p) for p in (*train_h5, *test_h5)],
        "derived_inputs_used": False,
    })
    write_json(output_root / "manifest.json", artifact_manifest(
        artifact_type="enigma_generation", source_h5=[], config={
            "epochs": epochs, "batch_size": batch_size, "max_lr": max_lr,
            "test_mse": test_mse, "seed": seed},
        extra={"predictions": str(images), "evaluation": str(evaluation)}))
    log.info("ENIGMA %s/%s complete: %s", dataset, subject, evaluation)
    return evaluation
