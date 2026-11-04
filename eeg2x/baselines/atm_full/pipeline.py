"""Complete-ATM orchestration shared by the benchmark's dataset runners.

The complete ATM (Li et al., NeurIPS 2024) is a *baseline* in this benchmark:
it keeps its own ATM-S encoder and is never combined with the swappable encoders
of the benchmark's Only-CLIP framework.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

log = logging.getLogger(__name__)

from eeg2x.frameworks.only_clip.inference import generate_stage
from eeg2x.baselines.atm_full.inference import (
    decode_low_level_conditions, generate_captions, generate_prior_embeddings,
)
from eeg2x.baselines.atm_full.training import train_low_level, train_prior, train_semantic
from eeg2x.evaluation.api import evaluate_predictions

SEED = 2000


def run_complete_atm(*, subject: str, dataset: str, profile_id: str,
                     h5_dir, artifact_root, feature_store, embeddings,
                     device: str = "cuda:0", seed: int = 2000,
                     overwrite: bool = False, atm_epochs: int = 40,
                     prior_epochs: int = 150, prior_batch_size: int = 1024,
                     prior_learning_rate: float = 1e-3,
                     prior_inference_batch_size: int = 64,
                     prior_inference_steps: int = 50,
                     prior_guidance_scale: float = 5.0,
                     low_level_epochs: int = 200, low_level_batch_size: int = 30,
                     low_level_learning_rate: float = 1e-3,
                     low_level_inference_batch_size: int = 64,
                     semantic_epochs: int = 30, semantic_batch_size: int = 32,
                     semantic_learning_rate: float = 1e-3,
                     caption_batch_size: int = 32) -> Path:
    """Complete ATM: Diffusion Prior U-Net high level + low-level + semantic."""
    args = SimpleNamespace(
        device=device, overwrite=overwrite, seed=seed,
        prior_epochs=prior_epochs, prior_batch_size=prior_batch_size,
        prior_learning_rate=prior_learning_rate,
        prior_inference_batch_size=prior_inference_batch_size,
        prior_inference_steps=prior_inference_steps,
        prior_guidance_scale=prior_guidance_scale,
        low_level_epochs=low_level_epochs, low_level_batch_size=low_level_batch_size,
        low_level_learning_rate=low_level_learning_rate,
        low_level_inference_batch_size=low_level_inference_batch_size,
        semantic_epochs=semantic_epochs, semantic_batch_size=semantic_batch_size,
        semantic_learning_rate=semantic_learning_rate,
        caption_batch_size=caption_batch_size)
    encoder, root, h5, store = "atm_s", Path(artifact_root), Path(h5_dir), Path(feature_store)
    """Complete ATM: prior U-Net high level + low-level + semantic branches."""
    profile = profile_id
    prior_dir = root / "prior"
    prior_checkpoint = train_prior(
        train_embeddings=embeddings["train"], output_dir=prior_dir / "model",
        epochs=args.prior_epochs, batch_size=args.prior_batch_size,
        learning_rate=args.prior_learning_rate, device=args.device, seed=SEED)
    prior_predictions = generate_prior_embeddings(
        test_embeddings=embeddings["test"], prior_checkpoint=prior_checkpoint,
        output_h5=prior_dir / "test_clip_predictions.h5",
        batch_size=args.prior_inference_batch_size,
        num_inference_steps=args.prior_inference_steps,
        guidance_scale=args.prior_guidance_scale, device=args.device, seed=SEED,
        overwrite=args.overwrite)

    results = {}
    results["clip"] = generate_stage(
        stage="clip", subject_id=subject, h5_dir=h5,
        prior_predictions=prior_predictions,
        output_dir=root / "generation" / "clip", num_inference_steps=4,
        seed=SEED, device=args.device, overwrite=args.overwrite,
        profile_id=profile)

    low_checkpoint = train_low_level(
        subject_id=subject, h5_dir=h5, feature_store=store,
        output_dir=root / "low_level" / "model", epochs=args.low_level_epochs,
        batch_size=args.low_level_batch_size,
        learning_rate=args.low_level_learning_rate, device=args.device, seed=SEED,
        profile_id=profile)
    low_manifest = decode_low_level_conditions(
        subject_id=subject, h5_dir=h5, low_level_checkpoint=low_checkpoint,
        output_dir=root / "low_level" / "conditions",
        batch_size=args.low_level_inference_batch_size, device=args.device,
        overwrite=args.overwrite, profile_id=profile)
    results["low_level"] = generate_stage(
        stage="low_level", subject_id=subject, h5_dir=h5,
        prior_predictions=prior_predictions,
        low_level_conditions=low_manifest.parent,
        output_dir=root / "generation" / "low_level",
        num_inference_steps=4, seed=SEED, device=args.device,
        overwrite=args.overwrite, profile_id=profile)

    semantic_checkpoint = train_semantic(
        subject_id=subject, h5_dir=h5, feature_store=store,
        output_dir=root / "semantic" / "model", epochs=args.semantic_epochs,
        batch_size=args.semantic_batch_size,
        learning_rate=args.semantic_learning_rate, device=args.device, seed=SEED,
        profile_id=profile)
    captions = generate_captions(
        prior_predictions=prior_predictions,
        semantic_checkpoint=semantic_checkpoint,
        output_json=root / "semantic" / "captions.json",
        batch_size=args.caption_batch_size, device=args.device,
        overwrite=args.overwrite)
    results["semantic"] = generate_stage(
        stage="semantic", subject_id=subject, h5_dir=h5,
        prior_predictions=prior_predictions,
        low_level_conditions=low_manifest.parent, captions_json=captions,
        output_dir=root / "generation" / "semantic",
        num_inference_steps=4, seed=SEED, device=args.device,
        overwrite=args.overwrite, profile_id=profile)

    for stage, predictions in results.items():
        evaluate_predictions(
            task="image_generation", h5_dir=h5, predictions_path=predictions,
            output_path=root / "evaluation" / f"generation_{stage}.json",
            metrics=None)
    (root / "run_config.json").write_text(json.dumps({
        "experiment_id": root.name, "dataset": dataset, "subject_id": subject,
        "task": "image_generation", "encoder": encoder, "model": "atm_full",
        "preprocessing_profile": profile,
        "pipeline": ("EEG -> ATM encoder -> Diffusion Prior U-Net (Stage I) -> CLIP "
                     "embedding -> SDXL-Turbo + IP-Adapter; low-level VAE-latent and "
                     "semantic caption branches as in the ATM paper"),
        "prior_steps": args.prior_inference_steps,
        "prior_guidance_scale": args.prior_guidance_scale,
        "prior_epochs": args.prior_epochs, "seed": SEED, "pythonhashseed": SEED,
        "low_level_epochs": args.low_level_epochs,
        "semantic_epochs": args.semantic_epochs, "device": args.device,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("%s/%s atm_full complete: %s", subject, encoder, list(results))
    return results["semantic"]


