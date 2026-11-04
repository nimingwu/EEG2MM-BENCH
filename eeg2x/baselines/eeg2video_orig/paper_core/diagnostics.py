"""Quality diagnostics for the three EEG conditioning predictors."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as functional

from .config import RUN_ROOT, SETTINGS
from .data import TEXT_EMBEDDINGS_PATH, VIDEO_LATENTS_PATH, dynamic_labels, read_manifest
from .inference import predict_eeg_conditions


def _mean_train_latent(latents: np.ndarray) -> np.ndarray:
    total = np.zeros(latents.shape[1:], dtype=np.float64)
    for begin in range(0, SETTINGS.train_samples, 32):
        batch = np.asarray(
            latents[begin : begin + 32], dtype=np.float32
        )
        total += batch.sum(axis=0, dtype=np.float64)
    return (total / SETTINGS.train_samples).astype(np.float32)


def _seq2seq_diagnostics(predicted: torch.Tensor) -> dict:
    latents = np.load(VIDEO_LATENTS_PATH, mmap_mode="r")
    target = torch.from_numpy(
        np.asarray(
            latents[
                SETTINGS.train_samples : SETTINGS.train_samples + SETTINGS.test_samples
            ],
            dtype=np.float32,
        )
    )
    prediction = predicted.float()
    train_mean = torch.from_numpy(_mean_train_latent(latents))
    frame_dims = (0, 2, 3, 4)
    prediction_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    cosine = functional.cosine_similarity(prediction_flat, target_flat, dim=1)
    across_sample_std = prediction.std(dim=0, unbiased=False).mean(dim=(1, 2, 3))
    frame_mse = ((prediction - target) ** 2).mean(dim=frame_dims)
    return {
        "test_mse": float(functional.mse_loss(prediction, target)),
        "zero_mse": float(target.square().mean()),
        "train_mean_mse": float((target - train_mean.unsqueeze(0)).square().mean()),
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std(unbiased=False)),
        "framewise_mse": [float(value) for value in frame_mse],
        "per_frame_across_sample_std": [
            float(value) for value in across_sample_std
        ],
        "first_frame_across_sample_std": float(across_sample_std[0]),
        "first_frame_varies_across_samples": bool(across_sample_std[0] > 0.0),
        "beats_zero_baseline": bool(
            functional.mse_loss(prediction, target) < target.square().mean()
        ),
        "beats_train_mean_baseline": bool(
            functional.mse_loss(prediction, target)
            < (target - train_mean.unsqueeze(0)).square().mean()
        ),
    }


def _semantic_diagnostics(predicted: torch.Tensor) -> dict:
    embeddings = np.load(TEXT_EMBEDDINGS_PATH, mmap_mode="r")
    target = torch.from_numpy(
        np.asarray(
            embeddings[
                SETTINGS.train_samples : SETTINGS.train_samples + SETTINGS.test_samples
            ],
            dtype=np.float32,
        )
    )
    prediction = predicted.float()
    prediction_flat = prediction.flatten(1)
    target_flat = target.flatten(1)
    cosine = functional.cosine_similarity(prediction_flat, target_flat, dim=1)

    test_rows = read_manifest()[
        SETTINGS.train_samples : SETTINGS.train_samples + SETTINGS.test_samples
    ]
    concepts = torch.tensor([row["concept"] for row in test_rows], dtype=torch.long)
    prototypes = torch.empty(
        SETTINGS.concepts, target_flat.shape[1], dtype=torch.float32
    )
    for concept in range(SETTINGS.concepts):
        mask = concepts == concept
        if int(mask.sum()) != SETTINGS.repeats_per_concept:
            raise ValueError(
                f"Test concept {concept} has {int(mask.sum())} clips, expected "
                f"{SETTINGS.repeats_per_concept}"
            )
        prototypes[concept] = target_flat[mask].mean(dim=0)
    similarities = functional.normalize(prediction_flat, dim=1) @ functional.normalize(
        prototypes, dim=1
    ).T
    ranking = similarities.argsort(dim=1, descending=True)
    top1 = ranking[:, 0] == concepts
    top5 = (ranking[:, :5] == concepts.unsqueeze(1)).any(dim=1)
    return {
        "test_mse": float(functional.mse_loss(prediction, target)),
        "zero_mse": float(target.square().mean()),
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std(unbiased=False)),
        "concept_top1": float(top1.float().mean()),
        "concept_top5": float(top5.float().mean()),
        "concept_chance_top1": 1.0 / SETTINGS.concepts,
        "concept_chance_top5": 5.0 / SETTINGS.concepts,
        "target_protocol": SETTINGS.semantic_target_mode,
        "feature_protocol": SETTINGS.semantic_feature_mode,
    }


def _dynamic_diagnostics(predicted: torch.Tensor) -> dict:
    target = torch.from_numpy(
        dynamic_labels()[
            SETTINGS.train_samples : SETTINGS.train_samples + SETTINGS.test_samples
        ]
    ).long()
    prediction = predicted.long()
    per_class_recall = []
    for label in range(2):
        mask = target == label
        per_class_recall.append(float((prediction[mask] == label).float().mean()))
    return {
        "accuracy": float((prediction == target).float().mean()),
        "balanced_accuracy": float(np.mean(per_class_recall)),
        "per_class_recall": {
            "slow": per_class_recall[0],
            "fast": per_class_recall[1],
        },
        "ground_truth_counts": {
            "slow": int((target == 0).sum()),
            "fast": int((target == 1).sum()),
        },
        "predicted_counts": {
            "slow": int((prediction == 0).sum()),
            "fast": int((prediction == 1).sum()),
        },
        "window_samples": SETTINGS.dynamic_input_samples,
        "windows_per_clip": SETTINGS.dynamic_windows,
    }


@torch.no_grad()
def evaluate_conditions() -> dict:
    """Evaluate block-7 conditions before spending time on DDIM inference."""
    if not torch.cuda.is_available():
        raise RuntimeError("Predictor diagnostics require CUDA for the semantic MLP")
    device = torch.device("cuda")
    predicted_latents, predicted_semantic, predicted_dynamic = predict_eeg_conditions(
        device
    )
    if predicted_semantic is None:
        raise RuntimeError("Full-model diagnostics require the Semantic Predictor")
    report = {
        "reproduction_revision": SETTINGS.reproduction_revision,
        "subject": SETTINGS.subject,
        "split": "block 7",
        "n_test": SETTINGS.test_samples,
        "seq2seq": _seq2seq_diagnostics(predicted_latents),
        "semantic": _semantic_diagnostics(predicted_semantic),
        "dynamic": _dynamic_diagnostics(predicted_dynamic),
    }
    output_path = RUN_ROOT / "condition_diagnostics.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report
