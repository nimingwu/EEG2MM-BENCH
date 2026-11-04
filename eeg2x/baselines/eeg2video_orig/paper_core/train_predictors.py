"""Independent 200-epoch trainers for Seq2Seq, semantic MLP and GLMNet."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from .config import (
    ABLATION_NO_SEMANTIC_DYNAMIC,
    CHECKPOINT_ROOT,
    RUN_ROOT,
    SETTINGS,
)
from .data import (
    ArrayDataset,
    FEATURE_WINDOW_EEG_PATH,
    TEXT_EMBEDDINGS_PATH,
    VIDEO_LATENTS_PATH,
    WINDOW_EEG_PATH,
    dynamic_labels,
    ensure_output_roots,
    extract_embeddings,
    prepare,
    read_manifest,
    semantic_de_features,
    seed_everything,
)
from .models import DynamicGLMNet, EEGToLatentSeq2Seq, SemanticPredictor


SEQ_CHECKPOINT = CHECKPOINT_ROOT / "seq2seq.pt"
SEMANTIC_CHECKPOINT = CHECKPOINT_ROOT / "semantic_predictor.pt"
DYNAMIC_CHECKPOINT = CHECKPOINT_ROOT / "dynamic_glmnet.pt"


def _completion_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".complete.json")


def _checkpoint_is_current(checkpoint_path: Path, task: str) -> bool:
    marker_path = _completion_path(checkpoint_path)
    if not checkpoint_path.exists() or not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        marker.get("task") == task
        and marker.get("epoch") == SETTINGS.predictor_epochs
        and marker.get("reproduction_revision") == SETTINGS.reproduction_revision
        and marker.get("checkpoint_bytes") == checkpoint_path.stat().st_size
    )


def _begin_training(checkpoint_path: Path, task: str, force: bool) -> bool:
    if not force and _checkpoint_is_current(checkpoint_path, task):
        print(f"{task} already trained: {checkpoint_path}")
        return False
    _completion_path(checkpoint_path).unlink(missing_ok=True)
    return True


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(values[: SETTINGS.train_samples], dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    return mean, std


def _normalized_copy(
    source: np.ndarray, mean: np.ndarray, std: np.ndarray, name: str
) -> np.ndarray:
    path = RUN_ROOT / name
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=source.shape
    )
    for begin in range(0, len(source), 32):
        output[begin : begin + 32] = (
            np.asarray(source[begin : begin + 32], dtype=np.float32) - mean
        ) / std
    output.flush()
    return output


def _loader(dataset, batch_size: int, seed: int):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=False,
    )


def _save_checkpoint(
    path: Path,
    model,
    epoch: int,
    mean: np.ndarray,
    std: np.ndarray,
    task: str,
    loss: float,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "normalization_mean": mean,
            "normalization_std": std,
            "task": task,
            "loss": loss,
            "optimizer": SETTINGS.predictor_optimizer,
            "learning_rate": SETTINGS.predictor_learning_rate,
            "scheduler": SETTINGS.predictor_scheduler,
            "epochs": SETTINGS.predictor_epochs,
            "paper_settings": SETTINGS.as_dict(),
        },
        path,
    )
    if epoch == SETTINGS.predictor_epochs:
        _completion_path(path).write_text(
            json.dumps(
                {
                    "task": task,
                    "epoch": epoch,
                    "reproduction_revision": SETTINGS.reproduction_revision,
                    "checkpoint_bytes": path.stat().st_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _write_history(task: str, history: list[dict]) -> None:
    path = CHECKPOINT_ROOT / f"{task}_history.json"
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _semantic_targets(text: np.ndarray) -> np.ndarray:
    """Apply the released one-caption-per-concept training recipe.

    The paper says BLIP is run on every middle frame, while the later author
    reproduction instructions say to select one caption for each class. The
    released semantic trainer implements the latter independently per block.
    Preserve that behavior without its hard-coded block-order indexing bug.
    """
    if SETTINGS.semantic_target_mode == "per_video_caption":
        return text[: SETTINGS.train_samples]
    if SETTINGS.semantic_target_mode not in {
        "block_concept_caption",
        "global_concept_caption",
    }:
        raise ValueError(
            f"Unsupported semantic target mode: {SETTINGS.semantic_target_mode}"
        )
    rows = read_manifest()[: SETTINGS.train_samples]
    targets = np.empty_like(text[: SETTINGS.train_samples])
    if SETTINGS.semantic_target_mode == "global_concept_caption":
        for concept in range(SETTINGS.concepts):
            concept_indices = [
                i for i, row in enumerate(rows) if row["concept"] == concept
            ]
            expected = SETTINGS.train_blocks * SETTINGS.repeats_per_concept
            if len(concept_indices) != expected:
                raise ValueError(
                    f"Concept {concept} has {len(concept_indices)} clips, expected {expected}"
                )
            targets[concept_indices] = text[concept_indices[0]]
        return targets
    for block in range(1, SETTINGS.train_blocks + 1):
        block_indices = [i for i, row in enumerate(rows) if row["block"] == block]
        for concept in range(SETTINGS.concepts):
            concept_indices = [
                i for i in block_indices if rows[i]["concept"] == concept
            ]
            if len(concept_indices) != SETTINGS.repeats_per_concept:
                raise ValueError(
                    f"Block {block}, concept {concept} has {len(concept_indices)} clips"
                )
            targets[concept_indices] = text[concept_indices[0]]
    return targets


def _limit_cpu_threads() -> None:
    """Cap torch's CPU thread pool before any loader is built.

    Same failure as Scheme D's ``_limit_cpu_threads`` (see
    ``model/eeg2video_core/scheme_d_training.py``): this chain's loader also
    uses ``num_workers=0``, so the collate stacks the batch in the main
    process, and on a many-core host torch's default thread count makes that
    pathological.  Measured here with the chain's own batch size of 32::

        torch.stack(32 tensors)   52 threads -> 1005 ms
                                   8 threads ->   32 ms

    At 178 batches per epoch that is ~179 s of assembly per epoch against
    minutes of compute, with the GPU idle throughout.
    """

    import torch

    wanted = int(os.environ.get("EEG2VIDEO_CPU_THREADS", "8"))
    before = torch.get_num_threads()
    if wanted > 0 and before > wanted:
        torch.set_num_threads(wanted)
        print(f"torch CPU threads: {before} -> {wanted} (the default starves "
              f"the DataLoader on this core count)", flush=True)


def train_seq2seq(force: bool = False) -> None:
    if not _begin_training(SEQ_CHECKPOINT, "seq2seq", force):
        return
    if not VIDEO_LATENTS_PATH.exists():
        extract_embeddings()
    import torch
    import torch.nn.functional as functional

    seed_everything(SETTINGS.predictor_seed)
    _limit_cpu_threads()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eeg = np.load(WINDOW_EEG_PATH, mmap_mode="r")
    mean, std = _normalization(eeg)
    normalized = _normalized_copy(eeg, mean, std, "seq2seq_eeg_normalized.float32.npy")
    latents = np.load(VIDEO_LATENTS_PATH, mmap_mode="r")
    dataset = ArrayDataset(
        normalized[: SETTINGS.train_samples], latents[: SETTINGS.train_samples]
    )
    loader = _loader(dataset, SETTINGS.predictor_batch_size, SETTINGS.predictor_seed)
    model = EEGToLatentSeq2Seq().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS.predictor_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SETTINGS.predictor_epochs * len(loader)
    )
    history: list[dict] = []
    for epoch in range(1, SETTINGS.predictor_epochs + 1):
        started = time.time()
        model.train()
        total_loss = 0.0
        samples = 0
        for eeg_batch, latent_batch in loader:
            eeg_batch = eeg_batch.to(device, dtype=torch.float32, non_blocking=True)
            latent_batch = latent_batch.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(eeg_batch, latent_batch)
            loss = functional.mse_loss(prediction, latent_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * eeg_batch.shape[0]
            samples += eeg_batch.shape[0]
        epoch_loss = total_loss / samples
        row = {
            "epoch": epoch,
            "mse": epoch_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"Seq2Seq epoch {epoch:03d}/200 mse={epoch_loss:.8f} "
            f"lr={row['learning_rate']:.10f}",
            flush=True,
        )
        _write_history("seq2seq", history)
        if epoch % 10 == 0 or epoch == SETTINGS.predictor_epochs:
            _save_checkpoint(SEQ_CHECKPOINT, model, epoch, mean, std, "seq2seq", epoch_loss)


def train_semantic(force: bool = False) -> None:
    if not _begin_training(SEMANTIC_CHECKPOINT, "semantic", force):
        return
    if not TEXT_EMBEDDINGS_PATH.exists():
        extract_embeddings()
    import torch
    import torch.nn.functional as functional

    seed_everything(SETTINGS.semantic_seed)
    _limit_cpu_threads()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = semantic_de_features().reshape(-1, SETTINGS.eeg_channels * 5)
    mean, std = _normalization(features)
    normalized = _normalized_copy(
        features, mean, std, "semantic_de_normalized.float32.npy"
    )
    text = np.load(TEXT_EMBEDDINGS_PATH, mmap_mode="r")
    semantic_targets = _semantic_targets(text)
    dataset = ArrayDataset(
        normalized[: SETTINGS.train_samples], semantic_targets
    )
    loader = _loader(dataset, SETTINGS.predictor_batch_size, SETTINGS.semantic_seed)
    model = SemanticPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS.predictor_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SETTINGS.predictor_epochs * len(loader)
    )
    history: list[dict] = []
    for epoch in range(1, SETTINGS.predictor_epochs + 1):
        started = time.time()
        model.train()
        total_loss = 0.0
        samples = 0
        for feature_batch, target_batch in loader:
            feature_batch = feature_batch.to(device, dtype=torch.float32, non_blocking=True)
            target_batch = target_batch.to(device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(feature_batch)
            loss = functional.mse_loss(prediction, target_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * feature_batch.shape[0]
            samples += feature_batch.shape[0]
        epoch_loss = total_loss / samples
        row = {
            "epoch": epoch,
            "mse": epoch_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"Semantic epoch {epoch:03d}/200 mse={epoch_loss:.8f} "
            f"lr={row['learning_rate']:.10f}",
            flush=True,
        )
        _write_history("semantic", history)
        if epoch % 10 == 0 or epoch == SETTINGS.predictor_epochs:
            _save_checkpoint(
                SEMANTIC_CHECKPOINT, model, epoch, mean, std, "semantic", epoch_loss
            )


def train_dynamic(force: bool = False) -> None:
    if not _begin_training(DYNAMIC_CHECKPOINT, "dynamic_glmnet", force):
        return
    if not FEATURE_WINDOW_EEG_PATH.exists():
        prepare()
    import torch
    import torch.nn.functional as functional

    seed_everything(SETTINGS.predictor_seed)
    _limit_cpu_threads()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    windows = np.load(FEATURE_WINDOW_EEG_PATH, mmap_mode="r")
    # Dataset geometry, not a constant: SEED-DV's 2 s trials carry 3 feature
    # windows, ds005662's 600 ms trials carry 1.  Deriving it here matters --
    # allocating the buffer with the wrong count used to broadcast silently
    # (a (B,1,C,T) batch assigning into a (B,3,C,T) slice), so training
    # "succeeded" on data replicated 3x while inference then failed the
    # reshape.  ``SETTINGS.dynamic_windows`` stays SEED-DV's value for the
    # checkpoint-compatibility check.
    windows_per_clip = int(windows.shape[1])
    train_windows = np.asarray(
        windows[: SETTINGS.train_samples], dtype=np.float32
    ).reshape(-1, SETTINGS.eeg_channels, SETTINGS.dynamic_input_samples)
    mean = train_windows.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = train_windows.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, np.float32(1e-6))
    normalized_path = RUN_ROOT / "dynamic_eeg_normalized.float32.npy"
    normalized = np.lib.format.open_memmap(
        normalized_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(windows),
            windows_per_clip,
            SETTINGS.eeg_channels,
            SETTINGS.dynamic_input_samples,
        ),
    )
    for begin in range(0, len(windows), 32):
        batch = np.asarray(windows[begin : begin + 32], dtype=np.float32)
        if batch.shape[1] != windows_per_clip:
            raise ValueError(
                f"feature-window count changed mid-file: {batch.shape[1]} vs "
                f"{windows_per_clip}; refusing to broadcast")
        normalized[begin : begin + len(batch)] = (batch - mean) / std
    normalized.flush()
    train_inputs = normalized[: SETTINGS.train_samples].reshape(
        -1, 1, SETTINGS.eeg_channels, SETTINGS.dynamic_input_samples
    )
    labels = dynamic_labels()
    dataset = ArrayDataset(
        train_inputs,
        np.repeat(labels[: SETTINGS.train_samples], windows_per_clip),
    )
    loader = _loader(dataset, SETTINGS.dynamic_batch_size, SETTINGS.predictor_seed)
    model = DynamicGLMNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS.predictor_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=SETTINGS.predictor_epochs * len(loader)
    )
    history: list[dict] = []
    for epoch in range(1, SETTINGS.predictor_epochs + 1):
        started = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        samples = 0
        for eeg_batch, label_batch in loader:
            eeg_batch = eeg_batch.to(device, dtype=torch.float32, non_blocking=True)
            label_batch = label_batch.to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(eeg_batch)
            loss = functional.cross_entropy(logits, label_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * eeg_batch.shape[0]
            correct += int((logits.argmax(1) == label_batch).sum())
            samples += eeg_batch.shape[0]
        epoch_loss = total_loss / samples
        row = {
            "epoch": epoch,
            "cross_entropy": epoch_loss,
            "train_accuracy": correct / samples,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"Dynamic GLMNet epoch {epoch:03d}/200 ce={epoch_loss:.8f} "
            f"acc={row['train_accuracy']:.4f} lr={row['learning_rate']:.10f}",
            flush=True,
        )
        _write_history("dynamic", history)
        if epoch % 10 == 0 or epoch == SETTINGS.predictor_epochs:
            _save_checkpoint(
                DYNAMIC_CHECKPOINT, model, epoch, mean, std, "dynamic_glmnet", epoch_loss
            )


def train_all(force: bool = False) -> None:
    ensure_output_roots()
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    train_seq2seq(force=force)
    if ABLATION_NO_SEMANTIC_DYNAMIC:
        marker = CHECKPOINT_ROOT / "predictors_ablation.json"
        marker.write_text(
            json.dumps(
                {
                    "subject": SETTINGS.subject,
                    "semantic_predictor": False,
                    "dynamic_predictor": False,
                    "seq2seq": True,
                    "epochs": SETTINGS.predictor_epochs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            "ablation: Semantic Predictor and DynamicGLMNet disabled; "
            "only Seq2Seq was trained",
            flush=True,
        )
        return
    train_semantic(force=force)
    train_dynamic(force=force)
