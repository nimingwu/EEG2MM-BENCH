"""Training and benchmark prediction routines for the ATM paper pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from eeg2x.protocol import sha256_file
from eeg2x.rawdata import require_benchmark_output
from eeg2x.data import (
    ATMH5Dataset,
    ATM_PROFILE,
    LUNA_ATM_PROFILE,
    LUNA_NICE_PROFILE,
    NICE_PROFILE,
    NOD_ATM_PROFILE,
    NOD_REVE_ATM_PROFILE,
    NOD_LUNA_ATM_PROFILE,
    NOD_LABRAM_ATM_PROFILE,
    REVE_ATM_PROFILE,
    REVE_NICE_PROFILE,
    FeatureStore,
    artifact_manifest,
    discover_subject_h5,
    h5_fingerprint,
    sample_id_for_h5_row,
    write_json,
)
from eeg2x.baselines.atm_full.diffusion_prior import DiffusionPriorUNet
from eeg2x.frameworks.only_clip.low_level import DEFAULT_LOW_LEVEL_EPOCHS, LowLevelEEGEncoder
from eeg2x.baselines.atm_full.model import ATMS
from eeg2x.frameworks.only_clip.model import (
    generation_alignment_loss,
    retrieval_alignment_loss,
)
from eeg2x.frameworks.only_clip.semantic import DEFAULT_SEMANTIC_EPOCHS, PixelProjector


log = logging.getLogger(__name__)
DEFAULT_ATM_EPOCHS = 40
DEFAULT_ATM_BATCH_SIZE = 64
DEFAULT_PRIOR_EPOCHS = 150
DEFAULT_PRIOR_BATCH_SIZE = 1024


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


def _atomic_torch_save(value: object, path: str | Path) -> Path:
    output = require_benchmark_output(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, output)
    return output


def _checkpoint_manifest(
    checkpoint: Path,
    *,
    artifact_type: str,
    source_h5: Sequence[Dict[str, object]],
    config: Dict[str, object],
    extra: Dict[str, object] | None = None,
) -> Path:
    value = artifact_manifest(
        artifact_type=artifact_type, source_h5=source_h5,
        config=config, extra={
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            **(extra or {}),
        })
    return write_json(checkpoint.with_suffix(checkpoint.suffix + ".manifest.json"), value)


def load_state_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    expected_type: str | None = None,
) -> Dict[str, object]:
    checkpoint = require_benchmark_output(checkpoint_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid ATM checkpoint: {checkpoint}")
    if expected_type is not None and payload.get("artifact_type") != expected_type:
        raise ValueError(
            f"Checkpoint {checkpoint} is {payload.get('artifact_type')!r}, "
            f"expected {expected_type!r}")
    if bool(payload.get("derived_inputs_used", True)):
        raise ValueError(f"Checkpoint used forbidden external derived inputs: {checkpoint}")
    model.load_state_dict(payload["model_state"])
    return payload


def load_atm_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_type: str,
) -> tuple[ATMS, Dict[str, object]]:
    """Rebuild the saved ATM encoder variant before restoring its state."""
    checkpoint = require_benchmark_output(checkpoint_path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid ATM checkpoint: {checkpoint}")
    if payload.get("artifact_type") != expected_type:
        raise ValueError(
            f"Checkpoint {checkpoint} is {payload.get('artifact_type')!r}, "
            f"expected {expected_type!r}")
    if bool(payload.get("derived_inputs_used", True)):
        raise ValueError(f"Checkpoint used forbidden external derived inputs: {checkpoint}")
    saved = payload.get("model_config", {"encoder": "atm_s", "encoder_config": {}})
    if not isinstance(saved, dict):
        raise ValueError(f"ATM checkpoint has invalid model_config: {checkpoint}")
    encoder = str(saved.get("encoder", "atm_s"))
    encoder_config = saved.get("encoder_config", {})
    if not isinstance(encoder_config, dict):
        raise ValueError(f"ATM checkpoint has invalid encoder_config: {checkpoint}")
    if encoder in {"reve", "luna"}:
        pretrained = saved.get("pretrained", {})
        if not isinstance(pretrained, dict) or not bool(
                pretrained.get("pretrained_loaded", False)):
            raise ValueError(
                f"{encoder.upper()} checkpoint does not prove pretrained "
                f"initialization: {checkpoint}")
    model = ATMS(encoder=encoder, encoder_config=encoder_config)
    model.load_state_dict(payload["model_state"])
    return model, payload


@dataclass(frozen=True)
class ATMTrainConfig:
    subject_id: str
    mode: str
    profile_id: str
    epochs: int = DEFAULT_ATM_EPOCHS
    batch_size: int = DEFAULT_ATM_BATCH_SIZE
    learning_rate: float = 3e-4
    seed: int = 2000
    encoder: str = "atm_s"
    model_config: Dict[str, object] | None = None


def _h5_encoder_contract(paths: Sequence[Path]) -> Dict[str, object]:
    channel_names: List[str] | None = None
    n_samples: int | None = None
    sampling_rate: float | None = None
    for path in paths:
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


def _resolve_profile(mode: str, encoder: str, profile_id: str | None) -> str:
    if mode not in ("retrieval", "generation"):
        raise ValueError(f"Unsupported ATM training mode: {mode!r}")
    if profile_id is None:
        if encoder == "reve":
            return REVE_NICE_PROFILE if mode == "retrieval" else REVE_ATM_PROFILE
        if encoder == "luna":
            return LUNA_NICE_PROFILE if mode == "retrieval" else LUNA_ATM_PROFILE
        if encoder == "labram":
            return NOD_LABRAM_ATM_PROFILE if mode == "generation" else NICE_PROFILE
        return NICE_PROFILE if mode == "retrieval" else ATM_PROFILE
    valid = {
        "retrieval": {NICE_PROFILE, REVE_NICE_PROFILE, LUNA_NICE_PROFILE},
        "generation": {
            ATM_PROFILE, REVE_ATM_PROFILE, LUNA_ATM_PROFILE, NOD_ATM_PROFILE,
            NOD_REVE_ATM_PROFILE, NOD_LUNA_ATM_PROFILE,
            NOD_LABRAM_ATM_PROFILE,
        },
    }
    if profile_id not in valid[mode]:
        raise ValueError(f"Profile {profile_id!r} is not valid for ATM {mode}")
    if encoder == "reve" and profile_id not in {
            REVE_NICE_PROFILE, REVE_ATM_PROFILE, NOD_REVE_ATM_PROFILE}:
        raise ValueError("REVE must use an explicit 200 Hz REVE profile")
    if encoder == "luna" and profile_id not in {
            LUNA_NICE_PROFILE, LUNA_ATM_PROFILE, NOD_LUNA_ATM_PROFILE}:
        raise ValueError("LUNA must use an explicit 256 Hz padded LUNA profile")
    return profile_id


def _atm_loss(model: ATMS, predicted: Tensor, target: Tensor, mode: str) -> Tensor:
    if mode == "retrieval":
        target = F.normalize(target, dim=-1)
        return retrieval_alignment_loss(
            predicted, target, model.logit_scale, model.loss_func)
    if mode == "generation":
        return generation_alignment_loss(
            predicted, target, model.logit_scale, model.loss_func)
    raise ValueError(f"Unsupported ATM training mode: {mode!r}")


@torch.inference_mode()
def evaluate_atm(model: ATMS, dataset: ATMH5Dataset, mode: str,
                 device: torch.device, batch_size: int = 64) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predicted_batches: List[Tensor] = []
    target_batches: List[Tensor] = []
    losses: List[float] = []
    for batch in loader:
        predicted = model(
            batch["eeg"].to(device), batch["subject_id"].to(device)).float()
        target = batch["target"].to(device).float()
        losses.append(float(_atm_loss(model, predicted, target, mode).item()))
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
    return {
        "loss": float(np.mean(losses)),
        "top1": float(top1.item()),
        "top5": float(top5.item()),
    }


def train_atm(
    *,
    subject_id: str,
    h5_dir: str | Path,
    feature_store: str | Path,
    output_dir: str | Path,
    mode: str,
    epochs: int = DEFAULT_ATM_EPOCHS,
    batch_size: int = DEFAULT_ATM_BATCH_SIZE,
    learning_rate: float = 3e-4,
    device: str = "cuda:0",
    seed: int = 2000,
    cache_eeg: bool = True,
    encoder: str = "atm_s",
    encoder_config: Dict[str, object] | None = None,
    profile_id: str | None = None,
) -> Path:
    """Train the released ATM objective with a protocol-declared EEG encoder."""
    if epochs <= 0 or batch_size <= 1:
        raise ValueError("ATM epochs must be positive and batch_size must exceed one")
    profile_id = _resolve_profile(mode, encoder, profile_id)
    output = require_benchmark_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=profile_id, split="train")
    test_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=profile_id, split="test")
    input_contract = _h5_encoder_contract(train_paths + test_paths)
    requested_encoder_config = dict(encoder_config or {})
    supplied_names = requested_encoder_config.get("channel_names")
    if supplied_names is not None and supplied_names != input_contract["channel_names"]:
        raise ValueError("Configured encoder channel_names disagree with HDF5")
    requested_encoder_config.update(input_contract)
    seed_everything(seed)
    model = ATMS(
        encoder=encoder, encoder_config=requested_encoder_config)
    model_config = model.checkpoint_model_config()
    config = ATMTrainConfig(
        subject_id=subject_id, mode=mode, profile_id=profile_id,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        seed=seed, encoder=encoder, model_config=model_config)
    train_data = ATMH5Dataset(
        train_paths, feature_store, profile_id=profile_id,
        cache_eeg=cache_eeg, cache_features=True)
    test_data = ATMH5Dataset(
        test_paths, feature_store, profile_id=profile_id,
        cache_eeg=True, cache_features=True)
    target_device = _device(device)
    model = model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True, generator=generator)
    history: List[Dict[str, float | int]] = []
    sources = train_data.fingerprints + test_data.fingerprints
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            n_batches = 0
            for batch in loader:
                eeg = batch["eeg"].to(target_device, non_blocking=True)
                subject = batch["subject_id"].to(target_device, non_blocking=True)
                target = batch["target"].to(target_device, non_blocking=True).float()
                optimizer.zero_grad(set_to_none=True)
                predicted = model(eeg, subject).float()
                loss = _atm_loss(model, predicted, target, mode)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item())
                n_batches += 1
            metrics = evaluate_atm(model, test_data, mode, target_device, batch_size)
            row: Dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": loss_sum / max(n_batches, 1),
                "test_loss": metrics["loss"],
                "test_top1": metrics["top1"],
                "test_top5": metrics["top5"],
            }
            history.append(row)
            write_json(output / "training_history.json", history)
            log.info(
                "%s ATM epoch %d/%d train=%.5f test=%.5f top1=%.4f top5=%.4f",
                mode, epoch, epochs, row["train_loss"], row["test_loss"],
                row["test_top1"], row["test_top5"])
            if epoch % 5 == 0:
                _atomic_torch_save({
                    "artifact_type": f"atm_{mode}",
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "config": asdict(config),
                    "model_config": model_config,
                    "source_h5": sources,
                    "feature_store": str(require_benchmark_output(feature_store)),
                    "derived_inputs_used": False,
                }, output / "checkpoints" / f"epoch-{epoch:03d}.pt")
        checkpoint = _atomic_torch_save({
            "artifact_type": f"atm_{mode}",
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epochs,
            "config": asdict(config),
            "model_config": model_config,
            "source_h5": sources,
            "feature_store": str(require_benchmark_output(feature_store)),
            "metrics": history[-1],
            "derived_inputs_used": False,
        }, output / "final.pt")
        _checkpoint_manifest(
            checkpoint, artifact_type=f"atm_{mode}", source_h5=sources,
            config=asdict(config), extra={
                "feature_store": str(require_benchmark_output(feature_store)),
                "final_metrics": history[-1],
            })
        return checkpoint
    finally:
        train_data.close()
        test_data.close()


def predict_retrieval(
    *,
    subject_id: str,
    h5_dir: str | Path,
    feature_store: str | Path,
    checkpoint: str | Path,
    output_json: str | Path,
    batch_size: int = 64,
    device: str = "cuda:0",
    profile_id: str | None = None,
) -> Path:
    """Write protocol predictions with the test concepts as candidate authority."""
    model, checkpoint_payload = load_atm_checkpoint(
        checkpoint, expected_type="atm_retrieval")
    saved_config = checkpoint_payload.get("config", {})
    if profile_id is None:
        profile_id = str(saved_config.get("profile_id", NICE_PROFILE))
    test_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=profile_id, split="test")
    dataset = ATMH5Dataset(
        test_paths, feature_store, profile_id=profile_id,
        cache_eeg=True, cache_features=True)
    target_device = _device(device)
    model.eval().to(target_device)
    candidates = torch.stack([
        dataset[index]["target"] for index in range(len(dataset))])
    candidates = F.normalize(candidates.float(), dim=-1).to(target_device)
    predictions: Dict[str, Dict[str, List[int]]] = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    offset = 0
    try:
        with torch.inference_mode():
            for batch in loader:
                eeg_features = model(
                    batch["eeg"].to(target_device),
                    batch["subject_id"].to(target_device)).float()
                ranked = (
                    model.logit_scale * eeg_features @ candidates.T
                ).argsort(dim=1, descending=True).cpu().tolist()
                for local_row, indices in enumerate(ranked):
                    sample_id = sample_id_for_h5_row(
                        test_paths[0], offset + local_row)
                    predictions[sample_id] = {"ranked_indices": indices}
                offset += len(ranked)
        value = {
            "candidate_pool_ids": dataset.labels(),
            "predictions": predictions,
            "provenance": {
                "model": {
                    "reve": "atm_reve", "luna": "atm_luna",
                }.get(model.encoder_name, "atm_s"),
                "model_config": model.checkpoint_model_config(),
                "checkpoint": str(require_benchmark_output(checkpoint)),
                "source_h5": dataset.fingerprints,
                "feature_store": str(require_benchmark_output(feature_store)),
                "derived_inputs_used": False,
            },
        }
        return write_json(output_json, value)
    finally:
        dataset.close()


def extract_atm_embeddings(
    *,
    subject_id: str,
    h5_dir: str | Path,
    feature_store: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    batch_size: int = 256,
    device: str = "cuda:0",
    profile_id: str | None = None,
) -> Dict[str, Path]:
    """Derive generation-ATM embeddings from benchmark HDF5, never old PT files."""
    output = require_benchmark_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model, checkpoint_payload = load_atm_checkpoint(
        checkpoint, expected_type="atm_generation")
    saved_config = checkpoint_payload.get("config", {})
    if profile_id is None:
        profile_id = str(saved_config.get("profile_id", ATM_PROFILE))
    target_device = _device(device)
    model.eval().to(target_device)
    written: Dict[str, Path] = {}
    for split in ("train", "test"):
        paths = discover_subject_h5(
            h5_dir, subject_id, profile_id=profile_id, split=split)
        dataset = ATMH5Dataset(
            paths, feature_store, profile_id=profile_id,
            cache_eeg=True, cache_features=True)
        destination = output / f"{split}_embeddings.h5"
        temporary = destination.with_suffix(".h5.tmp")
        if temporary.exists():
            temporary.unlink()
        string_dtype = h5py.string_dtype(encoding="utf-8")
        try:
            with h5py.File(temporary, "w") as h5:
                eeg_ds = h5.create_dataset(
                    "eeg_embedding", shape=(len(dataset), 1024), dtype=np.float16,
                    chunks=(min(256, len(dataset)), 1024))
                clip_ds = h5.create_dataset(
                    "clip_target", shape=(len(dataset), 1024), dtype=np.float16,
                    chunks=(min(256, len(dataset)), 1024))
                ids = h5.create_dataset(
                    "stimulus_id", shape=(len(dataset),), dtype=string_dtype)
                labels = h5.create_dataset(
                    "event_label", shape=(len(dataset),), dtype=string_dtype)
                loader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                offset = 0
                with torch.inference_mode():
                    for batch in loader:
                        embeddings = model(
                            batch["eeg"].to(target_device),
                            batch["subject_id"].to(target_device)).float().cpu().numpy()
                        stop = offset + len(embeddings)
                        eeg_ds[offset:stop] = embeddings.astype(np.float16)
                        clip_ds[offset:stop] = batch["target"].numpy().astype(np.float16)
                        ids[offset:stop] = np.asarray(batch["stimulus_id"], dtype=object)
                        labels[offset:stop] = np.asarray(batch["label"], dtype=object)
                        offset = stop
                h5.attrs["schema"] = "eeg2x-atm-eeg-embeddings-v1"
                h5.attrs["subject_id"] = subject_id
                h5.attrs["split"] = split
                h5.attrs["source_h5_json"] = json.dumps(dataset.fingerprints, sort_keys=True)
                h5.attrs["atm_checkpoint"] = str(require_benchmark_output(checkpoint))
                h5.attrs["atm_checkpoint_sha256"] = sha256_file(checkpoint)
                h5.attrs["model_config_json"] = json.dumps(
                    model.checkpoint_model_config(), sort_keys=True)
                h5.attrs["derived_inputs_used"] = False
            os.replace(temporary, destination)
            write_json(destination.with_suffix(".h5.manifest.json"), artifact_manifest(
                artifact_type="atm_eeg_embeddings",
                source_h5=dataset.fingerprints,
                config={
                    "subject_id": subject_id, "split": split,
                    "batch_size": batch_size, "profile_id": profile_id,
                    "model_config": model.checkpoint_model_config(),
                },
                extra={
                    "artifact": str(destination),
                    "artifact_sha256": sha256_file(destination),
                    "atm_checkpoint": str(require_benchmark_output(checkpoint)),
                    "feature_store": str(require_benchmark_output(feature_store)),
                }))
            written[split] = destination
        finally:
            if temporary.exists():
                temporary.unlink()
            dataset.close()
    return written


class EmbeddingH5Dataset(Dataset):
    def __init__(self, path: str | Path, *, cache: bool = True):
        self.path = require_benchmark_output(path)
        with h5py.File(self.path, "r") as h5:
            if bool(h5.attrs.get("derived_inputs_used", True)):
                raise ValueError(f"Invalid ATM embedding provenance: {self.path}")
            self.source_h5 = json.loads(h5.attrs["source_h5_json"])
            self.length = len(h5["eeg_embedding"])
            if cache:
                self.conditions = np.asarray(h5["eeg_embedding"][:], dtype=np.float32)
                self.targets = np.asarray(h5["clip_target"][:], dtype=np.float32)
            else:
                self.conditions = None
                self.targets = None
        self.handle: h5py.File | None = None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        if self.conditions is not None:
            condition = self.conditions[index]
            target = self.targets[index]
        else:
            if self.handle is None:
                self.handle = h5py.File(self.path, "r")
            condition = np.asarray(self.handle["eeg_embedding"][index], dtype=np.float32)
            target = np.asarray(self.handle["clip_target"][index], dtype=np.float32)
        return {
            "condition": torch.from_numpy(condition),
            "target": torch.from_numpy(target),
        }


def train_prior(
    *,
    train_embeddings: str | Path,
    output_dir: str | Path,
    epochs: int = DEFAULT_PRIOR_EPOCHS,
    batch_size: int = DEFAULT_PRIOR_BATCH_SIZE,
    learning_rate: float = 1e-3,
    condition_dropout: float = 0.1,
    device: str = "cuda:0",
    seed: int = 2000,
) -> Path:
    """Train the released conditional DDPM prior in CLIP embedding space."""
    from diffusers import DDPMScheduler
    from diffusers.optimization import get_cosine_schedule_with_warmup

    if epochs <= 0 or not 0 <= condition_dropout < 1:
        raise ValueError("Invalid prior training configuration")
    output = require_benchmark_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = EmbeddingH5Dataset(train_embeddings, cache=True)
    seed_everything(seed)
    target_device = _device(device)
    model = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(target_device)
    scheduler = DDPMScheduler()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True, generator=torch.Generator().manual_seed(seed))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=500,
        num_training_steps=len(loader) * epochs)
    config = {
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate,
        "condition_dropout": condition_dropout,
        "scheduler": "DDPMScheduler", "seed": seed,
    }
    history: List[Dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch in loader:
            condition = batch["condition"].to(target_device)
            target = batch["target"].to(target_device)
            # The released code drops the condition for an entire minibatch.
            used_condition = None if torch.rand((), device=target_device) < condition_dropout else condition
            noise = torch.randn_like(target)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (len(target),),
                device=target_device)
            perturbed = scheduler.add_noise(target, noise, timesteps)
            optimizer.zero_grad(set_to_none=True)
            predicted_noise = model(perturbed, timesteps, used_condition)
            loss = F.mse_loss(predicted_noise, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            loss_sum += float(loss.item())
        row: Dict[str, float | int] = {
            "epoch": epoch, "train_loss": loss_sum / max(len(loader), 1)}
        history.append(row)
        write_json(output / "training_history.json", history)
        log.info("prior epoch %d/%d loss=%.6f", epoch, epochs, row["train_loss"])
        if epoch % 25 == 0:
            _atomic_torch_save({
                "artifact_type": "atm_diffusion_prior",
                "model_state": model.state_dict(), "epoch": epoch,
                "config": config, "source_h5": data.source_h5,
                "embedding_artifact": str(data.path),
                "derived_inputs_used": False,
            }, output / "checkpoints" / f"epoch-{epoch:03d}.pt")
    checkpoint = _atomic_torch_save({
        "artifact_type": "atm_diffusion_prior",
        "model_state": model.state_dict(), "epoch": epochs,
        "config": config, "source_h5": data.source_h5,
        "embedding_artifact": str(data.path),
        "metrics": history[-1], "derived_inputs_used": False,
    }, output / "final.pt")
    _checkpoint_manifest(
        checkpoint, artifact_type="atm_diffusion_prior",
        source_h5=data.source_h5, config=config,
        extra={"embedding_artifact": str(data.path), "final_metrics": history[-1]})
    return checkpoint


def train_low_level(
    *,
    subject_id: str,
    h5_dir: str | Path,
    feature_store: str | Path,
    output_dir: str | Path,
    epochs: int = DEFAULT_LOW_LEVEL_EPOCHS,
    batch_size: int = 30,
    learning_rate: float = 1e-3,
    device: str = "cuda:0",
    seed: int = 2000,
    cache_eeg: bool = True,
    profile_id: str = ATM_PROFILE,
) -> Path:
    """Train the paper's direct EEG-to-unscaled-SDXL-latent decoder."""
    if epochs <= 0:
        raise ValueError("Low-Level epochs must be positive")
    output = require_benchmark_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=profile_id, split="train")
    test_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=profile_id, split="test")
    train_data = ATMH5Dataset(
        train_paths, feature_store, profile_id=profile_id,
        feature_kind="vae_latent", cache_eeg=cache_eeg, cache_features=True)
    test_data = ATMH5Dataset(
        test_paths, feature_store, profile_id=profile_id,
        feature_kind="vae_latent", cache_eeg=True, cache_features=True)
    seed_everything(seed)
    target_device = _device(device)
    model = LowLevelEEGEncoder(
        n_chans=int(train_data[0]["eeg"].shape[0]),
        n_samples=int(train_data[0]["eeg"].shape[1])).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_data, batch_size=20, shuffle=False, num_workers=0)
    config = {
        "subject_id": subject_id, "epochs": epochs,
        "default_epochs": DEFAULT_LOW_LEVEL_EPOCHS,
        "batch_size": batch_size, "learning_rate": learning_rate,
        "loss": "MAE", "scheduler": "CosineAnnealingLR",
        "latent_scaling": "unscaled", "seed": seed,
    }
    history: List[Dict[str, float | int]] = []
    sources = train_data.fingerprints + test_data.fingerprints
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss = 0.0
            for batch in train_loader:
                eeg = batch["eeg"].to(target_device)
                target = batch["target"].to(target_device).float()
                optimizer.zero_grad(set_to_none=True)
                predicted = model(eeg).float()
                loss = F.l1_loss(predicted, target)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
            scheduler.step()
            model.eval()
            test_loss = 0.0
            with torch.inference_mode():
                for batch in test_loader:
                    predicted = model(batch["eeg"].to(target_device)).float()
                    test_loss += float(F.l1_loss(
                        predicted, batch["target"].to(target_device).float()).item())
            row: Dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss / max(len(train_loader), 1),
                "test_loss": test_loss / max(len(test_loader), 1),
            }
            history.append(row)
            write_json(output / "training_history.json", history)
            log.info(
                "low-level epoch %d/%d train=%.6f test=%.6f",
                epoch, epochs, row["train_loss"], row["test_loss"])
            if epoch % 5 == 0:
                _atomic_torch_save({
                    "artifact_type": "atm_low_level",
                    "model_state": model.state_dict(), "epoch": epoch,
                    "config": config, "source_h5": sources,
                    "feature_store": str(require_benchmark_output(feature_store)),
                    "derived_inputs_used": False,
                }, output / "checkpoints" / f"epoch-{epoch:03d}.pt")
        checkpoint = _atomic_torch_save({
            "artifact_type": "atm_low_level",
            "model_state": model.state_dict(), "epoch": epochs,
            "config": config, "source_h5": sources,
            "feature_store": str(require_benchmark_output(feature_store)),
            "metrics": history[-1], "derived_inputs_used": False,
        }, output / "final.pt")
        _checkpoint_manifest(
            checkpoint, artifact_type="atm_low_level", source_h5=sources,
            config=config, extra={
                "feature_store": str(require_benchmark_output(feature_store)),
                "final_metrics": history[-1],
            })
        return checkpoint
    finally:
        train_data.close()
        test_data.close()


class _IndexedArrays(Dataset):
    def __init__(self, first: np.ndarray, second: np.ndarray, indices: Sequence[int]):
        self.first = first
        self.second = second
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor]:
        row = self.indices[item]
        return (
            torch.from_numpy(np.asarray(self.first[row], dtype=np.float32)).unsqueeze(0),
            torch.from_numpy(np.asarray(self.second[row], dtype=np.float32)),
        )


def _feature_rows_for_h5(store: FeatureStore, paths: Iterable[str | Path]) -> List[int]:
    rows: List[int] = []
    seen: set[str] = set()
    for path_value in paths:
        path = require_benchmark_output(path_value)
        with h5py.File(path, "r") as h5:
            for value in h5["stimulus_id"][:]:
                stimulus_id = value.decode() if isinstance(value, bytes) else str(value)
                if stimulus_id not in seen:
                    rows.append(store.row(stimulus_id))
                    seen.add(stimulus_id)
    return rows


def train_semantic(
    *,
    subject_id: str,
    h5_dir: str | Path,
    feature_store: str | Path,
    output_dir: str | Path,
    epochs: int = DEFAULT_SEMANTIC_EPOCHS,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cuda:0",
    seed: int = 2000,
    profile_id: str = ATM_PROFILE,
) -> Path:
    """Train the released unique-image CLIP-to-GIT token projector."""
    output = require_benchmark_output(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=profile_id, split="train")
    test_paths = discover_subject_h5(
        h5_dir, subject_id, profile_id=profile_id, split="test")
    store = FeatureStore(feature_store, ("clip_vit_h", "git_tokens"))
    train_rows = _feature_rows_for_h5(store, train_paths)
    test_rows = _feature_rows_for_h5(store, test_paths)
    with h5py.File(store.path, "r") as h5:
        clip_features = np.asarray(h5["clip_vit_h"][:], dtype=np.float32)
        # Retain float16 in host RAM; each minibatch is converted before transfer.
        git_features = np.asarray(h5["git_tokens"][:])
    train_data = _IndexedArrays(clip_features, git_features, train_rows)
    test_data = _IndexedArrays(clip_features, git_features, test_rows)
    seed_everything(seed)
    target_device = _device(device)
    dtype = torch.bfloat16 if target_device.type == "cuda" else torch.float32
    model = PixelProjector().to(device=target_device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=0,
        drop_last=True, generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    config = {
        "subject_id": subject_id, "epochs": epochs,
        "batch_size": batch_size, "learning_rate": learning_rate,
        "loss": "MSE", "input": "unique_raw_image_clip_vit_h",
        "target": "frozen_git_vision_tokens", "seed": seed,
        "profile_id": profile_id,
    }
    history: List[Dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(target_device, dtype=dtype)
            targets = targets.to(target_device, dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
        row: Dict[str, float | int] = {
            "epoch": epoch, "train_loss": loss_sum / max(len(train_loader), 1)}
        history.append(row)
        write_json(output / "training_history.json", history)
        log.info("semantic epoch %d/%d loss=%.6f", epoch, epochs, row["train_loss"])
    model.eval()
    test_loss = 0.0
    with torch.inference_mode():
        for inputs, targets in test_loader:
            test_loss += float(F.mse_loss(
                model(inputs.to(target_device, dtype=dtype)),
                targets.to(target_device, dtype=dtype)).item())
    final_metrics = {**history[-1], "test_loss": test_loss / max(len(test_loader), 1)}
    sources = [h5_fingerprint(path) for path in train_paths + test_paths]
    checkpoint = _atomic_torch_save({
        "artifact_type": "atm_semantic_projector",
        "model_state": model.float().state_dict(), "epoch": epochs,
        "config": config, "source_h5": sources,
        "feature_store": str(store.path), "metrics": final_metrics,
        "derived_inputs_used": False,
    }, output / "final.pt")
    _checkpoint_manifest(
        checkpoint, artifact_type="atm_semantic_projector",
        source_h5=sources, config=config, extra={
            "feature_store": str(store.path), "final_metrics": final_metrics})
    store.close()
    return checkpoint
