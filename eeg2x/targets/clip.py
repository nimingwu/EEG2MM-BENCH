"""Extract ATM image targets from raw ThingsEEG stimuli inside the benchmark."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import h5py
import numpy as np
import torch
from PIL import Image

from eeg2x.protocol import sha256_file
from eeg2x.rawdata import require_benchmark_output
from eeg2x.data import StimulusRow, h5_fingerprint, unique_stimuli, write_json


log = logging.getLogger(__name__)
HF_CACHE = Path("/datas/hf-cache/hub")
MODEL_IDS = {
    "clip_vit_h": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "vae_latent": "stabilityai/stable-diffusion-xl-base-1.0:vae",
    "git_tokens": "microsoft/git-large-coco",
}


def _revision(model_id: str) -> str:
    repo = model_id.split(":", 1)[0]
    path = HF_CACHE / ("models--" + repo.replace("/", "--")) / "refs" / "main"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "unknown"


def _manifest_path(store_path: Path) -> Path:
    return store_path.with_suffix(store_path.suffix + ".manifest.json")


def _json_attr(value: object) -> object:
    """Convert HDF5/NumPy scalar attributes to JSON-native values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _source_identity(paths: Sequence[str | Path]) -> List[Dict[str, object]]:
    return [h5_fingerprint(path) for path in paths]


def _prepare_store(
    store_path: Path,
    stimuli: Sequence[StimulusRow],
    source_h5: Sequence[Dict[str, object]],
    *,
    overwrite: bool,
) -> None:
    if overwrite and store_path.exists():
        store_path.unlink()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    ids = [row.stimulus_id for row in stimuli]
    uris = [row.uri for row in stimuli]
    if store_path.exists():
        with h5py.File(store_path, "r") as h5:
            actual = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in h5["stimulus_id"][:]
            ]
            if actual != ids:
                raise ValueError(
                    f"Existing feature store row order differs: {store_path}; "
                    "use --overwrite only after auditing the target")
            actual_uris = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in h5["stimulus_uri"][:]
            ]
            if actual_uris != uris:
                raise ValueError(f"Existing feature store URIs differ: {store_path}")
            actual_source = json.loads(h5.attrs["source_h5_json"])
            if actual_source != list(source_h5):
                raise ValueError(
                    f"Existing feature store has different source HDF5 inputs: "
                    f"{store_path}")
        return

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(store_path, "w") as h5:
        h5.create_dataset(
            "stimulus_id", data=np.asarray(ids, dtype=object), dtype=string_dtype)
        h5.create_dataset(
            "stimulus_uri", data=np.asarray(uris, dtype=object), dtype=string_dtype)
        h5.create_dataset(
            "stimulus_sha256", data=np.asarray(["" for _ in ids], dtype=object),
            dtype=string_dtype)
        h5.create_dataset(
            "event_label", data=np.asarray([row.label for row in stimuli], dtype=object),
            dtype=string_dtype)
        h5.create_dataset(
            "event_object_number",
            data=np.asarray([row.concept_number for row in stimuli], dtype=np.int32))
        h5.create_dataset(
            "event_stim_number",
            data=np.asarray([row.stimulus_number for row in stimuli], dtype=np.int32))
        h5.attrs["schema"] = "eeg2x-atm-image-features-v1"
        h5.attrs["n_stimuli"] = len(stimuli)
        h5.attrs["source_h5_json"] = json.dumps(source_h5, sort_keys=True)
        h5.attrs["derived_inputs_used"] = False


def _verify_hashes(store_path: Path, stimuli: Sequence[StimulusRow]) -> None:
    """Re-hash every raw image, including rows from a resumed extraction."""
    with h5py.File(store_path, "r+") as h5:
        saved = h5["stimulus_sha256"]
        for index, row in enumerate(stimuli):
            previous = saved[index]
            previous_text = (
                previous.decode("utf-8") if isinstance(previous, bytes) else str(previous))
            digest = sha256_file(row.uri)
            if row.sha256 and row.sha256.lower() != digest:
                raise ValueError(f"Raw stimulus hash mismatch: {row.uri}")
            if previous_text and previous_text.lower() != digest:
                raise ValueError(
                    f"Raw stimulus changed after feature extraction began: {row.uri}")
            if not previous_text:
                saved[index] = digest
            for source_uri in row.source_uris:
                if not Path(source_uri).is_file():
                    raise FileNotFoundError(source_uri)
                sha256_file(source_uri)  # verify the source remains readable
            if (index + 1) % 500 == 0:
                h5.flush()


def _ensure_dataset(
    h5: h5py.File,
    kind: str,
    shape: Sequence[int],
    dtype: np.dtype,
    *,
    overwrite: bool,
) -> h5py.Dataset:
    if kind in h5 and overwrite:
        del h5[kind]
    if kind in h5:
        dataset = h5[kind]
        if tuple(dataset.shape) != tuple(shape):
            raise ValueError(
                f"Existing {kind} shape is {dataset.shape}, expected {tuple(shape)}")
        return dataset
    chunks = (1, *shape[1:]) if len(shape) > 2 else (min(256, shape[0]), shape[1])
    dataset = h5.create_dataset(kind, shape=shape, dtype=dtype, chunks=chunks)
    dataset.attrs["complete"] = False
    dataset.attrs["completed_rows"] = 0
    dataset.attrs["model_id"] = MODEL_IDS[kind]
    dataset.attrs["model_revision"] = _revision(MODEL_IDS[kind])
    dataset.attrs["derived_inputs_used"] = False
    return dataset


def _batches(stimuli: Sequence[StimulusRow], start: int, batch_size: int):
    for first in range(start, len(stimuli), batch_size):
        yield first, stimuli[first:first + batch_size]


def _require_finite(values: torch.Tensor | np.ndarray, kind: str, first: int) -> None:
    """Reject corrupt frozen-model targets before they enter the feature store."""
    finite = torch.isfinite(values).all().item() if isinstance(values, torch.Tensor) \
        else bool(np.isfinite(values).all())
    if not finite:
        raise RuntimeError(
            f"Frozen-model feature {kind!r} contains non-finite values "
            f"in the batch beginning at row {first}")


def _validate_complete_dataset(dataset: h5py.Dataset, kind: str) -> None:
    """Do not trust a stale complete marker without checking stored values."""
    rows_per_chunk = max(1, min(256, 16 * 1024 * 1024 // max(
        int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize, 1)))
    for first in range(0, dataset.shape[0], rows_per_chunk):
        _require_finite(dataset[first:first + rows_per_chunk], kind, first)


def _pil_batch(rows: Sequence[StimulusRow]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for row in rows:
        with Image.open(row.uri) as source:
            images.append(source.convert("RGB"))
    return images


def _extract_clip(
    store_path: Path,
    stimuli: Sequence[StimulusRow],
    *,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
    seed: int,
) -> None:
    import open_clip

    # The release uses the training transform (including RandomResizedCrop) and
    # unnormalised encode_image output. A per-stimulus seed makes that exact
    # transform reproducible when extraction resumes part-way through a store.
    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", precision="fp32",
        device=device, cache_dir=str(HF_CACHE))
    model.eval()

    def transform_uri(uri: str, transform_id: str) -> torch.Tensor:
        row_seed = int.from_bytes(hashlib.sha256(
            f"{seed}:{transform_id}".encode("utf-8")).digest()[:8], "big")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(row_seed)
            with Image.open(uri) as source:
                return preprocess_train(source.convert("RGB"))

    with h5py.File(store_path, "r+") as h5:
        output = _ensure_dataset(
            h5, "clip_vit_h", (len(stimuli), 1024), np.float32,
            overwrite=overwrite)
        seed_scheme = "sha256(seed:stimulus_id)[:8]"
        has_progress = (
            bool(output.attrs.get("complete", False))
            or int(output.attrs.get("completed_rows", 0)) > 0
        )
        if has_progress:
            saved_seed = output.attrs.get("augmentation_seed")
            saved_scheme = output.attrs.get("augmentation_seed_scheme")
            try:
                seed_matches = int(saved_seed) == int(seed)
            except (TypeError, ValueError):
                seed_matches = False
            if not seed_matches or str(_json_attr(saved_scheme)) != seed_scheme:
                raise ValueError(
                    "Existing CLIP features were extracted with a different or "
                    f"unknown augmentation seed policy: {store_path}; use "
                    "--overwrite only after auditing the target"
                )
        output.attrs["model_name"] = "ViT-H-14"
        output.attrs["pretrained"] = "laion2b_s32b_b79k"
        output.attrs["preprocessing"] = (
            "open_clip.preprocess_train:RandomResizedCrop(224,scale=(0.9,1.0))")
        output.attrs["augmentation_seed"] = int(seed)
        output.attrs["augmentation_seed_scheme"] = seed_scheme
        output.attrs["embedding_normalized"] = False
        if bool(output.attrs.get("complete", False)):
            _validate_complete_dataset(output, "clip_vit_h")
            log.info("clip_vit_h is already complete in %s", store_path)
            return
        start = int(output.attrs.get("completed_rows", 0))
        for first, rows in _batches(stimuli, start, batch_size):
            # For an aggregated NOD test class, encode each of the four
            # session images and average their frozen CLIP vectors.  This
            # mirrors the EEG mean across sessions while keeping one candidate
            # row per held-out class.  ThingsEEG rows have a single URI.
            inputs = torch.stack([
                transform_uri(uri, f"{row.stimulus_id}:{j}")
                for row in rows
                for j, uri in enumerate(row.source_uris or (row.uri,))
            ])
            with torch.inference_mode():
                features = model.encode_image(inputs.to(device))
            values_all = features.float().cpu().numpy()
            values_list = []
            offset = 0
            for row in rows:
                count = len(row.source_uris) or 1
                values_list.append(values_all[offset:offset + count].mean(axis=0))
                offset += count
            stop = first + len(rows)
            values = np.stack(values_list, axis=0)
            _require_finite(values, "clip_vit_h", first)
            output[first:stop] = values
            output.attrs["completed_rows"] = stop
            h5.flush()
            log.info("clip_vit_h %d/%d", stop, len(stimuli))
        output.attrs["complete"] = True
    del model, preprocess_train
    torch.cuda.empty_cache()
    gc.collect()


def _extract_vae(
    store_path: Path,
    stimuli: Sequence[StimulusRow],
    *,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> None:
    from diffusers import AutoencoderKL
    from diffusers.image_processor import VaeImageProcessor

    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae",
        variant="fp16", cache_dir=str(HF_CACHE), local_files_only=True,
        # The release loads the fp16 weight variant but performs VAE work in
        # torch.float. Encoding SDXL's VAE in fp16 overflows on ThingsEEG.
        torch_dtype=torch.float32).eval().to(device)
    processor = VaeImageProcessor(vae_scale_factor=8)
    with h5py.File(store_path, "r+") as h5:
        output = _ensure_dataset(
            h5, "vae_latent", (len(stimuli), 4, 64, 64), np.float16,
            overwrite=overwrite)
        output.attrs["latent_scaling"] = "unscaled"
        output.attrs["posterior_statistic"] = "mode"
        output.attrs["image_size"] = 512
        output.attrs["compute_dtype"] = "float32"
        if bool(output.attrs.get("complete", False)):
            _validate_complete_dataset(output, "vae_latent")
            log.info("vae_latent is already complete in %s", store_path)
            return
        start = int(output.attrs.get("completed_rows", 0))
        for first, rows in _batches(stimuli, start, batch_size):
            pixels = processor.preprocess(
                _pil_batch(rows), height=512, width=512).to(
                    device, dtype=torch.float32)
            with torch.inference_mode():
                latents = vae.encode(pixels).latent_dist.mode()
            _require_finite(latents, "vae_latent", first)
            values = latents.float().cpu().numpy().astype(np.float16)
            _require_finite(values, "vae_latent float16 storage", first)
            stop = first + len(rows)
            output[first:stop] = values
            output.attrs["completed_rows"] = stop
            h5.flush()
            log.info("vae_latent %d/%d", stop, len(stimuli))
        output.attrs["complete"] = True
    del vae, processor
    torch.cuda.empty_cache()
    gc.collect()


def _extract_git(
    store_path: Path,
    stimuli: Sequence[StimulusRow],
    *,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> None:
    from transformers import AutoProcessor, GitForCausalLM

    processor = AutoProcessor.from_pretrained(
        "microsoft/git-large-coco", cache_dir=str(HF_CACHE), local_files_only=True)
    model = GitForCausalLM.from_pretrained(
        "microsoft/git-large-coco", cache_dir=str(HF_CACHE),
        local_files_only=True, torch_dtype=torch.float16).eval().to(device)
    vision = model.git.image_encoder
    with h5py.File(store_path, "r+") as h5:
        output = _ensure_dataset(
            h5, "git_tokens", (len(stimuli), 257, 1024), np.float16,
            overwrite=overwrite)
        output.attrs["target"] = "frozen_git_vision_last_hidden_state"
        if bool(output.attrs.get("complete", False)):
            _validate_complete_dataset(output, "git_tokens")
            log.info("git_tokens is already complete in %s", store_path)
            return
        start = int(output.attrs.get("completed_rows", 0))
        for first, rows in _batches(stimuli, start, batch_size):
            pixels = processor(images=_pil_batch(rows), return_tensors="pt").pixel_values
            with torch.inference_mode():
                tokens = vision(pixels.to(device, dtype=torch.float16)).last_hidden_state
            if tokens.shape[1:] != (257, 1024):
                raise RuntimeError(f"Unexpected GIT token shape: {tuple(tokens.shape)}")
            _require_finite(tokens, "git_tokens", first)
            values = tokens.float().cpu().numpy().astype(np.float16)
            _require_finite(values, "git_tokens float16 storage", first)
            stop = first + len(rows)
            output[first:stop] = values
            output.attrs["completed_rows"] = stop
            h5.flush()
            log.info("git_tokens %d/%d", stop, len(stimuli))
        output.attrs["complete"] = True
    del model, vision, processor
    torch.cuda.empty_cache()
    gc.collect()


def _write_manifest(store_path: Path) -> Path:
    with h5py.File(store_path, "r") as h5:
        ids = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in h5["stimulus_id"][:]
        ]
        kinds = {}
        for kind in MODEL_IDS:
            if kind in h5:
                kinds[kind] = {
                    "shape": list(h5[kind].shape),
                    "dtype": str(h5[kind].dtype),
                    "complete": bool(h5[kind].attrs.get("complete", False)),
                    "model_id": _json_attr(
                        h5[kind].attrs.get("model_id", "")),
                    "model_revision": _json_attr(
                        h5[kind].attrs.get("model_revision", "")),
                    "preprocessing": _json_attr(
                        h5[kind].attrs.get("preprocessing", "")),
                    "augmentation_seed": _json_attr(
                        h5[kind].attrs.get("augmentation_seed", "")),
                    "augmentation_seed_scheme": _json_attr(
                        h5[kind].attrs.get("augmentation_seed_scheme", "")),
                    "embedding_normalized": _json_attr(
                        h5[kind].attrs.get("embedding_normalized", "")),
                }
        source_h5 = json.loads(h5.attrs["source_h5_json"])
    row_map_hash = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()
    return write_json(_manifest_path(store_path), {
        "artifact_type": "atm_raw_image_features",
        "store": str(store_path),
        "n_stimuli": len(ids),
        "stimulus_id_to_row": {value: index for index, value in enumerate(ids)},
        "row_map_sha256": row_map_hash,
        "source_h5": source_h5,
        "features": kinds,
        "derived_inputs_used": False,
    })


def extract_features(
    h5_paths: Sequence[str | Path],
    store_path: str | Path,
    *,
    kinds: Sequence[str] = ("clip_vit_h", "vae_latent", "git_tokens"),
    batch_size: int = 16,
    device: str = "cuda:0",
    overwrite: bool = False,
    seed: int = 2000,
) -> Path:
    """Recompute all requested frozen-model targets from raw stimulus images."""
    unknown = sorted(set(kinds) - set(MODEL_IDS))
    if unknown:
        raise ValueError(f"Unknown ATM feature kinds: {unknown}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output = require_benchmark_output(store_path)
    stimuli = unique_stimuli(h5_paths)
    if not stimuli:
        raise ValueError("No raw ThingsEEG stimuli found in the selected HDF5 files")
    source = _source_identity(h5_paths)
    if any(item["derived_inputs_used"] for item in source):
        raise ValueError("A selected HDF5 file declares derived inputs")
    _prepare_store(output, stimuli, source, overwrite=overwrite)
    _verify_hashes(output, stimuli)
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for feature extraction but is unavailable")
    extractors = {
        "clip_vit_h": _extract_clip,
        "vae_latent": _extract_vae,
        "git_tokens": _extract_git,
    }
    for kind in kinds:
        kwargs = {
            "batch_size": batch_size,
            "device": target_device,
            "overwrite": overwrite,
        }
        if kind == "clip_vit_h":
            kwargs["seed"] = seed
        extractors[kind](output, stimuli, **kwargs)
        _write_manifest(output)
    return _write_manifest(output)
