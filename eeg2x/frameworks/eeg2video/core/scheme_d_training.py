"""Training and loading utilities for Scheme D.

Scheme D deliberately lives beside, rather than inside, the paper-literal
predictor trainer:

Stage 1:
    normalized time-domain EEG [B, 7, 62, 100]
    -> EEGNet applied independently to the seven windows
    -> mean over windows
    -> trainable projection head
    -> predicted CLIP context [B, 77, 768]
    -> MSE against frozen CLIP text embeddings

Stage 2:
    frozen Stage-1 predictor
    -> tokenwise Linear(768, 768)
    -> MSE against the same CLIP text embeddings

The target extraction and data split remain those of the cleaned EEG2Video
project: blocks 1--6 are training and block 7 is held out for evaluation.

Encoder-swappable extension (EEG2X-Benchmark-v2, additive)
----------------------------------------------------------
The EEG2Video benchmark swaps Scheme D's EEGNet for a pretrained foundation
model (LaBraM / REVE / LUNA).  Rather than fork this file, the three entry
points take two optional arguments:

* ``predictor_factory`` -- a zero-argument callable returning the predictor
  module.  Defaults to :class:`SchemeDSemanticPredictor`, so every existing
  caller keeps the original EEGNet model and the original arithmetic.
* ``predictor_meta`` -- a dict describing the encoder, stored inside the
  checkpoint next to the weights.

``MODEL_INPUT_NORMALIZATION`` is the second axis of the branch: when the
loader's preprocessing profile has already produced model-ready data (LaBraM's
amplitude scaling, LUNA's resampling), it is set to ``"none"`` and this file's
train-split standardisation -- which would otherwise cancel that scaling -- is
skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup

from .config import CHECKPOINT_ROOT, RUN_ROOT, SETTINGS
from .data import (
    TEXT_EMBEDDINGS_PATH,
    WINDOW_EEG_PATH,
    ArrayDataset,
    ensure_output_roots,
    extract_embeddings,
    prepare,
    seed_everything,
)
from .scheme_d import (
    CLIP_HIDDEN_SIZE,
    CLIP_SEQUENCE_LENGTH,
    SchemeDLinearAdapter,
    SchemeDSemanticPredictor,
    scheme_d_mse,
)


SCHEME_ROOT = CHECKPOINT_ROOT
STAGE_ONE_CHECKPOINT = SCHEME_ROOT / "stage_one.pt"
STAGE_TWO_CHECKPOINT = SCHEME_ROOT / "stage_two_linear.pt"


def _complete_path(path: Path) -> Path:
    return path.with_suffix(".complete.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _is_complete(path: Path, task: str, epochs: int, max_train_samples: int | None) -> bool:
    marker_path = _complete_path(path)
    if not path.exists() or not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_samples = SETTINGS.train_samples if max_train_samples is None else int(max_train_samples)
    return (
        marker.get("task") == task
        and marker.get("epoch") == epochs
        and marker.get("requested_epochs") == epochs
        and marker.get("train_samples") == expected_samples
        and marker.get("scheme") == "D"
        and marker.get("checkpoint_bytes") == path.stat().st_size
    )


#: How the model should treat the arrays the loader wrote.
#:
#: ``zscore_train_split`` -- Scheme D's own standardisation, fitted on the
#:     training split and stored in the checkpoint.  Used by EEGNet and REVE.
#: ``none`` -- the preprocessing branch already produced a model-ready
#:     representation (LaBraM's amplitude scaling, LUNA's resampling), so a
#:     second standardisation would cancel it.
MODEL_INPUT_NORMALIZATION = os.environ.get(
    "EEG2VIDEO_MODEL_INPUT_NORMALIZATION", "zscore_train_split"
)


def _window_spec_of(predictor_meta: dict[str, Any] | None):
    """Window geometry described by the encoder's checkpoint metadata."""

    if not predictor_meta:
        return None
    keys = ("n_windows", "window_samples", "n_chans")
    if all(k in predictor_meta for k in keys[:2]):
        return {k: predictor_meta[k] for k in keys if k in predictor_meta}
    return None


def _resolve_normalization(mode: str | None) -> str:
    mode = mode or MODEL_INPUT_NORMALIZATION
    if mode not in ("zscore_train_split", "none"):
        raise ValueError(
            f"model_input_normalization must be 'zscore_train_split' or "
            f"'none', got {mode!r}"
        )
    return mode


def _resolve_train_samples(value: int | None) -> int:
    """Rows belonging to the train split (the paper's constant by default)."""

    return int(SETTINGS.train_samples) if value is None else int(value)


def _normalization(values: np.ndarray, mode: str | None = None,
                   train_samples: int | None = None,
                   ) -> tuple[np.ndarray, np.ndarray]:
    if _resolve_normalization(mode) == "none":
        width = values.shape[1:]
        return np.zeros((1,) + width, np.float32), np.ones((1,) + width, np.float32)
    train = np.asarray(values[: _resolve_train_samples(train_samples)], dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True).astype(np.float32)
    std = train.std(axis=0, keepdims=True).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-6))


def _normalized_eeg(eeg: np.ndarray, mode: str | None = None,
                    train_samples: int | None = None,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _resolve_normalization(mode) == "none":
        # Identity: the branch already normalised. Mean 0 / std 1 are still
        # recorded so inference reproduces exactly this (no-op) transform.
        mean, std = _normalization(eeg, mode, train_samples)
        return eeg, mean, std

    mean, std = _normalization(eeg, mode, train_samples)
    path = RUN_ROOT / "scheme_d_eeg_normalized.float32.npy"
    expected = (len(eeg),) + tuple(eeg.shape[1:])
    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if existing.shape == expected and existing.dtype == np.float32:
            return existing, mean, std
    output = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=expected
    )
    for begin in range(0, len(eeg), 32):
        chunk = np.asarray(eeg[begin : begin + 32], dtype=np.float32)
        output[begin : begin + len(chunk)] = (chunk - mean) / std
    output.flush()
    return output, mean, std


def _loader(dataset, batch_size: int, seed: int):
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def _ensure_targets(window_spec: dict[str, Any] | None = None,
                    expected_rows: int | None = None,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Load the windowed EEG and the frozen CLIP targets.

    The EEG window geometry is the encoder branch's, not a constant: EEGNet
    writes ``(1400, 7, 62, 100)``, LaBraM writes ``(1400, 3, 62, 200)``.  The
    declared ``window_spec`` is checked when supplied; otherwise the array's
    own shape is accepted as long as the sample count and the CLIP targets
    line up.  A mismatch between the two is what would silently train a model
    on the wrong layout, so that is what is checked here.
    """

    ensure_output_roots()
    if not WINDOW_EEG_PATH.exists():
        prepare()
    if not TEXT_EMBEDDINGS_PATH.exists():
        extract_embeddings()
    eeg = np.load(WINDOW_EEG_PATH, mmap_mode="r")
    text = np.load(TEXT_EMBEDDINGS_PATH, mmap_mode="r")
    # The split sizes are the dataset's, not the SEED-DV paper's: ds005662 has
    # 630 train + 90 test where SEED-DV has 1200 + 200.  ``expected_rows`` comes
    # from the adapter, which counts the HDF5 rows it just materialised.
    expected = (int(expected_rows) if expected_rows is not None
                else SETTINGS.train_samples + SETTINGS.test_samples)
    if eeg.ndim != 4:
        raise ValueError(f"Unexpected Scheme D EEG cache shape: {eeg.shape}")
    if eeg.shape[0] != expected:
        raise ValueError(
            f"EEG cache holds {eeg.shape[0]} clips, expected {expected}"
        )
    # Channel count is the dataset's: SEED-DV has 62 ESI electrodes, ds005662
    # has 64 BioSemi ones.  The spec carries it when the adapter supplied one.
    expected_channels = (int(window_spec["n_chans"])
                         if window_spec and "n_chans" in window_spec
                         else SETTINGS.eeg_channels)
    if eeg.shape[2] != expected_channels:
        raise ValueError(
            f"EEG cache has {eeg.shape[2]} channels, expected "
            f"{expected_channels}"
        )
    if window_spec is not None:
        want = (int(window_spec["n_windows"]), expected_channels,
                int(window_spec["window_samples"]))
        if tuple(eeg.shape[1:]) != want:
            raise ValueError(
                f"EEG cache is {tuple(eeg.shape[1:])} but the encoder branch "
                f"declares {want}"
            )
    if text.shape != (expected, CLIP_SEQUENCE_LENGTH, CLIP_HIDDEN_SIZE):
        raise ValueError(f"Unexpected CLIP target shape: {text.shape}")
    return eeg, text


def _save_checkpoint(
    path: Path,
    model,
    epoch: int,
    task: str,
    loss: float,
    normalization_mean: np.ndarray | None = None,
    normalization_std: np.ndarray | None = None,
    requested_epochs: int | None = None,
    train_samples: int | None = None,
    predictor_meta: dict[str, Any] | None = None,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "epoch": epoch,
        "task": task,
        "loss": float(loss),
        "optimizer": "Adam",
        "learning_rate": SETTINGS.predictor_learning_rate,
        "scheduler": "cosine",
        "epochs": int(requested_epochs or SETTINGS.predictor_epochs),
        "paper_settings": SETTINGS.as_dict(),
        "scheme_d": {
            "stage_one_input": "[B,7,62,100]",
            "stage_one_output": "[B,77,768]",
            "stage_two_output": "[B,77,768]",
            "eegnet": "EEGNet-8,2, F1=8, D=2, F2=16, kern_length=64",
            "window_aggregation": "mean over seven 500-ms windows",
            "loss": "torch.nn.functional.mse_loss",
        },
    }
    if predictor_meta is not None:
        # Records which encoder produced this checkpoint and the window
        # geometry it consumed, so a checkpoint cannot be silently reloaded
        # against a different encoder's preprocessing branch.
        state["encoder"] = dict(predictor_meta)
        state["model_input_normalization"] = _resolve_normalization(None)
    if normalization_mean is not None:
        state["normalization_mean"] = normalization_mean
        state["normalization_std"] = normalization_std
    torch.save(state, path)
    if requested_epochs is not None and epoch == requested_epochs:
        _write_json(
            _complete_path(path),
            {
                "task": task,
                "epoch": epoch,
                "requested_epochs": requested_epochs,
                "train_samples": train_samples,
                "checkpoint_bytes": path.stat().st_size,
                "scheme": "D",
            },
        )


def _write_history(name: str, history: list[dict[str, Any]]) -> None:
    _write_json(SCHEME_ROOT / f"{name}_history.json", {"history": history})


def _training_device() -> "torch.device":
    """CUDA, or a loud refusal -- never a silent fallback to CPU.

    This was ``torch.device("cuda" if torch.cuda.is_available() else "cpu")``,
    and that one line is the difference between a run that finishes and one
    that never does: with no GPU visible the training happily proceeds on the
    CPU and the log looks exactly like a healthy one -- ``Scheme D stage 1
    epoch N/200``, printed every ten minutes instead of every five seconds.

    Measured on ds008257's geometry (5722 samples, labram, 3x200 windows): one
    epoch is **~5 s on a 4090 and ~580 s on a 64-core CPU**.  A factor of a
    hundred leaves no trace in the log, so it has to be refused here.

    ``EEG2VIDEO_ALLOW_CPU=1`` is the deliberate escape hatch, and it says so in
    the log.
    """

    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("EEG2VIDEO_ALLOW_CPU") == "1":
        print("WARNING: no CUDA device visible and EEG2VIDEO_ALLOW_CPU=1; "
              "training on the CPU, which is ~100x slower", flush=True)
        return torch.device("cpu")
    raise RuntimeError(
        "Scheme D training needs a GPU and none is visible to torch "
        f"(torch.cuda.is_available() is False, CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}). Refusing to "
        "train on CPU -- it is ~100x slower and the log would not show it. "
        "Fix the container's GPU access, or set EEG2VIDEO_ALLOW_CPU=1 to "
        "accept that."
    )


def _limit_cpu_threads() -> int:
    """Cap torch's CPU thread pool, which the default size makes pathological.

    On a many-core host torch defaults to one thread per core, and the
    DataLoader's collate path is built from *tiny* per-sample ops: stacking one
    256-sample batch is 256 copies of a ~184 KB tensor.  Measured on a 52-core
    machine:

        torch.stack(256 tensors)   52 threads -> 7058 ms
                                    8 threads ->   23 ms     (300x)
                                    1 thread  ->   43 ms

    Seven seconds per batch against 0.27 s of GPU compute is why a track ran at
    ~8 minutes per epoch with the GPU sitting at **0 % utilisation** -- the
    device was starved, and nothing in the log said so.

    Override with ``EEG2VIDEO_CPU_THREADS`` (0 leaves torch's default alone).
    """

    import torch

    wanted = int(os.environ.get("EEG2VIDEO_CPU_THREADS", "8"))
    before = torch.get_num_threads()
    if wanted > 0 and before > wanted:
        torch.set_num_threads(wanted)
        print(f"torch CPU threads: {before} -> {wanted} "
              f"(the default starves the DataLoader on this core count)",
              flush=True)
    return torch.get_num_threads()


def train_stage_one(
    epochs: int = SETTINGS.scheme_d_stage_one_epochs,
    max_train_samples: int | None = None,
    batch_size: int = SETTINGS.predictor_batch_size,
    force: bool = False,
    predictor_factory=None,
    predictor_meta: dict[str, Any] | None = None,
    model_input_normalization: str | None = None,
    train_samples: int | None = None,
    total_samples: int | None = None,
) -> Path:
    """Train the encoder and the Stage-1 projection head with MSE."""
    import torch

    mode = _resolve_normalization(model_input_normalization)
    n_train = _resolve_train_samples(train_samples)
    eeg, text = _ensure_targets(_window_spec_of(predictor_meta), total_samples)
    normalized, mean, std = _normalized_eeg(eeg, mode, n_train)
    if max_train_samples is not None:
        n_train = min(n_train, int(max_train_samples))
    if n_train < 2:
        raise ValueError("Stage 1 needs at least two training samples")
    if (
        _is_complete(
            STAGE_ONE_CHECKPOINT,
            "scheme_d_stage_one",
            epochs,
            max_train_samples,
        )
        and not force
    ):
        print(f"Scheme D Stage 1 already complete: {STAGE_ONE_CHECKPOINT}", flush=True)
        return STAGE_ONE_CHECKPOINT

    seed_everything(SETTINGS.semantic_seed)
    _limit_cpu_threads()
    device = _training_device()
    # Materialise both arrays in RAM.  ``normalized`` is a memmap, and with
    # ``num_workers=0`` the DataLoader would otherwise read one sample at a
    # time from disk; on a shared network filesystem that dominated the epoch
    # (6.7 s of I/O against 1.6 s of GPU compute).  The values, their order and
    # the seeding are unchanged -- this only moves the data off the filesystem.
    dataset = ArrayDataset(
        np.asarray(normalized[:n_train], dtype=np.float32),
        np.asarray(text[:n_train], dtype=np.float32),
    )
    loader = _loader(dataset, batch_size, SETTINGS.semantic_seed)
    model = (predictor_factory or SchemeDSemanticPredictor)().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SETTINGS.predictor_learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * len(loader))
    )
    history: list[dict[str, Any]] = []
    SCHEME_ROOT.mkdir(parents=True, exist_ok=True)
    _complete_path(STAGE_ONE_CHECKPOINT).unlink(missing_ok=True)
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        total_loss = 0.0
        samples = 0
        for eeg_batch, target_batch in loader:
            eeg_batch = eeg_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            target_batch = target_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(eeg_batch)
            loss = scheme_d_mse(prediction, target_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * eeg_batch.shape[0]
            samples += eeg_batch.shape[0]
        row = {
            "epoch": epoch,
            "mse": total_loss / samples,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
            "samples": n_train,
        }
        history.append(row)
        _write_history("stage_one", history)
        print(
            f"Scheme D stage 1 epoch {epoch:03d}/{epochs} "
            f"mse={row['mse']:.8f} lr={row['learning_rate']:.10g}",
            flush=True,
        )
        if epoch % 10 == 0 or epoch == epochs:
            _save_checkpoint(
                STAGE_ONE_CHECKPOINT,
                model,
                epoch,
                "scheme_d_stage_one",
                row["mse"],
                mean,
                std,
                epochs,
                n_train,
                predictor_meta,
            )
    return STAGE_ONE_CHECKPOINT


def _load_stage_one(
    device,
    checkpoint: Path = STAGE_ONE_CHECKPOINT,
    predictor_factory=None,
) -> tuple[SchemeDSemanticPredictor, dict[str, Any]]:
    import torch

    if not checkpoint.exists():
        raise FileNotFoundError(f"Stage-1 checkpoint is missing: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = (predictor_factory or SchemeDSemanticPredictor)()
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


def train_stage_two(
    epochs: int = SETTINGS.scheme_d_stage_two_epochs,
    max_train_samples: int | None = None,
    batch_size: int = SETTINGS.predictor_batch_size,
    force: bool = False,
    predictor_factory=None,
    predictor_meta: dict[str, Any] | None = None,
    model_input_normalization: str | None = None,
    train_samples: int | None = None,
    total_samples: int | None = None,
) -> Path:
    """Freeze Stage 1 and train only the tokenwise Linear adapter."""
    import torch

    mode = _resolve_normalization(model_input_normalization)
    n_train = _resolve_train_samples(train_samples)
    eeg, text = _ensure_targets(_window_spec_of(predictor_meta), total_samples)
    normalized, mean, std = _normalized_eeg(eeg, mode, n_train)
    if max_train_samples is not None:
        n_train = min(n_train, int(max_train_samples))
    if not _is_complete(
        STAGE_ONE_CHECKPOINT,
        "scheme_d_stage_one",
        SETTINGS.scheme_d_stage_one_epochs,
        max_train_samples,
    ):
        train_stage_one(
            epochs=SETTINGS.scheme_d_stage_one_epochs,
            max_train_samples=max_train_samples,
            batch_size=batch_size,
            force=force,
            predictor_factory=predictor_factory,
            predictor_meta=predictor_meta,
            model_input_normalization=mode,
            train_samples=train_samples,
            total_samples=total_samples,
        )
    if (
        _is_complete(
            STAGE_TWO_CHECKPOINT,
            "scheme_d_stage_two_linear",
            epochs,
            max_train_samples,
        )
        and not force
    ):
        print(f"Scheme D Stage 2 already complete: {STAGE_TWO_CHECKPOINT}", flush=True)
        return STAGE_TWO_CHECKPOINT

    seed_everything(SETTINGS.predictor_seed)
    _limit_cpu_threads()
    device = _training_device()
    stage_one, stage_one_state = _load_stage_one(device, predictor_factory=predictor_factory)
    for parameter in stage_one.parameters():
        parameter.requires_grad_(False)
    # Materialise both arrays in RAM.  ``normalized`` is a memmap, and with
    # ``num_workers=0`` the DataLoader would otherwise read one sample at a
    # time from disk; on a shared network filesystem that dominated the epoch
    # (6.7 s of I/O against 1.6 s of GPU compute).  The values, their order and
    # the seeding are unchanged -- this only moves the data off the filesystem.
    dataset = ArrayDataset(
        np.asarray(normalized[:n_train], dtype=np.float32),
        np.asarray(text[:n_train], dtype=np.float32),
    )
    loader = _loader(dataset, batch_size, SETTINGS.predictor_seed)
    adapter = SchemeDLinearAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=SETTINGS.predictor_learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=SETTINGS.scheme_d_stage_two_warmup_steps,
        num_training_steps=max(1, epochs * len(loader)),
    )
    history: list[dict[str, Any]] = []
    SCHEME_ROOT.mkdir(parents=True, exist_ok=True)
    _complete_path(STAGE_TWO_CHECKPOINT).unlink(missing_ok=True)
    for epoch in range(1, epochs + 1):
        started = time.time()
        adapter.train()
        total_loss = 0.0
        samples = 0
        for eeg_batch, target_batch in loader:
            eeg_batch = eeg_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            target_batch = target_batch.to(device=device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                predicted = stage_one(eeg_batch)
            optimizer.zero_grad(set_to_none=True)
            adapted = adapter(predicted)
            loss = scheme_d_mse(adapted, target_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * eeg_batch.shape[0]
            samples += eeg_batch.shape[0]
        row = {
            "epoch": epoch,
            "mse": total_loss / samples,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
            "samples": n_train,
            "frozen_stage_one": True,
        }
        history.append(row)
        _write_history("stage_two", history)
        print(
            f"Scheme D stage 2 epoch {epoch:03d}/{epochs} "
            f"mse={row['mse']:.8f} lr={row['learning_rate']:.10g}",
            flush=True,
        )
        if epoch % 10 == 0 or epoch == epochs:
            _save_checkpoint(
                STAGE_TWO_CHECKPOINT,
                adapter,
                epoch,
                "scheme_d_stage_two_linear",
                row["mse"],
                stage_one_state.get("normalization_mean", mean),
                stage_one_state.get("normalization_std", std),
                epochs,
                n_train,
                predictor_meta,
            )
    del stage_one
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return STAGE_TWO_CHECKPOINT


@torch.no_grad()
def predict_stage_d(
    eeg_windows: np.ndarray,
    stage_one_checkpoint: Path = STAGE_ONE_CHECKPOINT,
    stage_two_checkpoint: Path = STAGE_TWO_CHECKPOINT,
    batch_size: int = SETTINGS.predictor_batch_size,
    predictor_factory=None,
    model_input_normalization: str | None = None,
):
    """Run both trained stages and return a CPU float tensor [N,77,768]."""
    import torch

    _resolve_normalization(model_input_normalization)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_one, state = _load_stage_one(device, stage_one_checkpoint, predictor_factory)
    adapter = SchemeDLinearAdapter()
    adapter_state = torch.load(stage_two_checkpoint, map_location="cpu", weights_only=False)
    adapter.load_state_dict(adapter_state["model"])
    adapter = adapter.to(device).eval()
    mean = np.asarray(state["normalization_mean"], dtype=np.float32)
    std = np.asarray(state["normalization_std"], dtype=np.float32)
    normalized = (np.asarray(eeg_windows, dtype=np.float32) - mean) / std
    outputs = []
    for begin in range(0, len(normalized), batch_size):
        batch = torch.from_numpy(normalized[begin : begin + batch_size]).to(
            device=device, dtype=torch.float32
        )
        outputs.append(adapter(stage_one(batch)).float().cpu())
    return torch.cat(outputs, dim=0)


def evaluate_semantic_holdout(predictor_factory=None,
                              model_input_normalization: str | None = None,
                              train_samples: int | None = None,
                              total_samples: int | None = None,
                              predictor_meta: dict[str, Any] | None = None,
                              ) -> dict[str, float]:
    """Evaluate both semantic stages on the untouched block-7 samples."""
    import torch

    mode = _resolve_normalization(model_input_normalization)
    n_train = _resolve_train_samples(train_samples)
    # The window spec carries the channel count; without it _ensure_targets
    # falls back to the SEED-DV constant of 62 and rejects ds005662's 64.
    eeg, text = _ensure_targets(_window_spec_of(predictor_meta), total_samples)
    test_eeg = np.asarray(eeg[n_train:], dtype=np.float32)
    target = torch.from_numpy(np.asarray(text[n_train:], dtype=np.float32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage_one, state = _load_stage_one(device, predictor_factory=predictor_factory)
    adapter_state = torch.load(STAGE_TWO_CHECKPOINT, map_location="cpu", weights_only=False)
    mean = np.asarray(state["normalization_mean"], dtype=np.float32)
    std = np.asarray(state["normalization_std"], dtype=np.float32)
    normalized = (test_eeg - mean) / std
    one_outputs = []
    two_outputs = []
    adapter_model = SchemeDLinearAdapter().to(device)
    adapter_model.load_state_dict(adapter_state["model"])
    adapter_model.eval()
    with torch.no_grad():
        for begin in range(0, len(normalized), SETTINGS.predictor_batch_size):
            batch = torch.from_numpy(normalized[begin : begin + SETTINGS.predictor_batch_size]).to(
                device=device, dtype=torch.float32
            )
            one = stage_one(batch)
            one_outputs.append(one.float().cpu())
            two_outputs.append(adapter_model(one).float().cpu())
    one_pred = torch.cat(one_outputs)
    two_pred = torch.cat(two_outputs)
    result = {
        "stage_one_test_mse": float(torch.mean((one_pred - target) ** 2)),
        "stage_two_test_mse": float(torch.mean((two_pred - target) ** 2)),
        "test_samples": len(target),
    }
    _write_json(RUN_ROOT / "scheme_d_semantic_holdout.json", result)
    return result


def main_args(
    stage: str,
    epochs: int | None = None,
    max_train_samples: int | None = None,
    batch_size: int | None = None,
    force: bool = False,
) -> None:
    """Programmatic equivalent of the command-line entry point."""
    actual_epochs = int(epochs) if epochs is not None else None
    actual_batch_size = int(batch_size or SETTINGS.predictor_batch_size)
    if stage in ("one", "both"):
        train_stage_one(
            actual_epochs or SETTINGS.scheme_d_stage_one_epochs,
            max_train_samples,
            actual_batch_size,
            force,
        )
    if stage in ("two", "both"):
        train_stage_two(
            actual_epochs or SETTINGS.scheme_d_stage_two_epochs,
            max_train_samples,
            actual_batch_size,
            force,
        )
    if stage == "eval":
        print(json.dumps(evaluate_semantic_holdout(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("one", "two", "both", "eval"), required=True)
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the stage-specific default (stage 1: 200; stage 2: 150).",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=SETTINGS.predictor_batch_size)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main_args(
        args.stage,
        args.epochs,
        args.max_train_samples,
        args.batch_size,
        args.force,
    )


if __name__ == "__main__":
    main()
