"""Paper-aligned 200-epoch Tune-A-Video training on all 1200 train clips."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from einops import rearrange
from torch.utils.data import DataLoader

from .vendor.unet import UNet3DConditionModel

from .config import (
    ABLATION_NO_SEMANTIC_DYNAMIC,
    CHECKPOINT_ROOT,
    SHARED_TUNE_ROOT,
    SD14_ROOT,
    SETTINGS,
    TUNE_TRAINABLE_MODULES,
)
from .data import (
    UNCONDITIONAL_EMBEDDING_PATH,
    VIDEO_FRAMES_PATH,
    TuneVideoDataset,
    ensure_output_roots,
    prepare,
    seed_everything,
)


TUNE_ROOT = SHARED_TUNE_ROOT or (CHECKPOINT_ROOT / "tune_a_video")
RESUME_PATH = TUNE_ROOT / "training_state.pt"
HISTORY_PATH = TUNE_ROOT / "history.json"
FINAL_UNET_PATH = TUNE_ROOT / "unet"
CONFIG_PATH = TUNE_ROOT / "paper_config.json"


def _trainable_parameters(unet) -> list[torch.nn.Parameter]:
    unet.requires_grad_(False)
    selected: list[torch.nn.Parameter] = []
    selected_names: list[str] = []
    for name, module in unet.named_modules():
        if name.endswith(TUNE_TRAINABLE_MODULES):
            selected_names.append(name)
            for parameter in module.parameters():
                parameter.requires_grad_(True)
                selected.append(parameter)
    if not selected:
        raise RuntimeError("No Tune-A-Video attention parameters were selected")
    (TUNE_ROOT / "trainable_modules.json").write_text(
        json.dumps(selected_names, indent=2), encoding="utf-8"
    )
    return selected


def _save_training_state(
    unet,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    history: list[dict],
) -> None:
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in unet.state_dict().items()
        if any(token in name for token in TUNE_TRAINABLE_MODULES)
    }
    torch.save(
        {
            "epoch": epoch,
            "trainable_state": trainable_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "paper_settings": SETTINGS.as_dict(),
        },
        RESUME_PATH,
    )
    HISTORY_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def train_tune_a_video(force: bool = False) -> None:
    """Run the complete paper training; this function never shortens the schedule."""
    ensure_output_roots()
    TUNE_ROOT.mkdir(parents=True, exist_ok=True)
    if FINAL_UNET_PATH.exists() and not force:
        print(f"paper Tune-A-Video already trained: {FINAL_UNET_PATH}")
        return
    if not VIDEO_FRAMES_PATH.exists():
        prepare()

    from diffusers import AutoencoderKL, DDPMScheduler
    from diffusers.optimization import get_scheduler
    from transformers import CLIPTextModel, CLIPTokenizer

    seed_everything(SETTINGS.tune_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Paper Tune-A-Video training requires CUDA")
    device = torch.device("cuda")
    tokenizer = None
    text_encoder = None
    unconditional_embedding = None
    if ABLATION_NO_SEMANTIC_DYNAMIC:
        if not UNCONDITIONAL_EMBEDDING_PATH.exists():
            raise FileNotFoundError(
                "The fixed unconditional embedding is missing: "
                f"{UNCONDITIONAL_EMBEDDING_PATH}"
            )
        unconditional_embedding = torch.from_numpy(
            np.load(UNCONDITIONAL_EMBEDDING_PATH).astype(np.float16)
        ).to(device=device)
    else:
        tokenizer = CLIPTokenizer.from_pretrained(
            str(SD14_ROOT), subfolder="tokenizer", local_files_only=True
        )
        text_encoder = CLIPTextModel.from_pretrained(
            str(SD14_ROOT), subfolder="text_encoder", local_files_only=True
        ).to(device=device, dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(
        str(SD14_ROOT), subfolder="vae", local_files_only=True
    ).to(device=device, dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained(
        str(SD14_ROOT), subfolder="scheduler", local_files_only=True
    )
    unet = UNet3DConditionModel.from_pretrained_2d(
        str(SD14_ROOT), subfolder="unet"
    ).to(device)
    if text_encoder is not None:
        text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    trainable = _trainable_parameters(unet)
    unet.enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention()

    dataset = TuneVideoDataset(tokenizer)
    generator = torch.Generator().manual_seed(SETTINGS.tune_seed)
    loader = DataLoader(
        dataset,
        batch_size=SETTINGS.tune_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=SETTINGS.tune_learning_rate,
        betas=(SETTINGS.tune_adam_beta1, SETTINGS.tune_adam_beta2),
        weight_decay=SETTINGS.tune_weight_decay,
        eps=SETTINGS.tune_adam_epsilon,
    )
    total_steps = SETTINGS.tune_epochs * len(loader)
    scheduler = get_scheduler(
        SETTINGS.tune_scheduler,
        optimizer=optimizer,
        num_warmup_steps=SETTINGS.tune_warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    start_epoch = 1
    history: list[dict] = []
    if RESUME_PATH.exists() and not force:
        state = torch.load(RESUME_PATH, map_location="cpu")
        unet.load_state_dict(state["trainable_state"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        if HISTORY_PATH.exists():
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        print(f"resuming paper Tune-A-Video at epoch {start_epoch}", flush=True)

    config = {
        "paper": "NeurIPS 2024 EEG2Video Table 2, 40 classes, Full Model",
        "subject": SETTINGS.subject,
        "semantic_guidance": not ABLATION_NO_SEMANTIC_DYNAMIC,
        "conditioning": (
            "fixed_unconditional_embedding"
            if ABLATION_NO_SEMANTIC_DYNAMIC
            else "caption_CLIP_embedding"
        ),
        "base_model": str(SD14_ROOT),
        "training_samples": len(dataset),
        "epochs": SETTINGS.tune_epochs,
        "batch_size": SETTINGS.tune_batch_size,
        "learning_rate": SETTINGS.tune_learning_rate,
        "lr_scheduler": SETTINGS.tune_scheduler,
        "warmup_steps": SETTINGS.tune_warmup_steps,
        "optimizer": "AdamW",
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "trainable_modules": list(TUNE_TRAINABLE_MODULES),
        "seed": SETTINGS.tune_seed,
        "video": {
            "frames": SETTINGS.video_frames,
            "fps": SETTINGS.video_fps,
            "width": SETTINGS.video_width,
            "height": SETTINGS.video_height,
        },
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    for epoch in range(start_epoch, SETTINGS.tune_epochs + 1):
        started = time.time()
        unet.train()
        accumulated_loss = 0.0
        samples = 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(
                device=device, dtype=torch.float16, non_blocking=True
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                frames = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(frames).latent_dist.sample()
                latents = rearrange(
                    latents,
                    "(b f) c h w -> b c f h w",
                    f=SETTINGS.video_frames,
                )
                latents = latents * SETTINGS.latent_scale
                if ABLATION_NO_SEMANTIC_DYNAMIC:
                    encoder_hidden_states = unconditional_embedding.expand(
                        latents.shape[0], -1, -1
                    )
                else:
                    prompt_ids = batch["prompt_ids"].to(
                        device=device, non_blocking=True
                    )
                    encoder_hidden_states = text_encoder(prompt_ids)[0]
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0,
                noise_scheduler.num_train_timesteps,
                (latents.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            if noise_scheduler.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unsupported prediction type: {noise_scheduler.prediction_type}"
                )
            with torch.cuda.amp.autocast(dtype=torch.float16):
                prediction = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = functional.mse_loss(
                    prediction.float(), target.float(), reduction="mean"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if epoch == start_epoch and samples == 0:
                missing_gradients = [
                    name
                    for name, parameter in unet.named_parameters()
                    if parameter.requires_grad and parameter.grad is None
                ]
                nonfinite_gradients = [
                    name
                    for name, parameter in unet.named_parameters()
                    if parameter.requires_grad
                    and parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all().item()
                ]
                if missing_gradients or nonfinite_gradients:
                    raise RuntimeError(
                        "Invalid first-batch Tune-A-Video gradients: "
                        f"missing={missing_gradients}, nonfinite={nonfinite_gradients}"
                    )
            torch.nn.utils.clip_grad_norm_(trainable, SETTINGS.tune_max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if epoch == start_epoch and samples == 0:
                print(
                    "Tune-A-Video first batch verified: "
                    f"pixels={tuple(pixel_values.shape)} latents={tuple(latents.shape)} "
                    f"loss={float(loss.detach()):.8f} "
                    f"peak_cuda_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}",
                    flush=True,
                )
            accumulated_loss += float(loss.detach()) * latents.shape[0]
            samples += latents.shape[0]
        epoch_loss = accumulated_loss / samples
        row = {
            "epoch": epoch,
            "mse": epoch_loss,
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(
            f"Tune-A-Video epoch {epoch:03d}/200 mse={epoch_loss:.8f} "
            f"lr={row['learning_rate']:.10f}",
            flush=True,
        )
        _save_training_state(unet, optimizer, scheduler, scaler, epoch, history)

    unet.save_pretrained(FINAL_UNET_PATH)
    print(f"paper Tune-A-Video complete: {FINAL_UNET_PATH}")
