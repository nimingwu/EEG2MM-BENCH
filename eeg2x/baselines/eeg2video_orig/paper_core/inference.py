"""Full-model DANA and 100-step DDIM inference for the subject's block 7."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from eeg2x.frameworks.eeg2video import noise as noise_control

from .vendor.unet import UNet3DConditionModel

from .config import (
    ABLATION_NO_SEMANTIC_DYNAMIC,
    CHECKPOINT_ROOT,
    NEGATIVE_EMBEDDING_PATH,
    RUN_ROOT,
    SD14_ROOT,
    SETTINGS,
)
from .data import (
    FEATURE_WINDOW_EEG_PATH,
    UNCONDITIONAL_EMBEDDING_PATH,
    WINDOW_EEG_PATH,
    read_manifest,
    semantic_de_features,
    seed_everything,
)
from .models import DynamicGLMNet, EEGToLatentSeq2Seq, SemanticPredictor
from .train_predictors import (
    DYNAMIC_CHECKPOINT,
    SEMANTIC_CHECKPOINT,
    SEQ_CHECKPOINT,
)
from .train_video_diffusion import FINAL_UNET_PATH


def _load_predictor(model, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Required paper checkpoint is missing: {checkpoint_path}")
    # These checkpoints are produced locally by train_predictors.py and include
    # NumPy normalization arrays, which PyTorch's weights-only loader rejects.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(state["epoch"]) != SETTINGS.predictor_epochs:
        raise ValueError(
            f"{checkpoint_path} is epoch {state['epoch']}; paper requires 200 epochs"
        )
    checkpoint_settings = state.get("paper_settings", {})
    if checkpoint_settings.get("reproduction_revision") != SETTINGS.reproduction_revision:
        raise ValueError(
            f"{checkpoint_path} is not a {SETTINGS.reproduction_revision} predictor; "
            "retrain it for 200 epochs"
        )
    if (
        state.get("task") == "semantic"
        and checkpoint_settings.get("semantic_target_mode")
        != SETTINGS.semantic_target_mode
    ):
        raise ValueError(
            f"{checkpoint_path} uses a different semantic target protocol; "
            "retrain the Semantic Predictor for 200 epochs"
        )
    if (
        state.get("task") == "semantic"
        and checkpoint_settings.get("semantic_feature_mode", "mean_overlap_3")
        != SETTINGS.semantic_feature_mode
    ):
        raise ValueError(
            f"{checkpoint_path} uses a different semantic feature protocol; "
            "retrain the Semantic Predictor for 200 epochs"
        )
    if state.get("task") == "dynamic_glmnet" and (
        checkpoint_settings.get("dynamic_input_samples")
        != SETTINGS.dynamic_input_samples
        or checkpoint_settings.get("dynamic_windows") != SETTINGS.dynamic_windows
    ):
        raise ValueError(
            f"{checkpoint_path} uses a legacy dynamic input protocol; retrain "
            "the one-second-window GLMNet predictor for 200 epochs"
        )
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state


@torch.no_grad()
def predict_eeg_conditions(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    seq2seq, seq_state = _load_predictor(
        EEGToLatentSeq2Seq(), SEQ_CHECKPOINT, device
    )
    test_slice = slice(SETTINGS.train_samples, SETTINGS.train_samples + SETTINGS.test_samples)

    seq_eeg = np.asarray(np.load(WINDOW_EEG_PATH, mmap_mode="r")[test_slice])
    seq_eeg = (seq_eeg - seq_state["normalization_mean"]) / seq_state[
        "normalization_std"
    ]
    # Noise control, injected after the predictor's own normalisation so the
    # replacement sits at the scale the network reads.
    seq_eeg = noise_control.maybe_noise(seq_eeg)

    latent_batches: list[torch.Tensor] = []
    for begin in range(0, SETTINGS.test_samples, SETTINGS.predictor_batch_size):
        seq_batch = torch.from_numpy(seq_eeg[begin : begin + SETTINGS.predictor_batch_size]).to(
            device=device, dtype=torch.float32
        )
        latent_batches.append(seq2seq.predict(seq_batch).cpu())
    del seq2seq
    predicted_latents = torch.cat(latent_batches)
    if ABLATION_NO_SEMANTIC_DYNAMIC:
        # No Semantic Predictor and no Dynamic Predictor are used in this
        # ablation. DANA keeps its stochastic noise process but uses the
        # neutral slow branch for every clip instead of a learned label.
        dynamic_labels = torch.zeros(SETTINGS.test_samples, dtype=torch.long)
        return predicted_latents, None, dynamic_labels

    semantic, semantic_state = _load_predictor(
        SemanticPredictor(), SEMANTIC_CHECKPOINT, device
    )
    dynamic, dynamic_state = _load_predictor(
        DynamicGLMNet(), DYNAMIC_CHECKPOINT, device
    )
    de = np.asarray(semantic_de_features()[test_slice]).reshape(
        SETTINGS.test_samples, -1
    )
    de = (de - semantic_state["normalization_mean"]) / semantic_state[
        "normalization_std"
    ]
    dynamic_eeg = np.asarray(
        np.load(FEATURE_WINDOW_EEG_PATH, mmap_mode="r")[test_slice]
    )
    # How many feature windows one trial carries is dataset geometry, not a
    # constant: SEED-DV's 2 s trials give 3, ds005662's 600 ms trials give 1.
    # Read it off the array so the reshape below cannot disagree with the data
    # -- ``SETTINGS.dynamic_windows`` is SEED-DV's value and is left alone
    # because the checkpoint-compatibility check compares against it.
    windows_per_clip = int(dynamic_eeg.shape[1])
    dynamic_eeg = (dynamic_eeg - dynamic_state["normalization_mean"]) / dynamic_state[
        "normalization_std"
    ]
    # Noise control (second site -- the Dynamic GLMNet reads its own EEG array,
    # normalised with its own state).
    dynamic_eeg = noise_control.maybe_noise(dynamic_eeg)
    semantic_batches: list[torch.Tensor] = []
    dynamic_batches: list[torch.Tensor] = []
    dynamic_eeg = dynamic_eeg.reshape(
        -1, SETTINGS.eeg_channels, SETTINGS.dynamic_input_samples
    )
    for begin in range(0, SETTINGS.test_samples, SETTINGS.predictor_batch_size):
        de_batch = torch.from_numpy(de[begin : begin + SETTINGS.predictor_batch_size]).to(
            device=device, dtype=torch.float32
        )
        semantic_batches.append(semantic(de_batch).cpu())
    for begin in range(0, len(dynamic_eeg), SETTINGS.dynamic_batch_size):
        dyn_batch = torch.from_numpy(
            dynamic_eeg[begin : begin + SETTINGS.dynamic_batch_size]
        ).to(device=device, dtype=torch.float32)
        dynamic_batches.append(dynamic(dyn_batch.unsqueeze(1)).cpu())
    del semantic, dynamic
    torch.cuda.empty_cache()
    dynamic_logits = torch.cat(dynamic_batches).reshape(
        SETTINGS.test_samples, windows_per_clip, 2
    ).mean(dim=1)
    return predicted_latents, torch.cat(semantic_batches), dynamic_logits.argmax(dim=1)


def dana_latent(
    predicted_latent: torch.Tensor,
    dynamic_label: torch.Tensor,
    beta_schedule: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Apply the released DANA forward process at a sampled time t.

    The project implementation constructs a 500-step linear beta schedule,
    samples one timestep per clip, and then starts the 100-step DDIM reverse
    chain from the resulting latent.  This is the paper's equation (2) with
    ``alpha_T`` evaluated at that sampled forward-noising time.
    """
    batch, channels, frames, height, width = predicted_latent.shape
    timesteps = torch.randint(
        0,
        beta_schedule.numel(),
        (batch,),
        generator=generator,
        device=predicted_latent.device,
        dtype=torch.long,
    )
    alpha = torch.cumprod(1.0 - beta_schedule, dim=0)[timesteps]
    alpha = alpha.to(
        device=predicted_latent.device,
        dtype=predicted_latent.dtype,
    ).reshape(-1, 1, 1, 1, 1)
    diverse_noise = torch.randn(
        predicted_latent.shape,
        generator=generator,
        device=predicted_latent.device,
        dtype=predicted_latent.dtype,
    )
    static_noise = torch.randn(
        (batch, channels, 1, height, width),
        generator=generator,
        device=predicted_latent.device,
        dtype=predicted_latent.dtype,
    ).expand_as(predicted_latent)
    beta = torch.where(
        dynamic_label.bool(),
        torch.tensor(
            SETTINGS.dana_beta_fast,
            device=predicted_latent.device,
            dtype=predicted_latent.dtype,
        ),
        torch.tensor(
            SETTINGS.dana_beta_slow,
            device=predicted_latent.device,
            dtype=predicted_latent.dtype,
        ),
    ).reshape(-1, 1, 1, 1, 1)
    mixed_noise = beta.sqrt() * static_noise + (1.0 - beta).sqrt() * diverse_noise
    return alpha.sqrt() * predicted_latent + (
        1.0 - alpha
    ).sqrt() * mixed_noise


def _save_gif(frames: np.ndarray, path: Path) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, list(frames), duration=1.0 / SETTINGS.video_fps, loop=0)


@torch.no_grad()
def infer(seed: int = SETTINGS.inference_seed, force: bool = False) -> None:
    if not FINAL_UNET_PATH.exists():
        raise FileNotFoundError(
            f"Paper-trained Tune-A-Video UNet is missing: {FINAL_UNET_PATH}"
        )
    # A noise run writes under its own directory -- see the sibling note in
    # ``eeg2video_eegvid_model``; the frames file name is fixed further down.
    output_root = RUN_ROOT / "inference" / f"seed_{seed}"
    if noise_control.enabled():
        output_root = output_root.parent / f"{output_root.name}.noise"
    output_root.mkdir(parents=True, exist_ok=True)
    output_frames_path = output_root / "generated_frames.uint8.npy"
    completion_path = output_root / "inference.complete.json"
    complete = False
    if output_frames_path.exists() and completion_path.exists():
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            complete = (
                completion.get("reproduction_revision")
                == SETTINGS.reproduction_revision
                and completion.get("seed") == seed
                and completion.get("samples") == SETTINGS.test_samples
                and completion.get("output_bytes") == output_frames_path.stat().st_size
            )
        except (OSError, json.JSONDecodeError):
            complete = False
    if complete and not force:
        print(f"paper inference already complete: {output_frames_path}")
        return
    completion_path.unlink(missing_ok=True)

    from diffusers import AutoencoderKL, DDIMScheduler

    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Paper DDIM inference requires CUDA")
    device = torch.device("cuda")
    predicted_latents, semantic_embeddings, dynamic_labels = predict_eeg_conditions(device)
    condition_payload = {
        "latents": predicted_latents.numpy().astype(np.float16),
        "dynamic": dynamic_labels.numpy().astype(np.int64),
    }
    if semantic_embeddings is not None:
        condition_payload["semantic"] = semantic_embeddings.numpy().astype(np.float16)
    np.savez(output_root / "predicted_conditions.npz", **condition_payload)

    unet = UNet3DConditionModel.from_pretrained(
        str(FINAL_UNET_PATH), local_files_only=True, torch_dtype=torch.float16
    ).to(device)
    # Tune-A-Video's released inference path uses the same memory-efficient
    # attention implementation as training.
    try:
        unet.enable_xformers_memory_efficient_attention()
    except Exception as exc:
        print(
            f"xFormers is unavailable for this PyTorch/CUDA build; using native attention: {exc}",
            flush=True,
        )
    vae = AutoencoderKL.from_pretrained(
        str(SD14_ROOT),
        subfolder="vae",
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(device)
    scheduler = DDIMScheduler.from_pretrained(
        str(SD14_ROOT), subfolder="scheduler", local_files_only=True
    )
    scheduler.set_timesteps(SETTINGS.ddim_inference_steps, device=device)
    unet.eval()
    vae.eval()
    if not ABLATION_NO_SEMANTIC_DYNAMIC and not NEGATIVE_EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"The released negative prompt embedding is missing: {NEGATIVE_EMBEDDING_PATH}"
        )
    unconditional_path = (
        UNCONDITIONAL_EMBEDDING_PATH
        if ABLATION_NO_SEMANTIC_DYNAMIC
        else NEGATIVE_EMBEDDING_PATH
    )
    unconditional = torch.from_numpy(np.load(unconditional_path).astype(np.float16)).to(device)
    # DANA is precomputed independently in the released project (seed 3407),
    # while the DDIM inference seed controls the generated reverse trajectories.
    dana_generator = torch.Generator(device=device).manual_seed(SETTINGS.dana_seed)
    ddim_generator = torch.Generator(device=device).manual_seed(seed)
    dana_beta_schedule = torch.linspace(
        0.0001,
        0.02,
        SETTINGS.dana_time_steps,
        device=device,
        dtype=torch.float32,
    )
    frames_output = np.lib.format.open_memmap(
        output_frames_path,
        mode="w+",
        dtype=np.uint8,
        shape=(
            SETTINGS.test_samples,
            SETTINGS.video_frames,
            SETTINGS.video_height,
            SETTINGS.video_width,
            3,
        ),
    )
    for sample_id in range(SETTINGS.test_samples):
        clean = rearrange(
            predicted_latents[sample_id : sample_id + 1],
            "b f c h w -> b c f h w",
        ).to(device=device, dtype=torch.float16)
        label = dynamic_labels[sample_id : sample_id + 1].to(device)
        latents = dana_latent(clean, label, dana_beta_schedule, dana_generator)
        if ABLATION_NO_SEMANTIC_DYNAMIC:
            guidance = unconditional
        else:
            condition = semantic_embeddings[sample_id : sample_id + 1].to(
                device=device, dtype=torch.float16
            )
            guidance = torch.cat((unconditional, condition), dim=0)
        for timestep in scheduler.timesteps:
            if ABLATION_NO_SEMANTIC_DYNAMIC:
                model_input = scheduler.scale_model_input(latents, timestep)
                noise_prediction = unet(
                    model_input, timestep, encoder_hidden_states=guidance
                ).sample
            else:
                model_input = torch.cat((latents, latents), dim=0)
                model_input = scheduler.scale_model_input(model_input, timestep)
                noise_prediction = unet(
                    model_input, timestep, encoder_hidden_states=guidance
                ).sample
                noise_unconditional, noise_conditional = noise_prediction.chunk(2)
                noise_prediction = noise_unconditional + SETTINGS.guidance_scale * (
                    noise_conditional - noise_unconditional
                )
            latents = scheduler.step(
                noise_prediction, timestep, latents, eta=0.0, generator=ddim_generator
            ).prev_sample
        decoded_input = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded = vae.decode(decoded_input / SETTINGS.latent_scale).sample
        decoded = (
            (decoded / 2 + 0.5)
            .clamp(0, 1)
            .mul(255)
            .round()
            .byte()
            .reshape(1, SETTINGS.video_frames, 3, 288, 512)
            .permute(0, 1, 3, 4, 2)
            .cpu()
            .numpy()[0]
        )
        frames_output[sample_id] = decoded
        _save_gif(decoded, output_root / "videos" / f"{sample_id:03d}.gif")
        frames_output.flush()
        # The denominator is the actual clip count: it was hard-coded to 200
        # (SEED-DV's test split) and printed "090/200" on ds005662's 90.
        print(f"DDIM sample {sample_id + 1:03d}/{SETTINGS.test_samples}", flush=True)

    metadata = {
        "reproduction_revision": SETTINGS.reproduction_revision,
        "subject": SETTINGS.subject,
        "test_block": 7,
        "samples": SETTINGS.test_samples,
        "seed": seed,
        "ddim_steps": SETTINGS.ddim_inference_steps,
        "guidance_scale": 1.0 if ABLATION_NO_SEMANTIC_DYNAMIC else SETTINGS.guidance_scale,
        "semantic_guidance": not ABLATION_NO_SEMANTIC_DYNAMIC,
        "semantic_predictor": not ABLATION_NO_SEMANTIC_DYNAMIC,
        "dynamic_predictor": not ABLATION_NO_SEMANTIC_DYNAMIC,
        "dana_dynamic_label": (
            "fixed_slow" if ABLATION_NO_SEMANTIC_DYNAMIC else "DynamicGLMNet"
        ),
        "dana": {
            "time_steps": SETTINGS.dana_time_steps,
            "seed": SETTINGS.dana_seed,
            "beta_fast": SETTINGS.dana_beta_fast,
            "beta_slow": SETTINGS.dana_beta_slow,
        },
        "test_concepts": [
            row["concept_one_based"] for row in read_manifest()[SETTINGS.train_samples :]
        ],
        "validated_unet": str(FINAL_UNET_PATH),
    }
    (output_root / "inference.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    completion_path.write_text(
        json.dumps(
            {
                "reproduction_revision": SETTINGS.reproduction_revision,
                "subject": SETTINGS.subject,
                "seed": seed,
                "samples": SETTINGS.test_samples,
                "output_bytes": output_frames_path.stat().st_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"{'ablation no-semantic/dynamic' if ABLATION_NO_SEMANTIC_DYNAMIC else 'paper Full Model'} "
        f"inference complete: {output_root}"
    )
