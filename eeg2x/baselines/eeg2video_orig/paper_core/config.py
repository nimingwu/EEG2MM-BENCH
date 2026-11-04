"""Immutable settings stated by the paper or its released implementation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from eeg2x.protocol import SEED, derive_seed


# Vendored copy (EEG2X-Benchmark-v2).  The original resolved its run and
# checkpoint roots relative to the EEG2Video project, which from inside the
# benchmark lands *outside* it -- and the benchmark requires every artefact to
# stay in-tree.  The roots are therefore anchored to this package, and all
# other path logic is unchanged.
NEW_ROOT = Path(__file__).resolve().parent          # model/eeg2video_paper
SOURCE_ROOT = NEW_ROOT.parent                       # model/
PROJECT_ROOT = SOURCE_ROOT.parent                   # EEG2X-Benchmark-v2
BENCHMARK_ROOT = PROJECT_ROOT
DATA_ROOT = Path("/datas/eeg_benchmark_project/SEED-DV")
HF_ROOT = Path("/datas/hf-cache")
SUBJECT_ID = int(os.environ.get("EEG2VIDEO_SUBJECT", "1"))
FULL_SUBJECT_RUN = os.environ.get("EEG2VIDEO_FULL_SUBJECT_RUN", "0") == "1"
ABLATION_NO_SEMANTIC_DYNAMIC = (
    os.environ.get("EEG2VIDEO_ABLATION_NO_SEMANTIC_DYNAMIC", "0") == "1"
)
CUSTOM_RUN_ROOT = os.environ.get("EEG2VIDEO_RUN_ROOT")
CUSTOM_CHECKPOINT_ROOT = os.environ.get("EEG2VIDEO_CHECKPOINT_ROOT")
if CUSTOM_RUN_ROOT and CUSTOM_CHECKPOINT_ROOT:
    RUN_ROOT = Path(CUSTOM_RUN_ROOT).resolve()
    CHECKPOINT_ROOT = Path(CUSTOM_CHECKPOINT_ROOT).resolve()
else:
    # Always inside the benchmark; subject-scoped so tracks never collide.
    _TRACK_ROOT = PROJECT_ROOT / "runs" / "eeg2video_paper" / f"sub{SUBJECT_ID:02d}"
    RUN_ROOT = _TRACK_ROOT
    CHECKPOINT_ROOT = _TRACK_ROOT / "checkpoints"
SHARED_RUN_ROOT = Path(
    os.environ.get("EEG2VIDEO_SHARED_RUN_ROOT", str(RUN_ROOT))
).resolve()
SHARED_TUNE_ROOT = Path(
    os.environ.get("EEG2VIDEO_SHARED_TUNE_ROOT", "")
).resolve() if os.environ.get("EEG2VIDEO_SHARED_TUNE_ROOT") else None
SD14_ROOT = (
    HF_ROOT
    / "hub"
    / "models--CompVis--stable-diffusion-v1-4"
    / "snapshots"
    / "133a221b8aa7292a167afc5127cb63fb5005638b"
)
# Vendored alongside the diffusion half (see model/eeg2video_paper/vendor/).
NEGATIVE_EMBEDDING_PATH = NEW_ROOT / "vendor" / "negative.npy"


@dataclass(frozen=True)
class PaperSettings:
    reproduction_revision: str = os.environ.get(
        "EEG2VIDEO_REPRODUCTION_REVISION", "paper-literal-caption-v1"
    )
    subject: int = SUBJECT_ID
    sampling_rate: int = 200
    # Geometry is dataset-dependent and therefore env-overridable: SEED-DV
    # ships 2 s clips on 62 electrodes, ds005662 ships 0.9 s epochs on 64.  The
    # paper's numbers stay the defaults, so the SEED-DV run is unchanged.
    eeg_channels: int = int(os.environ.get("EEG2VIDEO_EEG_CHANNELS", "62"))
    clip_seconds: int = 2
    eeg_window_samples: int = 100
    eeg_stride_samples: int = 50
    eeg_tokens: int = int(os.environ.get("EEG2VIDEO_EEG_TOKENS", "7"))
    feature_window_samples: int = 200
    feature_stride_samples: int = 100
    feature_windows: int = int(os.environ.get("EEG2VIDEO_FEATURE_WINDOWS", "3"))
    # Appendix B.1 uses one-second windows with 500 ms overlap for EEG
    # classification and feature extraction. Dynamic GLMNet consumes the
    # resulting three windows and averages their clip-level logits.
    dynamic_input_samples: int = 200
    dynamic_windows: int = 3
    video_fps: int = 3
    video_frames: int = 6
    video_width: int = 512
    video_height: int = 288
    latent_channels: int = 4
    latent_height: int = 36
    latent_width: int = 64
    latent_scale: float = 0.18215
    train_blocks: int = 6
    test_block: int = 7
    samples_per_block: int = 200
    concepts: int = 40
    repeats_per_concept: int = 5
    eeg_embedding_dim: int = 512
    transformer_heads: int = 4
    transformer_encoder_layers: int = 2
    transformer_decoder_layers: int = 4
    transformer_ff_dim: int = 2048
    transformer_dropout: float = 0.1
    transformer_norm_first: bool = True
    predictor_batch_size: int = 32
    dynamic_batch_size: int = 256
    predictor_epochs: int = 200
    predictor_learning_rate: float = 5e-4
    predictor_optimizer: str = "Adam"
    predictor_scheduler: str = "cosine"
    semantic_hidden_dim: int = 10000
    semantic_relu_layers: int = 4
    # The released reproduction recipe selects one caption per concept. The
    # implementation has one such caption available in every training block,
    # so each block/concept caption is repeated across its five clips.
    semantic_target_mode: str = os.environ.get(
        "EEG2VIDEO_SEMANTIC_TARGET_MODE", "per_video_caption"
    )
    # The paper specifies three overlapping one-second DE windows, while the
    # released semantic trainer averages two non-overlapping one-second
    # windows. Keep the paper-literal interpretation as the default and make
    # the released behavior available as an isolated protocol ablation.
    semantic_feature_mode: str = os.environ.get(
        "EEG2VIDEO_SEMANTIC_FEATURE_MODE", "mean_overlap_3"
    )
    glm_embedding_dim: int = 64
    optical_flow_threshold: float = 1.799
    dana_beta_fast: float = 0.2
    dana_beta_slow: float = 0.3
    # The released DANA implementation samples its forward-noising time from
    # a separate 500-step linear schedule before the 100-step DDIM reverse run.
    dana_time_steps: int = 500
    dana_seed: int = derive_seed("generation_noise")
    diffusion_train_timesteps: int = 1000
    ddim_inference_steps: int = 100
    guidance_scale: float = 12.5
    tune_epochs: int = 200
    tune_batch_size: int = 10
    tune_learning_rate: float = 3e-5
    tune_scheduler: str = "cosine"
    tune_warmup_steps: int = 0
    tune_gradient_accumulation: int = 1
    tune_weight_decay: float = 1e-2
    tune_adam_beta1: float = 0.9
    tune_adam_beta2: float = 0.999
    tune_adam_epsilon: float = 1e-8
    tune_max_grad_norm: float = 1.0
    tune_seed: int = SEED
    predictor_seed: int = SEED
    semantic_seed: int = SEED
    inference_seed: int = derive_seed("generation_noise")
    metric_trials: int = 100

    @property
    def train_samples(self) -> int:
        # SEED-DV computes these from its block geometry (6 x 200 = 1200 train,
        # 200 test).  A dataset with a different split -- ds005662 holds out one
        # 90-clip block from a 720-trial set -- sets them directly.
        override = os.environ.get("EEG2VIDEO_TRAIN_SAMPLES")
        return int(override) if override else self.train_blocks * self.samples_per_block

    @property
    def test_samples(self) -> int:
        override = os.environ.get("EEG2VIDEO_TEST_SAMPLES")
        return int(override) if override else self.samples_per_block

    def as_dict(self) -> dict:
        return asdict(self)


SETTINGS = PaperSettings()

BLOCK_NAMES = (
    "1st_10min",
    "2nd_10min",
    "3rd_10min",
    "4th_10min",
    "5th_10min",
    "6th_10min",
    "7th_10min",
)

OCCIPITAL_CHANNELS = tuple(range(50, 62))
TUNE_TRAINABLE_MODULES = ("attn1.to_q", "attn2.to_q", "attn_temp")
