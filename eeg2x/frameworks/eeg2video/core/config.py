"""Immutable settings stated by the paper or its released implementation."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path


NEW_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = NEW_ROOT.parent
PROJECT_ROOT = NEW_ROOT.parents[1]
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
elif FULL_SUBJECT_RUN:
    RUN_ROOT = PROJECT_ROOT / "runs" / "full_subjects" / f"sub{SUBJECT_ID:02d}"
    CHECKPOINT_ROOT = (
        PROJECT_ROOT / "checkpoints" / "full_subjects" / f"sub{SUBJECT_ID:02d}"
    )
else:
    RUN_ROOT = PROJECT_ROOT / "runs" / "paper_sub01_table2_full_model"
    CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints" / "paper_sub01_table2_full_model"
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
# The paper-literal constants that used to live here (the released negative
# embedding and the author's Tune-A-Video checkpoint root) were only used by
# the Seq2Seq / Tune-A-Video path, which this project does not contain.

# --- Scheme D video generator: the official AnimateDiff ---------------------
# The diffusers AnimateDiffPipeline path produced unusable output; see
# 实验报告8.txt and AnimateDiff/experiments/INDEX.md.  Generation now runs
# through the vendored official implementation in
# EEG2Video_clean_d/EEG2Video/EEG2Video_New/animatediff_official/.
ANIMATEDIFF_BASE_MODEL = (
    HF_ROOT
    / "hub"
    / "models--stable-diffusion-v1-5--stable-diffusion-v1-5"
    / "snapshots"
    / "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
)
ANIMATEDIFF_MOTION_MODULE = HF_ROOT / "animatediff_ckpt" / "mm_sd_v15_v2.ckpt"
# The official training config pins sample_n_frames: 16 and the temporal
# attention was never trained below that.  Measured: 6 frames is unusable,
# 8-12 is seed-dependent, 16-24 is stable.  This project generates at exactly
# the training length, 16, and subsamples down to Scheme D's 6-frame output,
# which yields the even 3-frame stride (0, 3, 6, 9, 12, 15).
ANIMATEDIFF_GENERATION_FRAMES = 16
ANIMATEDIFF_OUTPUT_FRAMES = 6
ANIMATEDIFF_STEPS = 25
ANIMATEDIFF_GUIDANCE_SCALE = 7.5
#: Classifier-free guidance needs a real negative branch.  The old Scheme D
#: path used zeros, which is not equivalent and visibly degrades the result.
ANIMATEDIFF_NEGATIVE_PROMPT = "bad quality, worse quality"


@dataclass(frozen=True)
class PaperSettings:
    reproduction_revision: str = os.environ.get(
        "EEG2VIDEO_REPRODUCTION_REVISION", "paper-literal-caption-v1"
    )
    subject: int = SUBJECT_ID
    sampling_rate: int = 200
    eeg_channels: int = 62
    clip_seconds: int = 2
    eeg_window_samples: int = 100
    eeg_stride_samples: int = 50
    eeg_tokens: int = 7
    feature_window_samples: int = 200
    feature_stride_samples: int = 100
    feature_windows: int = 3
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
    scheme_d_stage_one_epochs: int = 200
    scheme_d_stage_two_epochs: int = 150
    scheme_d_stage_two_warmup_steps: int = 500
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
    # --- random seeds ---------------------------------------------------
    # This project pins EVERY random seed to one value, 2000.  Upstream left
    # some of them unset (the Seq2Seq trainer used whatever torch seeded
    # itself with, and the released evaluator did not seed NumPy at all);
    # those are now fixed as well.  The individual names are kept because
    # existing code and checkpoint metadata refer to them.
    seed: int = 2000
    tune_seed: int = 2000
    dana_seed: int = 2000
    predictor_seed: int = 2000
    semantic_seed: int = 2000
    inference_seed: int = 2000
    noise_seed: int = 2000
    metric_trials: int = 100

    @property
    def train_samples(self) -> int:
        return self.train_blocks * self.samples_per_block

    @property
    def test_samples(self) -> int:
        return self.samples_per_block

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
