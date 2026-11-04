"""Registry — central lookup for tasks, models, and their configurations.

Adding a new task:      add an entry to TASK_REGISTRY
Adding a new model:     add an entry to MODEL_REGISTRY
Adding a new metric:    add to METRIC_REGISTRY in metrics/__init__.py
"""

from __future__ import annotations

from typing import Any, Dict, List


# ====================================================================
# Task registry
# ====================================================================

_H5_SCHEMA_IMAGE = {
    "fields": ["data", "event_label", "event_object_number",
               "event_stim_number", "channel_names"],
    "attrs": ["dataset", "subject_id", "sampling_rate", "n_channels", "n_epochs"],
}

_H5_SCHEMA_VIDEO = {
    "fields": ["data", "event_concept", "event_caption", "channel_names"],
    "attrs": ["dataset", "subject_id", "sampling_rate", "n_channels", "n_epochs"],
}


def _image_generation_metrics() -> List[str]:
    """图像生成任务的 8 项指标，从任务契约里读，不在两处各写一份。

    v2 把指标清单同时写在 ``benchmark_eval`` 的注册表和 ``eeg2image_task_interface``
    里，两处已经漂移。v3 让 :mod:`eeg2x.tasks.image` 的 ``TaskSpec`` 做权威定义，
    评测层只引用它。
    """
    from eeg2x.tasks.image import TASK_REGISTRY as IMAGE_TASK_REGISTRY
    spec = IMAGE_TASK_REGISTRY["image_generation"]
    return [spec.primary_metric, *spec.secondary_metrics]


def _sentence_generation_metrics() -> List[str]:
    """文本生成的 15 项 SemKey 指标，从 metrics 注册表读。"""
    from eeg2x.metrics.semkey_metrics import default_metrics

    return sorted(default_metrics())

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "image_retrieval": {
        # Only final concept-level test files are benchmark evaluation inputs.
        "h5_patterns": ["thingseeg_sub-??_test_epoched.h5",
                        "nod_*_test_epoched.h5",
                        "grootswagers_*_test_epoched.h5",
                        "grootswagers2025_*_test_epoched.h5"],
        "gt_type": "int",
        "default_metrics": ["top1_accuracy", "top5_accuracy", "map"],
        "h5_schema": _H5_SCHEMA_IMAGE,
        "pred_required": ["ranked_indices"],
    },
    "image_generation": {
        # ATM has a distinct EEG profile and filename namespace.
        "h5_patterns": ["thingseeg_atm_sub-??_test_epoched.h5",
                        "grootswagers_atm_*_test_epoched.h5",
                        "grootswagers_labram_atm_*_test_epoched.h5",
                        "grootswagers_reve_atm_*_test_epoched.h5",
                        "grootswagers_luna_atm_*_test_epoched.h5",
                        "grootswagers2025_atm_*_test_epoched.h5",
                        "grootswagers2025_labram_atm_*_test_epoched.h5",
                        "grootswagers2025_reve_atm_*_test_epoched.h5",
                        "grootswagers2025_luna_atm_*_test_epoched.h5",
                        "nod_atm_sub-??_test_epoched.h5",
                        "nod_labram_atm_sub-??_test_epoched.h5",
                        "nod_reve_atm_sub-??_test_epoched.h5",
                        "nod_luna_atm_sub-??_test_epoched.h5"],
        "gt_type": "image_path",
        "default_metrics": _image_generation_metrics(),
        "h5_schema": _H5_SCHEMA_IMAGE,
        "pred_required": ["path", "data_ref"],   # 二选一
    },
    "concept_classification": {
        "h5_patterns": ["thingseeg_sub-??_test_epoched.h5",
                        "grootswagers_cls_*_test_epoched.h5",
                        "grootswagers2025_cls_*_test_epoched.h5",
                        "nod_*_test_epoched.h5"],
        "gt_type": "int",
        "default_metrics": ["top1_accuracy", "top5_accuracy"],
        "h5_schema": _H5_SCHEMA_IMAGE,
        "pred_required": ["predicted_concept"],
    },
    "video_generation": {
        # 三套数据集共用 schema、切分约定与七项指标；它们在通道数、试次长度、
        # 试次数量与刺激时钟上不同。模式必须互不重叠，否则一个目录里混放多个
        # 数据集时会给指标喂错真值。
        "h5_patterns": ["seed_dv_*_epoched.h5", "ds005662_*_epoched.h5",
                        "ds008257_*_epoched.h5"],
        # 真值就是测试 H5 里的 gt_frames；评测把它转成
        # h5://<file>#gt_frames[<row>] 引用，由 metrics/video_generation.py 解析。
        "gt_field": "gt_frames",
        "gt_type": "video_frames_ref",
        "default_metrics": [
            "video_2way_accuracy", "video_40way_accuracy",
            "frame_2way_accuracy", "frame_40way_accuracy", "video_ssim",
            "video_psnr", "video_hue_pcc",
        ],
        "h5_schema": _H5_SCHEMA_VIDEO,
        "pred_required": ["path"],
    },
    "sentence_generation": {
        # v3 的文本分支沿用 GLIM 的四套数据集。`zuco_*` 只匹配 zuco1
        # （zuco 后面必须是下划线），所以 zuco2、chisco、tmnred 各自显式列出。
        "h5_patterns": ["roamm_*_sentences.h5", "speech_*_sentences.h5",
                        "zuco_*_sentences.h5",
                        "zuco1_glim_*_sentences.h5", "zuco1_labram_*_sentences.h5",
                        "zuco1_luna_*_sentences.h5", "zuco1_reve_*_sentences.h5",
                        "zuco2_glim_*_sentences.h5", "zuco2_labram_*_sentences.h5",
                        "zuco2_luna_*_sentences.h5", "zuco2_reve_*_sentences.h5",
                        "chisco_glim_*_sentences.h5", "chisco_labram_*_sentences.h5",
                        "chisco_luna_*_sentences.h5", "chisco_reve_*_sentences.h5",
                        "tmnred_glim_*_sentences.h5", "tmnred_labram_*_sentences.h5",
                        "tmnred_luna_*_sentences.h5", "tmnred_reve_*_sentences.h5"],
        "gt_field": "sentence_text",
        "gt_type": "str",
        "default_metrics": _sentence_generation_metrics(),
        "h5_schema": {
            "fields": ["data", "sentence_text", "channel_names"],
            "attrs": ["dataset", "subject_id", "sampling_rate",
                      "n_channels", "n_epochs"],
        },
        "pred_required": ["generated_text"],
    },
    "word_decoding": {
        "h5_patterns": ["speech_*_words.h5", "roamm_*_words.h5"],
        "gt_field": "word_text",
        "gt_type": "str",
        "default_metrics": ["wer"],
        "h5_schema": {
            "fields": ["data", "word_text", "channel_names"],
            "attrs": ["dataset", "subject_id", "sampling_rate",
                      "n_channels", "n_epochs"],
        },
        "pred_required": ["predicted_word"],
    },
    "speech_reconstruction": {
        "h5_patterns": ["speechbrain_*_epoched.h5", "karaone_*_epoched.h5"],
        "gt_field": "text",
        "gt_type": "str",
        "default_metrics": ["mcd", "stoi", "wer", "cer"],
        # NOTE: 尚未接线到 benchmark_eval（见 extract_ground_truths）——
        # 原则 3/7：注册 = 可评测，接线前评测会显式报错。
        "h5_schema": {
            "fields": ["eeg", "text", "channel_names"],
            "attrs": ["dataset", "subject_id", "eeg_sampling_rate",
                      "n_channels", "n_epochs"],
        },
        "pred_required": ["path"],
    },
    "speech_detection": {
        "h5_patterns": ["karaone_*_epoched.h5"],
        "gt_field": "text",
        "gt_type": "str",
        "default_metrics": ["top1_accuracy"],
        # NOTE: 尚未接线到 benchmark_eval（见 extract_ground_truths）——
        # 原则 3/7：注册 = 可评测，接线前评测会显式报错。
        "h5_schema": {
            "fields": ["eeg", "text", "channel_names"],
            "attrs": ["dataset", "subject_id", "eeg_sampling_rate",
                      "n_channels", "n_epochs"],
        },
        "pred_required": ["predicted_state"],
    },
    "speech_retrieval": {
        "h5_patterns": ["brennan2019_*_epoched.h5"],
        # GT = 词实例 id（word_order，每行唯一）——实例级匹配，避免重复词
        # 字符串匹配虚高（专用指标 speech_* 见 metrics/speech_retrieval.py）
        "gt_field": "word_order",
        "gt_type": "str",
        "default_metrics": [
            # 实例级（instance-exact）
            "speech_top1_accuracy", "speech_top5_accuracy",
            "speech_top10_accuracy", "speech_median_retrieval_rank",
            "speech_map",
            # 词类型级 instance-agg（对齐 NeuralBench full-retrieval）
            "speech_top1_accuracy_instance-agg",
            "speech_top5_accuracy_instance-agg",
            "speech_top10_accuracy_instance-agg",
            "speech_median_retrieval_rank_instance-agg",
            # 词类型级 subject-agg-v2（对齐 NeuralBench *_subject-agg headline metric）
            "speech_top1_accuracy_subject-agg-v2",
            "speech_top5_accuracy_subject-agg-v2",
            "speech_median_retrieval_rank_subject-agg-v2",
            # 词类型级 subject-agg-v2 + size=250（对齐 NeuralBench headline）
            "speech_top1_accuracy_subject-agg-v2_size250",
            "speech_top5_accuracy_subject-agg-v2_size250",
            "speech_median_retrieval_rank_subject-agg-v2_size250",
        ],
        "h5_schema": {
            "fields": ["data", "word_text", "word_order", "channel_names"],
            "attrs": ["dataset", "subject_id", "sampling_rate",
                      "n_channels", "n_epochs", "epoch_type"],
        },
        "pred_required": ["ranked_indices"],
    },
}


def get_task_config(task: str) -> Dict[str, Any]:
    if task not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{task}'. Available: {list(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[task]


def list_tasks() -> List[str]:
    return list(TASK_REGISTRY.keys())


# ====================================================================
# Model registry
# ====================================================================

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Complete ATM (three branches) as an independent model next to the
    # benchmark's Only-CLIP variant: Stage I is the Diffusion Prior U-Net
    # (model/atm/prior_unet.py) instead of a single Linear(1024,1024) map, and
    # the low-level VAE-latent and semantic GIT-caption branches are trained and
    # evaluated as well.
    "atm_full": {
        "class_path": "model.atm.full_pipeline",
        "description": (
            "Complete ATM reconstruction (Li et al., NeurIPS 2024): ATM-S encoder "
            "with the Diffusion Prior U-Net high-level branch plus the low-level "
            "and semantic branches"),
        "tasks": ["image_generation"],
        "runner": "run_rsvp_atm_seed2000.py --model atm_full",
        "preprocessing_profiles": {
            "image_generation": "thingseeg-image-atm-v1",
        },
    },
    # AVDE as an independent baseline: a LaBraM encoder fine-tuned with a
    # bidirectional contrastive objective (Stage 1) feeding a VAR "next-scale
    # prediction" transformer over multi-scale VQ-VAE tokens (Stage 2).  Like
    # the complete ATM it keeps its own encoder and never swaps with the
    # benchmark's Only-CLIP framework.
    "avde": {
        "class_path": "model.avde.pipeline",
        "description": (
            "AVDE reconstruction (Dai et al., ICLR 2026): contrastive LaBraM "
            "encoder plus an autoregressive next-scale-prediction VAR decoder "
            "with a frozen multi-scale VQ-VAE"),
        "tasks": ["image_generation"],
        "runner": "run_avde_seed2000.py",
        "default_encoder": "labram",
        "preprocessing_profiles": {
            # The benchmark's existing 200 Hz / 200-sample LaBraM contract; it
            # reproduces AVDE's own preprocessing for this input.
            "image_generation": "nod-image-labram-atm-v1",
        },
        "environment": "/datas/linshishangchuan/avde",
    },
    # ENIGMA as an independent baseline: a spatio-temporal CNN encoder with
    # per-subject linear alignment layers and an MLP projector onto the CLIP
    # ViT-H/14 space, decoded by SDXL-Turbo + IP-Adapter.  Like the complete ATM
    # and AVDE it keeps its own encoder and never swaps with the framework.
    "enigma": {
        "class_path": "model.enigma.pipeline",
        "description": (
            "ENIGMA reconstruction (Kneeland et al., 2026): spatio-temporal CNN "
            "with per-subject latent alignment and an MLP projector to CLIP "
            "ViT-H/14, decoded by SDXL-Turbo + IP-Adapter"),
        "tasks": ["image_generation"],
        "runner": "run_enigma_seed2000.py",
        "default_encoder": "enigma_cnn",
        "preprocessing_profiles": {
            # ENIGMA's own preprocessing is this same contract; no new profile.
            "image_generation": "thingseeg-image-atm-v1",
        },
        "environment": "/opt/conda/envs/eeg2x-atm",
    },
    "atm_s": {
        "class_path": "model.atm_benchmark.ATMBenchmarkModel",
        "description": (
            "ATM-S channel-wise iTransformer EEG encoder with the guided "
            "diffusion reconstruction pipeline (Li et al., NeurIPS 2024)"),
        "tasks": ["image_retrieval", "image_generation"],
        "runner": "run_atm_benchmark.py",
        "preprocessing_profiles": {
            "image_retrieval": "thingseeg-image-nice-v1",
            "image_generation": "thingseeg-image-atm-v1",
        },
    },
    "atm_reve": {
        "class_path": "model.atm_benchmark.ATMReveBenchmarkModel",
        "description": (
            "ATM CLIP-alignment and guided-diffusion pipeline with the "
            "pretrained REVE-base EEG encoder and official attention pooling"),
        "tasks": ["image_retrieval", "image_generation"],
        "runner": "run_atm_reve_benchmark.py",
        "default_encoder": "reve",
        "preprocessing_profiles": {
            "image_retrieval": "thingseeg-image-reve-nice-v1",
            "image_generation": "thingseeg-image-reve-atm-v1",
        },
        "pretrained_model_id": "brain-bzh/reve-base",
        "position_model_id": "brain-bzh/reve-positions",
    },
    "atm_luna": {
        "class_path": "model.atm_benchmark.ATMLunaBenchmarkModel",
        "description": (
            "ATM CLIP-alignment and guided-diffusion pipeline with the "
            "official pretrained LUNA-Base encoder, 3D electrode locations, "
            "and fine-tuning attention pooling"),
        "tasks": ["image_retrieval", "image_generation"],
        "runner": "run_atm_luna_benchmark.py",
        "default_encoder": "luna",
        "preprocessing_profiles": {
            "image_retrieval": "thingseeg-image-luna-nice-v1",
            "image_generation": "thingseeg-image-luna-atm-v1",
        },
        "pretrained_model_id": "PulpBio/LUNA",
        "pretrained_weight_filename": "Base/LUNA_base.safetensors",
    },
    "text_baseline": {
        "class_path": "model.baseline_text.BaselineText",
        "description": "PCA(100) + LogisticRegression for EEG-to-Text",
    },
    "image_baseline": {
        "class_path": "model.baseline_image.BaselineImageClassify",
        "description": "PCA(100) + LogisticRegression for NOD-EEG concept classification",
    },
    "brain2image_vae": {
        "class_path": "model.brain2image.Brain2ImageVAE",
        "description": "LSTM encoder + VAE decoder for EEG-to-Image generation (Brain2Image, MM 2017)",
    },
    "brain2image_gan": {
        "class_path": "model.brain2image.Brain2ImageGAN",
        "description": "LSTM encoder + conditional GAN for EEG-to-Image generation (Brain2Image, MM 2017)",
    },
    "eegnet_retrieval": {
        "class_path": "model.eegnet_image.EEGNetRetrieval",
        "description": "EEGNet-8,2 + contrastive learning for zero-shot image retrieval (EEGNet, NICE, ATM)",
        "default_encoder": "eegnet",
    },
    "conformer_retrieval": {
        "class_path": "model.conformer_image.ConformerRetrieval",
        "description": "EEG-Conformer + contrastive learning for zero-shot image retrieval (Song et al., 2022, NICE)",
        "default_encoder": "conformer",
    },
    "nice_retrieval": {
        "class_path": "model.nice_image.NICERetrieval",
        "description": "NICE TSConv + contrastive learning for zero-shot image retrieval (Song et al., ICLR 2024)",
        "default_encoder": "tsconv",
    },
    "brain_translator": {
        "class_path": "model.brain_translator.BrainTranslatorModel",
        "description": "BrainTranslator: MTE + BART-large for open-vocabulary EEG-to-Text decoding (Wang & Ji, AAAI 2022)",
    },
    "eegnet_speech_retrieval": {
        "class_path": "model.eegnet_speech.EEGNetSpeechRetrieval",
        "description": "EEGNet-8,2 + InfoNCE contrastive learning for word-level EEG-to-Speech retrieval (Brennan2019, wav2vec2-large-xlsr-53 embedding space)",
        "default_encoder": "eegnet",
    },
    "labram_speech_retrieval": {
        "class_path": "model.labram.LabramSpeechRetrieval",
        "description": "Pretrained LaBraM fine-tune + linear probe for word-level EEG-to-Speech retrieval (Jiang et al., ICLR 2024; NeuralBench)",
        "default_encoder": "labram",
    },
}


def get_model_config(name: str) -> Dict[str, Any]:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def list_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())
