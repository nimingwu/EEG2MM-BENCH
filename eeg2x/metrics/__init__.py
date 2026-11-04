"""
Metrics module — uniform evaluation interface for the EEG benchmark.

Pre-built metrics
-----------------
  Top-1 Accuracy       retrieval / classification
  Top-5 Accuracy       retrieval / classification
  mAP                  retrieval
  CLIP Similarity      image generation / retrieval
  PixCorr              image generation
  SSIM                 image generation
  AlexNet(2)           image generation
  AlexNet(5)           image generation
  Inception            image generation
  SwAV                 image generation (lower is better)
  FID                  image generation (lower is better)
  BLEU                 sentence generation
  ROUGE-L              sentence generation
  METEOR               sentence generation
  BERTScore            sentence generation
  WER                  sentence generation / word decoding (lower is better)

Adding a new metric
-------------------
1. Create a file ``metrics/<name>.py``.
2. Define a class inheriting ``Metric`` (see ``metrics/base.py``).
3. Add it to ``METRIC_REGISTRY`` below.

Usage
-----
>>> from metrics import evaluate
>>> scores = evaluate(
...     task="image_retrieval",
...     predictions=predictions,
...     ground_truths=ground_truths,
...     metrics=["top1_accuracy", "top5_accuracy"],
... )
>>> print(scores)
"""

from .base import Metric
from .topk_accuracy import TopKAccuracy
from .clip_similarity import CLIPSimilarity
from .pixcorr import PixCorr
from .ssim import SSIM
from .perceptual_similarity import AlexNet2, AlexNet5, InceptionSim, SwAVSim
from .fid import FID
from .map import MeanAveragePrecision
from .text_metrics import BLEU, ROUGEL, METEOR, WordErrorRate
from .bert_score import BERTScore
from .mcd import MCD
from .stoi import STOI
from .cer import CER
from .retrieval_rank import RetrievalRank
from .speech_retrieval import (
    SpeechTopKAccuracy, SpeechRetrievalRank, SpeechMeanAveragePrecision,
    _WordTypeTopKAccuracy, _WordTypeRetrievalRank,
    _WordTypeSubjectAggTopKAccuracy, _WordTypeSubjectAggRetrievalRank,
)
from .video_generation import (
    VideoNWayAccuracy, FrameNWayAccuracy, VideoSSIM, VideoPSNR, VideoHuePCC,
)
from .semkey_metrics import default_metrics as _semkey_metrics

# ---------------------------------------------------------------------------
# Registry  — add new metrics here
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, Metric] = {
    # === EEG-to-Image: Retrieval ===
    "top1_accuracy": TopKAccuracy(k=1),
    "top5_accuracy": TopKAccuracy(k=5),
    "map":           MeanAveragePrecision(),

    # === EEG-to-Image: Generation ===
    "clip_similarity": CLIPSimilarity(),
    "pixcorr":         PixCorr(),
    "ssim":            SSIM(),
    "alexnet_2":       AlexNet2(),
    "alexnet_5":       AlexNet5(),
    "inception":       InceptionSim(),
    "swav":            SwAVSim(),
    "fid":             FID(),

    # === EEG-to-Text: SemKey 套件（arXiv 2603.03312v2）===
    # 15 项：MTV 与 standard 两种 BLEU-1/2、2/4/10/24-way 检索、DIST-1/2、
    # 内容召回、head entropy、Frechet distance、SBERT cosine、长度 MAE。
    **_semkey_metrics(),

    # === EEG-to-Video: Generation ===
    # 七项基础指标，与 tab:raw_video_recon 的七列一致。每项另有 shuffled 与
    # delta 两个版本：数据集内视觉场景固定，绝对数值被刺激相似度抬高，真正
    # 归因于 EEG 配对的是 matched 减 shuffled 的差。
    "video_2way_accuracy":  VideoNWayAccuracy(n_way=2),
    "video_40way_accuracy": VideoNWayAccuracy(n_way=40),
    "frame_2way_accuracy":  FrameNWayAccuracy(n_way=2),
    "frame_40way_accuracy": FrameNWayAccuracy(n_way=40),
    "video_ssim":           VideoSSIM(),
    "video_psnr":           VideoPSNR(),
    "video_hue_pcc":        VideoHuePCC(),

    "video_2way_accuracy_shuffled":  VideoNWayAccuracy(n_way=2, mode="shuffled"),
    "video_40way_accuracy_shuffled": VideoNWayAccuracy(n_way=40, mode="shuffled"),
    "frame_2way_accuracy_shuffled":  FrameNWayAccuracy(n_way=2, mode="shuffled"),
    "frame_40way_accuracy_shuffled": FrameNWayAccuracy(n_way=40, mode="shuffled"),
    "video_ssim_shuffled":           VideoSSIM(mode="shuffled"),
    "video_psnr_shuffled":           VideoPSNR(mode="shuffled"),
    "video_hue_pcc_shuffled":        VideoHuePCC(mode="shuffled"),

    "video_2way_accuracy_delta":  VideoNWayAccuracy(n_way=2, mode="delta"),
    "video_40way_accuracy_delta": VideoNWayAccuracy(n_way=40, mode="delta"),
    "frame_2way_accuracy_delta":  FrameNWayAccuracy(n_way=2, mode="delta"),
    "frame_40way_accuracy_delta": FrameNWayAccuracy(n_way=40, mode="delta"),
    "video_ssim_delta":           VideoSSIM(mode="delta"),
    "video_psnr_delta":           VideoPSNR(mode="delta"),
    "video_hue_pcc_delta":        VideoHuePCC(mode="delta"),

    # === EEG-to-Text: Sentence generation ===
    "bleu_4":    BLEU(max_order=4),
    "bleu_1":    BLEU(max_order=1),
    "rouge_l":   ROUGEL(),
    "meteor":    METEOR(),
    "bertscore": BERTScore(),
    "wer":       WordErrorRate(),
    "cer":       CER(),

    # === EEG-to-Speech: Speech reconstruction ===
    "mcd":       MCD(),
    "stoi":      STOI(),

    # === EEG-to-Speech: Retrieval（实例级，专用指标，见 metrics/speech_retrieval.py）===
    "speech_top1_accuracy": SpeechTopKAccuracy(k=1),
    "speech_top5_accuracy": SpeechTopKAccuracy(k=5),
    "speech_top10_accuracy": SpeechTopKAccuracy(k=10),
    "speech_median_retrieval_rank": SpeechRetrievalRank("median"),
    "speech_mean_retrieval_rank":   SpeechRetrievalRank("mean"),
    "speech_std_retrieval_rank":    SpeechRetrievalRank("std"),
    "speech_map": SpeechMeanAveragePrecision(),

    # === Speech retrieval: subject-aggregated variants (NeuralBench *_subject-agg) ===
    "speech_top1_accuracy_subject-agg": SpeechTopKAccuracy(k=1, subject_agg=True),
    "speech_top5_accuracy_subject-agg": SpeechTopKAccuracy(k=5, subject_agg=True),
    "speech_top10_accuracy_subject-agg": SpeechTopKAccuracy(k=10, subject_agg=True),
    "speech_median_retrieval_rank_subject-agg": SpeechRetrievalRank("median", subject_agg=True),
    "speech_mean_retrieval_rank_subject-agg":   SpeechRetrievalRank("mean", subject_agg=True),
    "speech_std_retrieval_rank_subject-agg":    SpeechRetrievalRank("std", subject_agg=True),

    # === Speech retrieval: word-type instance-aggregated (NeuralBench *_instance-agg) ===
    "speech_top1_accuracy_instance-agg": _WordTypeTopKAccuracy(k=1),
    "speech_top5_accuracy_instance-agg": _WordTypeTopKAccuracy(k=5),
    "speech_top10_accuracy_instance-agg": _WordTypeTopKAccuracy(k=10),
    "speech_median_retrieval_rank_instance-agg": _WordTypeRetrievalRank("median"),
    "speech_mean_retrieval_rank_instance-agg":   _WordTypeRetrievalRank("mean"),
    "speech_std_retrieval_rank_instance-agg":    _WordTypeRetrievalRank("std"),

    # === Speech retrieval: word-type subject-aggregated (NeuralBench *_subject-agg with instance agg) ===
    "speech_top1_accuracy_subject-agg-v2": _WordTypeSubjectAggTopKAccuracy(k=1),
    "speech_top5_accuracy_subject-agg-v2": _WordTypeSubjectAggTopKAccuracy(k=5),
    "speech_top10_accuracy_subject-agg-v2": _WordTypeSubjectAggTopKAccuracy(k=10),
    "speech_median_retrieval_rank_subject-agg-v2": _WordTypeSubjectAggRetrievalRank("median"),
    "speech_mean_retrieval_rank_subject-agg-v2":   _WordTypeSubjectAggRetrievalRank("mean"),
    "speech_std_retrieval_rank_subject-agg-v2":    _WordTypeSubjectAggRetrievalRank("std"),

    # === Speech retrieval: word-type instance-aggregated + size=250 (NeuralBench *_size=250) ===
    "speech_top1_accuracy_instance-agg_size250": _WordTypeTopKAccuracy(k=1, size_250=True),
    "speech_top5_accuracy_instance-agg_size250": _WordTypeTopKAccuracy(k=5, size_250=True),
    "speech_top10_accuracy_instance-agg_size250": _WordTypeTopKAccuracy(k=10, size_250=True),
    "speech_median_retrieval_rank_instance-agg_size250": _WordTypeRetrievalRank("median", size_250=True),
    "speech_mean_retrieval_rank_instance-agg_size250":   _WordTypeRetrievalRank("mean", size_250=True),
    "speech_std_retrieval_rank_instance-agg_size250":    _WordTypeRetrievalRank("std", size_250=True),

    # === Speech retrieval: word-type subject-aggregated + size=250 ===
    "speech_top1_accuracy_subject-agg-v2_size250": _WordTypeSubjectAggTopKAccuracy(k=1, size_250=True),
    "speech_top5_accuracy_subject-agg-v2_size250": _WordTypeSubjectAggTopKAccuracy(k=5, size_250=True),
    "speech_top10_accuracy_subject-agg-v2_size250": _WordTypeSubjectAggTopKAccuracy(k=10, size_250=True),
    "speech_median_retrieval_rank_subject-agg-v2_size250": _WordTypeSubjectAggRetrievalRank("median", size_250=True),
    "speech_mean_retrieval_rank_subject-agg-v2_size250":   _WordTypeSubjectAggRetrievalRank("mean", size_250=True),
    "speech_std_retrieval_rank_subject-agg-v2_size250":    _WordTypeSubjectAggRetrievalRank("std", size_250=True),
}


def get_metric(name: str) -> Metric:
    """Return a metric by name. Raises KeyError if unknown."""
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {list(_METRIC_REGISTRY)}"
        )
    return _METRIC_REGISTRY[name]


def list_metrics(task: str = None) -> dict[str, Metric]:
    """Return metrics (optionally filtered by task applicability)."""
    if task is None:
        return _METRIC_REGISTRY.copy()
    return {
        k: m for k, m in _METRIC_REGISTRY.items()
        if task in m.applicable
    }


# ---------------------------------------------------------------------------
# Convenience: evaluate a batch against selected metrics
# ---------------------------------------------------------------------------

def evaluate(task: str,
             predictions: list,
             ground_truths: list,
             metrics: list[str] = None,
             **kwargs) -> dict[str, float]:
    """Run a set of metrics over predictions and ground truths.

    Parameters
    ----------
    task : str
        One of "image_retrieval", "image_generation", "concept_classification",
        "sentence_generation", "word_decoding".
    predictions : list
        Model outputs, one per sample.
    ground_truths : list
        Ground-truth labels, one per sample.
    metrics : list[str], optional
        Metric names to compute. If None, uses all applicable metrics.
    **kwargs
        Extra args passed to each metric's ``batch_compute``.

    Returns
    -------
    scores : dict[str, float]
        e.g. {"top1_accuracy": 0.632, "clip_similarity": 0.781}
    """
    if metrics is None:
        applicable = list_metrics(task)
        metric_names = list(applicable.keys())
    else:
        metric_names = metrics

    scores = {}
    for name in metric_names:
        m = get_metric(name)
        if task not in m.applicable:
            raise ValueError(
                f"Metric '{name}' does not apply to task '{task}'. "
                f"Applicable tasks: {m.applicable}"
            )
        try:
            scores[name] = m.batch_compute(predictions, ground_truths, **kwargs)
        except Exception as exc:
            scores[name] = float("nan")
            import logging
            logging.getLogger(__name__).warning(
                f"Metric '{name}' failed: {exc}"
            )
    return scores
