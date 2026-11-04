"""EEG-to-Video Table-2 metrics (EEG2Video, NeurIPS 2024).

The N-way decision rule and the SSIM computation are copied **verbatim** from
the working implementation in ``EEG2Video_clean_d_v2``
(``paper_reproduction/metrics.py``); only the outer ``Metric`` interface is new.
Keeping the arithmetic byte-identical is the point: numbers computed here must
be directly comparable with the ones the EEG2Video project already reports.

The released rule, restated because it is easy to get wrong
-----------------------------------------------------------
* Ground truth is *not* a single class.  The frozen ImageNet/VideoMAE
  classifier is run on the ground-truth video, and its **top-3 classes** are
  all accepted as correct.
* Distractors are drawn from the classes *outside* those three.
* A trial counts as a hit if any one of the three accepted classes is ranked
  first among {that class} + distractors.
* Two-way means one distractor, 40-way means 39 distractors, and every metric
  averages 100 random trials per sample.

Consequently the chance level is ``3 / (3 + n_way - 1)`` -- 0.75 for 2-way and
3/42 for 40-way -- not 1/n_way.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from .base import Metric
from eeg2x.protocol import SEED, derive_seed

log = logging.getLogger(__name__)

N_TRIALS = 100

# ---------------------------------------------------------------------------
# matched / shuffled / delta
# ---------------------------------------------------------------------------
# Every SEED-DV and ds005662 metric is reported three ways:
#
#   matched   prediction scored against its own ground truth (the number the
#             paper reports)
#   shuffled  the same prediction scored against a *deranged* ground truth --
#             a permutation with no fixed point, so no clip is ever paired with
#             itself
#   delta     matched - shuffled, the part of the score that is actually
#             attributable to the pairing
#
# The shuffled control exists because both datasets hold their visual scene
# constant: ds005662 says so outright ("visual elements ... were kept
# consistent"), and its 360 non-target stimuli are 90 videos in four flips.
# When every clip looks like every other clip, a frozen classifier will score
# a random pairing highly, so the absolute number is not evidence of decoding.
# The delta is.
#
# The derangement seed is derived from the benchmark's single seed rather than
# invented, per core/seed_protocol.py.
SHUFFLE_SEED = derive_seed("metric_trials")

MODE_MATCHED = "matched"
MODE_SHUFFLED = "shuffled"
MODE_DELTA = "delta"


def fixed_derangement(size: int, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """A permutation of ``range(size)`` with no fixed point.

    Deterministic in ``seed`` so every metric sees the same pairing, and every
    rerun sees the same pairing again.  For ``size < 2`` a derangement does not
    exist; the identity is returned and the caller reports a zero delta.
    """

    if size < 2:
        return np.arange(size)
    # ``derive_seed`` returns a 64-bit value; numpy's legacy RandomState only
    # accepts 32-bit seeds.  Reducing modulo 2**32 keeps the derangement
    # deterministic and reproducible without touching the seed protocol.
    rng = np.random.RandomState(int(seed) % (2 ** 32))
    for _ in range(1000):
        order = rng.permutation(size)
        if not np.any(order == np.arange(size)):
            return order
    raise RuntimeError(f"no derangement found for size {size}")  # pragma: no cover


def _shuffled(values: List[Any], seed: int = SHUFFLE_SEED) -> List[Any]:
    order = fixed_derangement(len(values), seed)
    return [values[i] for i in order]


def _cache_key(tag: str, predictions: Any, ground_truths: Any) -> tuple:
    """Identity of this evaluation batch, stable across the metric objects.

    ``benchmark_eval`` hands the *same two lists* to every registered metric,
    so caching on their ``id`` lets matched and shuffled be computed once and
    reused by all three modes instead of recomputed per mode -- which matters,
    because each recomputation re-runs the frozen ViT and VideoMAE.
    """

    return (tag, id(predictions), id(ground_truths),
            len(predictions), len(ground_truths))


_SCORE_CACHE: dict = {}

#: Resolved ``(predicted, ground_truth)`` pairs, keyed on the identity of the
#: two lists the evaluator hands to every metric.
#:
#: ``_paired`` was re-run for each of the four ``nway-<branch>-<n_way>`` cache
#: keys and, inside each, once for the matched pass and once for the shuffled
#: one -- so every clip was re-read from disk and re-materialised eight times
#: per evaluation.  Holding the resolved arrays makes their identity stable,
#: which is what lets ``_logits_for_clip`` below memoise on it.
_PAIR_CACHE: dict = {}

#: Frozen-classifier logits per clip pair.  ``_logits_for_clip`` returns the
#: image and video logits together, but the score cache is keyed on branch and
#: n_way, so without this the image half was recomputed for the video branch
#: (and the video half for the frame branch), and the prediction half twice per
#: pass.  Measured on a 200-sample track that was ~8x the ViT and VideoMAE
#: forwards actually needed, and the duplicates -- not the model -- are what
#: kept evaluation busy for minutes.
_LOGITS_CACHE: dict = {}


def _cached_pairs(predictions: Any, ground_truths: Any):
    """``list(_paired(...))``, computed once per (predictions, ground_truths)."""

    key = (id(predictions), id(ground_truths), len(predictions),
           len(ground_truths))
    hit = _PAIR_CACHE.get(key)
    if hit is None:
        hit = list(_paired(predictions, ground_truths))
        _PAIR_CACHE[key] = hit
    return hit


def _pick(matched: List[float], shuffled: List[float], mode: str) -> float:
    if not matched:
        return float("nan")
    if mode == MODE_SHUFFLED:
        return float(np.mean(shuffled))
    if mode == MODE_DELTA:
        return float(np.mean(matched) - np.mean(shuffled))
    return float(np.mean(matched))
#: Accepted ground-truth classes, as in the released evaluator.
GT_TOP_K = 3

#: Local frozen checkpoints, matching EEG2Video_clean_d_v2.
VIT_ROOT = (
    Path("/datas/hf-cache/hub/models--google--vit-base-patch16-224")
    / "snapshots" / "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"
)
VIDEOMAE_ROOT = (
    Path("/datas/hf-cache/hub/models--MCG-NJU--videomae-base-finetuned-kinetics")
    / "snapshots" / "488eb9a0565f257b32866000305c8178965eb9f6"
)


# ---------------------------------------------------------------------------
# sample resolution
# ---------------------------------------------------------------------------

def _resolve(ref: Any) -> np.ndarray:
    """Return one video clip as uint8 ``(T, H, W, 3)``.

    Accepts a plain ``.npy`` path, an ``h5://<file>#<dataset>[<index>]``
    reference (how the loader records ground truth), or an array.
    """

    if isinstance(ref, np.ndarray):
        return np.asarray(ref, dtype=np.uint8)
    if isinstance(ref, dict):
        for key in ("data", "path", "ref", "video"):
            if key in ref:
                return _resolve(ref[key])
        raise ValueError(f"cannot resolve video reference {ref!r}")

    text = str(ref)
    if text.startswith("h5://"):
        import h5py

        body = text[len("h5://"):]
        path, _, fragment = body.partition("#")
        name, _, index = fragment.partition("[")
        index = int(index.rstrip("]")) if index else 0
        with h5py.File(path, "r") as h5:
            if name not in h5:
                raise KeyError(f"{name!r} not in {path}")
            return np.asarray(h5[name][index], dtype=np.uint8)

    # "<path>#<index>" addresses one clip inside a stacked .npy of videos.
    index = None
    if "#" in text and not text.startswith("h5://"):
        text, _, tail = text.partition("#")
        if tail.strip():
            index = int(tail.strip())

    path = Path(text)
    if not path.is_file():
        raise FileNotFoundError(f"video reference does not exist: {path}")
    if path.suffix == ".npy":
        array = np.asarray(np.load(path, mmap_mode="r"))
        if index is not None:
            array = np.asarray(array[index])
    elif path.suffix in (".mp4", ".gif", ".avi"):
        import imageio.v2 as imageio

        array = np.asarray(imageio.mimread(str(path)))
    else:
        raise ValueError(f"unsupported video reference: {text}")
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 4:
        raise ValueError(f"expected (T,H,W,3) video, got {array.shape}")
    return array.astype(np.uint8)


def _resize_to_reference(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match resolution and frame count to the reference clip, as the
    released evaluator does by reading same-shaped GT and generated arrays."""

    if frames.shape[1:3] != reference.shape[1:3]:
        from PIL import Image

        size = (reference.shape[2], reference.shape[1])  # (W, H)
        frames = np.stack([
            np.asarray(Image.fromarray(f).resize(size, Image.BILINEAR))
            for f in frames
        ])
    if frames.shape[0] != reference.shape[0]:
        take = min(frames.shape[0], reference.shape[0])
        frames = frames[:take]
        reference = reference[:take]
    return frames


# ---------------------------------------------------------------------------
# frozen classifiers (shared, loaded once)
# ---------------------------------------------------------------------------

_CLASSIFIERS: Optional[dict] = None


def _ensure_classifiers(num_frames: int = 6) -> dict:
    """Load the frozen ViT and VideoMAE pair once per frame count.

    ``num_frames`` must be passed through: VideoMAE's tubelet embedding is
    built for a fixed frame count, and leaving the default in place while
    feeding 6 frames fails with a positional-embedding size mismatch.  The
    EEG2Video project passes ``num_frames=SETTINGS.video_frames`` (6) for the
    same reason.
    """

    global _CLASSIFIERS
    if _CLASSIFIERS is not None and _CLASSIFIERS["num_frames"] == num_frames:
        return _CLASSIFIERS

    import functools

    import torch

    # ``transformers``' image processors call ``is_vision_available()`` once per
    # image.  That function interrogates installed package metadata, and on a
    # 200-sample clip it ran 136,046 times -- 544 s of a 1610 s evaluation, a
    # third of the wall clock, spent asking whether Pillow is installed.  It is
    # a constant, so memoise it here, before any processor is constructed.
    from transformers.utils import import_utils as _iu
    if not isinstance(getattr(_iu.is_vision_available, "__wrapped__", None), type(None)) \
            or not hasattr(_iu.is_vision_available, "cache_info"):
        _iu.is_vision_available = functools.lru_cache(maxsize=1)(
            _iu.is_vision_available)

    from transformers import (
        ViTForImageClassification,
        ViTImageProcessor,
        VideoMAEForVideoClassification,
        VideoMAEImageProcessor,
    )

    if not VIT_ROOT.exists() or not VIDEOMAE_ROOT.exists():
        raise FileNotFoundError(
            f"frozen evaluator weights missing: {VIT_ROOT} / {VIDEOMAE_ROOT}"
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    _CLASSIFIERS = {
        "device": device,
        "num_frames": int(num_frames),
        "image_processor": ViTImageProcessor.from_pretrained(
            str(VIT_ROOT), local_files_only=True),
        "image_model": ViTForImageClassification.from_pretrained(
            str(VIT_ROOT), local_files_only=True).to(device=device, dtype=dtype).eval(),
        "video_processor": VideoMAEImageProcessor.from_pretrained(
            str(VIDEOMAE_ROOT), local_files_only=True),
        "video_model": VideoMAEForVideoClassification.from_pretrained(
            str(VIDEOMAE_ROOT), local_files_only=True,
            num_frames=int(num_frames)).to(device=device, dtype=dtype).eval(),
    }
    log.info("loaded frozen ViT + VideoMAE (%d frames) on %s", num_frames, device)
    return _CLASSIFIERS


# ---------------------------------------------------------------------------
# N-way trial loop -- copied verbatim from EEG2Video_clean_d_v2
# ---------------------------------------------------------------------------

def _n_way_top_k(prediction, target_classes, n_way: int, trials: int,
                 top_k: int, generator: np.random.RandomState) -> float:
    """Match the released evaluator's target ordering and N-way decision."""

    import torch

    target_classes = target_classes.cpu().numpy().astype(np.int64)
    target_set = set(target_classes.tolist())
    candidates = np.asarray(
        [i for i in range(prediction.numel()) if i not in target_set])
    if n_way - 1 > len(candidates):
        raise ValueError(f"Cannot draw {n_way} classes from {prediction.numel()} outputs")
    correct = 0
    for _ in range(trials):
        distractors = generator.choice(candidates, n_way - 1, replace=False)
        hit = False
        for target_class in target_classes:
            indices = np.concatenate((np.asarray([target_class]), distractors))
            scores = prediction[torch.from_numpy(indices)].numpy()
            if 0 in np.argsort(scores)[-top_k:]:
                hit = True
                break
        if hit:
            correct += 1
    return correct / trials


def _logits_for_clip(predicted: np.ndarray, ground_truth: np.ndarray):
    """Return per-frame ViT logits for both clips and the VideoMAE logits."""

    key = (id(predicted), id(ground_truth))
    hit = _LOGITS_CACHE.get(key)
    if hit is not None:
        return hit

    import torch

    bundle = _ensure_classifiers(int(predicted.shape[0]))
    device = bundle["device"]
    image_processor = bundle["image_processor"]
    image_model = bundle["image_model"]
    video_processor = bundle["video_processor"]
    video_model = bundle["video_model"]
    model_dtype = next(image_model.parameters()).dtype

    image_scores, image_targets = [], []
    for frame in range(predicted.shape[0]):
        pred = image_processor(images=predicted[frame], return_tensors="pt")
        gt = image_processor(images=ground_truth[frame], return_tensors="pt")
        with torch.no_grad():
            pred_logits = image_model(
                **{k: v.to(device=device, dtype=model_dtype) for k, v in pred.items()}
            ).logits[0]
            gt_logits = image_model(
                **{k: v.to(device=device, dtype=model_dtype) for k, v in gt.items()}
            ).logits[0]
        image_scores.append(pred_logits.softmax(-1).float().cpu())
        image_targets.append(gt_logits.argsort(descending=False).flatten()[-GT_TOP_K:].cpu())

    pred_input = video_processor(list(predicted), return_tensors="pt")
    gt_input = video_processor(list(ground_truth), return_tensors="pt")
    video_dtype = next(video_model.parameters()).dtype
    with torch.no_grad():
        pred_logits = video_model(
            **{k: v.to(device=device, dtype=video_dtype) for k, v in pred_input.items()}
        ).logits[0]
        gt_logits = video_model(
            **{k: v.to(device=device, dtype=video_dtype) for k, v in gt_input.items()}
        ).logits[0]
    video_scores = pred_logits.softmax(-1).float().cpu()
    video_targets = gt_logits.argsort(descending=False)[-GT_TOP_K:].cpu()
    result = (image_scores, image_targets, video_scores, video_targets)
    # The arrays stay alive in _PAIR_CACHE, so their ids remain valid.
    _LOGITS_CACHE[key] = result
    return result


def _paired(predictions: List[Any], ground_truths: List[Any]):
    for pred, gt in zip(predictions, ground_truths):
        if pred is None or gt is None:
            continue
        try:
            predicted = _resolve(pred)
            ground_truth = _resolve(gt)
        except Exception as exc:  # noqa: BLE001 - reported, not silent
            log.warning("skipping unresolvable sample: %s", exc)
            continue
        if predicted.size == 0 or ground_truth.size == 0:
            continue
        yield _resize_to_reference(predicted, ground_truth), ground_truth


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class _ClipNWayAccuracy(Metric):
    """Shared N-way accuracy over either the frame branch or the video branch."""

    branch = "frame"          # overridden by VideoNWayAccuracy
    n_way = 2                 # overridden by the registered instances
    mode = MODE_MATCHED       # matched / shuffled / delta

    @property
    def applicable(self) -> List[str]:
        return ["video_generation"]

    def _key(self) -> str:
        base = f"{self.branch}_{self.n_way}way_accuracy"
        return base if self.mode == MODE_MATCHED else f"{base}_{self.mode}"

    @property
    def name(self) -> str:
        return self._key()

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      seed: int = SEED, trials: int = N_TRIALS, **kwargs) -> float:
        matched, shuffled = self._matched_and_shuffled(
            predictions, ground_truths, seed, trials)
        return _pick(matched, shuffled, self.mode)

    def _matched_and_shuffled(self, predictions, ground_truths, seed, trials):
        """Per-sample scores against own GT and against a deranged GT.

        Both passes use a RandomState seeded identically, so sample *i* draws
        the same distractors in each: the delta then isolates the pairing
        rather than mixing in a different distractor sample.
        """

        key = _cache_key(f"nway-{self.branch}-{self.n_way}",
                         predictions, ground_truths)
        cached = _SCORE_CACHE.get(key)
        if cached is not None:
            return cached

        pairs = _cached_pairs(predictions, ground_truths)
        if not pairs:
            _SCORE_CACHE[key] = ([], [])
            return [], []
        predicted_list = [p for p, _ in pairs]
        truth_list = [g for _, g in pairs]
        shuffled_list = _shuffled(truth_list)

        def score(preds, truths):
            rng = np.random.RandomState(seed)
            out: List[float] = []
            for predicted, ground_truth in zip(preds, truths):
                image_scores, image_targets, video_scores, video_targets = \
                    _logits_for_clip(predicted, ground_truth)
                if self.branch == "frame":
                    for one_score, one_target in zip(image_scores, image_targets):
                        out.append(_n_way_top_k(
                            one_score, one_target, self.n_way, trials, 1, rng))
                else:
                    out.append(_n_way_top_k(
                        video_scores, video_targets, self.n_way, trials, 1, rng))
            return out

        result = (score(predicted_list, truth_list),
                  score(predicted_list, shuffled_list))
        _SCORE_CACHE[key] = result
        return result

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return self.batch_compute([predictions], [ground_truths], **kwargs)


class VideoNWayAccuracy(_ClipNWayAccuracy):
    """VideoMAE N-way accuracy: 2-way and 40-way."""

    branch = "video"

    def __init__(self, n_way: int, mode: str = MODE_MATCHED):
        self.n_way = int(n_way)
        self.mode = mode


class FrameNWayAccuracy(_ClipNWayAccuracy):
    """Per-frame ViT N-way accuracy, averaged over the clip's frames."""

    branch = "frame"

    def __init__(self, n_way: int, mode: str = MODE_MATCHED):
        self.n_way = int(n_way)
        self.mode = mode


class VideoSSIM(Metric):
    """Per-frame SSIM between generated and ground-truth clips, then averaged."""

    mode = MODE_MATCHED

    def __init__(self, mode: str = MODE_MATCHED):
        self.mode = mode

    @property
    def name(self) -> str:
        base = "video_ssim"
        return base if self.mode == MODE_MATCHED else f"{base}_{self.mode}"

    @property
    def applicable(self) -> List[str]:
        return ["video_generation"]

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        matched, shuffled = self._matched_and_shuffled(predictions, ground_truths)
        return _pick(matched, shuffled, self.mode)

    def _matched_and_shuffled(self, predictions, ground_truths):
        key = _cache_key("ssim", predictions, ground_truths)
        cached = _SCORE_CACHE.get(key)
        if cached is not None:
            return cached

        from skimage.metrics import structural_similarity

        pairs = list(_paired(predictions, ground_truths))
        if not pairs:
            _SCORE_CACHE[key] = ([], [])
            return [], []

        def score(truths):
            values: List[float] = []
            for (predicted, _), ground_truth in zip(pairs, truths):
                for frame in range(predicted.shape[0]):
                    values.append(
                        structural_similarity(
                            predicted[frame], ground_truth[frame],
                            channel_axis=-1, data_range=255,
                        )
                    )
            return values

        result = (score([g for _, g in pairs]),
                  score(_shuffled([g for _, g in pairs])))
        _SCORE_CACHE[key] = result
        return result

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return self.batch_compute([predictions], [ground_truths], **kwargs)


# ---------------------------------------------------------------------------
# MindCine's two extra pixel-level metrics
# ---------------------------------------------------------------------------
# MindCine (arXiv:2601.18192) reports seven numbers; five of them are the
# metrics above.  The other two are PSNR and Hue-pcc, listed as "pixel-level"
# alongside SSIM, and both copied here from the released evaluator
# (``MindCine/Tune-A-Video/40_class_run_metrics.py``) so their values can sit in
# the same column of a table as the paper's.
#
# Hue-pcc is copied *arithmetic for arithmetic*, and that matters, because the
# released code does not compute what the paper says it does:
#
#   paper  "hue-based Pearson correlation coefficient [22] (Hue-pcc)", citing
#          Swain & Ballard, *Color indexing* -- i.e. a correlation.
#   code   a **cosine similarity** between the two hue planes; its own comment
#          says so ("计算余弦相似度").
#   code   it also feeds ``imageio``'s **RGB** frames to ``cv2.COLOR_BGR2HSV``,
#          so R and B are swapped before the hue is taken.
#
# The difference is not academic: on two consecutive ground-truth frames of one
# clip the three readings are 0.9947 (released), 0.9632 (channel order
# corrected) and 0.9029 (a true Pearson correlation).  Reproducing the released
# arithmetic is the choice here -- nothing in this repository can check that
# either of the other two reproduces the paper's table -- and the divergence is
# recorded in the experiment report rather than smoothed over.
#
# Known interaction with the EEG-Video-Action branch: three of its thirteen
# stimuli are letterboxed, so part of every frame is black by construction.
# That is identical in prediction and ground truth for a matched pairing, but a
# generated clip that fills the padding differently is penalised by both of
# these metrics.  Read the delta there, as everywhere else on these datasets.


class VideoPSNR(Metric):
    """Per-frame PSNR between generated and ground-truth clips, then averaged.

    Copied from the released evaluator's ``psnr_score_only``/``psnr``:
    ``20 * log10(255 / sqrt(MSE))`` over the whole frame, with a perfect match
    reported as 100 rather than infinity.
    """

    mode = MODE_MATCHED

    def __init__(self, mode: str = MODE_MATCHED):
        self.mode = mode

    @property
    def name(self) -> str:
        base = "video_psnr"
        return base if self.mode == MODE_MATCHED else f"{base}_{self.mode}"

    @property
    def applicable(self) -> List[str]:
        return ["video_generation"]

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        matched, shuffled = self._matched_and_shuffled(predictions, ground_truths)
        return _pick(matched, shuffled, self.mode)

    def _matched_and_shuffled(self, predictions, ground_truths):
        key = _cache_key("psnr", predictions, ground_truths)
        cached = _SCORE_CACHE.get(key)
        if cached is not None:
            return cached

        pairs = list(_paired(predictions, ground_truths))
        if not pairs:
            _SCORE_CACHE[key] = ([], [])
            return [], []

        def score(truths):
            values: List[float] = []
            for (predicted, _), ground_truth in zip(pairs, truths):
                for frame in range(predicted.shape[0]):
                    values.append(_psnr_frame(predicted[frame], ground_truth[frame]))
            return values

        result = (score([g for _, g in pairs]),
                  score(_shuffled([g for _, g in pairs])))
        _SCORE_CACHE[key] = result
        return result

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return self.batch_compute([predictions], [ground_truths], **kwargs)


def _psnr_frame(img1: np.ndarray, img2: np.ndarray) -> float:
    """The released ``psnr()``, unchanged apart from the return type.

    ``mse < 1e-10`` -> 100 is the released cap: an identical frame would
    otherwise be ``+inf``, and a single one would poison the mean.
    """

    mse = float(np.mean((np.asarray(img1, dtype=np.float64)
                         - np.asarray(img2, dtype=np.float64)) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


class VideoHuePCC(Metric):
    """Per-frame hue similarity between clips, then averaged.

    The released ``calculate_hue_similarity``, reproduced as written: a cosine
    similarity of the two hue planes, with the source frames handed to
    ``cv2.COLOR_BGR2HSV`` unchanged.  See the block comment above for why the
    metric's name and its arithmetic disagree, and why the arithmetic won.
    """

    mode = MODE_MATCHED

    def __init__(self, mode: str = MODE_MATCHED):
        self.mode = mode

    @property
    def name(self) -> str:
        base = "video_hue_pcc"
        return base if self.mode == MODE_MATCHED else f"{base}_{self.mode}"

    @property
    def applicable(self) -> List[str]:
        return ["video_generation"]

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        matched, shuffled = self._matched_and_shuffled(predictions, ground_truths)
        return _pick(matched, shuffled, self.mode)

    def _matched_and_shuffled(self, predictions, ground_truths):
        key = _cache_key("hue_pcc", predictions, ground_truths)
        cached = _SCORE_CACHE.get(key)
        if cached is not None:
            return cached

        pairs = list(_paired(predictions, ground_truths))
        if not pairs:
            _SCORE_CACHE[key] = ([], [])
            return [], []

        def score(truths):
            values: List[float] = []
            for (predicted, _), ground_truth in zip(pairs, truths):
                for frame in range(predicted.shape[0]):
                    values.append(
                        _hue_similarity_frame(predicted[frame], ground_truth[frame]))
            return values

        result = (score([g for _, g in pairs]),
                  score(_shuffled([g for _, g in pairs])))
        _SCORE_CACHE[key] = result
        return result

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return self.batch_compute([predictions], [ground_truths], **kwargs)


#: Warn at most once per process: a hue-less frame is a property of the
#: prediction, not of one sample, and a per-sample warning would be megabytes.
_HUE_DEGENERATE_WARNED = False


def _hue_similarity_frame(img1: np.ndarray, img2: np.ndarray) -> float:
    """The released ``calculate_hue_similarity``, with one defined edge case.

    Verbatim otherwise: ``COLOR_BGR2HSV`` on whatever array is handed in, and
    the cosine taken over the flattened, ``/ 360.0``-scaled hue planes.  The
    scaling is kept even though OpenCV reports 8-bit hue over 0..179 and a
    cosine is scale-invariant, so the constant cannot change the result.

    **The one addition.**  A frame with no hue at all -- black, or any
    greyscale image, where OpenCV reports H = 0 for every pixel -- makes both
    norms zero and the released expression return ``nan``.  That ``nan``
    averages into a ``nan`` track score, which reads as a broken metric rather
    than as a hue-less prediction.  Such a frame scores 0.0 (no hue agreement)
    and is reported once.
    """

    global _HUE_DEGENERATE_WARNED

    import cv2

    hue1 = cv2.cvtColor(np.ascontiguousarray(img1), cv2.COLOR_BGR2HSV)[:, :, 0]
    hue2 = cv2.cvtColor(np.ascontiguousarray(img2), cv2.COLOR_BGR2HSV)[:, :, 0]
    hue1 = hue1.astype(np.float64) / 360.0
    hue2 = hue2.astype(np.float64) / 360.0
    norm1 = float(np.sqrt(np.sum(hue1 ** 2)))
    norm2 = float(np.sqrt(np.sum(hue2 ** 2)))
    if norm1 == 0.0 or norm2 == 0.0:
        if not _HUE_DEGENERATE_WARNED:
            _HUE_DEGENERATE_WARNED = True
            log.warning(
                "video_hue_pcc: a frame has no hue (black or greyscale), so "
                "the cosine is undefined; scoring it 0.0. Further occurrences "
                "are not reported.")
        return 0.0
    return float(np.sum(hue1 * hue2) / (norm1 * norm2))
