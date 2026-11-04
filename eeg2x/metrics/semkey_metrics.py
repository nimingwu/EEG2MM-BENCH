"""
SemKey evaluation metrics (arXiv 2603.03312v2).

The paper's argument is that n-gram overlap alone cannot separate a model that
decodes EEG from one that leans on the frozen language model's prior, so it
scores sentence generation along four axes:

* **Generation (Table 2)** — BLEU under two reference protocols:
  ``MTV`` (the 8 rewritten/simplified variants of the stimulus sentence) and
  ``Standard`` (the raw stimulus text only).  The gap between them is the
  "BLEU Trap": Standard BLEU is nearly blind to whether the input is real EEG
  or noise.
* **Retrieval (Table 1)** — N-way retrieval accuracy: sample N (prediction,
  reference) pairs, embed both with SBERT, and check whether each prediction's
  nearest reference is its own.  Chance is 1/N, so this is a direct read-out
  of whether the generated text carries reference-specific information.
* **Substance / Diversity (Table 1)** — content-word recall, Dist-1/Dist-2,
  head entropy.  Diversity is NOT "higher is better": natural English repeats
  function words heavily (human ZuCo references sit near Dist-1 4.7% /
  Dist-2 9.6%), so a value far above that indicates degenerate text rather
  than rich text — the paper reports 99.6% Dist-2 for noise input.
* **Quality (Table 1)** — Fréchet distance between prediction and reference
  SBERT embedding distributions (lower is better).

Self-BLEU is deliberately **not** implemented here (excluded by request); the
paper's own runners often skip it because it is O(N^2).

Implementation notes
--------------------
* Metric maths are ported from ``SemKey/evaluation/evaluate.py`` and kept
  faithful, including the use of ``torchmetrics``' ``bleu_score`` and the
  per-sample-then-averaged reduction.  Values are therefore comparable to the
  paper's tables rather than to this benchmark's existing ``bleu_*`` metrics,
  which use nltk's smoothed ``sentence_bleu``.
* **Every heavy dependency is imported inside the method that needs it.**
  ``metrics/__init__.py`` imports this module eagerly, and the benchmark's
  own environment does not ship ``bert_score`` / ``sentence_transformers`` /
  ``pycocoevalcap``; a module-level import would break ``import metrics`` for
  every task.  ``metrics/bert_score.py`` sets the same precedent.
* N-way retrieval, Dist-n, head entropy and FD are corpus-level: call
  ``batch_compute``, not ``compute``.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .base import Metric
from .tokenization import retokenize

log = logging.getLogger(__name__)

# NLTK corpora needed by Dist-n / content recall / head entropy.  The env may
# only have some of them; both known locations are added if present.
_NLTK_CANDIDATES = (
    os.environ.get("EEG2X_NLTK_DATA", ""),
    "/datas/eeg_benchmark_project/fuxian/fuxian-SEMKEY/SemKey/data/zuco_preprocessed_dataframe/nltk_data",
    "/root/nltk_data",
)

# Cached SBERT handles, keyed by (model_name, device) -- SemKey instantiates
# the encoder once per metric family; sharing it keeps the cost the same while
# avoiding four redundant loads of an identical model.
_SBERT_CACHE: Dict[str, Any] = {}


def _ensure_nltk() -> None:
    # Delegates to the shared resolver so this module and text_metrics.py cannot
    # drift apart on where the NLTK corpora live.
    from .tokenization import ensure_nltk
    ensure_nltk()


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sbert_cache_dir() -> str:
    """Shared HF cache, matching the `model_cache_dir` the EEG encoders use.

    Left to itself, sentence-transformers falls back to
    ~/.cache/torch/sentence_transformers and would download ~420 MB into $HOME
    -- on this box $HOME is small enough that filling it has already broken
    runs once.  Pointing at the same cache as everything else also means the
    weights survive an env rebuild.
    """
    return os.environ.get("EEG2X_HF_CACHE", "/datas/hf-cache")


def _get_sbert(model_name: str, device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer
    device = device or _device()
    key = f"{model_name}|{device}"
    if key not in _SBERT_CACHE:
        _SBERT_CACHE[key] = SentenceTransformer(
            model_name, device=device, cache_folder=_sbert_cache_dir())
    return _SBERT_CACHE[key]


def _as_str_list(values) -> List[str]:
    out = []
    for v in values:
        if v is None:
            out.append("")
        elif isinstance(v, float) and v != v:  # NaN
            out.append("")
        else:
            out.append(str(v))
    return out


# ===================================================================
# Generation — BLEU under the MTV and Standard reference protocols
# ===================================================================

class SemKeyBLEU(Metric):
    """BLEU-N averaged per sample, under one of the two reference protocols.

    Parameters
    ----------
    order : int
        n-gram order (1 or 2 for the paper's tables; 1-4 supported).
    protocol : {"mtv", "standard"}
        ``mtv`` scores against ``kwargs["mtv_references"]`` (the 8 variants);
        ``standard`` scores against the single ``sentence_text`` ground truth.
    """

    def __init__(self, order: int = 1, protocol: str = "mtv"):
        if protocol not in ("mtv", "standard"):
            raise ValueError(f"unknown protocol {protocol!r}")
        self.order = int(order)
        self.protocol = protocol

    @property
    def name(self) -> str:
        return f"semkey_bleu{self.order}_{self.protocol}"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        refs = kwargs.get("references")
        if self.protocol == "mtv":
            if not refs:
                raise ValueError(
                    "semkey MTV BLEU needs kwargs['references'] (the variant "
                    "list per sample); pass mtv_references from the HDF5 "
                    "target_texts_json field")
            return self._bleu(predictions, list(refs))
        return self._bleu(predictions, [ground_truths])

    def _bleu(self, pred: str, refs: List[str]) -> float:
        from torchmetrics.functional.text import bleu_score
        # torchmetrics tokenises on whitespace.  Chinese has none, so a raw
        # Chinese sentence is ONE token and any two different sentences score
        # 0.0 regardless of quality (measured: 0.0000 vs 0.5969 once split into
        # characters).  Pre-split so the library's own split recovers our tokens.
        return float(bleu_score([retokenize(pred) or " "],
                                [[retokenize(r) for r in refs]],
                                n_gram=self.order).item())

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        import numpy as np
        preds = _as_str_list(predictions)
        gts = _as_str_list(ground_truths)
        mtv = kwargs.get("mtv_references")
        if self.protocol == "mtv" and not mtv:
            raise ValueError(
                "semkey MTV BLEU requires the HDF5 'target_texts_json' field; "
                "this dataset does not provide it")
        scores = []
        for i, (p, g) in enumerate(zip(preds, gts)):
            refs: List[str] = list(mtv[i]) if self.protocol == "mtv" else [g]
            try:
                scores.append(self._bleu(p, refs))
            except Exception as exc:
                log.warning("semkey BLEU failed for sample %d: %s", i, exc)
        return float(np.mean(scores)) if scores else 0.0


# ===================================================================
# Retrieval — N-way accuracy (corpus level; chance = 1/N)
# ===================================================================

class SemKeyNwayRetrieval(Metric):
    """N-way retrieval accuracy over SBERT embeddings (paper's Table 1)."""

    def __init__(self, n_ways: int = 2, num_trials: int = 10000,
                 model_name: str = "all-mpnet-base-v2"):
        self.n = int(n_ways)
        self.num_trials = int(num_trials)
        self.model_name = model_name

    @property
    def name(self) -> str:
        return f"semkey_nway_{self.n}"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        raise NotImplementedError(
            "semkey N-way retrieval is a corpus-level metric; call batch_compute")

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        import torch
        preds = _as_str_list(predictions)
        gts = _as_str_list(ground_truths)
        total = len(preds)
        if total < self.n:
            log.warning("not enough samples (%d) for %d-way retrieval", total, self.n)
            return 0.0
        model = _get_sbert(self.model_name, kwargs.get("device"))
        pred_emb = model.encode(preds, batch_size=64, convert_to_tensor=True,
                                show_progress_bar=False, normalize_embeddings=True)
        tgt_emb = model.encode(gts, batch_size=64, convert_to_tensor=True,
                               show_progress_bar=False, normalize_embeddings=True)
        torch.random.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        accs = []
        for _ in range(self.num_trials):
            idx = torch.randperm(total)[: self.n]
            sim = torch.mm(pred_emb[idx], tgt_emb[idx].transpose(0, 1))
            correct = (sim.argmax(dim=1) == torch.arange(self.n, device=sim.device)).sum().item()
            accs.append(correct / self.n)
        return float(sum(accs) / len(accs))


# ===================================================================
# Substance / Diversity (corpus level)
# ===================================================================

class SemKeyContentRecall(Metric):
    """Proportion of reference content words recovered by the prediction.

    Stopwords are excluded so that fluent-but-empty text cannot score well.
    """

    @property
    def name(self) -> str:
        return "semkey_content_recall"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return self.batch_compute([predictions], [ground_truths], **kwargs)

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        _ensure_nltk()
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        stop = set(stopwords.words("english"))
        preds = _as_str_list(predictions)
        gts = _as_str_list(ground_truths)
        total, valid = 0.0, 0
        for pred, ref in zip(preds, gts):
            ref_c = [w.lower() for w in word_tokenize(ref) if w.isalnum() and w.lower() not in stop]
            pred_c = [w.lower() for w in word_tokenize(pred) if w.isalnum() and w.lower() not in stop]
            if not ref_c:
                continue
            total += sum((Counter(pred_c) & Counter(ref_c)).values()) / len(ref_c)
            valid += 1
        return float(total / valid) if valid else 0.0


class SemKeyDistN(Metric):
    """Dist-N: unique n-grams / total n-grams over the whole generated corpus.

    NOT an error signal on its own -- compare against the reference corpus
    (see the module docstring); degenerate text scores far above human text.
    """

    def __init__(self, n: int = 1):
        self.n = int(n)

    @property
    def name(self) -> str:
        return f"semkey_dist_{self.n}"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        raise NotImplementedError(
            "semkey Dist-N is a corpus-level metric; call batch_compute")

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        _ensure_nltk()
        from nltk.tokenize import word_tokenize
        grams: List[Tuple[str, ...]] = []
        for sent in _as_str_list(predictions):
            toks = word_tokenize(sent.lower())
            if len(toks) < self.n:
                continue
            grams.extend(tuple(toks[i:i + self.n]) for i in range(len(toks) - self.n + 1))
        return float(len(set(grams)) / len(grams)) if grams else 0.0


class SemKeyHeadEntropy(Metric):
    """Shannon entropy (bits) of the first-two-token prefix distribution.

    Low entropy means the model opens every sentence the same way; very high
    entropy on noise input reflects unstructured output rather than variety.
    """

    @property
    def name(self) -> str:
        return "semkey_head_entropy"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        raise NotImplementedError(
            "semkey head entropy is a corpus-level metric; call batch_compute")

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        import numpy as np
        _ensure_nltk()
        from nltk.tokenize import word_tokenize
        prefixes = []
        for s in _as_str_list(predictions):
            toks = word_tokenize(s.lower())
            if len(toks) >= 2:
                prefixes.append(tuple(toks[:2]))
        if not prefixes:
            return 0.0
        counts = Counter(prefixes)
        total = sum(counts.values())
        return float(-sum((c / total) * np.log2(c / total) for c in counts.values()))


# ===================================================================
# Quality — Fréchet distance between SBERT embedding distributions
# ===================================================================

class SemKeyFrechetDistance(Metric):
    """Fréchet distance between prediction and reference SBERT embeddings.

    Lower is better.  Requires more samples than embedding dimensions to make
    the covariance estimate meaningful; the paper's test splits satisfy this,
    but small subsets will not.
    """

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name

    @property
    def name(self) -> str:
        return "semkey_frechet_distance"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        raise NotImplementedError(
            "semkey Frechet distance is a corpus-level metric; call batch_compute")

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        import numpy as np
        import scipy.linalg
        preds = _as_str_list(predictions)
        gts = _as_str_list(ground_truths)
        model = _get_sbert(self.model_name, kwargs.get("device"))
        a = np.asarray(model.encode(preds, batch_size=64, convert_to_tensor=False,
                                    show_progress_bar=False), dtype=np.float64)
        b = np.asarray(model.encode(gts, batch_size=64, convert_to_tensor=False,
                                    show_progress_bar=False), dtype=np.float64)
        return float(_frechet(a, b))


def _frechet(act1, act2) -> float:
    """Fréchet distance with SemKey's numerical-stability fixes."""
    import numpy as np
    import scipy.linalg
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    ssdiff = float(np.sum((mu1 - mu2) ** 2.0))
    eps = 1e-6
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + float(np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return max(0.0, fid)


# ===================================================================
# Semantic similarity (per-sample averaged)
# ===================================================================

class SemKeySBERTCosine(Metric):
    """Mean cosine similarity between prediction and reference SBERT vectors."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model_name = model_name

    @property
    def name(self) -> str:
        return "semkey_sbert_cosine"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return self.batch_compute([predictions], [ground_truths], **kwargs)

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        import torch
        import torch.nn.functional as F
        preds = _as_str_list(predictions)
        gts = _as_str_list(ground_truths)
        if not preds:
            return 0.0
        model = _get_sbert(self.model_name, kwargs.get("device"))
        pe = model.encode(preds, batch_size=64, convert_to_tensor=True,
                          show_progress_bar=False)
        te = model.encode(gts, batch_size=64, convert_to_tensor=True,
                          show_progress_bar=False)
        return float(F.cosine_similarity(pe, te, dim=1).mean().item())


class SemKeyLengthMAE(Metric):
    """Mean absolute error of sentence length in words (diagnostic)."""

    @property
    def name(self) -> str:
        return "semkey_length_mae"

    @property
    def applicable(self) -> List[str]:
        return ["sentence_generation"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return self.batch_compute([predictions], [ground_truths], **kwargs)

    def batch_compute(self, predictions: List[Any], ground_truths: List[Any],
                      **kwargs) -> float:
        import numpy as np
        _ensure_nltk()
        from nltk.tokenize import word_tokenize
        diffs = []
        for p, g in zip(_as_str_list(predictions), _as_str_list(ground_truths)):
            diffs.append(abs(len(word_tokenize(p)) - len(word_tokenize(g))))
        return float(np.mean(diffs)) if diffs else 0.0


# ------------------------------------------------------------------
# Registry-facing constructor list
# ------------------------------------------------------------------

def default_metrics() -> Dict[str, Metric]:
    """Metric instances SemKey reports, keyed by registry name.

    Self-BLEU is intentionally absent.
    """
    out: Dict[str, Metric] = {}
    for order in (1, 2):
        for proto in ("mtv", "standard"):
            m = SemKeyBLEU(order=order, protocol=proto)
            out[m.name] = m
    for n in (2, 4, 10, 24):
        m = SemKeyNwayRetrieval(n_ways=n)
        out[m.name] = m
    for n in (1, 2):
        m = SemKeyDistN(n=n)
        out[m.name] = m
    for m in (SemKeyContentRecall(), SemKeyHeadEntropy(),
              SemKeyFrechetDistance(), SemKeySBERTCosine(), SemKeyLengthMAE()):
        out[m.name] = m
    return out
