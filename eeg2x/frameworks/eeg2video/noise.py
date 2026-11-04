"""The noise control: run inference with the EEG replaced by Gaussian noise.

This is not a robustness metric and not an SNR sweep.  It is a **control**: it
asks whether a model is using the EEG at all.  Every trial's EEG is replaced by
noise of the same shape while the captions, labels and prompts stay untouched;
if the model still produces plausible video, its output is coming from the
language/visual prior rather than from the brain signal.

Ported from ``EEG2X-Benchmark-v2-GS/NOISE_EXPERIMENT.md``, which in turn follows
GLIM's ``eval_noise_input``.  Three of that document's conventions are load
bearing and are enforced here:

1. **A noise run never overwrites a clean one.**  Callers go on passing the
   clean path; :func:`variant_path` adds a ``_noise`` suffix.
2. **Condition metadata does not go inside the predictions JSON.**
   ``benchmark_eval.py`` reads every top-level key as a sample id, so a
   ``{"noise": "gaussian"}`` entry would be scored as a sample.  It goes in a
   sidecar ``<stem>.meta.json`` instead.
3. **The seed is fixed and recorded**, and one generator serves the whole run,
   so the same batch size and data order reproduce the same noise.

One deliberate deviation from the reference document
----------------------------------------------------
GLIM replaces the EEG with ``standard_normal`` outright.  That is only a fair
control when the stored EEG is already at unit scale, and in this benchmark it
is not: the ``eegnet``/``reve``/``eegvid_nice`` branches hold raw EEG at a
standard deviation of ~1e-4 (SEED-DV 1.06e-4, ds008257 9.7e-6), while the
``labram``/``luna`` branches hold it standardised to ~1.0.  Literal N(0, 1)
would therefore hand the raw branches noise **ten thousand times larger** than
the signal -- a scale attack, not an EEG-content control, and it would answer a
different question than the one asked.

So :func:`maybe_noise` matches each element's own mean and standard deviation
along the last axis.  On the already-standardised branches that is exactly
N(0, 1) and agrees with the reference; on the raw branches it is the same
experiment at the right magnitude.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

#: ``--noise none`` (the default) runs the clean condition.
MODE_NONE = "none"
#: ``--noise gaussian`` replaces the EEG with scale-matched Gaussian noise.
MODE_GAUSSIAN = "gaussian"
MODES = (MODE_NONE, MODE_GAUSSIAN)

#: Appended to every output path a noise run writes.
SUFFIX = "_noise"

_mode: str = MODE_NONE
_seed: Optional[int] = None
_rng: Optional[np.random.Generator] = None


def configure(mode: str, seed: Optional[int] = None) -> None:
    """Arm or disarm the control.  Call once, before inference.

    ``seed`` defaults to the benchmark seed when omitted; it is recorded in the
    sidecar so a run can be reproduced.
    """

    global _mode, _seed, _rng
    if mode not in MODES:
        raise ValueError(f"unknown noise mode {mode!r}; expected one of {list(MODES)}")
    _mode = mode
    _seed = None if seed is None else int(seed)
    # One generator for the whole run, on the numpy side.  Sampling here rather
    # than in torch also sidesteps the reference document's second pitfall: a
    # CPU torch.Generator cannot fill a CUDA tensor, and making the sampling
    # device-dependent would give a different control on every device.
    _rng = np.random.default_rng(_seed) if mode == MODE_GAUSSIAN else None


def add_cli_arguments(parser: Any) -> None:
    """Give a chain driver the two flags, identically.

    Every driver gets the same ``--noise`` / ``--noise-seed`` pair from here
    rather than each spelling its own, so a new baseline cannot end up with a
    subtly different control (a different default seed, a missing suffix).
    """

    parser.add_argument(
        "--noise", choices=MODES, default=MODE_NONE,
        help="noise control: replace each trial's EEG with scale-matched "
             "Gaussian noise and write the predictions and report under "
             "*_noise names, leaving the clean artefacts alone")
    parser.add_argument(
        "--noise-seed", type=int, default=None,
        help="seed for the noise draw (default: the benchmark SEED)")


def configure_from_args(args: Any, default_seed: int,
                        logger: Any = None) -> None:
    """Arm the control from parsed CLI flags, and say so loudly when armed."""

    configure(getattr(args, "noise", MODE_NONE),
              default_seed if getattr(args, "noise_seed", None) is None
              else args.noise_seed)
    if enabled():
        message = (
            "NOISE CONTROL ARMED -- every trial's EEG is replaced with "
            f"scale-matched Gaussian noise (seed {_seed}). Predictions and "
            "report go to *_noise paths; the clean artefacts are untouched.")
        if logger is not None:
            logger.warning(message)
        else:
            print("WARNING: " + message, flush=True)


def mode() -> str:
    return _mode


def enabled() -> bool:
    return _mode != MODE_NONE


def seed() -> Optional[int]:
    return _seed


def maybe_noise(array: Any) -> Any:
    """The clean array, or a scale-matched noise replacement for it.

    The replacement is drawn along the last axis: a ``(N, C, T)`` clip gets one
    mean and one standard deviation per trial and channel, so each element
    keeps the magnitude the model was trained to read.  Arrays whose last axis
    is not time (MindCine's flat ``(N, 310)`` DE features, say) match on their
    per-row statistics instead, which is the same rule applied to what is
    there.
    """

    if not enabled() or array is None:
        return array

    x = np.asarray(array)
    if x.size == 0:
        return array

    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    # A constant channel has no scale to match; leave it at unit scale rather
    # than dividing by zero.
    std = np.where(std > 0, std, 1.0)
    drawn = _generator().standard_normal(x.shape).astype(np.float32)
    return (drawn * std + mean).astype(x.dtype, copy=False)


def _generator() -> np.random.Generator:
    global _rng
    if _rng is None:
        _rng = np.random.default_rng(_seed)
    return _rng


def variant_path(path: Any) -> Path:
    """``preds_video_generation.json`` -> ``..._noise.json`` when armed.

    Returns ``path`` untouched in the clean condition, so callers can pass the
    result straight to the evaluator without branching.
    """

    p = Path(path)
    if not enabled():
        return p
    return p.with_name(f"{p.stem}{SUFFIX}{p.suffix}")


def describe() -> Dict[str, Any]:
    """The conditioning block written into every sidecar."""

    return {
        "noise": _mode,
        "noise_seed": _seed,
        "noise_scale": ("matched to each element's own mean/std along the last "
                        "axis" if enabled() else None),
        "reference": ("EEG2X-Benchmark-v2-GS/NOISE_EXPERIMENT.md; GLIM "
                      "eval_noise_input"),
    }


def write_meta(predictions_path: Any, **extra: Any) -> Path:
    """Write ``<stem>.meta.json`` beside a predictions file.

    ``extra`` carries whatever the calling baseline knows about the run -- h5
    directory, checkpoint, dataset, subject.  Only written when armed, so a
    clean run leaves exactly the artefacts it always did.
    """

    path = Path(predictions_path)
    if not enabled():
        return path
    meta = dict(describe())
    meta.update(extra)
    out = path.with_name(f"{path.stem}.meta.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return out
