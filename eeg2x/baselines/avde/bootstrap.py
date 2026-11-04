"""Benchmark-side bootstrap for the AVDE baseline.

AVDE (Dai et al., ICLR 2026, arXiv:2602.22555) is a two-stage autoregressive
decoder: a LaBraM encoder fine-tuned with a bidirectional contrastive objective
(Stage 1) followed by a VAR "next-scale prediction" transformer that turns the
EEG embedding into multi-scale VQ-VAE tokens (Stage 2).

The code below ``model/avde`` is a near-verbatim copy of the released
implementation (https://github.com/ddicee/avde).  That implementation uses flat
top-level imports (``import dist``, ``from models import ...``, ``from utils
import ...``, ``from eeg_data.dataset import ...``).  Rewriting every import
would touch every file and make the copy diverge from upstream, so this module
instead puts the copied tree on ``sys.path``.  No module outside ``model/avde``
imports those top-level names (verified by grep over the repository), so the
insertion cannot shadow a benchmark module.

Everything the baseline *writes* stays inside EEG2X-Benchmark-v2-test: the
copied code only ever receives output directories that were already passed
through :func:`core.paths.require_benchmark_output`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


AVDE_ROOT = Path(__file__).resolve().parent
BENCHMARK_ROOT = AVDE_ROOT.parents[2]   # v3: eeg2x/baselines/avde -> 仓库根
#: Frozen upstream weights shipped with the AVDE release.  They are inputs, not
#: benchmark products, and are kept next to the copied code so the baseline is
#: runnable from inside the repository.
PRETRAINED_DIR = AVDE_ROOT / "pretrained"
LABRAM_CKPT = PRETRAINED_DIR / "labram-base.pth"
VAE_CKPT = PRETRAINED_DIR / "vae_ch160v4096z32.pth"
VAR_CKPT = PRETRAINED_DIR / "var_d16.pth"


def install() -> Path:
    """Put the copied AVDE tree and the benchmark root on ``sys.path``."""
    for path in (str(AVDE_ROOT), str(BENCHMARK_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return AVDE_ROOT


def require_pretrained() -> Path:
    """Fail before a run starts if the frozen upstream weights are missing."""
    missing = [str(path) for path in (LABRAM_CKPT, VAE_CKPT, VAR_CKPT)
               if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"AVDE baseline is missing frozen weights: {missing}")
    return PRETRAINED_DIR


def torchrun_env(port: int) -> dict:
    """Environment for a single-process ``torchrun`` stage.

    AVDE's training loops call ``torch.distributed`` barriers and all-reduces
    unconditionally, which requires a real (if one-process) process group.
    ``torchrun --nproc_per_node=1`` supplies exactly that, so the copied code
    needs no single-process special case.
    """
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "2000"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(port)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BENCHMARK_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH")
        else [str(BENCHMARK_ROOT)])
    return env
