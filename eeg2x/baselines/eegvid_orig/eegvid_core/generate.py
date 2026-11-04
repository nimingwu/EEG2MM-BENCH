"""Synthesise frames with a trained VideoEEGStyleGAN-ADA, for the benchmark.

The generator is conditioned on ``cat([psi, pe(frame)])`` -- the EEG feature
plus a sinusoidal encoding of the frame index (the paper's Listing 1) -- so
producing a clip means running the generator once per frame with the clip's psi
and that frame's index.  Both the generator and its conditioning are exactly
what :mod:`eeg2video_eegvid.train_gan` trained.

The clip is the paper's own shape -- eight frames at 128x128.  The benchmark
scores six frames at 288x512, so :func:`prepare_for_scoring` makes both changes
at the very end; training or generating at the benchmark's shape would not be
EEGVid.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .dataset import (BENCHMARK_N_FRAMES, N_FRAMES, PE_DIM,
                      build_conditions, select_benchmark_frames)

LOG = logging.getLogger("eeg2video_eegvid.generate")

VENDOR = Path(__file__).resolve().parent / "vendor" / "stylegan_ada"

#: The benchmark's frame geometry (the evaluator's ground truth).
TARGET_HEIGHT = 288
TARGET_WIDTH = 512


def find_snapshot(run_dir: Path) -> Path:
    """The highest-numbered ``network-snapshot-*.pkl`` under ``run_dir``."""

    snapshots = sorted(Path(run_dir).glob("network-snapshot-*.pkl"))
    if not snapshots:
        raise FileNotFoundError(
            f"no network-snapshot-*.pkl under {run_dir}; train the generator first")
    return snapshots[-1]


@torch.no_grad()
def generate_frames(
    run_dir: Path,
    psi: np.ndarray,
    *,
    n_frames: int = N_FRAMES,
    pe_dim: int = PE_DIM,
    batch_size: int = 16,
    seed: int = 0,
    device: str = "cuda",
    snapshot: Optional[Path] = None,
    truncation_psi: float = 1.0,
) -> np.ndarray:
    """``(N, D)`` EEG features -> ``(N, n_frames, H, W, 3)`` uint8 at 128x128.

    One generator call per frame: the frame index enters through the
    conditioning vector, which is the whole point of the paper's temporal
    conditioning (its ablation shows the generator produces static, fading
    output without it).
    """

    if VENDOR not in [Path(p) for p in sys.path]:
        sys.path.insert(0, str(VENDOR))
    import dnnlib
    import legacy

    psi = np.asarray(psi, dtype=np.float32)
    snapshot = Path(snapshot) if snapshot is not None else find_snapshot(run_dir)
    LOG.info("loading generator from %s", snapshot)

    with dnnlib.util.open_url(str(snapshot)) as handle:
        generator = legacy.load_network_pkl(handle)["G_ema"].to(device).eval()
    if generator.img_resolution != 128:
        LOG.warning("generator resolution is %d, expected 128",
                    generator.img_resolution)

    conditions = build_conditions(
        torch.as_tensor(psi, dtype=torch.float32), n_frames, pe_dim)
    n_clips = psi.shape[0]

    # Seeded per clip so a rerun reproduces the same z, and different clips do
    # not share one latent -- the benchmark pins generation noise to its seed
    # protocol.
    generator_state = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(n_clips * n_frames, generator.z_dim,
                          generator=generator_state)

    out = np.zeros((n_clips * n_frames, generator.img_resolution,
                    generator.img_resolution, 3), dtype=np.uint8)
    for begin in range(0, n_clips * n_frames, batch_size):
        chunk = slice(begin, begin + batch_size)
        z = latents[chunk].to(device)
        c = conditions[chunk].to(device)
        images = generator(z, c, truncation_psi=truncation_psi,
                           noise_mode="const")
        # (-1, 1) float -> (0, 255) uint8 HWC, as stylegan2-ada's own
        # ``save_image_grid`` does with drange=[-1, 1].
        images = (images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        out[chunk] = images.permute(0, 2, 3, 1).cpu().numpy()
        if (begin // batch_size) % 5 == 0:
            LOG.info("generated %d/%d frames", min(begin + batch_size,
                                                   n_clips * n_frames),
                     n_clips * n_frames)

    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out.reshape(n_clips, n_frames, *out.shape[1:])


def prepare_for_scoring(frames: np.ndarray,
                        n_scored: int = BENCHMARK_N_FRAMES) -> np.ndarray:
    """The paper's 8-frame clip -> the benchmark's 6 scored frames @288x512.

    Two benchmark-side adaptations in one place: the temporal selection
    (:func:`~eeg2video_eegvid.dataset.select_benchmark_frames`) and the
    resolution change.  The generator itself stays at 8 frames and 128x128.
    """

    return upsample_to_benchmark(select_benchmark_frames(frames, n_scored))


def upsample_to_benchmark(frames: np.ndarray,
                          height: int = TARGET_HEIGHT,
                          width: int = TARGET_WIDTH) -> np.ndarray:
    """``(N, F, 128, 128, 3)`` -> ``(N, F, 288, 512, 3)`` uint8.

    The evaluator's ground truth is 288x512; EEGVid's generator is 128x128 by
    construction, so this is a protocol adaptation rather than a change to the
    model.  Bicubic, which is what the benchmark's own frame loader uses to
    reach this geometry.
    """

    import cv2

    frames = np.asarray(frames, dtype=np.uint8)
    n, f, h, w, c = frames.shape
    if (h, w) == (height, width):
        return frames
    out = np.empty((n, f, height, width, c), dtype=np.uint8)
    flat_in = frames.reshape(-1, h, w, c)
    flat_out = out.reshape(-1, height, width, c)
    for i in range(flat_in.shape[0]):
        flat_out[i] = cv2.resize(flat_in[i], (width, height),
                                 interpolation=cv2.INTER_CUBIC)
    return out
