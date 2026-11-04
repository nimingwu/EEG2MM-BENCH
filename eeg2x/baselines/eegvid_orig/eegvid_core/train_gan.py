"""Train VideoEEGStyleGAN-ADA, the EEGVid generator.

EEGVid ships no code, but its generator *is* the authors' earlier
EEGStyleGAN-ADA (WACV 2024, github.com/prajwalsingh/EEGStyleGAN-ADA) with a
temporal conditioning input added, and that repo is open.  This module drives
the vendored copy of it with the hyperparameters EEGVid's own Listing 2 gives
for SEED-DV:

    'cifar': dict(ref_gpus=1, kimg=2500, mb=16, mbstd=4, fmaps=1.0,
                  lrate=0.0015, gamma=2, ema=10, ramp=0.05, map=4)

The one substantive change is the conditioning vector: EEGStyleGAN-ADA feeds the
generator a 128-d EEG feature, EEGVid feeds it ``cat([psi(1024), pe(frame)])``
= 1045-d, which :mod:`eeg2video_eegvid.dataset` builds.  ``c_dim`` therefore
becomes 1045, and nothing else in the network changes -- both generator and
discriminator already consume ``c`` as a continuous vector through their mapping
networks, so a longer vector needs no architectural edit.

Notes on running the vendored code on this machine (RTX 4090, torch 2.1):

* Its two CUDA kernels (``bias_act``, ``upfirdn2d``) are JIT-compiled on first
  use, which needs ``ninja`` on PATH, ``CUDA_HOME`` set, and
  ``TORCH_CUDA_ARCH_LIST`` naming the real architecture (sm_89 here).  The
  reference fallback is *not* equivalent -- ``_upfirdn2d_ref`` rejects the
  4-D filters the network passes -- so a failed build is a hard failure, not a
  slow path.
* ``conv2d_gradfix`` disables itself with a warning on torch > 1.9; that is
  expected and only costs the fused-op speedup.
* On a multi-GPU box its ``FileBaton`` serialises the build and *blocks
  forever* on a stale lock left by an interrupted run; delete
  ``~/.cache/torch_extensions/*/<op>_plugin/lock`` if a build hangs at 0% CPU.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

LOG = logging.getLogger("eeg2video_eegvid.train_gan")

VENDOR = Path(__file__).resolve().parent / "vendor" / "stylegan_ada"

#: EEGVid Listing 2, the SEED-DV row.
SEED_DV_SPEC: Dict[str, Any] = dict(
    ref_gpus=1, kimg=2500, mb=16, mbstd=4, fmaps=1.0,
    lrate=0.0015, gamma=2, ema=10, ramp=0.05, map=4,
)


def _torch_env() -> None:
    """Everything the vendored CUDA build needs, set before torch compiles."""

    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        import torch

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            os.environ["TORCH_CUDA_ARCH_LIST"] = f"{cap[0]}.{cap[1]}"
    bin_dir = str(Path(sys.executable).parent)
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def _distributed_launch() -> Tuple[int, int]:
    """``(world_size, rank)`` from torchrun's environment.

    ``torchrun`` exports ``WORLD_SIZE``/``RANK``/``LOCAL_RANK``; a plain
    ``python train_gan.py`` exports none of them and gets ``(1, 0)``, so the
    single-process path is unchanged.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size <= 1:
        return 1, 0
    return world_size, int(os.environ.get("RANK", "0") or "0")


def _init_process_group(world_size: int, rank: int) -> None:
    """Join the NCCL group and bind this process to its own card.

    Nothing in the vendored loop does this.  Upstream stylegan2-ada initialises
    the group inside ``launch_training``, which was stripped when the tree was
    vendored, so a multi-process launch would otherwise reach
    ``DistributedDataParallel`` with no process group and die there.

    ``training_loop`` binds ``device = torch.device('cuda', rank)``, so each
    rank must see *all* the cards -- do not set ``CUDA_VISIBLE_DEVICES`` when
    launching this.
    """

    import torch
    import torch.distributed as dist

    if world_size <= 1 or dist.is_initialized():
        return
    device_count = torch.cuda.device_count()
    if device_count < world_size:
        raise RuntimeError(
            f"WORLD_SIZE={world_size} but only {device_count} CUDA device(s) "
            "are visible; launch one process per card"
        )
    torch.cuda.set_device(rank % device_count)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def train_gan(
    run_dir: Path,
    frames_path: Path,
    psi_path: Path,
    labels_path: Optional[Path] = None,
    *,
    spec: Optional[Dict[str, Any]] = None,
    feature_dim: int = 1024,
    pe_dim: int = 10,
    seed: int = 0,
    force: bool = False,
    snapshot_ticks: int = 10,
    resume: bool = True,
) -> Dict[str, Any]:
    """Train the generator.  Returns a summary dict."""

    _torch_env()
    sys.path.insert(0, str(VENDOR))
    import torch
    import dnnlib
    from training import training_loop

    from .dataset import conditioning_dim

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    marker = run_dir / "gan.complete.json"
    if marker.is_file() and not force:
        LOG.info("generator already trained: %s", marker)
        return json.loads(marker.read_text(encoding="utf-8"))

    spec = dict(SEED_DV_SPEC if spec is None else spec)
    # Resume support.  stylegan2-ada's ``resume_pkl`` restores G/D/G_ema
    # weights but neither the optimizer state nor the kimg counter -- the loop
    # always counts from zero -- so the remaining budget is passed as
    # ``total_kimg`` instead.  Note the Adam moments do restart from scratch;
    # that is a property of the vendored loop, not a choice made here, and it
    # shows up as a short transient in the loss after a resume.
    resume_pkl = None
    if resume:
        snapshots = sorted(Path(run_dir).glob("network-snapshot-*.pkl"))
        if snapshots:
            resume_pkl = snapshots[-1]
            done = int(resume_pkl.stem.rsplit("-", 1)[-1])
            remaining = int(spec["kimg"]) - done
            LOG.info("resuming from %s (%d kimg done, %d to go)",
                     resume_pkl.name, done, remaining)
            if remaining <= 0:
                LOG.info("target kimg already reached; nothing to train")
                return {"run_dir": str(run_dir), "resumed_from": str(resume_pkl),
                        "kimg_done": done, "trained": 0}
            spec["kimg"] = remaining
            spec["_resumed_from"] = str(resume_pkl)
            spec["_kimg_already_done"] = done
    # stylegan2-ada runs the N highest-resolution blocks in fp16.  Its vendored
    # augmentation uses ``grid_sample``, whose backward is not available for
    # every mixed-precision path on torch 2.1; the authors' own pinned stack
    # (torch 1.7) cannot be installed on this GPU at all (sm_89 needs CUDA
    # 11.8+, torch 1.7 ships cu110).  Defaulting to 0 keeps the whole pipeline
    # fp32, which is the supported path.
    fp16_res = int(spec.get("fp16_res", 0))
    c_dim = conditioning_dim(feature_dim, pe_dim)
    LOG.info("training VideoEEGStyleGAN-ADA: c_dim=%d, spec=%s", c_dim, spec)

    num_gpus, rank = _distributed_launch()
    _init_process_group(num_gpus, rank)

    args = dnnlib.EasyDict()
    args.run_dir = str(run_dir)
    args.num_gpus = num_gpus
    args.random_seed = seed
    # The paper's Listing 2 keys map onto stylegan2-ada's arguments, which are
    # named differently: kimg -> total_kimg, ema -> ema_kimg, ramp ->
    # ema_rampup, and gamma is the R1 penalty inside loss_kwargs.
    #
    # ``mb`` is the *total* batch for one iteration and stays fixed; the number
    # of cards only changes how it is split.  The loop's round count is
    # ``batch_size // (batch_gpu * num_gpus)``, so a per-card batch that does
    # not divide evenly makes it run zero rounds -- training would proceed,
    # print ticks, and learn nothing.  Derive the per-card share from the real
    # GPU count and refuse anything that would collapse.
    per_card = num_gpus if num_gpus > 1 else int(spec["ref_gpus"])
    args.batch_size = spec["mb"]
    args.batch_gpu = spec["mb"] // per_card
    if args.batch_gpu < 1 or args.batch_size // (args.batch_gpu * num_gpus) < 1:
        raise ValueError(
            f"batch {args.batch_size} over {num_gpus} GPU(s) with per-card "
            f"batch_gpu={args.batch_gpu} gives "
            f"{args.batch_size // (args.batch_gpu * num_gpus)} rounds; the "
            "training loop would never execute. Lower batch_gpu."
        )
    if rank == 0:
        LOG.info("batch %d = %d x %d (per-card x GPUs), %d round(s)",
                 args.batch_size, args.batch_gpu, num_gpus,
                 args.batch_size // (args.batch_gpu * num_gpus))
    args.total_kimg = spec["kimg"]
    args.ema_kimg = spec["ema"]
    args.ema_rampup = spec["ramp"]
    args.ada_target = 0.6
    args.augment_p = 0.0
    args.cudnn_benchmark = True

    args.G_kwargs = dnnlib.EasyDict(
        class_name="training.networks.Generator", z_dim=512, w_dim=512,
        mapping_kwargs=dnnlib.EasyDict(num_layers=spec["map"]),
        synthesis_kwargs=dnnlib.EasyDict(channel_base=int(32768 * spec["fmaps"]),
                                         channel_max=512,
                                         num_fp16_res=fp16_res,
                                         conv_clamp=256),
    )
    args.D_kwargs = dnnlib.EasyDict(
        class_name="training.networks.Discriminator", block_kwargs=dnnlib.EasyDict(),
        mapping_kwargs=dnnlib.EasyDict(num_layers=spec["map"]),
        epilogue_kwargs=dnnlib.EasyDict(mbstd_group_size=spec["mbstd"]),
        channel_base=int(32768 * spec["fmaps"]), channel_max=512,
        num_fp16_res=fp16_res, conv_clamp=256, architecture="orig",
    )
    args.G_opt_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam", lr=spec["lrate"], betas=[0, 0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(
        class_name="torch.optim.Adam", lr=spec["lrate"], betas=[0, 0.99], eps=1e-8)
    # The CIFAR-style overrides EEGStyleGAN-ADA applies at 128x128: path-length
    # regularisation and style mixing are both switched off there, and "orig"
    # discriminator architecture (no residual skips).
    # Parameters are exactly this fork's StyleGAN2Loss signature: it takes
    # ``pl_weight``/``style_mixing_prob``/``r1_gamma`` but has no
    # ``lazy_regularization`` (the lazy-regularisation split lives in the
    # training loop's G_reg_interval/D_reg_interval instead).
    args.loss_kwargs = dnnlib.EasyDict(
        class_name="training.loss.StyleGAN2Loss",
        pl_weight=0.0, style_mixing_prob=0.0, r1_gamma=spec["gamma"])
    # ``augpipe=bgcfnc`` (the authors' own command line), expanded exactly the
    # way their train.py expands it.
    args.augment_kwargs = dnnlib.EasyDict(
        class_name="training.augment.AugmentPipe",
        xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1,
        brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
        imgfilter=1, noise=1, cutout=1)
    args.metrics = []
    args.training_set_kwargs = dnnlib.EasyDict(
        class_name="eeg2x.baselines.eegvid_orig.eegvid_core.dataset.EEGVidFrameDataset",
        frames_path=str(frames_path), psi_path=str(psi_path),
        labels_path=str(labels_path) if labels_path is not None else None,
        pe_dim=pe_dim)
    args.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True, num_workers=4, prefetch_factor=2)
    args.image_snapshot_ticks = snapshot_ticks
    args.network_snapshot_ticks = snapshot_ticks
    args.resume_pkl = str(resume_pkl) if resume_pkl is not None else None

    training_loop.training_loop(rank=rank, **args)

    summary = {
        "run_dir": str(run_dir),
        "c_dim": c_dim,
        "spec": {k: v for k, v in spec.items()},
        "objective": "VideoEEGStyleGAN-ADA (EEGVid arXiv:2505.21385 App. E)",
    }
    marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
