"""EEGVid (arXiv:2505.21385) — the paper's chain, as a benchmark model.

"Beyond Reconstruction: What EEG-to-Video Decoding Actually Recovers" ships no
code.  What it describes is:

    a triplet-loss EEG encoder (a modified NICE: channel graph attention ->
    temporal convolution -> linear, psi in R^1024, unit-normalised)
        -> a temporally conditioned GAN (VideoEEGStyleGAN-ADA: the authors'
           WACV 2024 EEGStyleGAN-ADA, itself open, plus a sinusoidal encoding
           of the frame index appended to the EEG feature)

and :mod:`model.eeg2video_eegvid` implements exactly that, with the vendored
generator under ``vendor/stylegan_ada``.

Provenance and the two things this adapter does not take from the paper:

* **Split.**  The paper uses EEG2Video's 7-fold protocol on SEED-DV (5 train /
  1 val / 1 test).  The benchmark's protocol is blocks 1-6 train, block 7 test;
  that is what runs here, so the numbers are the benchmark's, not the paper's.
* **Scoring shape.**  The generator is trained and run at the paper's shape --
  eight frames at 128x128, with its frame-index encoding spanning 0..7.  The
  benchmark's ground truth is six keyframes at 288x512, so at scoring time the
  six generated frames nearest in time to those keyframes are selected and
  upsampled.  Both steps live in
  :func:`~eeg2video_eegvid.generate.prepare_for_scoring`; nothing about the
  model itself is changed to fit the benchmark.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eeg2x.frameworks.eeg2video import noise as noise_control
from eeg2x.frameworks.eeg2video.core.model_base import BaseModel
from eeg2x.protocol import SEED
from eeg2x.seeding import seed_everything

log = logging.getLogger(__name__)


class EEGVidModel(BaseModel):
    """EEGVid, trained and evaluated under the benchmark protocol."""

    name = "eegvid"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = dict(config or {})
        self.subject = str(self.cfg.get("subject_id", "sub-01"))
        # EEGVid's encoder consumes a whole trace, not patches, so it reads the
        # dataset-native branch the loader wrote for it.
        self.h5_dir = Path(self.cfg.get("h5_dir", "./h5_seed_dv/eegvid_nice"))
        self.track_root = Path(self.cfg.get(
            "work_dir", f"./runs/eegvid/{self.subject}"))
        self.dataset_id = str(self.cfg.get("dataset_id", "seed_dv"))
        self._env_ready = False

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    def _ensure_env(self) -> None:
        if self._env_ready:
            return
        track = self.track_root.resolve()
        os.environ.update({
            "EEG2VIDEO_SUBJECT": self.subject.split("-")[-1].lstrip("0") or "1",
            "EEG2VIDEO_RUN_ROOT": str(track),
            "EEG2VIDEO_CHECKPOINT_ROOT": str(track / "checkpoints"),
            "EEG2VIDEO_SHARED_RUN_ROOT": str(track / "shared"),
        })
        self._env_ready = True

    def _split(self, split: str) -> List[Path]:
        path = self.h5_dir / f"{self.dataset_id}_{self.subject}_{split}_epoched.h5"
        if not path.is_file():
            raise FileNotFoundError(f"missing benchmark HDF5 {path}")
        return [path]

    @property
    def data_root(self) -> Path:
        return self.track_root / "shared" / "data"

    @property
    def encoder_path(self) -> Path:
        return self.track_root / "checkpoints" / "eegvid_encoder.pt"

    @property
    def gan_dir(self) -> Path:
        return self.track_root / "shared" / "gan"

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def train(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Frames -> triplet encoder -> psi -> temporally conditioned GAN."""

        self.cfg.update(config or {})
        self._ensure_env()
        seed_everything(SEED)
        force = bool(self.cfg.get("force", False))
        device = str(self.cfg.get("device", "cuda"))

        from .eegvid_core import dataset as eegvid_data
        from .eegvid_core import train_encoder, train_gan
        from .eegvid_core.train_gan import (
            _distributed_launch, _init_process_group)

        import torch

        h5 = {"train": self._split("train"), "test": self._split("test")}
        n_frames = int(self.cfg.get("n_frames", eegvid_data.N_FRAMES))
        resolution = int(self.cfg.get("resolution",
                                      eegvid_data.FRAME_RESOLUTION))
        self.data_root.mkdir(parents=True, exist_ok=True)

        # Under torchrun every rank runs this whole method, but only the
        # generator (step 4) is distributed.  Steps 1-3 are single-writer work
        # on fixed paths under data_root -- decoding 2.2 GB of frames into one
        # file, training one triplet encoder, writing one psi array.  Letting
        # four ranks do them is not a speedup: they decode the same clips into
        # the same file, and because ``device`` is plain ``cuda`` they all
        # resolve cuda:0 and drive GPU 0 out of memory before training starts.
        world_size, rank = _distributed_launch()
        _init_process_group(world_size, rank)

        if rank == 0:
            # 1. stimulus frames at the generator's resolution
            frames = eegvid_data.prepare_frames(
                h5["train"], self.data_root / "train_frames.uint8.npy",
                resolution=resolution, n_frames=n_frames, force=force)

            # 2. the triplet encoder
            encoder_ckpt = train_encoder.train_encoder(
                h5["train"], self.encoder_path, device=device, seed=SEED,
                epochs=int(self.cfg.get("encoder_epochs", train_encoder.PAPER_EPOCHS)),
                batch_size=int(self.cfg.get("encoder_batch_size", 256)),
                force=force)

            # 3. frozen psi for every training clip, and the class labels the
            #    generator's grid reports under (class_id * n_frames + frame).
            psi_path = self.data_root / "train_psi.npy"
            labels_path = self.data_root / "train_labels.npy"
            if force or not psi_path.is_file():
                encoder = train_encoder.load_encoder(encoder_ckpt, device=device)
                eeg, concepts = train_encoder.load_eeg_and_labels(h5["train"])
                psi_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(psi_path, self._encode(encoder, eeg, device))
                np.save(labels_path, concepts)
                log.info("encoded %d training clips -> psi %s",
                         eeg.shape[0], np.load(psi_path).shape)
        else:
            frames = encoder_ckpt = psi_path = labels_path = None

        # The ranks must not reach the generator before the artefacts it reads
        # exist, and must not race the summary write afterwards.
        if world_size > 1:
            torch.distributed.barrier()

        # 4. the generator
        gan = train_gan.train_gan(
            self.gan_dir,
            self.data_root / "train_frames.uint8.npy",
            self.data_root / "train_psi.npy",
            labels_path=self.data_root / "train_labels.npy",
            spec=self.cfg.get("gan_spec"),
            seed=SEED, force=force,
            snapshot_ticks=int(self.cfg.get("snapshot_ticks", 10)),
        )

        if world_size > 1:
            torch.distributed.barrier()

        summary = {
            # Only rank 0 ran steps 1-3, so only it knows these; the others
            # return the generator's half and let the caller's rank 0 write.
            "frames": (list(np.asarray(frames).shape)
                       if rank == 0 else None),
            "encoder": str(encoder_ckpt) if rank == 0 else None,
            "psi": str(psi_path) if rank == 0 else None,
            "gan": gan,
            "deviations": [
                "benchmark split (blocks 1-6 train / block 7 test), "
                "not the paper's 5/1/1",
                "8 frames generated per the paper, then the 6 nearest in "
                "time to the GT keyframes are scored",
                "128x128 generation upsampled to the benchmark's 288x512",
            ],
        }
        if rank == 0:
            (self.track_root / "train_summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    @staticmethod
    def _encode(encoder, eeg: np.ndarray, device: str,
                batch: int = 64) -> np.ndarray:
        """Run the frozen encoder over ``(N, C, T)`` in batches -> ``(N, D)``."""

        import torch

        out = []
        with torch.no_grad():
            for begin in range(0, eeg.shape[0], batch):
                chunk = torch.from_numpy(
                    eeg[begin:begin + batch]).to(device, dtype=torch.float32)
                out.append(encoder(chunk).cpu().numpy())
        return np.concatenate(out, axis=0)

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def load(self):
        from .eegvid_core.generate import find_snapshot

        self._ensure_env()
        missing = [p for p in (self.encoder_path,
                               self.data_root / "train_psi.npy")
                   if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "EEGVid is missing trained artefacts:\n  "
                + "\n  ".join(str(p) for p in missing) + "\nrun train() first")
        if not (self.gan_dir / "gan.complete.json").is_file():
            raise FileNotFoundError(
                f"no completed generator under {self.gan_dir}; run train() first")
        find_snapshot(self.gan_dir)          # raises if nothing to load
        self._loaded = True
        return self

    def generate(self, inputs: List[Any]) -> List[Dict[str, Any]]:
        if not getattr(self, "_loaded", False):
            self.load()

        from .eegvid_core import dataset as eegvid_data
        from .eegvid_core import generate as eegvid_generate
        from .eegvid_core import train_encoder

        device = str(self.cfg.get("device", "cuda"))
        n_frames = int(self.cfg.get("n_frames", eegvid_data.N_FRAMES))

        # Test clips only need their EEG -- the ground truth is already in the
        # HDF5, so no stimulus video is decoded here.
        encoder = train_encoder.load_encoder(self.encoder_path, device=device)
        eeg, _concepts = train_encoder.load_eeg_and_labels(self._split("test"))
        # Noise control, before the frozen encoder turns EEG into psi.  The
        # generator is conditioned on psi and never sees EEG again, so this is
        # the only point where the control can enter this chain.
        eeg = noise_control.maybe_noise(eeg)
        psi = self._encode(encoder, eeg, device)
        log.info("encoded %d test clips", psi.shape[0])

        frames = eegvid_generate.generate_frames(
            self.gan_dir, psi, n_frames=n_frames,
            batch_size=int(self.cfg.get("generate_batch_size", 16)),
            device=device, seed=SEED,
            truncation_psi=float(self.cfg.get("truncation_psi", 1.0)))
        # The generator stays at the paper's 8 frames / 128x128; only here do
        # the two benchmark adaptations happen.
        frames = eegvid_generate.prepare_for_scoring(frames)

        # A noise run writes under its own directory.  The file name below is
        # fixed, so a suffix on it applied from the driver cannot protect the
        # clean frames -- the directory has to move instead.
        out_dir = self.track_root / "generate"
        if noise_control.enabled():
            out_dir = out_dir.parent / f"{out_dir.name}.noise"
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_path = out_dir / "generated_frames.uint8.npy"
        np.save(frames_path, frames)
        (out_dir / "generation_meta.json").write_text(json.dumps({
            "model": self.name, "subject": self.subject,
            "generated_resolution": int(self.cfg.get("resolution",
                                                     eegvid_data.FRAME_RESOLUTION)),
            "scored_resolution": [int(frames.shape[2]), int(frames.shape[3])],
            "n_clips": int(frames.shape[0]), "n_frames": int(frames.shape[1]),
            "seed": SEED,
        }, indent=2), encoding="utf-8")
        log.info("wrote %s", frames_path)

        n_test = int(frames.shape[0])
        predictions: List[Dict[str, Any]] = []
        for sample in inputs:
            sample_id = str(getattr(sample, "sample_id", ""))
            if "_clip-" not in sample_id:
                raise ValueError(
                    f"unexpected sample id {sample_id!r}; expected the "
                    "benchmark's '<stem>_clip-<row>' form")
            row = int(sample_id.rsplit("_clip-", 1)[1])
            if not 0 <= row < n_test:
                raise IndexError(
                    f"sample {sample_id} maps to row {row}, outside the "
                    f"{n_test} generated clips")
            predictions.append({
                "sample_id": sample_id,
                "path": f"{frames_path}#{row}",
                "eegvid": {"manifest_row": row, "n_frames": int(frames.shape[1])},
            })
        return predictions

    def unload(self):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
