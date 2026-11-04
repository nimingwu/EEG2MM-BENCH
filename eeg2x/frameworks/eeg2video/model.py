"""Scheme D — EEG-to-Video model adapter (EEG2Video, NeurIPS 2024).

Reuse over reimplementation
---------------------------
Everything that decides a number is **copied verbatim** from
``EEG2Video_clean_d_v2`` into :mod:`model.eeg2video_core` (verified byte
identical):

* ``scheme_d.py``            EEGNet-8,2 + projection head + token-wise Linear
* ``scheme_d_training.py``   the two training stages and the block-7 holdout
* ``animatediff_generator.py`` the frozen official AnimateDiff generator
* ``animatediff/``           the vendored upstream AnimateDiff implementation

This adapter therefore contains no model arithmetic of its own.  It only

1. reads the benchmark HDF5 instead of the EEG2Video project's own npy caches,
2. materialises exactly the arrays ``scheme_d_training`` expects, so its
   verified training loop can be called unchanged,
3. calls the copied generator and returns benchmark-format predictions.

Condition contract
------------------
``EEG (n,62,400)`` -> seven 100-sample windows at stride 50 -> ``(n,7,62,100)``
-> EEGNet -> mean over windows -> projection head -> token-wise Linear
-> ``(n,77,768)`` -> AnimateDiff -> ``(n,6,288,512,3)`` uint8.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from . import noise as noise_control
from .core.model_base import BaseModel
from eeg2x.profiles_video import SEED_DV_ENCODER_CLIP
from eeg2x.protocol import SEED
from eeg2x.seeding import seed_everything

# The copied AnimateDiff implementation imports itself as ``animatediff.*``
# (upstream uses absolute imports), so its parent directory has to be
# importable too, not just ``model/``.

from .core.animatediff_generator import (  # noqa: E402
    AnimateDiffGenerator, subsample_indices,
)
from .core.scheme_d import (  # noqa: E402
    CLIP_HIDDEN_SIZE, CLIP_SEQUENCE_LENGTH, SchemeDSemanticPredictor,
    SchemeDLinearAdapter, scheme_d_mse,
)
from .core.scheme_d_encoders import (  # noqa: E402
    ENCODER_WINDOW_SPECS, PREPROCESSED_ENCODERS, SchemeDEncoderPredictor,
)

log = logging.getLogger(__name__)

#: Window layout, matching the loader's ``epoch`` step and the EEG2Video paper.
#: Retained as the default for callers that predate the encoder branch; the
#: authoritative geometry now travels in the HDF5 attrs written by the loader.
N_WINDOWS = 7
WINDOW_SAMPLES = 100
WINDOW_STRIDE = 50

STAGE_ONE_EPOCHS = 200
STAGE_TWO_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 5e-4

#: Encoders this adapter knows how to build.
SUPPORTED_ENCODERS = ("eegnet",) + tuple(
    e for e in ENCODER_WINDOW_SPECS if e != "eegnet"
)


def _windows_from_epoch(epochs: np.ndarray,
                        spec: Optional[Dict[str, int]] = None) -> np.ndarray:
    """``(n, 62, T)`` -> ``(n, n_windows, 62, window_samples)``.

    ``spec`` comes from the HDF5's ``encoder_window_specs_json`` /
    ``window_samples`` attrs, so the slicing matches the preprocessing branch
    that produced the file.  Defaults to the paper's 7 x 100 layout.
    """

    spec = dict(spec or {"window_samples": WINDOW_SAMPLES,
                         "stride_samples": WINDOW_STRIDE,
                         "n_windows": N_WINDOWS})
    window = int(spec["window_samples"])
    stride = int(spec["stride_samples"])
    expected = int(spec["n_windows"])
    starts = np.arange(0, epochs.shape[-1] - window + 1, stride)
    if len(starts) != expected:
        raise ValueError(
            f"expected {expected} windows of {window} samples from "
            f"{epochs.shape[-1]} samples at stride {stride}, got {len(starts)}"
        )
    return np.stack(
        [epochs[..., s:s + window] for s in starts], axis=1
    ).astype(np.float32)


def _branch_attrs(h5_path: Path) -> Dict[str, Any]:
    """Read the preprocessing branch's declaration back out of an HDF5."""

    import h5py

    with h5py.File(h5_path, "r") as h5:
        attrs = {
            "encoder": str(h5.attrs.get("encoder_id", "eegnet")),
            "model_input_normalization": str(
                h5.attrs.get("model_input_normalization", "zscore_train_split")
            ),
            "channel_names": [
                c.decode("utf-8") if isinstance(c, bytes) else str(c)
                for c in h5["channel_names"][:]
            ],
            "sampling_rate": float(h5.attrs.get("sampling_rate", 200.0)),
        }
        specs = h5.attrs.get("encoder_window_specs_json")
        if specs is not None:
            attrs["window_specs"] = json.loads(str(specs))
        if "encoder_id" in h5.attrs:
            attrs["window_spec"] = {
                "window_samples": int(h5.attrs["window_samples"]),
                "stride_samples": int(h5.attrs["window_stride_samples"]),
                "n_windows": int(h5.attrs["n_windows"]),
            }
    return attrs


class EEG2VideoModel(BaseModel):
    """Benchmark adapter for the Scheme D EEG-to-Video model."""

    name = "eeg2video"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = dict(config or {})
        self.encoder = str(self.cfg.get("encoder", "eegnet"))
        if self.encoder not in SUPPORTED_ENCODERS:
            raise KeyError(
                f"unsupported encoder {self.encoder!r}; "
                f"expected one of {list(SUPPORTED_ENCODERS)}"
            )
        self._device = torch.device(
            self.cfg.get("device")
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = Path(
            self.cfg.get("checkpoint_dir", "./checkpoints/eeg2video")
        )
        self._stage_one = None
        self._adapter: Optional[SchemeDLinearAdapter] = None
        self._normalization: Optional[tuple] = None
        self._channel_names: Optional[List[str]] = None
        self._window_spec: Optional[Dict[str, int]] = None
        self._model_input_normalization: str = str(
            self.cfg.get("model_input_normalization", "zscore_train_split")
        )
        self._branch_loaded = False

    def _ensure_branch_from_h5(self) -> None:
        """Load montage / window geometry from the HDF5 the branch wrote.

        Needed before the encoder is built for labram/reve (which select
        position embeddings and electrode coordinates by montage).  Reads the
        test split when present, else the train split.
        """

        if self._branch_loaded or not self.cfg.get("h5_dir"):
            return
        h5_dir = Path(self.cfg["h5_dir"])
        candidates: List[Path] = []
        for split in ("test", "train"):
            candidates = self._split_h5(h5_dir, split)
            if candidates:
                break
        if not candidates:
            return
        attrs = _branch_attrs(candidates[0])
        self._channel_names = attrs["channel_names"]
        self._window_spec = attrs.get("window_spec")
        self._model_input_normalization = attrs["model_input_normalization"]
        self._branch_loaded = True

    # ------------------------------------------------------------------
    # encoder branch plumbing
    # ------------------------------------------------------------------

    def _split_h5(self, h5_dir: Path, split: str,
                  subject: Optional[str] = None) -> List[Path]:
        """``seed_dv_<subject>_<split>_epoched.h5`` files under ``h5_dir``.

        The loader names files ``{dataset}_{subject}_{split}_epoched.h5``.
        Filtering on both parts matters: a cohort directory holds every
        subject, and training on more than the requested one would leak other
        subjects' blocks 1-6 into this subject's model.
        """

        subject = subject or self.cfg.get("subject_id")
        suffix = f"_{split}_epoched.h5"
        # Two datasets feed this task now, so the glob cannot hard-code one
        # prefix: SEED-DV writes ``seed_dv_*`` and ds005662 writes
        # ``ds005662_*``.  ``dataset_id`` pins it when the directory ever
        # holds both.
        dataset = self.cfg.get("dataset_id")
        pattern = f"{dataset}_*_epoched.h5" if dataset else "*_epoched.h5"
        files = [
            p for p in sorted(h5_dir.glob(pattern))
            if p.name.endswith(suffix)
            and (subject is None or f"_{subject}_" in p.name)
        ]
        if not files:
            raise FileNotFoundError(
                f"no {pattern} matching {suffix} and subject {subject!r} "
                f"under {h5_dir}"
            )
        return files

    def _train_h5(self, h5_dir: Path) -> List[Path]:
        """Training HDF5s of exactly this subject (never the whole cohort)."""

        return self._split_h5(h5_dir, "train")

    def _paper_geometry(self, spec: Dict[str, int]) -> bool:
        """True only for the paper's own [.,7,62,100] EEGNet layout."""

        return (
            int(spec["n_windows"]),
            len(self._channel_names or []) or 62,
            int(spec["window_samples"]),
        ) == (7, 62, 100)

    def _predictor_factory(self):
        """Callable the copied training loop instantiates.

        EEGNet is routed to the original ``SchemeDSemanticPredictor`` so the
        baseline's arithmetic stays byte-identical; the other encoders go
        through :class:`SchemeDEncoderPredictor`.
        """

        # Same geometry test the predictor itself applies: the paper's class is
        # only usable at its own [.,7,62,100], which is SEED-DV's EEGNet branch
        # and nothing else.  ds005662's 2x64x100 must go through the generic
        # path or it trips the paper class's assert.
        spec = self._window_spec or ENCODER_WINDOW_SPECS[self.encoder]
        if self.encoder == "eegnet" and self._paper_geometry(spec):
            return SchemeDSemanticPredictor
        kwargs = {
            "encoder": self.encoder,
            "window_spec": spec,
            "n_chans": len(self._channel_names or []) or 62,
            "channel_names": self._channel_names,
            "input_clip": SEED_DV_ENCODER_CLIP.get(self.encoder),
        }

        def factory():
            return SchemeDEncoderPredictor(**kwargs)

        return factory

    def _predictor_meta(self) -> Dict[str, Any]:
        # The geometry is whatever the preprocessing branch declared, not the
        # paper's constants: EEGNet runs the paper's 7x100 on SEED-DV but 2x100
        # on ds005662, where a trial is only 200 samples long.
        spec = self._window_spec or ENCODER_WINDOW_SPECS[self.encoder]
        if self.encoder == "eegnet":
            return {
                "encoder": "eegnet",
                "feature_dim": None,
                "n_windows": int(spec["n_windows"]),
                "window_samples": int(spec["window_samples"]),
                "stride_samples": int(spec["stride_samples"]),
                "n_chans": len(self._channel_names or []) or None,
            }
        return {
            "encoder": self.encoder,
            "n_windows": int(spec["n_windows"]),
            "window_samples": int(spec["window_samples"]),
            "stride_samples": int(spec["stride_samples"]),
            "n_chans": len(self._channel_names or []) or None,
            "channel_names": list(self._channel_names or []),
        }

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def train(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Train both stages on the benchmark HDF5 train split.

        The two arrays ``scheme_d_training`` consumes are rebuilt from the h5
        and written into a scratch run root, then the **copied** training
        functions are called unchanged.  That keeps the optimisation identical
        to the EEG2Video project instead of re-deriving it here.
        """

        self.cfg.update(config or {})
        h5_dir = Path(self.cfg["h5_dir"])
        files = self._train_h5(h5_dir)
        subject = str(self.cfg.get("subject_id") or f"sub-{self.cfg.get('subject', 1):02d}")
        work = Path(self.cfg.get("work_dir", "./runs/eeg2video_train"))
        run_root = work / subject
        shared_root = work / "shared"
        run_root.mkdir(parents=True, exist_ok=True)
        shared_root.mkdir(parents=True, exist_ok=True)

        import h5py

        branch = _branch_attrs(files[0])
        self._channel_names = branch["channel_names"]
        self._window_spec = branch.get("window_spec")
        if self.encoder == "eegnet" and branch["encoder"] != "eegnet":
            log.warning(
                "h5 %s was written for encoder %r but this model is 'eegnet'",
                files[0], branch["encoder"],
            )

        # Both splits, train first.  ``scheme_d_training`` holds out its last
        # ``test_samples`` rows for the block-7 semantic diagnostic, so it
        # requires exactly ``train_samples + test_samples`` windows; writing
        # only the train split (as the original adapter did) makes
        # ``_ensure_targets`` reject the cache with "holds 1200 clips,
        # expected 1400" before training can start.
        test_files = self._split_h5(h5_dir, "test")
        if len(test_files) != 1:
            raise FileNotFoundError(
                f"expected exactly one test HDF5 for {subject} under {h5_dir}, "
                f"found {len(test_files)}"
            )
        eeg_parts, captions, manifest_rows = [], [], []
        for h5_path in [*files, *test_files]:
            with h5py.File(h5_path, "r") as h5:
                eeg_parts.append(np.asarray(h5["data"][:], dtype=np.float32))
                for i in range(int(h5.attrs["n_epochs"])):
                    cap = h5["event_caption"][i]
                    captions.append(cap.decode() if isinstance(cap, bytes) else str(cap))
                    manifest_rows.append({"sample_id": len(manifest_rows), "caption": captions[-1]})
        if not eeg_parts:
            raise FileNotFoundError(f"no training HDF5 under {h5_dir}")

        epochs = np.concatenate(eeg_parts, axis=0)
        n_train = int(sum(int(h5py.File(p, "r").attrs["n_epochs"]) for p in files))
        n_total = len(epochs)
        log.info("%d train + %d test rows for %s/%s",
                 n_train, n_total - n_train, self.encoder, subject)
        windows = _windows_from_epoch(epochs, self._window_spec)
        np.save(run_root / "eeg_windows.float32.npy", windows)
        (run_root / "manifest.jsonl").write_text(
            "\n".join(json.dumps(r) for r in manifest_rows) + "\n", encoding="utf-8"
        )
        log.info("rebuilt %s from %d train + %d test h5 file(s) for %s/%s",
                 windows.shape, len(files), len(test_files), self.encoder, subject)

        embeddings = self._encode_captions(captions)
        np.save(shared_root / "text_embeddings.float16.npy", embeddings)

        # Point the copied trainer at the scratch layout and call it unchanged.
        # MODEL_INPUT_NORMALIZATION mirrors the branch's declaration: when the
        # loader already produced model-ready data, Scheme D's train-split
        # standardisation must not run again.
        os.environ.update({
            "EEG2VIDEO_SUBJECT": "1",
            "EEG2VIDEO_RUN_ROOT": str(run_root),
            "EEG2VIDEO_CHECKPOINT_ROOT": str(self.checkpoint_dir),
            "EEG2VIDEO_SHARED_RUN_ROOT": str(shared_root),
            "EEG2VIDEO_SHARED_TUNE_ROOT": str(self.checkpoint_dir / "shared"),
            "EEG2VIDEO_SEMANTIC_TARGET_MODE": "per_video_caption",
            "EEG2VIDEO_MODEL_INPUT_NORMALIZATION": (
                "none" if self.encoder in PREPROCESSED_ENCODERS
                else branch["model_input_normalization"]
            ),
        })
        seed_everything(SEED)
        from .core import scheme_d_training as trainer

        # ``eeg2video_core.config`` snapshots its env vars when it is imported,
        # and importing this module already pulls it in (via
        # animatediff_generator).  So the env update above cannot move
        # ``trainer.SCHEME_ROOT`` -- a run whose checkpoint root was not set
        # before import would silently write every track's checkpoints to the
        # same default directory and overwrite them.  Verify instead of trust.
        expected_root = self.checkpoint_dir.resolve()
        actual_root = Path(trainer.SCHEME_ROOT).resolve()
        if actual_root != expected_root:
            raise RuntimeError(
                "checkpoint root mismatch: the training module resolved "
                f"{actual_root} but this model was configured for "
                f"{expected_root}.\n"
                "EEG2VIDEO_RUN_ROOT / EEG2VIDEO_CHECKPOINT_ROOT must be set in "
                "the environment *before* eeg2video_core.config is imported "
                "(it snapshots them at import time). Set them, then start a "
                "fresh process -- see scripts/run_eeg2video_track.py, which "
                "does exactly that."
            )

        factory = self._predictor_factory()
        meta = self._predictor_meta()
        mode = ("none" if self.encoder in PREPROCESSED_ENCODERS
                else branch["model_input_normalization"])
        self._model_input_normalization = mode
        stage_one_epochs = int(self.cfg.get("stage_one_epochs", STAGE_ONE_EPOCHS))
        stage_two_epochs = int(self.cfg.get("stage_two_epochs", STAGE_TWO_EPOCHS))
        batch_size = int(self.cfg.get("batch_size", BATCH_SIZE))
        trainer.train_stage_one(
            epochs=stage_one_epochs, batch_size=batch_size,
            predictor_factory=factory, predictor_meta=meta,
            model_input_normalization=mode, train_samples=n_train,
            total_samples=n_total,
        )
        trainer.train_stage_two(
            epochs=stage_two_epochs, batch_size=batch_size,
            predictor_factory=factory, predictor_meta=meta,
            model_input_normalization=mode, train_samples=n_train,
            total_samples=n_total,
        )
        holdout = trainer.evaluate_semantic_holdout(
            predictor_factory=factory, model_input_normalization=mode,
            train_samples=n_train, total_samples=n_total, predictor_meta=meta,
        )
        log.info("training finished: %s", holdout)
        return {"holdout": holdout, "checkpoint_dir": str(self.checkpoint_dir),
                "encoder": self.encoder, "subject": subject}

    def _encode_captions(self, captions: List[str]) -> np.ndarray:
        """Frozen CLIP text encoder targets, ``(n, 77, 768)`` float16."""

        from transformers import CLIPTextModel, CLIPTokenizer

        from .core.config import SD14_ROOT

        device = self._device
        tokenizer = CLIPTokenizer.from_pretrained(
            str(SD14_ROOT), subfolder="tokenizer", local_files_only=True)
        encoder = CLIPTextModel.from_pretrained(
            str(SD14_ROOT), subfolder="text_encoder",
            torch_dtype=torch.float16, local_files_only=True).to(device).eval()
        out = np.empty((len(captions), CLIP_SEQUENCE_LENGTH, CLIP_HIDDEN_SIZE), np.float16)
        with torch.no_grad():
            for begin in range(0, len(captions), 32):
                ids = tokenizer(
                    captions[begin:begin + 32], padding="max_length",
                    max_length=tokenizer.model_max_length, truncation=True,
                    return_tensors="pt").input_ids.to(device)
                out[begin:begin + ids.shape[0]] = (
                    encoder(ids)[0].float().cpu().numpy().astype(np.float16))
        del encoder
        torch.cuda.empty_cache()
        return out

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def load(self):
        """Load the trained checkpoints (the generator is loaded lazily)."""

        self._ensure_branch_from_h5()
        stage_one = self.checkpoint_dir / "stage_one.pt"
        stage_two = self.checkpoint_dir / "stage_two_linear.pt"
        if not stage_one.is_file() or not stage_two.is_file():
            raise FileNotFoundError(
                f"trained checkpoints missing under {self.checkpoint_dir}; "
                "run train() first"
            )
        state = torch.load(stage_one, map_location="cpu", weights_only=False)
        # The checkpoint records the encoder it was trained with; a mismatch
        # means the weights and the window geometry do not belong together.
        saved = state.get("encoder") or {}
        if saved.get("encoder") and saved["encoder"] != self.encoder:
            raise ValueError(
                f"checkpoint {stage_one} was trained with encoder "
                f"{saved['encoder']!r}, but this model is {self.encoder!r}"
            )
        if saved.get("window_samples"):
            self._window_spec = {
                "n_windows": int(saved["n_windows"]),
                "window_samples": int(saved["window_samples"]),
                "stride_samples": int(saved["stride_samples"]),
            }
        if saved.get("channel_names") and not self._channel_names:
            self._channel_names = list(saved["channel_names"])
        self._model_input_normalization = state.get(
            "model_input_normalization", "zscore_train_split"
        )
        self._stage_one = self._predictor_factory()()
        self._stage_one.load_state_dict(state["model"])
        self._stage_one = self._stage_one.to(self._device).eval()
        adapter_state = torch.load(stage_two, map_location="cpu", weights_only=False)
        self._adapter = SchemeDLinearAdapter()
        self._adapter.load_state_dict(adapter_state["model"])
        self._adapter = self._adapter.to(self._device).eval()
        self._normalization = (
            np.asarray(state["normalization_mean"], dtype=np.float32),
            np.asarray(state["normalization_std"], dtype=np.float32),
        )
        self._generator: Optional[AnimateDiffGenerator] = None
        return self

    @torch.no_grad()
    def generate(self, inputs: List[Any]) -> List[Dict[str, Any]]:
        """Produce one clip per sample and return ``{"path": "<file>#<row>"}``."""

        if self._stage_one is None:
            self.load()
        out_dir = Path(self.cfg.get("output_dir", "./runs/eeg2video_generate"))
        out_dir.mkdir(parents=True, exist_ok=True)

        epochs = np.stack([np.asarray(s.eeg, dtype=np.float32) for s in inputs])
        if self._window_spec is None:
            # No checkpoint metadata to read (older run): fall back to the
            # branch's declared geometry, then to the paper's default.
            self._window_spec = ENCODER_WINDOW_SPECS.get(self.encoder)
        windows = _windows_from_epoch(epochs, self._window_spec)
        if self._channel_names is None:
            names = getattr(inputs[0], "channel_names", None)
            if names:
                self._channel_names = list(names)
        mean, std = self._normalization
        normalized = (windows - mean) / std
        # Noise control: replace the standardised EEG with noise carrying the
        # same per-element mean and standard deviation.  Injected here, after
        # the branch's own normalisation, because that is the point where the
        # scale no longer depends on which branch this is -- see
        # `core/noise_control.py`.
        normalized = noise_control.maybe_noise(normalized)
        conditions = []
        for begin in range(0, len(normalized), BATCH_SIZE):
            batch = torch.from_numpy(
                normalized[begin:begin + BATCH_SIZE]).to(self._device)
            conditions.append(self._adapter(self._stage_one(batch)).float().cpu())
        conditions = torch.cat(conditions)

        if self._generator is None:
            # float16 is the benchmark default for this task.  The copied core
            # defaults to float32, but on an A100 that costs ~22 s/clip against
            # ~5 s/clip, i.e. 94 h against 22.5 h for the 4-encoder x 20-subject
            # sweep.  Measured on one EEGNet track, the two dtypes differ by at
            # most 0.0056 on any of the five metrics (0.2 standard errors at
            # n=200) and by SSIM 0.96 pixel-wise, so the speed is taken and the
            # deviation recorded in the report.  Pass "float32" for the
            # released, protocol-exact path.
            dtype_name = str(self.cfg.get("generator_dtype", "float16"))
            dtype = {
                "float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            }.get(dtype_name)
            if dtype is None:
                raise ValueError(
                    f"unknown generator_dtype {dtype_name!r}; expected one of "
                    "float32/fp32/float16/fp16/bfloat16/bf16"
                )
            self._generator = AnimateDiffGenerator(self._device, dtype=dtype)
            self._generator_dtype = dtype_name
        frames = self._generator.generate(conditions, seed=SEED, progress=False)
        np.save(out_dir / "generated_frames.uint8.npy", frames)
        if self.cfg.get("export_videos", True):
            written = self.export_videos(frames, out_dir, resume=True)
            log.info("exported %d mp4 clip(s)", written)
        (out_dir / "generation_meta.json").write_text(
            json.dumps({
                "encoder": self.encoder,
                "subject": self.cfg.get("subject_id"),
                "generator_dtype": getattr(self, "_generator_dtype", "float16"),
                "model_input_normalization": self._model_input_normalization,
                "window_spec": self._window_spec,
                "seed": SEED,
                "n_clips": int(frames.shape[0]),
            }, indent=2, default=str),
            encoding="utf-8",
        )
        log.info("generated %s clips at %s (dtype=%s)", frames.shape, out_dir,
                 getattr(self, "_generator_dtype", "float32"))

        return [
            {"sample_id": inputs[i].sample_id,
             "path": f"{out_dir / 'generated_frames.uint8.npy'}#{i}"}
            for i in range(len(inputs))
        ]

    # ------------------------------------------------------------------
    # video export
    # ------------------------------------------------------------------

    def export_videos(self, frames, out_dir: Path, fps: int = 3,
                      resume: bool = True) -> int:
        """Write one mp4 per clip into ``<out_dir>/videos``.

        The stacked ``generated_frames.uint8.npy`` stays the scoring artefact
        (``preds_video_generation.json`` points at it); the mp4s are the same
        frames in a container a human can actually open.  Both live in the
        track's generate directory, so a track's outputs stay together.

        The evaluator already accepts mp4/gif predictions --
        ``metrics/video_generation.py::_resolve`` handles ``.mp4/.gif/.avi``
        through imageio -- so this is presentation, not a second protocol.
        """

        import imageio.v2 as imageio

        frames = np.asarray(frames)
        if frames.ndim != 5:
            raise ValueError(
                f"expected (N,T,H,W,3) clips, got {frames.shape}"
            )
        video_dir = Path(out_dir) / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        written = 0
        for index in range(frames.shape[0]):
            path = video_dir / f"clip_{index:04d}.mp4"
            if resume and path.is_file() and path.stat().st_size > 0:
                continue
            imageio.mimsave(
                path, [np.asarray(frame) for frame in frames[index]],
                fps=fps, quality=8, macro_block_size=None,
            )
            written += 1
        return written

    def unload(self):
        self._stage_one = None
        self._adapter = None
        self._generator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
