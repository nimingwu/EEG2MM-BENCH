"""EEG2Video (NeurIPS 2024) — the paper's original model chain, as a benchmark model.

This is the **baseline** half of the design: unlike ``model/eeg2video.py``
(Scheme D), whose EEG encoder is swappable for a foundation model, this model
replaces nothing.  It is the architecture the paper describes, end to end:

    EEG -> GLMNet (global + occipital local)
        -> Visual Seq2Seq            -> predicted video latents z0
        -> Semantic Predictor        -> predicted CLIP context e_t
        -> Dynamic GLMNet            -> Fast/Slow, which sets DANA's beta
        -> DANA noise + Tune-A-Video -> frames

Why a separate model rather than another encoder option: the point of the
benchmark is to compare *frameworks* as well as encoders.  Scheme D asks "does
a foundation-model encoder help this simple pipeline?"; this model asks "what
does the paper's own pipeline score on the same data, split and metrics?".

Provenance
----------
``model/eeg2video_paper/`` is copied from the EEG2Video project's
``paper_reproduction`` package (identical to ``EEG2Video_reproduction_clean``,
which differs only in two paper-pinning defaults that are adopted here:
``reproduction_revision="paper-literal-caption-v1"`` and
``semantic_target_mode="per_video_caption"``).  The diffusion half
(``vendor/``) is a byte-copy of the released Tune-A-Video UNet and the
EEG-conditioned pipeline.

Only two things are adapted, both forced by running inside a benchmark:

1. paths are anchored to the benchmark tree instead of the EEG2Video project,
   so no artefact can be written outside it;
2. the data step reads the benchmark HDF5 (``from_h5.prepare_from_h5``) rather
   than SEED-DV's raw ``.npy``, so the split, the sample order and the stimulus
   manifest all come from the benchmark's single source of truth.

The model arithmetic -- every layer, loss and sampling step -- is untouched.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from eeg2x.frameworks.eeg2video.core.model_base import BaseModel
from eeg2x.protocol import SEED
from eeg2x.seeding import seed_everything

log = logging.getLogger(__name__)

_PACKAGE = Path(__file__).resolve().parent / "paper_core"


class EEG2VideoPaperModel(BaseModel):
    """The paper's EEG2Video chain, consuming the benchmark HDF5."""

    name = "eeg2video_paper"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = dict(config or {})
        self.subject = str(self.cfg.get("subject_id", "sub-01"))
        self.h5_dir = Path(self.cfg.get("h5_dir", "./h5_seed_dv/eegnet"))
        # One track directory per subject, so two subjects never share a
        # checkpoint root (the paper chain's config resolves these once, at
        # import, which is exactly why the guard in _ensure_env exists).
        self.track_root = Path(self.cfg.get(
            "work_dir", f"./runs/eeg2video_paper/{self.subject}"))
        self._env_ready = False

    # ------------------------------------------------------------------
    # environment: must be set before the vendored config is imported
    # ------------------------------------------------------------------

    def _ensure_env(self) -> None:
        """Export the paper chain's roots, then import it.

        ``model/eeg2video_paper/config.py`` reads ``EEG2VIDEO_*`` once at import
        time -- the same snapshot trap the Scheme D adapter has.  Hence the
        package is imported lazily, only after the roots are in the
        environment, and :meth:`_verify_roots` re-checks the result rather than
        trusting it.
        """

        if self._env_ready:
            return
        track = self.track_root.resolve()
        # The Tune-A-Video fine-tune consumes stimulus videos only -- never EEG
        # and never a predictor output -- so one UNet serves every subject.
        # Its root is therefore benchmark-wide, not per-track: copying a 4.3 GB
        # UNet into 20 subject directories would cost 86 GB for no benefit.
        tune_root = Path(self.cfg.get(
            "tune_root",
            track.parent / "shared" / "tune_a_video")).resolve()
        os.environ.update({
            "EEG2VIDEO_SUBJECT": self.subject.split("-")[-1].lstrip("0") or "1",
            "EEG2VIDEO_RUN_ROOT": str(track),
            "EEG2VIDEO_CHECKPOINT_ROOT": str(track / "checkpoints"),
            "EEG2VIDEO_SHARED_RUN_ROOT": str(track / "shared"),
            "EEG2VIDEO_SHARED_TUNE_ROOT": str(tune_root),
            "EEG2VIDEO_SEMANTIC_TARGET_MODE": "per_video_caption",
            "EEG2VIDEO_REPRODUCTION_REVISION": "paper-literal-caption-v1",
        })
        self._env_ready = True

    def _verify_roots(self):
        """Import the package and confirm it agrees with the configured roots."""

        self._ensure_env()
        from .paper_core import config as paper_config
        from .paper_core import data as paper_data
        from .paper_core import inference as paper_inference
        from .paper_core import train_predictors, train_video_diffusion

        expected = self.track_root.resolve()
        if Path(paper_config.RUN_ROOT).resolve() != expected:
            raise RuntimeError(
                "EEG2VIDEO_RUN_ROOT was not in the environment when "
                f"eeg2video_paper.config was first imported: the chain resolved "
                f"{paper_config.RUN_ROOT} but this model is configured for "
                f"{expected}. Set the variables in a fresh process -- see "
                "scripts/run_eeg2video_track.py, which does exactly that."
            )
        return paper_config, paper_data, paper_inference, train_predictors, \
            train_video_diffusion

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def train(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Prepare from the HDF5, then train the chain the paper describes."""

        self.cfg.update(config or {})
        paper_config, paper_data, _inference, trainers, diffusion = \
            self._verify_roots()
        seed_everything(SEED)

        from .paper_core.from_h5 import prepare_from_h5

        summary = prepare_from_h5(
            h5_dir=self.h5_dir,
            run_root=paper_config.RUN_ROOT,
            shared_root=paper_config.SHARED_RUN_ROOT,
            subject=self.subject,
            data_root=paper_config.DATA_ROOT,
        )
        log.info("prepared from the benchmark h5: %s", summary)

        # Video latents and CLIP text embeddings still need the frozen VAE and
        # text encoder; the EEG side is already done.
        paper_data.extract_embeddings(
            force=bool(self.cfg.get("force", False)))
        trainers.train_all(force=bool(self.cfg.get("force", False)))

        # Tune-A-Video fine-tunes on stimulus videos only -- it never sees EEG
        # or the predictors -- so one trained UNet serves every subject.  Reuse
        # it unless the caller explicitly asks for a fresh fine-tune.
        result: Dict[str, Any] = {"prepared": summary}
        if self.cfg.get("train_unet", False):
            diffusion.train_tune_a_video(
                force=bool(self.cfg.get("force", False)))
            result["unet"] = "trained"
        else:
            result["unet"] = (
                "reused" if diffusion.FINAL_UNET_PATH.exists()
                else "MISSING -- set train_unet=True or point "
                     "EEG2VIDEO_SHARED_TUNE_ROOT at an existing fine-tune"
            )
        return result

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def load(self):
        """Verify the checkpoints the chain needs are present."""

        paper_config, paper_data, paper_inference, _t, diffusion = \
            self._verify_roots()
        missing = [
            p for p in (
                Path(paper_config.CHECKPOINT_ROOT) / "seq2seq.pt",
                Path(paper_config.CHECKPOINT_ROOT) / "semantic_predictor.pt",
                Path(paper_config.CHECKPOINT_ROOT) / "dynamic_glmnet.pt",
                Path(diffusion.FINAL_UNET_PATH),
                Path(paper_data.VIDEO_LATENTS_PATH),
                Path(paper_data.TEXT_EMBEDDINGS_PATH),
            ) if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "the paper chain is missing trained artefacts:\n  "
                + "\n  ".join(str(p) for p in missing)
                + "\nrun train() first"
            )
        self._loaded = True
        return self

    def generate(self, inputs: List[Any]) -> List[Dict[str, Any]]:
        """Run the paper chain's inference and map it to benchmark predictions.

        The chain reconstructs the whole held-out block in one pass (DANA and
        the DDIM loop are batch-shaped over the block), so this runs ``infer``
        once and then hands each input its own row.  Row order is the HDF5's,
        which is the order the manifest was built in.
        """

        if not getattr(self, "_loaded", False):
            self.load()
        paper_config, _data, paper_inference, _t, _d = self._verify_roots()

        seed = int(self.cfg.get("inference_seed", paper_config.SETTINGS.inference_seed))
        paper_inference.infer(seed=seed, force=bool(self.cfg.get("force", False)))

        output_root = Path(paper_config.RUN_ROOT) / "inference" / f"seed_{seed}"
        frames_path = output_root / "generated_frames.uint8.npy"
        if not frames_path.is_file():
            raise FileNotFoundError(
                f"inference produced no frames at {frames_path}")

        # ``benchmark_eval`` mints every sample id as "<h5 stem>_clip-<row>",
        # and ``infer`` writes one frame row per test sample in manifest order,
        # which is the HDF5's row order.  So the row index in the id is the row
        # index into the generated array -- no re-derivation needed.
        import h5py

        frames = np.load(frames_path, mmap_mode="r")
        n_test = int(frames.shape[0])
        # An ``h5_dir`` holds the whole cohort, not one subject -- 20 subjects x
        # 2 splits.  Narrow the glob by subject, or the file count is 20 and not
        # the one this track ran.
        pattern = f"*_{self.subject}_train_epoched.h5"
        train_files = sorted(self.h5_dir.glob(pattern))
        if len(train_files) != 1:
            raise FileNotFoundError(
                f"expected exactly one training HDF5 matching {pattern!r} under "
                f"{self.h5_dir}, found {len(train_files)}")
        with h5py.File(train_files[0], "r") as h5:
            n_train = int(h5.attrs["n_epochs"])

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
                "paper_chain": {
                    "inference_seed": seed,
                    "reproduction_revision":
                        paper_config.SETTINGS.reproduction_revision,
                    "semantic_target_mode":
                        paper_config.SETTINGS.semantic_target_mode,
                    "manifest_row": int(n_train + row),
                },
            })
        return predictions

    def unload(self):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
