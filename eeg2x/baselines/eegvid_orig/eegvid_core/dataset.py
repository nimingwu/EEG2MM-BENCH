"""Training data for VideoEEGStyleGAN-ADA: frames plus EEG conditioning.

The paper's generation recipe, from Appendix E and Listing 1:

    "we have used EEGStyleGAN-ADA with modification...  We further modified it
     to condition the temporal frames from the video with positional encoding
     on the EEG features.  This helps us synthesize video from an image-based
     StyleGAN network.  We have used sinusoidal positional encoding with an
     encoding dimension of 10."

Listing 1 shows the construction exactly: for each frame index ``f`` of a clip,
build ``encode_position_vectorized(f, 10)`` and append it to the EEG feature,

    eeg_feat = cat([psi, pe(f)])

``encode_position_vectorized(x, enc_dim)`` returns
``cat([x, sin(2^0 x), cos(2^0 x), ..., sin(2^9 x), cos(2^9 x)])`` -- 1 + 2*10 =
21 values -- so the generator's conditioning vector is 1024 + 21 = **1045**-d,
and every frame of a clip is a separate training item sharing the clip's psi.

Two benchmark adaptations, both forced and both recorded in the run summary:

* **Frame count.**  The paper synthesises eight frames per stimulus, so the
  model is trained and run with eight (``N_FRAMES``), frame indices 0..7.  The
  benchmark's ground truth is six keyframes over the same window, so scoring
  needs six: :func:`select_benchmark_frames` picks the six of the eight nearest
  in time to the ground-truth keyframes, rather than the model being trained on
  a six-frame horizon it was never meant to have.
* **Resolution.**  The paper's GAN is 128x128 and stays 128x128 here -- the
  frames are downsampled from the benchmark's 288x512 for training and
  upsampled back at scoring time, which is the paper's own pipeline rather than
  a change to it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import sys

# The manifest's cutting rule is the loader's, and it is reused rather than
# re-derived -- see ``stimulus_frame_indices``.
from eeg2x.preprocessing.ds005662 import stimulus_frame_indices

LOG = logging.getLogger("eeg2video_eegvid.dataset")

#: The paper's positional-encoding dimension ("an encoding dimension of 10").
PE_DIM = 10
#: The paper's frame resolution for SEED-DV ("8 frames at 128x128 resolution").
FRAME_RESOLUTION = 128
#: Frames generated per clip.  The paper's own number -- the model is trained
#: and run at this horizon, so its frame-index encoding spans 0..7.
N_FRAMES = 8
#: Frames the benchmark's ground truth has, and therefore how many are scored.
BENCHMARK_N_FRAMES = 6


def benchmark_frame_selection(n_generated: int = N_FRAMES,
                              n_scored: int = BENCHMARK_N_FRAMES
                              ) -> List[int]:
    """Which generated frames correspond to the scored keyframes.

    Both grids are evenly spaced over the same window, so frame ``i`` of
    ``n_scored`` sits at ``i / (n_scored - 1)`` of the way through and the
    nearest generated frame is ``round(i * (n_generated - 1) / (n_scored - 1))``.

    Computed rather than hard-coded so the rule stays visible and correct if
    either count changes.  For the paper's 8 against the benchmark's 6 this is
    ``[0, 1, 3, 4, 6, 7]``.
    """

    return [round(i * (n_generated - 1) / (n_scored - 1))
            for i in range(n_scored)]


def select_benchmark_frames(frames: np.ndarray,
                            n_scored: int = BENCHMARK_N_FRAMES) -> np.ndarray:
    """``(N, n_generated, H, W, C)`` -> ``(N, n_scored, H, W, C)``.

    The model keeps the paper's eight-frame output; this is the benchmark-side
    adaptation that lines it up with six ground-truth keyframes.
    """

    frames = np.asarray(frames)
    selection = benchmark_frame_selection(frames.shape[1], n_scored)
    return frames[:, selection]


def encode_position_vectorized(x: torch.Tensor, enc_dim: int) -> torch.Tensor:
    """Listing 1 of the paper, verbatim in behaviour.

    ``x`` is a scalar frame index; the result is
    ``cat([x, sin(2^0 x), cos(2^0 x), ..., sin(2^(d-1) x), cos(2^(d-1) x)])``,
    i.e. ``1 + 2*enc_dim`` values.
    """

    powers = torch.arange(enc_dim, device=x.device)
    frequencies = torch.pow(2.0, powers)
    xf = x.unsqueeze(-1) * frequencies
    sin_encodings = torch.sin(xf)
    cos_encodings = torch.cos(xf)
    encodings = torch.stack([sin_encodings, cos_encodings], dim=-1).flatten(-2)
    return torch.cat([x.unsqueeze(-1), encodings], dim=-1)


def conditioning_dim(feature_dim: int, pe_dim: int = PE_DIM) -> int:
    """1024-d EEG feature plus the 21-d frame-index encoding."""

    return feature_dim + 1 + 2 * pe_dim


def build_conditions(psi: torch.Tensor, n_frames: int = N_FRAMES,
                     pe_dim: int = PE_DIM) -> torch.Tensor:
    """``psi (N, D)`` -> ``(N * n_frames, D + 1 + 2*pe_dim)``.

    Row ``i * n_frames + f`` is clip ``i``'s EEG feature with frame ``f``'s
    positional encoding appended -- the paper's
    ``torch.cat([eeg_feat, temp_eeg_feat], dim=-1)``.
    """

    n, dim = psi.shape
    frames = torch.arange(n_frames, device=psi.device, dtype=torch.float32)
    pe = torch.stack([encode_position_vectorized(f, pe_dim) for f in frames])
    pe = pe.unsqueeze(0).expand(n, -1, -1)              # (N, F, 1 + 2*pe_dim)
    flat = psi.unsqueeze(1).expand(n, n_frames, dim)    # (N, F, D)
    return torch.cat([flat, pe], dim=-1).reshape(n * n_frames, -1)


def prepare_frames(h5_paths: Sequence[Path], out_path: Path,
                   resolution: int = FRAME_RESOLUTION,
                   n_frames: int = N_FRAMES,
                   force: bool = False) -> np.ndarray:
    """Stimulus frames at ``resolution``, ``(N, n_frames, R, R, 3)`` uint8.

    The manifest gives the six keyframes the evaluator scores; this samples
    ``n_frames`` evenly spaced frames *over the same span*, so the model sees
    the paper's eight-frame horizon covering exactly the window the ground
    truth covers (and :func:`select_benchmark_frames` can then pick the six
    that align with it).  Without the expansion the model would only ever see
    six positions while its frame-index encoding spans eight.
    """

    import cv2
    import h5py
    import json

    out_path = Path(out_path)
    if out_path.is_file() and not force:
        return np.load(out_path, mmap_mode="r")

    rows: List[Tuple[str, List[int], int, int]] = []
    for path in h5_paths:
        with h5py.File(path, "r") as h5:
            n = int(h5.attrs["n_epochs"])
            for i in range(n):
                transform = json.loads(
                    h5["stimulus_transform_json"][i].decode("utf-8"))
                uri = h5["stimulus_uri"][i].decode("utf-8")
                # Whichever cutting rule this dataset declared: SEED-DV and
                # EEG-Video-Action write ``frame_indices``, ds005662 does not
                # and its frames are spread over the whole file.  Reading the
                # key straight out of the manifest assumed the first rule and
                # died on ``KeyError: 'frame_indices'`` for the others.
                gt = stimulus_frame_indices(Path(uri), transform)
                # The GT keyframes are evenly spaced over the clip; resample
                # that same span onto the model's frame count.
                span_from, span_to = gt[0], gt[-1]
                indices = [round(span_from + (span_to - span_from) * j
                                 / (n_frames - 1)) for j in range(n_frames)]
                rows.append((uri, indices,
                             int(transform["height"]), int(transform["width"])))

    # One decoder held open per video, walked forward: the same reader the rest
    # of the benchmark uses, because re-opening per trial re-decodes the prefix.
    frames = np.zeros((len(rows), n_frames, resolution, resolution, 3),
                      dtype=np.uint8)
    current, capture, cursor = None, None, 0
    for index, (uri, indices, height, width) in enumerate(rows):
        if capture is None or uri != current or cursor > min(indices):
            if capture is not None:
                capture.release()
            capture = cv2.VideoCapture(str(uri))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open stimulus video {uri}")
            current, cursor = uri, 0
        wanted = {v: k for k, v in enumerate(indices)}
        got, last = 0, max(indices)
        while cursor <= last and got < len(wanted):
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError(f"{uri} ended before frame {last}")
            if cursor in wanted:
                resized = cv2.resize(frame[..., ::-1], (resolution, resolution),
                                     interpolation=cv2.INTER_AREA)
                frames[index, wanted[cursor]] = resized
                got += 1
            cursor += 1
        if (index + 1) % 200 == 0:
            LOG.info("decoded %d/%d clips", index + 1, len(rows))
    if capture is not None:
        capture.release()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, frames)
    LOG.info("wrote stimulus frames %s", frames.shape)
    return frames


class EEGVidFrameDataset:
    """One item per (clip, frame): a frame image and its 1045-d conditioning.

    Mirrors the interface stylegan2-ada's training loop expects:
    ``__getitem__`` -> ``(image_uint8_HWC, label)``, plus ``label_dim``,
    ``resolution``, ``num_channels``, ``has_labels`` and ``get_label``.
    """

    def __init__(self, frames_path, psi_path, labels_path=None,
                 n_frames: int = N_FRAMES, pe_dim: int = PE_DIM):
        # Constructed by ``dnnlib.util.construct_class_by_name(**kwargs)`` from
        # the training loop, so the signature is paths rather than arrays -- the
        # EEG conditioning is precomputed by the frozen encoder in a prepare
        # stage, exactly as EEGStyleGAN-ADA precomputes its own features.
        frames = np.load(Path(frames_path), mmap_mode="r")
        psi = np.load(Path(psi_path), mmap_mode="r")
        if frames.shape[0] != psi.shape[0]:
            raise ValueError(
                f"{frames_path} holds {frames.shape[0]} clips but {psi_path} "
                f"holds {psi.shape[0]}")
        self.frames_path = str(frames_path)
        self.psi_path = str(psi_path)
        self.resolution = int(frames.shape[2])
        self.num_channels = 3
        self.n_frames = int(n_frames)
        clips = frames.shape[0]

        conditions = build_conditions(
            torch.as_tensor(np.asarray(psi), dtype=torch.float32),
            n_frames, pe_dim).numpy()
        # (N, F, C, R, R) uint8 -- CHW, which is how EEGStyleGAN-ADA's own
        # dataset stores its images and what its training loop indexes.
        self.images = np.ascontiguousarray(
            frames.transpose(0, 1, 4, 2, 3), dtype=np.uint8)
        self.conditions = conditions.reshape(clips, n_frames, -1)
        self._n_clips = clips
        self.set_class_labels(
            np.load(Path(labels_path)) if labels_path is not None
            else np.arange(clips, dtype=np.int64))

        # The attribute set stylegan2-ada's training loop reads off a dataset.
        self.has_labels = True
        self.label_dim = int(conditions.shape[-1])
        self.label_shape = [self.label_dim]
        self.image_shape = [self.num_channels, self.resolution, self.resolution]

    def get_details(self) -> str:
        """One-line description, shown in the training log."""

        return (f"EEGVid frames: {self._n_clips} clips x {self.n_frames} frames, "
                f"{self.resolution}x{self.resolution}, "
                f"conditioning {self.label_dim}-d")

    def __len__(self) -> int:
        return self._n_clips * self.n_frames

    def _split(self, idx: int) -> Tuple[int, int]:
        return idx // self.n_frames, idx % self.n_frames

    def __getitem__(self, idx: int):
        clip, frame = self._split(idx)
        return (torch.from_numpy(np.ascontiguousarray(self.images[clip, frame])),
                torch.from_numpy(self.conditions[clip, frame].copy()))

    def get_label(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.conditions[self._split(idx)])

    def set_class_labels(self, concepts: np.ndarray) -> None:
        """``class_id * n_frames + frame`` per item, flattened (Listing 1).

        The training loop reads ``training_set.labels[idx]`` as a *tensor* per
        flat item index (``setup_snapshot_image_grid`` calls ``.detach()`` on
        it), so this is 1-D of length ``len(self)`` rather than per-clip.
        """

        per_clip = (np.asarray(concepts, dtype=np.int64)[:, None]
                    * self.n_frames
                    + np.arange(self.n_frames, dtype=np.int64)[None, :])
        self.labels = torch.from_numpy(per_clip.reshape(-1))
