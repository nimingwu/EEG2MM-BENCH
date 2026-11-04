"""EEG foundation-model encoders, wrapped to the benchmark encoder contract.

Each class here satisfies ``model/encoders.py``'s contract exactly:

    Encoder(n_chans=..., n_samples=..., **kwargs)
    forward(x: (B, C, T)) -> (B, feature_dim)
    .feature_dim : int

so they can be dropped into any model that uses ``build_encoder``.  The three
pretrained backbones (LaBraM, REVE, LUNA) are all "base" sizes, as required.

Each wrapper exists because the released models do **not** expose a pooled
feature vector; they expose a token grid, a reconstruction, or a logits head.
The pooling chosen here is stated per class and is part of the encoder's
contract, so a model consuming the encoder never has to know the difference.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from .labram_model import (
    build_labram_base,
    labram_input_chans,
    load_labram_base_checkpoint,
)
from .luna_model import (
    build_luna_base,
    load_luna_base_checkpoint,
    luna_encode,
)
from .reve_model import build_reve_base, load_reve_base_checkpoint
from .reve_position_bank import RevePositionBank

#: LaBraM's pretraining patch is 1 s at its 200 Hz clock.
LABRAM_PATCH_SAMPLES = 200
#: REVE's patch is 1 s at 200 Hz with 20 samples of overlap.
REVE_PATCH_SAMPLES = 200
#: LUNA's patch is 40 samples; the repo runs it at 256 Hz (its training clock).
LUNA_PATCH_SAMPLES = 40


class LabramOfficialEncoder(nn.Module):
    """LaBraM base (Jiang et al., ICLR 2024), minus the classification head.

    ``x`` is ``(B, C, T)`` with ``T`` a multiple of 200; it is reshaped to the
    ``(B, C, A, 200)`` patch grid the released ``NeuralTransformer`` expects.
    Features are the mean over all returned tokens (CLS included), matching the
    pooling the benchmark's braindecode LaBraM encoder already uses, so the two
    are interchangeable downstream.

    ``feature_dim`` = 200.
    """

    def __init__(self, n_chans: int = 62, n_samples: int = 200,
                 channel_names: Optional[Sequence[str]] = None,
                 checkpoint_path=None):
        super().__init__()
        if channel_names is None:
            raise ValueError(
                "LabramOfficialEncoder requires channel_names: LaBraM selects "
                "position embeddings by electrode index, so the montage must be "
                "known. Provide the h5's channel_names."
            )
        if n_samples % LABRAM_PATCH_SAMPLES != 0:
            raise ValueError(
                f"LaBraM needs a whole number of {LABRAM_PATCH_SAMPLES}-sample "
                f"patches, got n_samples={n_samples}"
            )
        self.n_chans = int(n_chans)
        self.n_samples = int(n_samples)
        self.n_patches = n_samples // LABRAM_PATCH_SAMPLES
        self.model = build_labram_base()
        if checkpoint_path is None:
            self.load_report = load_labram_base_checkpoint(self.model)
        else:
            self.load_report = load_labram_base_checkpoint(self.model, checkpoint_path)
        #: index 0 is the CLS slot, then one row per electrode.
        self.input_chans = labram_input_chans(list(channel_names))
        self.feature_dim = 200

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_chans:
            raise ValueError(f"expected (B,{self.n_chans},T), got {tuple(x.shape)}")
        batch = x.shape[0]
        grid = x.reshape(batch, self.n_chans, self.n_patches, LABRAM_PATCH_SAMPLES)
        tokens = self.model(
            grid, input_chans=self.input_chans, return_all_tokens=True
        )
        return tokens.mean(dim=1)


class ReveEncoder(nn.Module):
    """REVE base (``brain-bzh/reve-base``), encoder half only.

    REVE needs two things besides the signal: at least one whole 200-sample
    patch (its tokenizer is a bare ``unfold``, so anything shorter raises), and
    a ``(B, C, 3)`` electrode-coordinate tensor.  Coordinates are resolved once
    from the released position bank, in the order of ``channel_names``.

    ``clip`` applies REVE's pretraining input clamp to the (already
    standardised) signal: the released recipe z-scores per channel and clamps
    to +/-15 before the patch embedding.  ``feature_dim`` = 512.
    """

    def __init__(self, n_chans: int = 62, n_samples: int = 200,
                 channel_names: Optional[Sequence[str]] = None,
                 clip: float = 15.0, checkpoint_path=None,
                 position_bank=None):
        super().__init__()
        if channel_names is None:
            raise ValueError(
                "ReveEncoder requires channel_names: REVE conditions on 3-D "
                "electrode coordinates, and there is no channel-free fallback."
            )
        if n_samples < REVE_PATCH_SAMPLES:
            raise ValueError(
                f"REVE needs at least one {REVE_PATCH_SAMPLES}-sample patch, "
                f"got n_samples={n_samples}. Choose a longer window; REVE's "
                "tokenizer has no padding path."
            )
        self.n_chans = int(n_chans)
        self.n_samples = int(n_samples)
        self.clip = float(clip)
        self.model = build_reve_base()
        if checkpoint_path is None:
            self.load_report = load_reve_base_checkpoint(self.model)
        else:
            self.load_report = load_reve_base_checkpoint(self.model, checkpoint_path)
        bank = position_bank if position_bank is not None else RevePositionBank.from_release()
        positions = bank.channel_positions(list(channel_names))
        if positions.shape[0] != self.n_chans:
            raise ValueError(
                f"resolved {positions.shape[0]} electrode positions for "
                f"{self.n_chans} EEG channels; every channel must have one"
            )
        self.register_buffer("positions", positions.float(), persistent=False)
        self.position_source = "bank" if bank.has_all(list(channel_names)) else "bank+mne"
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_chans:
            raise ValueError(f"expected (B,{self.n_chans},T), got {tuple(x.shape)}")
        if x.shape[-1] < REVE_PATCH_SAMPLES:
            raise ValueError(
                f"REVE needs T >= {REVE_PATCH_SAMPLES}, got {x.shape[-1]}"
            )
        if self.clip > 0:
            x = x.clamp(-self.clip, self.clip)
        pos = self.positions.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.device)
        hidden = self.model(x, pos)                # (B, C, n_patches, 512)
        return hidden.flatten(1, 2).mean(dim=1)    # (B, 512)


class LunaEncoder(nn.Module):
    """LUNA base (PulpBio), encoder path only.

    ``x`` is ``(B, C, T)`` with ``T`` a multiple of 40.  LUNA normalises
    **per channel within each sample** (``(x - mean_t) / (std_t + 1e-8)``) --
    that is the released pipeline's only input normalisation, and it is
    stateless, so it lives here rather than in a fitted preprocessing step.

    ``feature_dim`` = embed_dim * num_queries = 64 * 4 = 256.
    """

    def __init__(self, n_chans: int = 62, n_samples: int = 480,
                 channel_names: Optional[Sequence[str]] = None,
                 channel_locations: Optional[torch.Tensor] = None,
                 normalize: bool = True, checkpoint_path=None):
        super().__init__()
        if n_samples % LUNA_PATCH_SAMPLES != 0:
            raise ValueError(
                f"LUNA needs a whole number of {LUNA_PATCH_SAMPLES}-sample "
                f"patches, got n_samples={n_samples}"
            )
        if channel_locations is None:
            if channel_names is None:
                raise ValueError(
                    "LunaEncoder requires channel_names or explicit "
                    "channel_locations: LUNA conditions on 3-D coordinates."
                )
            from .luna_channel_embeddings import get_channel_locations

            channel_locations = torch.tensor(
                get_channel_locations(list(channel_names)), dtype=torch.float32
            )
        if channel_locations.shape != (n_chans, 3):
            raise ValueError(
                f"channel_locations must be ({n_chans}, 3), got "
                f"{tuple(channel_locations.shape)}"
            )
        self.n_chans = int(n_chans)
        self.n_samples = int(n_samples)
        self.normalize = bool(normalize)
        self.model = build_luna_base()
        if checkpoint_path is None:
            self.load_report = load_luna_base_checkpoint(self.model)
        else:
            self.load_report = load_luna_base_checkpoint(self.model, checkpoint_path)
        self.register_buffer(
            "channel_locations", channel_locations.float(), persistent=False
        )
        self.feature_dim = int(self.model.embed_dim * self.model.num_queries)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.n_chans:
            raise ValueError(f"expected (B,{self.n_chans},T), got {tuple(x.shape)}")
        if self.normalize:
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True)
            x = (x - mean) / (std + 1e-8)
        locs = self.channel_locations.unsqueeze(0).expand(x.shape[0], -1, -1)
        latent = luna_encode(self.model, x, locs.to(x.device))
        return latent.mean(dim=1)


#: ``ENCODER_REGISTRY`` entries contributed by the vendored foundation models.
PRETRAINED_ENCODER_REGISTRY = {
    "labram": {
        "builder": LabramOfficialEncoder,
        "extra_kwargs": ("channel_names", "checkpoint_path"),
        "description": "LaBraM base, reference implementation (Jiang et al., "
                       "ICLR 2024); feature_dim=200, needs channel_names",
    },
    "reve": {
        "builder": ReveEncoder,
        "extra_kwargs": ("channel_names", "clip", "checkpoint_path"),
        "description": "REVE base (brain-bzh), encoder half; feature_dim=512, "
                       "needs channel_names (3-D electrodes) and T>=200",
    },
    "luna": {
        "builder": LunaEncoder,
        "extra_kwargs": ("channel_names", "channel_locations", "normalize",
                         "checkpoint_path"),
        "description": "LUNA base (PulpBio), encoder path; feature_dim=256, "
                       "needs channel_names or coordinates and T%40==0",
    },
}
