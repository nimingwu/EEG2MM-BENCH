"""GLIM text reconstruction with LUNA-base or REVE-base EEG encoders."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors

from .core.glim import GLIM
from .core.modules import encoder_width

log = logging.getLogger(__name__)


def _fixed_positions(n_channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Deterministic pseudo-3D montage for anonymised ZuCo channels."""
    idx = torch.arange(n_channels, device=device, dtype=dtype)
    golden = 0.6180339887498949
    z = 1.0 - 2.0 * (idx + 0.5) / n_channels
    radius = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = 2.0 * torch.pi * torch.remainder(idx * golden, 1.0)
    return torch.stack((radius * torch.cos(theta), radius * torch.sin(theta), z), dim=-1)


class LunaEEGAdapter(nn.Module):
    """LUNA-base encoder followed by a GLIM-compatible token projection."""

    def __init__(self, pretrained_path: str, out_len: int = 96, out_dim: int = 256):
        super().__init__()
        from .third_party.luna.LUNA import LUNA

        self.backbone = LUNA(
            patch_size=40, embed_dim=64, num_heads=2, depth=8,
            num_queries=4, drop_path=0.0, num_classes=2,
        )
        state = load_safetensors(pretrained_path, device="cpu")
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        missing_encoder = [k for k in missing if not k.startswith("classifier.")]
        if missing_encoder:
            raise RuntimeError(f"LUNA-base missing encoder keys: {missing_encoder[:8]}")
        log.info("Loaded LUNA-base: missing=%d unexpected=%d", len(missing), len(unexpected))
        self.out_len = int(out_len)
        self.proj = nn.Linear(64 * 4, out_dim)

    def forward(self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None,
                p=None, need_weights: bool = False):
        # GLIM is (B,T,C); LUNA is (B,C,T).
        x = eeg.transpose(1, 2)
        b, c, t = x.shape
        # Only the TIME axis is fixed: it must match the temporal positional
        # encoding this adapter was built for.  The channel axis is free --
        # `_fixed_positions(c, ...)` below derives a deterministic pseudo-3D
        # montage from whatever `c` is, so the adapter was always
        # channel-count-agnostic; the `c != 128` guard was an artifact of ZuCo
        # being the only dataset wired up (it pads 104 measured -> 128).
        # DATA_PROTOCOL.md:31 is explicit that the protocol does not fix the
        # channel count.
        if t != 1280:
            raise ValueError(f"LUNA adapter expects (B,1280,C), got {tuple(eeg.shape)}")
        if mask is not None:
            x = x * mask.to(x.dtype).unsqueeze(1)
        pos = _fixed_positions(c, x.device, x.dtype).unsqueeze(0).expand(b, -1, -1)
        # Reuse the official token preparation logic, but stop before its decoder.
        x0, loc_emb = self.backbone.prepare_tokens(x, pos, mask=None)
        x0, _ = self.backbone.cross_attn(x0)
        x0 = x0.reshape(b, t // 40, self.backbone.num_queries * self.backbone.embed_dim)
        for block in self.backbone.blocks:
            x0 = block(x0)
        latent = self.backbone.norm(x0)
        latent = self.proj(latent)
        return F.adaptive_avg_pool1d(latent.transpose(1, 2), self.out_len).transpose(1, 2), {}


class ReveEEGAdapter(nn.Module):
    """REVE-base encoder with deterministic positions and GLIM token output.

    ``normalize`` (default True) applies a per-trial, per-channel z-score over
    time before the backbone.  This is not cosmetic: REVE's token is
    ``Linear(patch) + bias`` added to a **LayerNorm'd** positional embedding,
    so the position term is pinned to O(1) and is identical for every trial.
    Feed it an EEG whose amplitude is orders of magnitude below that (the
    chisco/tmnred h5 branches carry raw volts, std ~2e-6, while ZuCo's carry
    ~2.1 and the image benchmark's REVE h5 ~0.94) and the signal term vanishes
    under the constant one: the encoder becomes a **constant function of the
    trial**, so the decoder can only learn the language prior.  Measured on
    tmnred: trial-to-trial cosine 1.0000 without, 0.9375 with; see
    eval_logs/diag_reve_wiring.py (outside the benchmark).

    This is deliberately local to the REVE branch -- regenerating the h5 files
    would change labram/luna/glim inputs too, and those runs are already
    produced.  ZuCo is unaffected in practice (its input is already O(1), and a
    z-score leaves an O(1) input O(1)).
    """

    def __init__(self, pretrained_path: str, out_len: int = 96, out_dim: int = 256,
                 normalize: bool = True, boundary_norm: str = "none",
                 norm_scale: float = 3.0, boundary_norm_kind: str = "l2"):
        super().__init__()
        from .third_party.reve.configuration_reve import ReveConfig
        from .third_party.reve.modeling_reve import Reve

        cfg = ReveConfig.from_json_file(str(Path(pretrained_path).with_name("config.json")))
        self.backbone = Reve(cfg)
        state = load_safetensors(pretrained_path, device="cpu")
        state.pop("cls_query_token", None)
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        allowed = {"cls_query_token", "final_layer.weight", "final_layer.bias"}
        bad = [k for k in missing if k not in allowed]
        if bad:
            raise RuntimeError(f"REVE-base missing encoder keys: {bad[:8]}")
        log.info("Loaded REVE-base: missing=%d unexpected=%d", len(missing), len(unexpected))
        self.out_len = int(out_len)
        self.normalize = bool(normalize)
        self.boundary_norm = str(boundary_norm)
        self.norm_scale = float(norm_scale)
        self.boundary_norm_kind = str(boundary_norm_kind)
        self.proj = nn.Linear(int(cfg.embed_dim), out_dim)

    def forward(self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None,
                p=None, need_weights: bool = False):
        if mask is not None:
            eeg = eeg * mask.to(eeg.dtype).unsqueeze(-1)
        # REVE uses (B,C,T), patch 200 with 20-sample overlap.
        x = eeg.transpose(1, 2)
        b, c, t = x.shape
        if self.normalize:
            # per-(trial, channel) z-score; padded steps are excluded from the
            # statistics and re-zeroed afterwards.
            m = (torch.ones_like(x) if mask is None
                 else mask.to(x.dtype).unsqueeze(1).expand(b, c, t))
            n = m.sum(dim=2, keepdim=True).clamp_min(1.0)
            mu = (x * m).sum(dim=2, keepdim=True) / n
            var = ((x - mu) ** 2 * m).sum(dim=2, keepdim=True) / n
            x = ((x - mu) / var.clamp_min(1e-12).sqrt()) * m
        # Time axis fixed, channel axis free -- see the LUNA adapter above.
        if t != 2000:
            raise ValueError(f"REVE adapter expects (B,2000,C), got {tuple(eeg.shape)}")
        pos = _fixed_positions(c, x.device, x.dtype).unsqueeze(0).expand(b, -1, -1)
        tokens = self.backbone(x, pos=pos, return_output=True)
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[-1]
        tokens = self.proj(tokens)
        o = F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.out_len).transpose(1, 2)
        # The REVE branch has a much larger token scale than the frozen text
        # encoder's input interface.  Keep this opt-in so old REVE checkpoints
        # remain reproducible; the formal corrected branch uses l2 x 3.
        if self.boundary_norm == "l2":
            if self.boundary_norm_kind == "l2":
                o = F.normalize(o.float(), dim=-1).to(o.dtype) * self.norm_scale
            elif self.boundary_norm_kind == "rms":
                rms = o.float().square().mean(dim=-1, keepdim=True).add(1e-8).sqrt()
                o = (o.float() / rms * self.norm_scale).to(o.dtype)
            else:
                raise ValueError(f"unknown REVE boundary norm kind: {self.boundary_norm_kind}")
        elif self.boundary_norm != "none":
            raise ValueError(f"unknown REVE boundary norm: {self.boundary_norm}")
        return o, {}


class GLIMLunaText(GLIM):
    def __init__(self, luna_pretrained: str, **kwargs):
        kwargs.setdefault("input_eeg_len", 1280)
        kwargs.setdefault("use_null_prompt", True)
        # GLIM.__init__ always builds an EEGEncoder that this class then throws
        # away, but it is constructed BEFORE the replacement line, and it uses
        # `input_dim` as its hidden_dim, which num_heads must divide.  A dataset
        # whose channel count does not divide (TMNRED's 30) would raise there
        # and never reach the adapter.  The adapter reads the real channel count
        # off the tensor and ignores the prompt embedding entirely, so the
        # throwaway can be built at a compatible width.  Reported by the TMNRED
        # smoke test, which is the only place this could have surfaced.
        kwargs["input_dim"] = encoder_width(
            int(kwargs.get("input_dim", 128)), int(kwargs.get("num_heads", 8)))
        super().__init__(**kwargs)
        self.eeg_encoder = LunaEEGAdapter(luna_pretrained)
        self.save_hyperparameters(logger=True)


class GLIMReveText(GLIM):
    def __init__(self, reve_pretrained: str,
                 reve_boundary_norm: str = "none", reve_norm_scale: float = 3.0,
                 reve_normalize: bool = True, reve_boundary_norm_kind: str = "l2",
                 **kwargs):
        kwargs.setdefault("input_eeg_len", 2000)
        kwargs.setdefault("use_null_prompt", True)
        kwargs["input_dim"] = encoder_width(          # see GLIMLunaText above
            int(kwargs.get("input_dim", 128)), int(kwargs.get("num_heads", 8)))
        super().__init__(**kwargs)
        self.eeg_encoder = ReveEEGAdapter(
            reve_pretrained, normalize=reve_normalize,
            boundary_norm=reve_boundary_norm, norm_scale=reve_norm_scale,
            boundary_norm_kind=reve_boundary_norm_kind)
        self.save_hyperparameters(logger=True)


__all__ = ["LunaEEGAdapter", "ReveEEGAdapter", "GLIMLunaText", "GLIMReveText"]
