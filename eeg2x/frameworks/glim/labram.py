"""GLIM v2 with the EEG encoder replaced by official LaBraM-base.

Only the encoder changes relative to :mod:`model.glim_v2.glim`: prompts remain
null, the GLIM aligner and Flan-T5 decoder are untouched, and LaBraM's
pretrained backbone is loaded from the benchmark-local ``outputs`` tree.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.glim import GLIM
from .core.modules import encoder_width

log = logging.getLogger(__name__)


class LaBraMEEGAdapter(nn.Module):
    """Official LaBraM-base backbone followed by a fixed token adapter.

    Input is GLIM's ``(B, 2000, 128)`` at 200 Hz.  It is chunked into the
    official LaBraM layout ``(B, 128, 10, 200)``.  LaBraM returns one CLS plus
    1280 channel/time tokens; the CLS token is dropped and adaptive average
    pooling produces GLIM's 96 x 256 EEG sequence.
    """

    def __init__(self, pretrained_path: str, out_len: int = 96, out_dim: int = 256,
                 eeg_size: int = 2000, freeze_backbone: bool = False):
        super().__init__()
        from .third_party.labram.modeling_finetune import labram_base_patch200_200
        # ``num_classes=0`` is essential: otherwise the official factory's
        # default 1000-way head would be applied after ``return_all_tokens``.
        self.backbone = labram_base_patch200_200(EEG_size=eeg_size, init_values=0.1, num_classes=0)
        state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        student = {k[len("student."):]: v for k, v in state.items() if k.startswith("student.")}
        # The pretraining implementation calls the final LayerNorm ``norm``;
        # the official fine-tuning implementation exposes the same operation
        # as ``fc_norm`` when mean pooling is enabled.  Transfer it explicitly
        # so no backbone normalization is silently re-initialized.
        if "norm.weight" in student:
            student["fc_norm.weight"] = student["norm.weight"]
        if "norm.bias" in student:
            student["fc_norm.bias"] = student["norm.bias"]
        missing, unexpected = self.backbone.load_state_dict(student, strict=False)
        # Pretraining heads are intentionally absent from the fine-tuning
        # backbone.  Backbone parameters must not be silently shape-mismatched.
        shape_errors = []
        own = self.backbone.state_dict()
        for k, v in student.items():
            if k in own and own[k].shape != v.shape:
                shape_errors.append((k, tuple(v.shape), tuple(own[k].shape)))
        if shape_errors:
            raise RuntimeError(f"LaBraM checkpoint shape mismatch: {shape_errors[:3]}")
        log.info("Loaded LaBraM-base: missing=%d unexpected=%d", len(missing), len(unexpected))
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad_(False)
        self.proj = nn.Linear(200, out_dim)
        self.out_len = int(out_len)

    def forward(self, eeg: torch.Tensor, mask: Optional[torch.Tensor] = None, p=None, need_weights: bool = False):
        # GLIM convention: (B,T,C); zero padded samples are excluded before
        # chunking, while LaBraM itself remains structurally unchanged.
        if mask is not None:
            eeg = eeg * mask.to(eeg.dtype).unsqueeze(-1)
        b, t, c = eeg.shape
        # Time axis fixed (2000 = 10 patches x 200), channel axis free: LaBraM
        # takes (B, C, 10, 200) and selects position embeddings by channel, so
        # `c` was always a variable here.  The `c != 128` guard was an artifact
        # of ZuCo being the only dataset wired up (it pads 104 measured -> 128).
        # DATA_PROTOCOL.md:31 is explicit that the protocol does not fix the
        # channel count.
        if t != 2000:
            raise ValueError(f"LaBraM adapter expects (B,2000,C), got {tuple(eeg.shape)}")
        x = eeg.transpose(1, 2).reshape(b, c, 10, 200)
        # `input_chans` selects which of LaBraM's 128 pretrained channel
        # position embeddings to use.  LaBraM's `pos_embed` is literally
        # `zeros(1, 128 + 1, D)` in `third_party/labram/modeling_finetune.py`,
        # and its `forward_features` indexes it with `input_chans` --
        # `None` means "all 128", which only works when the input really has
        # 128 channels.  Index 0 is the CLS slot, so a c-channel input needs
        # `[0, 1, ..., c]`.
        #
        # The selection is positional, not name-based: LaBraM's own
        # name->index map is not consulted here, so this makes the same
        # (arbitrary) assumption the 128-channel path already made.  A
        # name-based mapping would be the principled upgrade if these runs ever
        # need to be comparable to LaBraM's published numbers.
        input_chans = list(range(c + 1))
        tokens = self.backbone(x, input_chans=input_chans, return_all_tokens=True)
        tokens = tokens[:, 1:, :]  # discard CLS; (B, 10*c, 200)
        tokens = self.proj(tokens)
        pooled = F.adaptive_avg_pool1d(tokens.transpose(1, 2), self.out_len).transpose(1, 2)
        return pooled, {}


class GLIMLaBraMText(GLIM):
    """GLIM v2 null-prompt model with a pretrained LaBraM EEG encoder."""

    def __init__(self, labram_pretrained: str = "outputs/glim_labram_text/pretrained/labram-base.pth",
                 freeze_labram: bool = False, **kwargs):
        # The GLIM constructor creates all unchanged modules and a temporary
        # EEGEncoder; replace only that component immediately afterwards.
        kwargs.setdefault("input_eeg_len", 2000)
        kwargs.setdefault("hidden_eeg_len", 96)
        kwargs.setdefault("input_dim", 128)
        kwargs.setdefault("hidden_dim", 256)
        kwargs.setdefault("use_null_prompt", True)
        # That temporary EEGEncoder uses `input_dim` as its hidden_dim, which
        # num_heads must divide -- and it is built BEFORE the replacement below,
        # so a channel count that does not divide (TMNRED's 30) raises there and
        # never reaches the adapter.  The adapter reads the real channel count
        # off the tensor, so the throwaway is built at a compatible width.
        kwargs["input_dim"] = encoder_width(
            int(kwargs["input_dim"]), int(kwargs.get("num_heads", 8)))
        super().__init__(**kwargs)
        path = Path(labram_pretrained)
        if not path.is_absolute(): path = Path(__file__).resolve().parent.parent / path
        if not path.exists(): raise FileNotFoundError(f"Missing benchmark-local LaBraM weight: {path}")
        self.eeg_encoder = LaBraMEEGAdapter(str(path), out_len=96, out_dim=256,
                                            eeg_size=2000, freeze_backbone=freeze_labram)
        self.labram_pretrained = str(path)
        self.freeze_labram = bool(freeze_labram)
        self.save_hyperparameters(logger=True)


__all__ = ["LaBraMEEGAdapter", "GLIMLaBraMText"]
