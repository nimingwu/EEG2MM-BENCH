"""Local stand-ins for the three ``timm`` symbols the LUNA code uses.

The BioFoundation copy of LUNA imports ``timm.layers.Mlp``,
``timm.layers.DropPath`` and ``timm.models.layers.trunc_normal_``.  ``timm``
is not part of the benchmark environment, and adding it would risk disturbing
the pinned torch/torchvision pair, so the three symbols are restated here.

They are transcribed from the timm implementations the released code was
written against; the forward arithmetic of ``Mlp`` and ``DropPath`` is
unchanged, and ``trunc_normal_`` delegates to ``torch.nn.init``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """timm's drop_path (stochastic depth), restated."""

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """timm's DropPath: per-sample stochastic depth on the residual branch."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class Mlp(nn.Module):
    """timm's ``Mlp``, transcribed from ``timm/layers/mlp.py``.

    The ordering matters and is easy to get wrong: the optional ``norm_layer``
    is built over the **hidden** dimension and runs *after* the activation and
    the first dropout, not before ``fc1``.  The released LUNA checkpoints prove
    it -- ``channel_location_embedder.0.norm.*`` is (128,) and
    ``cross_attn.ffn.norm.*`` is (256,), i.e. hidden widths, not input widths.

        fc1 -> act -> drop1 -> norm -> fc2 -> drop2
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
    ):
        super().__init__()
        if use_conv:
            raise NotImplementedError("LUNA never builds a convolutional Mlp")
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop) if isinstance(drop, (int, float)) else tuple(drop)
        bias = (bias, bias) if isinstance(bias, bool) else tuple(bias)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


#: The SEED 62-channel montage, used to rebuild LUNA's
#: ``SEED_PRETRAINING_CHANNEL_LIST``.  The upstream file imports it from
#: ``torcheeg``; SEED-DV's own channel table is the same montage, so it is
#: inlined here and ``torcheeg`` is not needed.
SEED_CHANNEL_LIST = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ",
    "O2", "CB2",
]
