"""EEGVid's EEG encoder: modified NICE, GAT -> temporal conv -> linear.

The EEGVid paper (arXiv:2505.21385, "Beyond Reconstruction: What EEG-to-Video
Decoding Actually Recovers") ships no code, but describes its encoder in one
sentence:

    "We use a modified NICE EEG [52] network. ... it consists of a
     graph-attention layer (GAT) [57], followed by temporal convolution and a
     linear layer that produces a psi in R^1024 dimensional feature."

and adds that psi is unit-normalised and trained with

    "the triplet loss [46] with a multi-similarity miner ... Adam with learning
     rate 3e-4 and weight decay 1e-4 ... trained for 300 epochs."

[52] is NICE (Song et al., "Decoding natural images from EEG for object
recognition", arXiv:2308.13234), whose released implementation this module
follows.  NICE's own encoder is ``Enc_eeg`` = ``ResidualAdd(EEG_GAT)`` ->
``PatchEmbedding(tsconv)`` -> head; the paper says it replaced the head with a
linear projection to 1024 and normalises the result.  The pieces used here are
therefore NICE's, with only the changes the paper states.

What the paper does **not** specify, and the choice made here:

* **Layer sizes.**  NICE's released defaults are kept -- ``emb_size=40`` for the
  temporal branch, dropout 0.3 (GAT) / 0.5 (tsconv).  The paper gives no
  numbers, so inventing them would be less faithful than reusing [52]'s.
* **The GAT graph.**  NICE hard-codes a fully connected 64-channel graph and
  calls ``.cuda()`` at construction.  Here the graph is fully connected over
  *this dataset's* channel count and registered as a buffer, so it follows the
  module's device and works for SEED-DV's 62 channels as well as ds005662's 64.
  That is NICE's own topology, not a simplification of it.
* **Output.**  NICE ends in a class head; EEGVid replaces it with a linear
  projection to 1024 plus L2 normalisation, as the paper states.

Everything downstream of this encoder -- the triplet-with-multi-similarity
training objective -- lives in :mod:`eeg2video_eegvid.train_encoder`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

#: NICE's released temporal branch width (its ``emb_size``).
DEFAULT_EMB_SIZE = 40
#: EEGVid states this dimension explicitly: "psi in R^1024".
DEFAULT_FEATURE_DIM = 1024


def _channel_edge_index(n_channels: int) -> torch.Tensor:
    """Fully connected channel graph with no self loops, as ``(2, E)``.

    NICE builds exactly this and then calls ``.cuda()`` on it in ``__init__``;
    returning it as a buffer instead keeps it on whatever device the module is
    moved to, and parameterising the channel count makes it work for datasets
    that are not 64-channel.
    """

    pairs = [(i, j) for i in range(n_channels) for j in range(n_channels) if i != j]
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


class ChannelGraphAttention(nn.Module):
    """NICE's ``EEG_GAT``: attention across EEG channels.

    Each channel is a node whose feature vector is its whole time course, so the
    layer mixes information between channels while weighting them by relevance
    -- the paper's "The GAT layer combines information from all EEG channels
    into a shared representation while weighting them according to their
    relevance."
    """

    def __init__(self, n_channels: int, n_samples: int, dropout: float = 0.3):
        super().__init__()
        from torch_geometric.nn import GATConv

        self.n_channels = int(n_channels)
        self.conv = GATConv(in_channels=n_samples, out_channels=n_samples, heads=1)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("edge_index", _channel_edge_index(self.n_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, 1, C, T), matching NICE's layout for the conv branch.
        batch, channels, _ = x.shape
        if channels != self.n_channels:
            raise ValueError(
                f"expected {self.n_channels} channels, got {channels}")
        nodes = rearrange(x, "b c t -> (b c) t")
        nodes = self.conv(nodes, self.edge_index)
        x = rearrange(nodes, "(b c) t -> b 1 c t", b=batch)
        return self.dropout(x)


class TemporalConvBranch(nn.Module):
    """NICE's ``PatchEmbedding``: temporal conv -> pool -> spatial conv."""

    def __init__(self, n_channels: int, emb_size: int = DEFAULT_EMB_SIZE,
                 dropout: float = 0.5):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange("b e (h) (w) -> b (h w) e"),
        )

    @staticmethod
    def n_tokens(n_samples: int) -> int:
        """Temporal positions surviving ``conv(1,25)`` then ``avgpool(51/5)``."""

        return (n_samples - 25 + 1 - 51) // 5 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tsconv(x)             # (B, 40, 1, tokens)
        x = self.projection(x)         # (B, tokens, emb_size)
        return rearrange(x, "b t e -> b (t e)")


class EEGVidEncoder(nn.Module):
    """The paper's "modified NICE": GAT -> temporal conv -> linear -> 1024-d.

    Follows the benchmark's encoder contract: ``forward((B, C, T)) ->
    (B, feature_dim)`` with a ``feature_dim`` attribute, so it can be dropped
    into any model that takes a swappable EEG encoder.
    """

    def __init__(self, n_chans: int, n_samples: int,
                 feature_dim: int = DEFAULT_FEATURE_DIM,
                 emb_size: int = DEFAULT_EMB_SIZE,
                 gat_dropout: float = 0.3, conv_dropout: float = 0.5):
        super().__init__()
        self.n_chans = int(n_chans)
        self.n_samples = int(n_samples)
        self.feature_dim = int(feature_dim)

        self.gat = ChannelGraphAttention(self.n_chans, self.n_samples, gat_dropout)
        self.temporal = TemporalConvBranch(self.n_chans, emb_size, conv_dropout)
        tokens = self.temporal.n_tokens(self.n_samples)
        if tokens <= 0:
            raise ValueError(
                f"n_samples={self.n_samples} is too short for the temporal "
                f"branch (it needs at least 76 samples)")
        self.projection = nn.Linear(emb_size * tokens, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != self.n_chans:
            raise ValueError(
                f"expected {self.n_chans} channels, got {x.shape[1]}")
        x = self.gat(x)                       # (B, 1, C, T)
        x = self.temporal(x)                  # (B, emb_size * tokens)
        x = self.projection(x)                # (B, feature_dim)
        # "psi in R^1024, which is unit normalized" -- the paper normalises, and
        # the GAN is conditioned on this vector directly.
        return F.normalize(x, dim=-1)


def build_eegvid_encoder(n_chans: int, n_samples: int, **kwargs) -> EEGVidEncoder:
    return EEGVidEncoder(n_chans=n_chans, n_samples=n_samples, **kwargs)
