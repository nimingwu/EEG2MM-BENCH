"""Semantic-level CLIP-to-GIT projector used by ATM reconstruction."""
from __future__ import annotations

from einops.layers.torch import Rearrange
from torch import nn


class PixelProjector(nn.Sequential):
    def __init__(self, embedding_dim: int = 1024):
        super().__init__(
            Rearrange("b c l -> b l c"),
            nn.Linear(1, 257),
            nn.LayerNorm(257),
            Rearrange("b l c -> b c l"),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )


DEFAULT_SEMANTIC_EPOCHS = 30
