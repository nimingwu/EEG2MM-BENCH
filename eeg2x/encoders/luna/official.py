# Copyright (C) 2025 ETH Zurich, Switzerland
# SPDX-License-Identifier: Apache-2.0
"""LUNA-Base encoder copied from BioFoundation commit 68bd7cc.

Only the reconstruction decoder and classification logits are omitted. Module
names and tensor operations for the pretrained encoder are retained so the
official safetensors checkpoint loads without architecture conversion.
"""
from __future__ import annotations

import math

import torch
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from timm.layers import DropPath, Mlp, trunc_normal_
from torch import nn
from torch.nn import functional as F


def nerf_positional_encoding(coords: torch.Tensor, embed_size: int) -> torch.Tensor:
    batch, channels, dimensions = coords.shape
    frequencies = embed_size // (2 * dimensions)
    leftover = embed_size - frequencies * 2 * dimensions
    frequency_bands = 2.0 ** torch.arange(frequencies, device=coords.device).float()
    scaled = coords.unsqueeze(-1) * frequency_bands.view(1, 1, 1, -1)
    encoded = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1)
    encoded = encoded.permute(0, 1, 3, 2, 4).reshape(
        batch, channels, frequencies * dimensions * 2)
    if leftover:
        encoded = torch.cat([
            encoded,
            torch.zeros(
                batch, channels, leftover, device=coords.device, dtype=coords.dtype),
        ], dim=-1)
    return encoded


class FrequencyFeatureEmbedder(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        in_features = 2 * (patch_size // 2 + 1)
        self.frequency_to_embed = Mlp(
            in_features=in_features,
            hidden_features=4 * in_features,
            out_features=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, times = x.shape
        if times % self.patch_size:
            x = F.pad(x, (0, self.patch_size - times % self.patch_size))
            times = x.shape[-1]
        patches = x.view(batch, channels, times // self.patch_size, self.patch_size)
        frequency = torch.fft.rfft(patches, dim=-1)
        features = torch.cat((torch.abs(frequency), torch.angle(frequency)), dim=-1)
        embedded = self.frequency_to_embed(features)
        return rearrange(embedded, "B C t D -> B (C t) D")


class PatchEmbedNetwork(nn.Module):
    def __init__(self, embed_dim: int = 64, patch_size: int = 40):
        super().__init__()
        self.patch_size = patch_size
        out_channels = embed_dim // 4
        kernel_size = patch_size // 2
        self.proj_in = nn.Sequential(
            nn.Conv2d(
                1, out_channels, kernel_size=(1, kernel_size - 1),
                stride=(1, kernel_size // 2), padding=(0, kernel_size // 2 - 1)),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=(1, 3),
                stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=(1, 3),
                stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "B C (S P) -> B (C S) P", P=self.patch_size)
        x = self.proj_in(x.unsqueeze(1))
        return rearrange(x, "B E CS D -> B CS (D E)")


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        num_queries: int,
        input_embed_dim: int,
        output_embed_dim: int,
        num_heads: int,
        dropout_p: float = 0.1,
        ff_dim: int = 2048,
    ):
        super().__init__()
        self.query_embed = nn.Parameter(torch.randn(1, num_queries, input_embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            input_embed_dim, num_heads, dropout=dropout_p, batch_first=True)
        self.temparature = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.ffn = Mlp(
            input_embed_dim, ff_dim, output_embed_dim, act_layer=nn.GELU,
            drop=dropout_p, norm_layer=nn.LayerNorm)
        self.keys_norm = nn.LayerNorm(input_embed_dim)
        self.values_norm = nn.LayerNorm(input_embed_dim)
        self.queries_norm = nn.LayerNorm(input_embed_dim)
        self.query_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                input_embed_dim, nhead=num_heads, activation="gelu",
                dim_feedforward=ff_dim, batch_first=True, norm_first=True),
            num_layers=3,
        )

    def initialize_weights(self) -> None:
        nn.init.orthogonal_(self.query_embed, gain=1.0)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        queries = self.queries_norm(self.query_embed.repeat(x.shape[0], 1, 1))
        output, scores = self.cross_attention(
            query=queries, key=self.keys_norm(x), value=self.values_norm(x))
        output = self.ffn(output) + output
        return self.query_self_attn(output), scores


class RotarySelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.rotary_emb = RotaryEmbedding(dim=head_dim, learned_freq=False)
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_fn = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv_proj(x).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)
        attention = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attention = self.attn_drop_fn(attention) @ v
        attention = rearrange(attention, "B H N D -> B N (H D)")
        return self.proj_drop(self.proj(attention))


class FeedForwardBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout1(self.activation(self.fc1(x)))
        return self.dropout2(self.fc2(self.norm(x)))


class RotaryTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotarySelfAttentionBlock(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForwardBlock(dim, int(dim * mlp_ratio), dropout=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        return x + self.drop_path2(self.mlp(self.norm2(x)))


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


class LunaBackbone(nn.Module):
    """Official LUNA-Base pretrained encoder, without task-specific heads."""

    def __init__(
        self,
        patch_size: int = 40,
        num_queries: int = 4,
        embed_dim: int = 64,
        depth: int = 8,
        num_heads: int = 2,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.patch_size = patch_size
        self.patch_embed_size = embed_dim
        self.patch_embed = PatchEmbedNetwork(embed_dim, patch_size)
        self.freq_embed = FrequencyFeatureEmbedder(patch_size, embed_dim)
        self.channel_location_embedder = nn.Sequential(Mlp(
            in_features=embed_dim, hidden_features=embed_dim * 2,
            out_features=embed_dim, act_layer=nn.GELU, drop=0.0,
            norm_layer=nn.LayerNorm))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cross_attn = CrossAttentionBlock(
            num_queries, embed_dim, embed_dim, num_heads,
            ff_dim=int(mlp_ratio * embed_dim))
        self.blocks = nn.ModuleList([
            RotaryTransformerBlock(
                embed_dim * num_queries, num_heads * num_queries,
                mlp_ratio=mlp_ratio, qkv_bias=True, drop_path=drop_path)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim * num_queries)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        self.cross_attn.initialize_weights()
        trunc_normal_(self.mask_token, std=0.02)
        self.apply(_init_weights)
        for layer_id, layer in enumerate(self.blocks, start=1):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def prepare_tokens(
        self, x_signal: torch.Tensor, channel_locations: torch.Tensor,
    ) -> torch.Tensor:
        channels = channel_locations.shape[1]
        patches = x_signal.shape[-1] // self.patch_size
        tokenized = self.patch_embed(x_signal) + self.freq_embed(x_signal)
        channel_min = channel_locations.min(dim=1, keepdim=True).values
        channel_max = channel_locations.max(dim=1, keepdim=True).values
        locations = (channel_locations - channel_min) / (
            channel_max - channel_min + 1e-8)
        locations = nerf_positional_encoding(locations, self.patch_embed_size)
        locations = self.channel_location_embedder(locations)
        tokenized = rearrange(tokenized, "B (C t) D -> (B t) C D", C=channels)
        # All samples use the same montage, matching the official repeat logic.
        return tokenized + locations.repeat(patches, 1, 1)

    def forward(
        self, x_signal: torch.Tensor, channel_locations: torch.Tensor,
    ) -> torch.Tensor:
        batch = x_signal.shape[0]
        x = self.prepare_tokens(x_signal, channel_locations)
        x, _ = self.cross_attn(x)
        x = rearrange(x, "(B t) Q D -> B t (Q D)", B=batch)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class LunaTemporalPool(nn.Module):
    """Official fine-tuning aggregation query, with classification MLP removed."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 2):
        super().__init__()
        self.decoder_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.15)
        self.learned_agg = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.learned_agg.repeat(x.shape[0], 1, 1)
        return self.decoder_attn(query=query, key=x, value=x)[0][:, 0, :]


__all__ = ["LunaBackbone", "LunaTemporalPool"]
