"""Models used by the paper's three independent EEG prediction tasks."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import OCCIPITAL_CHANNELS, SETTINGS


class EEGWindowEncoder(nn.Module):
    """Released EEGNet-style encoder for one 500 ms, 62-channel window."""

    def __init__(self, output_dim: int = SETTINGS.eeg_embedding_dim) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(1, 16, (1, 64), bias=False),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, (SETTINGS.eeg_channels, 1), groups=16, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5),
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(64, 64, (1, 16), groups=64, bias=False),
            nn.Conv2d(64, 16, (1, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5),
        )
        self.embedding = nn.Linear(48, output_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.features(eeg).flatten(1))


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_length: int = 32) -> None:
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        encoding = torch.zeros(max_length, dim)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.encoding[:, : tokens.shape[1]].to(tokens.dtype)


class EEGToLatentSeq2Seq(nn.Module):
    """Two-layer encoder/four-layer autoregressive Transformer decoder."""

    latent_size = (
        SETTINGS.latent_channels
        * SETTINGS.latent_height
        * SETTINGS.latent_width
    )

    def __init__(self) -> None:
        super().__init__()
        dim = SETTINGS.eeg_embedding_dim
        self.eeg_encoder = EEGWindowEncoder(dim)
        self.position = SinusoidalPositionEncoding(dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=SETTINGS.transformer_heads,
            dim_feedforward=SETTINGS.transformer_ff_dim,
            dropout=SETTINGS.transformer_dropout,
            activation="relu",
            batch_first=True,
            norm_first=SETTINGS.transformer_norm_first,
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=SETTINGS.transformer_heads,
            dim_feedforward=SETTINGS.transformer_ff_dim,
            dropout=SETTINGS.transformer_dropout,
            activation="relu",
            batch_first=True,
            norm_first=SETTINGS.transformer_norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, SETTINGS.transformer_encoder_layers
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, SETTINGS.transformer_decoder_layers
        )
        self.latent_predictor = nn.Linear(dim, self.latent_size)

    def encode_eeg(self, eeg: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels, samples = eeg.shape
        if (tokens, channels, samples) != (
            SETTINGS.eeg_tokens,
            SETTINGS.eeg_channels,
            SETTINGS.eeg_window_samples,
        ):
            raise ValueError(f"Unexpected Seq2Seq EEG shape: {tuple(eeg.shape)}")
        encoded = self.eeg_encoder(eeg.reshape(batch * tokens, 1, channels, samples))
        encoded = encoded.reshape(batch, tokens, SETTINGS.eeg_embedding_dim)
        return self.transformer_encoder(self.position(encoded))

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(length, device=device)

    def forward(self, eeg: torch.Tensor, target_latents: torch.Tensor) -> torch.Tensor:
        """Run the released zero-start autoregressive decoder.

        The released trainer feeds each decoder hidden state back as the next
        token and uses the target only for the MSE loss.
        """
        if target_latents.shape[1:] != (
            SETTINGS.video_frames,
            SETTINGS.latent_channels,
            SETTINGS.latent_height,
            SETTINGS.latent_width,
        ):
            raise ValueError(f"Unexpected latent target shape: {tuple(target_latents.shape)}")
        memory = self.encode_eeg(eeg)
        decoder_tokens = torch.zeros(
            eeg.shape[0], 1, SETTINGS.eeg_embedding_dim,
            device=eeg.device, dtype=memory.dtype,
        )
        for _ in range(SETTINGS.video_frames):
            decoded = self.transformer_decoder(
                decoder_tokens,
                memory,
                tgt_mask=self.causal_mask(decoder_tokens.shape[1], eeg.device),
            )
            decoder_tokens = torch.cat((decoder_tokens, decoded[:, -1:]), dim=1)
        # Each predicted frame must depend on the EEG memory. The released
        # script accidentally projected [zero, h1, ..., h5], making frame 1 a
        # learned constant and discarding h6. Project [h1, ..., h6] instead.
        outputs = self.latent_predictor(decoder_tokens[:, 1:])
        return outputs.reshape(
            target_latents.shape[0], SETTINGS.video_frames,
            SETTINGS.latent_channels, SETTINGS.latent_height,
            SETTINGS.latent_width,
        )

    @torch.no_grad()
    def predict(self, eeg: torch.Tensor) -> torch.Tensor:
        memory = self.encode_eeg(eeg)
        decoder_tokens = torch.zeros(
            eeg.shape[0], 1, SETTINGS.eeg_embedding_dim,
            device=eeg.device, dtype=memory.dtype,
        )
        for _ in range(SETTINGS.video_frames):
            decoded = self.transformer_decoder(
                decoder_tokens,
                memory,
                tgt_mask=self.causal_mask(decoder_tokens.shape[1], eeg.device),
            )
            decoder_tokens = torch.cat((decoder_tokens, decoded[:, -1:]), dim=1)
        outputs = self.latent_predictor(decoder_tokens[:, 1:])
        return outputs.reshape(
            eeg.shape[0],
            SETTINGS.video_frames,
            SETTINGS.latent_channels,
            SETTINGS.latent_height,
            SETTINGS.latent_width,
        )


class SemanticPredictor(nn.Module):
    """Released 310-to-CLIP MLP with four ReLU hidden layers."""

    def __init__(self) -> None:
        super().__init__()
        hidden = SETTINGS.semantic_hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(SETTINGS.eeg_channels * 5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 77 * 768),
        )

    def forward(self, de_features: torch.Tensor) -> torch.Tensor:
        return self.mlp(de_features).reshape(-1, 77, 768)


class ShallowFeatureEncoder(nn.Module):
    """ShallowNet branch used as the global/local GLMNet encoder."""

    def __init__(self, channels: int, samples: int, embedding_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.Dropout(0.5),
        )
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, 1, channels, samples)).numel()
        self.embedding = nn.Linear(flat_dim, embedding_dim)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.features(eeg).flatten(1))


class DynamicGLMNet(nn.Module):
    """GLMNet: whole-head global branch plus occipital local branch."""

    def __init__(self) -> None:
        super().__init__()
        samples = SETTINGS.dynamic_input_samples
        dim = SETTINGS.glm_embedding_dim
        self.global_encoder = ShallowFeatureEncoder(SETTINGS.eeg_channels, samples, dim)
        self.local_encoder = ShallowFeatureEncoder(len(OCCIPITAL_CHANNELS), samples, dim)
        self.classifier = nn.Linear(dim * 2, 2)

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.shape[1:] != (
            1,
            SETTINGS.eeg_channels,
            SETTINGS.dynamic_input_samples,
        ):
            raise ValueError(f"Unexpected GLMNet EEG shape: {tuple(eeg.shape)}")
        global_features = self.global_encoder(eeg)
        local_features = self.local_encoder(eeg[:, :, OCCIPITAL_CHANNELS, :])
        return self.classifier(torch.cat((global_features, local_features), dim=1))
