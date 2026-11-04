"""Pretrained REVE adapter for the EEG2X encoder contract."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from .base import EncoderSpec


REVE_MODEL_ID = "brain-bzh/reve-base"
REVE_POSITION_MODEL_ID = "brain-bzh/reve-positions"
REVE_CACHE_DIR = "/datas/hf-cache"
REVE_SAMPLING_RATE = 200
REVE_N_SAMPLES = 200
REVE_FEATURE_DIM = 512


def _load_pretrained_model(model_id: str, **kwargs: Any) -> nn.Module:
    """Load one official REVE component without a random-init fallback."""
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "REVE requires transformers==4.56.2; use environment-atm-reve.yml"
        ) from exc
    try:
        return AutoModel.from_pretrained(model_id, trust_remote_code=True, **kwargs)
    except Exception as exc:
        cache_dir = kwargs.get("cache_dir", REVE_CACHE_DIR)
        raise RuntimeError(
            f"Could not load pretrained REVE component {model_id!r} from "
            f"Hugging Face/cache {cache_dir!r}. Accept the gated model terms, "
            "authenticate with Hugging Face, and run scripts/download_reve_weights.py. "
            "Random initialization is forbidden."
        ) from exc


def _commit_hash(model: nn.Module) -> Optional[str]:
    config = getattr(model, "config", None)
    value = getattr(config, "_commit_hash", None)
    return str(value) if value else None


class ReveEncoder(nn.Module):
    """Official pretrained REVE-base plus its official attention pooling.

    The adapter deliberately performs no resampling. EEG2X preprocessing profiles
    are the single authority for the 200 Hz input contract.
    """

    feature_dim = REVE_FEATURE_DIM

    def __init__(
        self,
        n_chans: int,
        n_samples: int,
        channel_names: Optional[List[str]] = None,
        sampling_rate: float = REVE_SAMPLING_RATE,
        pretrained_model_id: str = REVE_MODEL_ID,
        position_model_id: str = REVE_POSITION_MODEL_ID,
        cache_dir: str = REVE_CACHE_DIR,
        revision: Optional[str] = None,
        position_revision: Optional[str] = None,
        local_files_only: bool = False,
        freeze: bool = False,
    ):
        super().__init__()
        if float(sampling_rate) != float(REVE_SAMPLING_RATE):
            raise ValueError(
                f"REVE requires {REVE_SAMPLING_RATE} Hz input, "
                f"got sampling_rate={sampling_rate}"
            )
        if n_samples != REVE_N_SAMPLES:
            raise ValueError(
                f"REVE requires one second at 200 Hz ({REVE_N_SAMPLES} samples), "
                f"got n_samples={n_samples}"
            )
        if channel_names is None:
            raise ValueError(
                "ReveEncoder requires ordered channel_names for its electrode positions"
            )
        if len(channel_names) != n_chans:
            raise ValueError(
                f"REVE received {len(channel_names)} channel names for {n_chans} channels"
            )
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("REVE channel_names must be unique")
        cache_path = Path(cache_dir).expanduser().absolute()
        common_load_kwargs: Dict[str, Any] = {
            "cache_dir": str(cache_path),
            "local_files_only": bool(local_files_only),
        }
        position_load_kwargs = dict(common_load_kwargs)
        if position_revision is not None:
            position_load_kwargs["revision"] = position_revision
        position_bank = _load_pretrained_model(
            position_model_id, **position_load_kwargs)
        available = None
        if hasattr(position_bank, "get_all_positions"):
            available = set(position_bank.get_all_positions())
        elif hasattr(position_bank, "mapping"):
            available = set(position_bank.mapping)
        if available is not None:
            missing = [name for name in channel_names if name not in available]
            if missing:
                raise ValueError(
                    "REVE position bank has no coordinates for channels: "
                    + ", ".join(missing)
                )
        with torch.no_grad():
            positions = position_bank(list(channel_names))
        if not isinstance(positions, torch.Tensor) or positions.shape != (n_chans, 3):
            shape = tuple(positions.shape) if isinstance(positions, torch.Tensor) else None
            raise ValueError(
                f"REVE position bank returned {shape}, expected ({n_chans}, 3); "
                "unknown channels must not be silently dropped"
            )
        if not torch.isfinite(positions).all():
            raise ValueError("REVE position bank returned non-finite coordinates")

        model_load_kwargs = dict(common_load_kwargs)
        if revision is not None:
            model_load_kwargs["revision"] = revision
        model = _load_pretrained_model(pretrained_model_id, **model_load_kwargs)
        if not callable(getattr(model, "attention_pooling", None)):
            raise TypeError(
                f"Pretrained model {pretrained_model_id!r} has no attention_pooling()"
            )
        feature_dim = int(getattr(model, "embed_dim", REVE_FEATURE_DIM))
        if feature_dim != REVE_FEATURE_DIM:
            raise ValueError(
                f"ATM-REVE expects REVE-base feature_dim={REVE_FEATURE_DIM}, "
                f"got {feature_dim}"
            )

        self.model = model
        self.register_buffer("positions", positions.detach().float().clone())
        self.channel_names = list(channel_names)
        self.sampling_rate = float(sampling_rate)
        self.pretrained_model_id = pretrained_model_id
        self.position_model_id = position_model_id
        self.cache_dir = str(cache_path)
        self.revision = revision
        self.position_revision = position_revision
        self.resolved_model_revision = _commit_hash(model)
        self.resolved_position_revision = _commit_hash(position_bank)
        self.local_files_only = bool(local_files_only)
        self.freeze = bool(freeze)
        self.pretrained_loaded = True
        if self.freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, mode: bool = True) -> "ReveEncoder":
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1:] != (
                len(self.channel_names), REVE_N_SAMPLES):
            raise ValueError(
                "REVE expects [batch, channels, 200] matching channel_names, "
                f"got {tuple(eeg.shape)}"
            )
        positions = self.positions.to(device=eeg.device).unsqueeze(0).expand(
            eeg.shape[0], -1, -1)
        encoded = self.model(eeg, positions)
        if not isinstance(encoded, torch.Tensor) or encoded.ndim != 4:
            shape = tuple(encoded.shape) if isinstance(encoded, torch.Tensor) else None
            raise RuntimeError(f"REVE backbone returned invalid shape {shape}")
        pooled = self.model.attention_pooling(encoded)
        if pooled.shape != (eeg.shape[0], self.feature_dim):
            raise RuntimeError(
                f"REVE attention pooling returned {tuple(pooled.shape)}, expected "
                f"({eeg.shape[0]}, {self.feature_dim})"
            )
        if not torch.isfinite(pooled).all():
            raise RuntimeError("REVE produced non-finite pooled features")
        return pooled

    def pretrained_metadata(self) -> Dict[str, Any]:
        return {
            "pretrained_loaded": self.pretrained_loaded,
            "pretrained_model_id": self.pretrained_model_id,
            "position_model_id": self.position_model_id,
            "requested_revision": self.revision,
            "requested_position_revision": self.position_revision,
            "resolved_model_revision": self.resolved_model_revision,
            "resolved_position_revision": self.resolved_position_revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "freeze": self.freeze,
            "channel_names": list(self.channel_names),
            "sampling_rate": self.sampling_rate,
        }


__all__ = [
    "ReveEncoder",
    "REVE_MODEL_ID",
    "REVE_POSITION_MODEL_ID",
    "REVE_CACHE_DIR",
    "REVE_SAMPLING_RATE",
    "REVE_N_SAMPLES",
    "REVE_FEATURE_DIM",
]


# ---------------------------------------------------------------------------
# v3 注册契约
# ---------------------------------------------------------------------------

_CONFIG_KEYS = (
    "sampling_rate", "channel_names", "pretrained_model_id", "position_model_id",
    "cache_dir", "revision", "position_revision", "local_files_only", "freeze",
)


def _builder(*, n_chans: int, n_samples: int, config: dict) -> nn.Module:
    kwargs = {k: config[k] for k in _CONFIG_KEYS if k in config}
    return ReveEncoder(n_chans=n_chans, n_samples=n_samples, **kwargs)


SPEC = EncoderSpec(
    name="reve",
    builder=_builder,
    feature_dim=512,
    required_packages=("mne",),
    needs_channel_names=True,
    description="Pretrained REVE-base with official electrode positions and "
                "attention pooling; requires 200 Hz and channel_names.",
)
