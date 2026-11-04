"""Pretrained LUNA-Base adapter for the EEG2X encoder contract."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from eeg2x.protocol import sha256_file
from .official import LunaBackbone, LunaTemporalPool
from .input import (
    LUNA_INPUT_SAMPLES,
    LUNA_PAD_RIGHT,
    LUNA_PATCH_SIZE,
    LUNA_SAMPLING_RATE,
    LUNA_VALID_SAMPLES,
)


LUNA_REPO_ID = "PulpBio/LUNA"
# The current official PulpBio/LUNA repository publishes the Base checkpoint
# at its root.  BioFoundation's older documentation still shows a Base/
# directory, which is not present in the Hub snapshot.
LUNA_WEIGHT_FILENAME = "LUNA_base.safetensors"
LUNA_CACHE_DIR = "/datas/hf-cache"
LUNA_FEATURE_DIM = 256


def _resolve_luna_weights(
    repo_id: str,
    filename: str,
    *,
    cache_dir: str,
    revision: Optional[str],
    local_files_only: bool,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "LUNA requires huggingface-hub; use environment-atm-luna.yml"
        ) from exc
    try:
        return Path(hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )).absolute()
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve official LUNA weights {repo_id!r}/{filename!r} "
            f"from cache {cache_dir!r}. Run scripts/download_luna_weights.py; "
            "random initialization is forbidden."
        ) from exc


def _load_luna_state(path: Path) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "LUNA requires safetensors; use environment-atm-luna.yml"
        ) from exc
    return dict(load_file(str(path), device="cpu"))


def _checkpoint_key(key: str) -> str:
    value = key
    while value.startswith("model.") or value.startswith("module."):
        value = value.split(".", 1)[1]
    return value


def _snapshot_revision(path: Path) -> Optional[str]:
    parts = path.parts
    try:
        index = parts.index("snapshots")
    except ValueError:
        return None
    return parts[index + 1] if index + 1 < len(parts) else None


def _channel_locations(channel_names: List[str]) -> torch.Tensor:
    try:
        import mne
    except ImportError as exc:
        raise RuntimeError("LUNA electrode locations require mne") from exc
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=LUNA_SAMPLING_RATE,
        ch_types=["eeg"] * len(channel_names),
    )
    try:
        info.set_montage(
            mne.channels.make_standard_montage("standard_1005"),
            match_case=False,
            match_alias={"cb1": "POO7", "cb2": "POO8"},
            on_missing="raise",
        )
    except Exception as exc:
        raise ValueError(
            "LUNA could not map all configured channels to standard_1005 coordinates"
        ) from exc
    positions = info.get_montage().get_positions()["ch_pos"]
    missing = [name for name in channel_names if name not in positions]
    if missing:
        raise ValueError("LUNA has no coordinates for channels: " + ", ".join(missing))
    value = torch.from_numpy(np.asarray(
        [positions[name] for name in channel_names], dtype=np.float32))
    if value.shape != (len(channel_names), 3) or not torch.isfinite(value).all():
        raise ValueError("LUNA channel coordinates are incomplete or non-finite")
    return value


class LunaEncoder(nn.Module):
    """Official pretrained LUNA-Base plus its fine-tuning attention pool."""

    feature_dim = LUNA_FEATURE_DIM

    def __init__(
        self,
        n_chans: int,
        n_samples: int,
        channel_names: Optional[List[str]] = None,
        sampling_rate: float = LUNA_SAMPLING_RATE,
        repo_id: str = LUNA_REPO_ID,
        weight_filename: str = LUNA_WEIGHT_FILENAME,
        cache_dir: str = LUNA_CACHE_DIR,
        revision: Optional[str] = None,
        local_files_only: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
        if float(sampling_rate) != float(LUNA_SAMPLING_RATE):
            raise ValueError(
                f"LUNA requires {LUNA_SAMPLING_RATE} Hz input metadata, "
                f"got sampling_rate={sampling_rate}"
            )
        if n_samples != LUNA_INPUT_SAMPLES:
            raise ValueError(
                "LUNA requires a one-second 256 Hz window right-padded to "
                f"{LUNA_INPUT_SAMPLES} samples, got n_samples={n_samples}"
            )
        if channel_names is None:
            raise ValueError("LunaEncoder requires ordered channel_names")
        if len(channel_names) != n_chans:
            raise ValueError(
                f"LUNA received {len(channel_names)} channel names for {n_chans} channels")
        if len(set(channel_names)) != len(channel_names):
            raise ValueError("LUNA channel_names must be unique")

        cache_path = Path(cache_dir).expanduser().absolute()
        weight_path = _resolve_luna_weights(
            repo_id,
            weight_filename,
            cache_dir=str(cache_path),
            revision=revision,
            local_files_only=bool(local_files_only),
        )
        state = {
            _checkpoint_key(key): value
            for key, value in _load_luna_state(weight_path).items()
        }
        backbone = LunaBackbone(
            patch_size=LUNA_PATCH_SIZE,
            num_queries=4,
            embed_dim=64,
            depth=8,
            num_heads=2,
            drop_path=0.1,
        )
        expected = set(backbone.state_dict())
        encoder_state = {key: value for key, value in state.items() if key in expected}
        missing = sorted(expected - set(encoder_state))
        unexpected = sorted(
            key for key in state
            if key not in expected and not key.startswith(
                ("decoder_head.", "channel_emb.", "classifier."))
        )
        if missing or unexpected:
            raise RuntimeError(
                "Official LUNA checkpoint does not exactly cover the encoder: "
                f"missing={missing}, unexpected={unexpected}"
            )
        try:
            backbone.load_state_dict(encoder_state, strict=True)
        except Exception as exc:
            raise RuntimeError(
                "Official LUNA checkpoint tensors do not match LUNA-Base"
            ) from exc

        self.backbone = backbone
        self.pool = LunaTemporalPool(embed_dim=LUNA_FEATURE_DIM, num_heads=2)
        self.register_buffer(
            "channel_locations", _channel_locations(list(channel_names)).unsqueeze(0))
        self.channel_names = list(channel_names)
        self.sampling_rate = float(sampling_rate)
        #: 输入契约里**有效 EEG** 的长度：后 ``LUNA_PAD_RIGHT`` 个采样是结构性零填充，
        #: 不是信号。下游工具（如噪声对照 model/noise_control.py）据此只替换信号段，
        #: 否则 forward 的零填充断言会拒绝整段被噪声覆盖的输入。
        self.valid_samples = LUNA_VALID_SAMPLES
        self.repo_id = repo_id
        self.weight_filename = weight_filename
        self.weight_path = str(weight_path)
        self.weight_sha256 = sha256_file(weight_path)
        self.cache_dir = str(cache_path)
        self.revision = revision
        self.resolved_revision = _snapshot_revision(weight_path)
        self.local_files_only = bool(local_files_only)
        self.freeze = bool(freeze)
        self.pretrained_loaded = True
        if self.freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> "LunaEncoder":
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        if eeg.ndim != 3 or eeg.shape[1:] != (
                len(self.channel_names), LUNA_INPUT_SAMPLES):
            raise ValueError(
                f"LUNA expects [batch, channels, {LUNA_INPUT_SAMPLES}], "
                f"got {tuple(eeg.shape)}")
        if torch.count_nonzero(eeg[..., -LUNA_PAD_RIGHT:]).item():
            raise ValueError("LUNA input must contain the declared 24-sample zero pad")
        locations = self.channel_locations.expand(eeg.shape[0], -1, -1)
        latent = self.backbone(eeg, locations)
        if latent.shape != (eeg.shape[0], LUNA_INPUT_SAMPLES // LUNA_PATCH_SIZE,
                            LUNA_FEATURE_DIM):
            raise RuntimeError(f"LUNA backbone returned invalid shape {tuple(latent.shape)}")
        pooled = self.pool(latent)
        if pooled.shape != (eeg.shape[0], LUNA_FEATURE_DIM):
            raise RuntimeError(f"LUNA pooling returned invalid shape {tuple(pooled.shape)}")
        if not torch.isfinite(pooled).all():
            raise RuntimeError("LUNA produced non-finite pooled features")
        return pooled

    def pretrained_metadata(self) -> Dict[str, Any]:
        return {
            "pretrained_loaded": self.pretrained_loaded,
            "pretrained_model_id": self.repo_id,
            "weight_filename": self.weight_filename,
            "weight_path": self.weight_path,
            "weight_sha256": self.weight_sha256,
            "requested_revision": self.revision,
            "resolved_model_revision": self.resolved_revision,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "freeze": self.freeze,
            "channel_names": list(self.channel_names),
            "sampling_rate": self.sampling_rate,
            "valid_samples": LUNA_VALID_SAMPLES,
            "input_samples": LUNA_INPUT_SAMPLES,
            "pad_right": LUNA_PAD_RIGHT,
            "temporal_pool_pretrained": False,
        }


__all__ = [
    "LunaEncoder",
    "LUNA_CACHE_DIR",
    "LUNA_FEATURE_DIM",
    "LUNA_REPO_ID",
    "LUNA_WEIGHT_FILENAME",
]
