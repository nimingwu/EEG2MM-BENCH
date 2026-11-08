"""LaBraM 编码器（Jiang et al., ICLR 2024）：预训练权重 + 通道名映射包装。

模型代码逐字来自 EEG2X-Benchmark-v2-test/model/encoders.py（``_adapt_pretrained_dimensions``
到 ``LabramEncoder``），数学部分未改；这里只补了模块头与 v3 的 ``EncoderSpec``。

**braindecode 版本属于复现性的一部分**：依赖 ``braindecode==1.3.2``。
1.6.1 / 1.7.0 / 1.8.1 的 ``Labram`` 缺少 ``fc_norm`` 与
``patch_embed.segment_patch``，加载 v2 checkpoint 会 key mismatch。
该版本号取自 HF 快照 config.json 的 ``braindecode_version`` 字段，
安装见 ``environments/image-braindecode.txt``。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg2x.protocol import sha256_file

from .base import EncoderSpec

log = logging.getLogger("eeg2x.encoders.labram")

LABRAM_PRETRAINED = "braindecode/labram-pretrained"
LABRAM_N_TIMES = 3000        # 预训练输入 3000 样本（@200Hz，15 patches）
LABRAM_N_CHANS = 128         # 预训练 128 通道模板
LABRAM_PATCH_SIZE = 200      # 200Hz x 1s
LABRAM_EMBED_DIM = 200       # LaBraM hidden dim
LABRAM_CACHE_DIR = "/datas/hf-cache"
LABRAM_SAMPLING_RATE = 200

# labram.json：数字 easycap 通道名 -> 标准名
_CH_MAP_PATH = Path(__file__).resolve().parent / "channel_mappings" / "labram.json"


def _adapt_pretrained_dimensions(
    model: nn.Module, n_times: int, n_chans: Optional[int] = None
) -> Tuple[int, int]:
    """适配预训练 LaBraM 到不同 n_times（对齐 neuraltrain 同款实现）。

    词窗 600 样本 = 3 patches（600 % 200 == 0，无需 pad/truncate）；
    temporal_embedding 从预训练 (1, 15+1, 200) 截断到 (1, 3+1, 200)，
    patch_embed 元数据同步。返回 (pad_right, truncate_right)。
    """
    for attr in ("patch_size", "n_times", "patch_embed"):
        if not hasattr(model, attr):
            raise AttributeError(
                f"braindecode Labram has no attribute {attr!r} — braindecode "
                f"internal layout changed?"
            )
    patch_embed_first = model.patch_embed[0]
    if not hasattr(patch_embed_first, "n_patchs"):
        raise AttributeError("model.patch_embed[0] has no 'n_patchs'")

    patch_size: int = model.patch_size
    pad_right = truncate_right = 0
    if n_times < patch_size:
        pad_right = patch_size - n_times
    elif n_times % patch_size != 0:
        truncate_right = n_times % patch_size
    effective_n_times = n_times + pad_right - truncate_right

    actual_n_patches = effective_n_times // patch_size
    pretrained_n_patches: int = patch_embed_first.n_patchs

    if actual_n_patches != pretrained_n_patches:
        log.info(f"Adapting pretrained LaBraM from n_times={model.n_times} "
                 f"({pretrained_n_patches} patches) to n_times={effective_n_times} "
                 f"({actual_n_patches} patches)")
        if hasattr(model, "temporal_embedding"):
            if actual_n_patches < pretrained_n_patches:
                n_keep = actual_n_patches + 1
                model.temporal_embedding = nn.Parameter(
                    model.temporal_embedding.data[:, :n_keep, :])
            else:
                old_emb = model.temporal_embedding.data
                cls_token = old_emb[:, :1, :]
                patch_tokens = old_emb[:, 1:, :].permute(0, 2, 1)
                patch_tokens = F.interpolate(
                    patch_tokens, size=actual_n_patches, mode="linear",
                    align_corners=False).permute(0, 2, 1)
                model.temporal_embedding = nn.Parameter(
                    torch.cat([cls_token, patch_tokens], dim=1))
        model._n_times = effective_n_times
        model.n_path = actual_n_patches
        patch_embed_first.n_patchs = actual_n_patches
        patch_embed_first.n_times = effective_n_times

    # Braindecode 1.3.x stores the channel count both on the model and on the
    # segment patcher.  The pretrained checkpoint must first be instantiated
    # with its native 128-channel position table so that every weight loads,
    # then these two pieces of reshape metadata must match the real EEG input.
    if n_chans is not None:
        if n_chans <= 0:
            raise ValueError(f"LaBraM requires n_chans > 0, got {n_chans}")
        model._n_chans = int(n_chans)
        patch_embed_first.n_chans = int(n_chans)

    return pad_right, truncate_right


def _load_channel_mapping() -> Dict[str, str]:
    """labram.json：数字 easycap 通道名 → LABRAM_CHANNEL_ORDER 标准名。"""
    with open(_CH_MAP_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_labram_snapshot(
    repo_id: str,
    *,
    cache_dir: str,
    revision: Optional[str],
    local_files_only: bool,
) -> Tuple[Path, str, Path]:
    """Resolve a complete LaBraM Hub snapshot without making a network call.

    Passing the resolved directory, instead of the Hub repo id, to
    ``Labram.from_pretrained`` makes offline execution an enforceable property.
    The returned weight path is also used for checkpoint provenance.
    """
    if not local_files_only:
        raise ValueError(
            "LaBraM loading is strictly offline for this benchmark; "
            "local_files_only must be True"
        )
    repo_parts = repo_id.split("/") if isinstance(repo_id, str) else []
    if (len(repo_parts) != 2 or any(
            not part or part in {".", ".."} or "\\" in part
            for part in repo_parts)):
        raise ValueError(f"Invalid LaBraM Hugging Face repo id: {repo_id!r}")

    cache_path = Path(cache_dir).expanduser().absolute()
    repo_path = cache_path / f"models--{repo_id.replace('/', '--')}"
    requested = revision or "main"
    requested_path = Path(requested)
    if requested_path.is_absolute() or any(
            part in {"", ".", ".."} for part in requested_path.parts):
        raise ValueError(f"Invalid LaBraM revision: {requested!r}")

    # A caller may pin the immutable snapshot commit directly.  Otherwise use
    # the corresponding local ref (including refs with slash-separated names).
    direct_snapshot = repo_path / "snapshots" / requested
    if direct_snapshot.is_dir():
        resolved_revision = requested
    else:
        ref_path = repo_path / "refs" / requested
        if not ref_path.is_file():
            raise FileNotFoundError(
                f"LaBraM revision {requested!r} is not present in local cache "
                f"{repo_path}; network fallback is disabled"
            )
        resolved_revision = ref_path.read_text(encoding="utf-8").strip()
        resolved_path = Path(resolved_revision)
        if (not resolved_revision or resolved_path.is_absolute() or any(
                part in {"", ".", ".."} for part in resolved_path.parts)):
            raise RuntimeError(
                f"Invalid revision stored in LaBraM cache ref {ref_path}"
            )

    snapshot_path = repo_path / "snapshots" / resolved_revision
    config_path = snapshot_path / "config.json"
    safetensors_path = snapshot_path / "model.safetensors"
    pytorch_path = snapshot_path / "pytorch_model.bin"
    if not snapshot_path.is_dir() or not config_path.is_file():
        raise FileNotFoundError(
            f"Incomplete LaBraM snapshot {snapshot_path}: config.json is required; "
            "network fallback is disabled"
        )
    if safetensors_path.is_file():
        weight_path = safetensors_path
    elif pytorch_path.is_file():
        weight_path = pytorch_path
    else:
        raise FileNotFoundError(
            f"Incomplete LaBraM snapshot {snapshot_path}: model.safetensors or "
            "pytorch_model.bin is required; network fallback is disabled"
        )
    return snapshot_path.absolute(), resolved_revision, weight_path.absolute()


def _labram_channel_authority(snapshot_path: Path) -> Dict[str, str]:
    """Read case-insensitive channel aliases from the cached model config."""
    config_path = snapshot_path / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read LaBraM channel metadata from {config_path}"
        ) from exc
    entries = config.get("chs_info")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"LaBraM config {config_path} has no non-empty chs_info")

    authority: Dict[str, str] = {}
    for entry in entries:
        name = entry.get("ch_name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"LaBraM config {config_path} has an invalid chs_info entry")
        canonical = name.strip()
        key = canonical.upper()
        if key in authority and authority[key] != canonical:
            raise ValueError(
                f"LaBraM config {config_path} has ambiguous channel {canonical!r}"
            )
        authority[key] = canonical
    return authority


def _normalize_labram_channel_names(
    channel_names: List[str], channel_authority: Dict[str, str]
) -> List[str]:
    """Map dataset channel labels to names declared by the cached checkpoint."""
    ch_map = _load_channel_mapping()
    normalized: List[str] = []
    unknown: List[str] = []
    for raw_name in channel_names:
        if not isinstance(raw_name, str) or not raw_name.strip():
            unknown.append(repr(raw_name))
            continue
        input_name = raw_name.strip()
        mapped_name = ch_map.get(input_name, input_name)
        canonical = channel_authority.get(mapped_name.upper())
        if canonical is None:
            unknown.append(input_name)
        else:
            normalized.append(canonical)
    if unknown:
        raise ValueError(
            "LaBraM checkpoint has no position embedding for channels: "
            + ", ".join(unknown)
        )
    if len(set(normalized)) != len(normalized):
        duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
        raise ValueError(
            "LaBraM channel mapping produced duplicate positions: "
            + ", ".join(duplicates)
        )
    return normalized


class LabramChannelWrapper(nn.Module):
    """Map dataset labels to cached LaBraM position names and adapt time."""

    def __init__(self, model: nn.Module, union_ch_names: List[str],
                 pad_right: int = 0, truncate_right: int = 0,
                 channel_authority: Optional[Dict[str, str]] = None):
        super().__init__()
        self.model = model
        self._pad_right = pad_right
        self._truncate_right = truncate_right
        if channel_authority is None:
            ch_map = _load_channel_mapping()
            self._labram_names = [ch_map.get(name, name) for name in union_ch_names]
        else:
            self._labram_names = _normalize_labram_channel_names(
                union_ch_names, channel_authority)
        # Labram.forward() in the snapshot-declared braindecode==1.3.2 API
        # accepts ``input_chans`` (position-table indices), not channel names.
        # Position 0 is reserved for the CLS token; EEG channel i in chs_info
        # therefore uses position i+1.  Include CLS so the selected position
        # embedding has the same token count as [CLS + all EEG patch tokens].
        authority_order = {
            canonical.upper(): index + 1
            for index, canonical in enumerate(channel_authority.values())
        } if channel_authority is not None else {}
        if authority_order:
            self._input_chans = [0] + [
                authority_order[name.upper()] for name in self._labram_names
            ]
        else:
            # Kept only for interface-compatible tests/legacy callers.  Formal
            # benchmark construction always supplies the checkpoint authority.
            self._input_chans = [0] + list(range(1, len(self._labram_names) + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._pad_right > 0:
            x = F.pad(x, (0, self._pad_right))
        elif self._truncate_right > 0:
            x = x[..., :x.shape[-1] - self._truncate_right]
        return self.model(x, input_chans=self._input_chans,
                          return_all_tokens=True)


class Mean(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim)


class LabramEncoder(nn.Module):
    """预训练 LaBraM 编码器（去 probe 部分）→ (B, LABRAM_EMBED_DIM)。

    forward: (B, C, T) → Labram(return_all_tokens) → (B, n_tokens, 200)
             → Mean(dim=1) → (B, 200)
    channel_names 必填（braindecode 按名选择位置嵌入）。通道名会以缓存
    config.json 的 chs_info 为权威进行大小写标准化；未知或重复位置会失败。
    """

    def __init__(
        self,
        n_chans: int = 60,
        n_samples: int = 600,
        channel_names: Optional[List[str]] = None,
        sampling_rate: float = LABRAM_SAMPLING_RATE,
        pretrained_model_id: str = LABRAM_PRETRAINED,
        cache_dir: str = LABRAM_CACHE_DIR,
        revision: Optional[str] = None,
        local_files_only: bool = True,
        freeze: bool = False,
    ):
        super().__init__()
        if float(sampling_rate) != float(LABRAM_SAMPLING_RATE):
            raise ValueError(
                f"LaBraM requires {LABRAM_SAMPLING_RATE} Hz input, "
                f"got sampling_rate={sampling_rate}"
            )
        if n_chans <= 0:
            raise ValueError(f"LaBraM requires n_chans > 0, got {n_chans}")
        if n_samples <= 0:
            raise ValueError(f"LaBraM requires n_samples > 0, got {n_samples}")
        if channel_names is None:
            raise ValueError(
                "LabramEncoder requires channel_names (easycap 数字通道名) — "
                "braindecode 按名选择位置嵌入;请由下游模型从 h5/checkpoint 提供"
            )
        if len(channel_names) != n_chans:
            raise ValueError(
                f"LaBraM received {len(channel_names)} channel names for "
                f"{n_chans} channels"
            )
        raw_keys = [
            name.strip().upper() if isinstance(name, str) else repr(name)
            for name in channel_names
        ]
        if len(set(raw_keys)) != len(raw_keys):
            raise ValueError("LaBraM channel_names must be unique")

        cache_path = Path(cache_dir).expanduser().absolute()
        snapshot_path, resolved_revision, weight_path = _resolve_labram_snapshot(
            pretrained_model_id,
            cache_dir=str(cache_path),
            revision=revision,
            local_files_only=bool(local_files_only),
        )
        channel_authority = _labram_channel_authority(snapshot_path)
        labram_channel_names = _normalize_labram_channel_names(
            list(channel_names), channel_authority)
        from braindecode.models import Labram

        # 1. 预训练权重（构造参数匹配预训练形状 n_times=3000/n_chans=128）
        model = Labram.from_pretrained(
            str(snapshot_path),
            n_times=LABRAM_N_TIMES, n_chans=LABRAM_N_CHANS, n_outputs=0,
        )
        # 2. 时间维适配到词窗
        pad_right, truncate_right = _adapt_pretrained_dimensions(
            model, n_samples, n_chans=n_chans)
        # 3. 通道名包装（h5 通道名 → LaBraM 名）
        model = LabramChannelWrapper(
            model, list(channel_names), pad_right, truncate_right,
            channel_authority=channel_authority)
        self.wrapped = model
        self.mean_pool = Mean(dim=1)
        self.feature_dim = LABRAM_EMBED_DIM
        self.n_chans = int(n_chans)
        self.n_samples = int(n_samples)
        self.channel_names = list(channel_names)
        self.labram_channel_names = labram_channel_names
        self.sampling_rate = float(sampling_rate)
        self.pretrained_model_id = pretrained_model_id
        self.cache_dir = str(cache_path)
        self.revision = revision
        self.resolved_model_revision = resolved_revision
        self.snapshot_path = str(snapshot_path)
        self.weight_path = str(weight_path)
        self.weight_sha256 = sha256_file(weight_path)
        self.local_files_only = bool(local_files_only)
        self.freeze = bool(freeze)
        self.pretrained_loaded = True
        self.pad_right = int(pad_right)
        self.truncate_right = int(truncate_right)
        if self.freeze:
            self.wrapped.requires_grad_(False)
            self.wrapped.eval()

    def train(self, mode: bool = True) -> "LabramEncoder":
        super().train(mode)
        if self.freeze:
            self.wrapped.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.n_chans, self.n_samples):
            raise ValueError(
                f"LaBraM expects [batch, {self.n_chans}, {self.n_samples}], "
                f"got {tuple(x.shape)}"
            )
        out = self.wrapped(x)          # (B, n_tokens, 200) — 含 CLS 的所有 token
        return self.mean_pool(out)     # (B, 200) — Mean(dim=1)

    def pretrained_metadata(self) -> Dict[str, Any]:
        return {
            "pretrained_loaded": self.pretrained_loaded,
            "pretrained_model_id": self.pretrained_model_id,
            "requested_revision": self.revision,
            "resolved_model_revision": self.resolved_model_revision,
            "cache_dir": self.cache_dir,
            "snapshot_path": self.snapshot_path,
            "weight_path": self.weight_path,
            "weight_sha256": self.weight_sha256,
            "local_files_only": self.local_files_only,
            "freeze": self.freeze,
            "channel_names": list(self.channel_names),
            "labram_channel_names": list(self.labram_channel_names),
            "sampling_rate": self.sampling_rate,
            "n_samples": self.n_samples,
            "patch_size": LABRAM_PATCH_SIZE,
            "pad_right": self.pad_right,
            "truncate_right": self.truncate_right,
        }




# ---------------------------------------------------------------------------
# v3 注册契约
# ---------------------------------------------------------------------------

#: 从 config 透传给构造函数的键（其余键由框架丢弃，避免误传）。
_CONFIG_KEYS = (
    "sampling_rate", "channel_names", "pretrained_model_id", "cache_dir",
    "revision", "local_files_only", "freeze",
)


def _builder(*, n_chans: int, n_samples: int, config: dict) -> nn.Module:
    kwargs = {k: config[k] for k in _CONFIG_KEYS if k in config}
    kwargs.setdefault("pretrained_model_id", LABRAM_PRETRAINED)
    kwargs.setdefault("cache_dir", LABRAM_CACHE_DIR)
    return LabramEncoder(n_chans=n_chans, n_samples=n_samples, **kwargs)


SPEC = EncoderSpec(
    name="labram",
    builder=_builder,
    feature_dim=LABRAM_EMBED_DIM,
    required_packages=("braindecode",),
    needs_channel_names=True,
    description="Pretrained LaBraM (ICLR 2024) minus probe; mean-pooled tokens. "
                "Requires channel_names and braindecode==1.3.2.",
)
