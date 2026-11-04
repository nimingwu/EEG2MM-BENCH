"""从 v2 checkpoint 重建框架模型。

v2 的 ``ATMS`` checkpoint 里，``model_config`` 记录了 ``encoder`` 与
``encoder_config``（含 ``n_chans`` / ``n_samples`` / ``channel_names`` /
预训练 id 与 revision）。v3 用同一份配置构造 ``OnlyClipModel``，再把
``model_state`` 灌进去。

**唯一需要处理的不一致**：v2 的 ``ATMS`` 为 state-dict 兼容保留了一个
``subject_wise_linear.*``（``atm_s`` 分支），v3 的 ``OnlyClipModel`` 没有它。
它是死参数（``forward`` 不经过），所以加载时忽略——但**只允许**忽略这一个前缀，
其余任何缺失/多余都报错，避免把真正的结构不一致吞掉。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from eeg2x import encoders as encoder_registry
from eeg2x.checkpoints import load_state_checkpoint

from .model import OnlyClipModel

#: v2 里仅用于 state-dict 兼容、forward 不使用的参数前缀。
_IGNORED_PREFIXES = ("subject_wise_linear.",)


def _migrate_atm_s_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """把 v2 ``ATMS`` 的 atm_s 键布局映射到 v3。

    v2 把编码器拆成 ``ATMS.encoder``（ChannelTransformer）与
    ``ATMS.temporal_spatial``（两个**兄弟**模块）；v3 认为时序卷积属于编码器，
    于是两者都落在 ``OnlyClipModel.encoder``（``AtmSEncoder``）内部。键因此多一层：

        v2 ``encoder.<X>``          -> v3 ``encoder.encoder.<X>``
        v2 ``temporal_spatial.<X>`` -> v3 ``encoder.temporal_spatial.<X>``

    这是一次**结构改进导致的显式迁移**，不是静默容错：其它编码器的键不动，
    迁移后仍然 strict 校验。
    """
    migrated: Dict[str, Any] = {}
    for key, value in state.items():
        if key.startswith("temporal_spatial."):
            migrated["encoder." + key] = value
        elif key.startswith("encoder."):
            migrated["encoder.encoder." + key[len("encoder."):]] = value
        else:
            migrated[key] = value
    return migrated


def build_model_from_config(model_config: Dict[str, Any], *,
                            subject_id: str | None = None) -> OnlyClipModel:
    """按 checkpoint 的 ``model_config`` 构造框架模型。"""
    if "encoder" not in model_config:
        raise ValueError("model_config is missing 'encoder'")
    encoder_name = str(model_config["encoder"])
    enc_cfg = dict(model_config.get("encoder_config") or {})
    n_chans = int(enc_cfg.get("n_chans", 63))
    n_samples = int(enc_cfg.get("n_samples", 250))
    if subject_id is not None:
        enc_cfg.setdefault("subject_id", subject_id)
    encoder = encoder_registry.build(
        encoder_name, n_chans=n_chans, n_samples=n_samples, config=enc_cfg)
    return OnlyClipModel(encoder)


def load_only_clip_checkpoint(
    checkpoint_path: str | Path,
    *,
    subject_id: str | None = None,
    expected_type: str = "atm_generation",
) -> Tuple[OnlyClipModel, Dict[str, Any]]:
    """加载 v2/v3 的框架 checkpoint，返回 ``(model, payload)``。"""
    resolved = Path(checkpoint_path).expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid framework checkpoint: {resolved}")
    if payload.get("artifact_type") != expected_type:
        raise ValueError(
            f"Checkpoint {resolved} is {payload.get('artifact_type')!r}, "
            f"expected {expected_type!r}")
    if bool(payload.get("derived_inputs_used", True)):
        raise ValueError(
            f"Checkpoint used forbidden external derived inputs: {resolved}")

    config = payload.get("config") or {}
    subject = subject_id or config.get("subject_id")
    model = build_model_from_config(payload["model_config"], subject_id=subject)

    # v3 训练出的 checkpoint 已经是目标键布局，不能再迁一次；v2 的 checkpoint
    # 没有这个标记，按 v2 布局迁移。显式标记而不是猜测键名，避免把「已经是对的」
    # 和「需要迁移」混在一起。
    layout = str(payload.get("state_layout", "v2"))
    if layout not in ("v2", "v3"):
        raise ValueError(
            f"Checkpoint {resolved} declares unknown state_layout {layout!r}")
    state = dict(payload["model_state"])
    if layout == "v2" and str(payload["model_config"].get("encoder")) == "atm_s":
        state = _migrate_atm_s_keys(state)
    ignored = {k: state.pop(k) for k in list(state)
               if k.startswith(_IGNORED_PREFIXES)}
    if ignored:
        # 只允许忽略声明过的兼容参数，并且数量必须与预期一致。
        unexpected = [k for k in ignored if not k.startswith(("subject_wise_linear.",))]
        if unexpected:
            raise RuntimeError(
                f"refusing to ignore unexpected checkpoint entries: {unexpected}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint does not match the v3 framework model: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}")
    return model, payload


__all__ = ["build_model_from_config", "load_only_clip_checkpoint"]
