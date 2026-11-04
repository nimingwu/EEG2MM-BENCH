"""噪声对照（noise control）：冻结权重，把整段 EEG 换成 N(0,1)。

移植自 ``fuxian-GLIM/EEG2X-Benchmark-v2-GS``（该目录 ``NOISE_EXPERIMENT.md`` 是
完整说明；上游原型是 GLIM 的 ``eval_noise_input``，``data/datamodule.py:247-271``）。

它回答的问题不是"模型有多鲁棒"，而是一个**对照**：模型到底有没有在用 EEG？
把 EEG 整段换成标准正态噪声，文本 / 标签 / prompt / 图像目标全部不动，照常推理
与评测。若此时输出仍旧像样，说明模型靠的是先验而不是脑电信号。

**只用于重建（``image_generation``）**，检索任务不接线。

移植过来的三条约定（上游踩过坑，逐条照抄）：

1. **绝不覆盖干净条件的结果。** 噪声产物统一走 :func:`resolve_noise_dir` /
   :func:`resolve_noise_path`，落在 ``_noise`` 命名空间里；调用方照常传干净路径。
2. **条件元数据不放进预测 JSON。** 评测入口把 predictions JSON 的每个顶层 key 当
   样本 id，混进 ``{"noise": "gaussian"}`` 会被当成一条样本。条件信息写旁挂的
   ``*.meta.json``。
3. **种子固定且留痕。** 一个 ``torch.Generator`` 跑完全程 ⇒ 同 batch size、同数据
   顺序下重跑得到同一份噪声；种子随产物写进 meta，可复查。

上游文档 §8.7 的两个坑，这里按推荐方式规避：

* ``torch.empty`` / ``torch.empty_like`` 不接受 ``generator`` 参数——用
  ``torch.randn``。
* **CPU 上的 ``torch.Generator`` 不能直接给 CUDA 张量抽样**。本模块一律在 CPU 上
  采样再 ``.to(device)``，而不是建同设备生成器：这样同一个种子在不同设备上抽到的
  是**同一份**噪声，评测结果不会因为换了卡而变。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

#: ``none`` = 干净条件；``gaussian`` = 整段 EEG 换成 N(0,1)。
NOISE_CHOICES = ("none", "gaussian")

#: 噪声产物的目录名（干净条件与噪声条件在同一被试根目录下并列，结构上不可能互相覆盖）。
NOISE_SUBDIR = "noise_control"


def add_noise_arguments(parser: argparse.ArgumentParser, *, seed: int) -> None:
    """给重建推理入口加 ``--noise`` / ``--noise-seed``（默认关闭）。"""
    parser.add_argument(
        "--noise", choices=NOISE_CHOICES, default="none",
        help="噪声对照。'gaussian' 把整段 EEG 换成 N(0,1)（形状/dtype/设备不变），"
             "权重全部冻结，产物落到 noise_control/ 下，绝不覆盖干净结果")
    parser.add_argument(
        "--noise-seed", type=int, default=seed,
        help="噪声抽样种子，默认取 benchmark 全局种子，使这份噪声可复现且随产物留痕")


def noise_enabled(noise: str) -> bool:
    if noise not in NOISE_CHOICES:
        raise ValueError(f"Unknown noise condition {noise!r}; expected one of {NOISE_CHOICES}")
    return noise != "none"


def noise_generator(seed: int) -> torch.Generator:
    """CPU 生成器——见模块 docstring：跨设备抽到同一份噪声。"""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def inject_noise(tensor: torch.Tensor, generator: Optional[torch.Generator],
                 *, valid_samples: Optional[int] = None) -> torch.Tensor:
    """把 EEG 替换为 N(0,1)，保持形状、dtype 与设备。

    ``generator`` 为 ``None`` 时原样返回，因此调用方可以无条件调用本函数。
    每次调用消耗生成器状态，所以同一份噪声由 (种子, batch size, 数据顺序) 共同决定，
    与上游一致。

    ``valid_samples`` 是**契约允许被替换的信号长度**，用于那些输入里含有结构性
    零填充的编码器（目前只有 LUNA：280 = 256 有效 + 24 零填充）。后 24 个采样不是
    EEG 而是接口约定，替换掉会被 ``LunaEncoder.forward`` 的零填充断言直接拒绝；
    因此这里只对前 ``valid_samples`` 个采样抽噪声，尾部原样保持为 0。
    为 ``None`` 时整段替换（其余编码器没有这种契约）。
    """
    if generator is None:
        return tensor
    noise = torch.randn(tuple(tensor.shape), generator=generator,
                        dtype=torch.float32, device="cpu")
    if valid_samples is not None and valid_samples < tensor.shape[-1]:
        # 尾部是结构性零填充，不是信号：清零以维持编码器的输入契约。
        noise[..., valid_samples:] = 0.0
    return noise.to(device=tensor.device, dtype=tensor.dtype)


def resolve_noise_dir(root: str | Path, noise: str) -> Path:
    """噪声条件的产品目录；``none`` 时原样返回干净目录。"""
    root = Path(root)
    return root / NOISE_SUBDIR if noise_enabled(noise) else root


def resolve_noise_path(path: str | Path, noise: str) -> Path:
    """噪声条件的文件名；已含 ``noise`` 的路径不重复加后缀。"""
    path = Path(path)
    if noise_enabled(noise) and "noise" not in path.stem:
        path = path.with_name(f"{path.stem}_noise{path.suffix}")
    return path


def noise_meta(noise: str, noise_seed: int, **extra: Any) -> Dict[str, Any]:
    """条件溯源记录（写旁挂 meta 文件，见约定 2）。"""
    return {
        "noise": noise,
        "noise_seed": int(noise_seed),
        "weights": "frozen (eval() + no_grad())",
        "eeg": ("replaced by N(0,1) at float32 on CPU then moved to the target "
                "device" if noise_enabled(noise) else "untouched"),
        "task_scope": "image_generation only; retrieval paths are not instrumented",
        **extra,
    }


def write_noise_meta(path: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def validate_noise_outputs(clean: str | Path, noisy: str | Path) -> None:
    """端到端小样本验证：干净与噪声条件的输出不得逐条相同。

    上游文档 §8.5 要求的那条断言——用来证明注入真的生效了，而不是开关接错了地方。
    """
    clean_dir, noisy_dir = Path(clean), Path(noisy)
    if not clean_dir.is_dir() or not noisy_dir.is_dir():
        raise FileNotFoundError(
            f"Need both clean and noise image dirs to compare: {clean_dir}, {noisy_dir}")
    clean_names = sorted(path.name for path in clean_dir.glob("sample-*.png"))
    noisy_names = sorted(path.name for path in noisy_dir.glob("sample-*.png"))
    if not clean_names or clean_names != noisy_names:
        raise ValueError(
            f"Clean/noise image sets differ: {len(clean_names)} vs "
            f"{len(noisy_names)} file(s)")
    identical = [name for name in clean_names
                 if (clean_dir / name).read_bytes() == (noisy_dir / name).read_bytes()]
    if identical:
        raise AssertionError(
            f"{len(identical)}/{len(clean_names)} image(s) are byte-identical "
            f"between the clean and noise conditions (e.g. {identical[0]}); the "
            "noise injection did not take effect")
