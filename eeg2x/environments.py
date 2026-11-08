"""环境注册表：声明每个框架/基线需要哪些包，并检查当前解释器。

仓库不创建环境，也不写死解释器路径；本模块只回答两个问题：

* 当前解释器能满足哪些框架/编码器；
* 不满足时，缺的是哪些包、应该换用 ``environments/`` 下的哪一份说明。

检查只做 ``importlib.util.find_spec`` 与 ``importlib.metadata.version``，不导入
torch 等重包，因此可以在一台几乎空白的解释器里运行 ``benchmark.py env``。
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Dict, Iterable, List, Mapping, Tuple


@dataclass(frozen=True)
class EnvironmentSpec:
    name: str
    description: str
    requirements: Tuple[str, ...]
    file: str
    notes: str = ""


ENVIRONMENTS: Dict[str, EnvironmentSpec] = {
    "image": EnvironmentSpec(
        "image", "Only-CLIP 图像框架（atm_s / reve / enigma_cnn）与 ENIGMA",
        ("torch", "diffusers", "transformers", "timm", "mne", "h5py",
         "sklearn", "numpy", "PIL", "einops"),
        "environments/image.txt",
        notes="已知可用：/datas/linshishangchuan/11-3-image-env/eeg2x-atm"
              "（含 braindecode 与 rotary，五个编码器共用）；"
              "AVDE 历史环境：/datas/linshishangchuan/avde"),
    "image-braindecode": EnvironmentSpec(
        "image-braindecode", "image 环境加 LaBraM（braindecode==1.3.2）",
        ("torch", "diffusers", "transformers", "timm", "mne", "h5py",
         "sklearn", "numpy", "PIL", "einops", "braindecode"),
        "environments/image-braindecode.txt",
        notes="已知可用：/datas/linshishangchuan/11-3-image-env/eeg2x-atm"),
    "image-luna": EnvironmentSpec(
        "image-luna", "image 环境加 LUNA（rotary-embedding-torch==0.8.9）",
        ("torch", "diffusers", "transformers", "timm", "mne", "h5py",
         "sklearn", "numpy", "PIL", "einops", "rotary_embedding_torch"),
        "environments/image-luna.txt",
        notes="已知可用：/datas/linshishangchuan/11-3-image-env/eeg2x-atm"),
    "video": EnvironmentSpec(
        "video", "EEG-to-Video 框架与 EEGVid / EEG2Video Orig",
        ("torch", "diffusers", "transformers", "torch_geometric",
         "rotary_embedding_torch", "mne", "h5py", "sklearn", "cv2",
         "pandas", "einops", "PIL"),
        "environments/video.txt",
        notes="已知可用：/datas/linshishangchuan/eeg2video"),
    "text": EnvironmentSpec(
        "text", "GLIM 文本框架与 GLIM Orig",
        ("torch", "lightning", "transformers", "timm", "sentence_transformers",
         "torchmetrics", "nltk", "rouge_score", "bert_score", "h5py", "mne",
         "pandas"),
        "environments/text.txt",
        notes="已知可用：/datas/linshishangchuan/glim_2"),
    "retrieval": EnvironmentSpec(
        "retrieval", "EEGBench 检索（NeuralBench 栈，Python 3.12）",
        ("torch", "lightning", "transformers", "neuralbench", "neuraltrain",
         "neuralset", "neuralfetch", "h5py", "yaml", "sklearn", "mne",
         "pandas"),
        "environments/retrieval.txt",
        notes="已知可用：/datas/linshishangchuan/11-3-jiansuo/eegbench-retrieval；"
              "推理还需要 EEGBench 的数据缓存"),
}


def _candidate_names(model: str, encoder: str | None) -> Tuple[str, ...]:
    if model == "only_clip":
        if encoder == "labram":
            return ("image-braindecode",)
        if encoder == "luna":
            return ("image-luna",)
        return ("image",)
    if model in ("eeg2video", "eeg2video_orig", "eegvid_orig"):
        return ("video",)
    if model in ("glim", "glim_orig", "eeg2text_orig", "rag_orig"):
        return ("text",)
    if model in ("enigma", "avde"):
        return ("image",)
    if model == "eegbench":
        return ("retrieval",)
    return ()


def missing_packages(spec: EnvironmentSpec) -> List[str]:
    return [name for name in spec.requirements
            if importlib.util.find_spec(name) is None]


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name.replace("_", "-"))
    except metadata.PackageNotFoundError:
        return None


def candidates(model: str, encoder: str | None = None) -> List[EnvironmentSpec]:
    return [ENVIRONMENTS[name] for name in _candidate_names(model, encoder)]


def require_environment(model: str, encoder: str | None = None) -> EnvironmentSpec | None:
    """当前解释器满足时返回所用的环境说明；否则抛出带缺失包与文件名的错误。

    未知模型返回 ``None``，由调用方自己报“未知模型”。
    """
    options = candidates(model, encoder)
    if not options:
        return None
    reports = []
    for spec in options:
        missing = missing_packages(spec)
        if not missing:
            return spec
        reports.append(f"{spec.name}: missing {missing} (see {spec.file})")
    raise RuntimeError(
        f"{model}"
        + (f"/{encoder}" if encoder else "")
        + " cannot run in this interpreter. "
        + " | ".join(reports)
        + f". Current interpreter: {sys.executable}")


def report() -> Dict[str, object]:
    """``benchmark.py env`` 的机器可读输出。"""
    names = sorted({name for spec in ENVIRONMENTS.values()
                    for name in spec.requirements})
    packages = {name: package_version(name) for name in names}
    envs = {}
    for key, spec in ENVIRONMENTS.items():
        missing = missing_packages(spec)
        envs[key] = {
            "description": spec.description,
            "file": spec.file,
            "missing": missing,
            "ok": not missing,
            "notes": spec.notes,
        }
    runnable = {}
    for model, encoder in (
            ("only_clip", None), ("only_clip", "labram"), ("only_clip", "luna"),
            ("eeg2video", None), ("glim", None), ("enigma", None),
            ("avde", None), ("eeg2video_orig", None), ("eegvid_orig", None),
            ("glim_orig", None), ("eeg2text_orig", None), ("rag_orig", None),
            ("eegbench", None)):
        key = model if encoder is None else f"{model}:{encoder}"
        options = candidates(model, encoder)
        runnable[key] = {
            "environments": [spec.name for spec in options],
            "ok": any(not missing_packages(spec) for spec in options),
            "missing": [missing_packages(spec) for spec in options],
        }
    return {
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "packages": packages,
        "environments": envs,
        "runnable": runnable,
    }


__all__ = [
    "ENVIRONMENTS", "EnvironmentSpec", "candidates", "missing_packages",
    "package_version", "report", "require_environment",
]
