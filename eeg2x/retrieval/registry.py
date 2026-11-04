"""检索任务的登记表：EEGBench 的数据集、模型与轨道布局。

检索是独立的一族任务，模型与数据格式都与生成任务不同：EEGBench 建在
NeuralBench 上，每条轨道是一个 ``predictions.npz`` 加 ``test_samples.jsonl``。
本模块只登记路径与名字，不导入 NeuralBench，因此可以在任何解释器里使用。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


EEGBENCH_ROOT = Path(os.environ.get("EEG2X_EEGBENCH_ROOT", "/EEG_bench_open"))
EEGBENCH_CODE = EEGBENCH_ROOT / "EEGBench"

#: dataset key -> (workspace 目录名, launch 任务名, 模态)
DATASETS = {
    "things_eeg": ("THINGS_EEG", "THINGS-EEG", "image"),
    "nod_eeg": ("NOD_EEG", "NOD_EEG", "image"),
    "grootswagers2019_5hz": ("Grootswagers2019", "Grootswagers2019", "image", "5hz"),
    "grootswagers2019_20hz": ("Grootswagers2019", "Grootswagers2019", "image", "20hz"),
    "grootswagers2025": ("Grootswagers2025", "Grootswagers2025", "image"),
    "zuco1": ("ZuCo1", "ZuCo1", "text"),
    "zuco2": ("ZuCo2", "ZuCo2", "text"),
    "chisco": ("CHISCO", "Chisco", "text"),
    "tmnred": ("TMNRED", "TMNRED", "text"),
    "cofett": ("COFETT", "COFETT", "text"),
    "ds005662": ("ds005662", "ds005662", "video"),
    "seed_dv": ("SEED-DV", "SEED-DV", "video"),
    "emd": ("EMD", "EMD", "video"),
    "eeg_video_action": ("EEG-Video-Action", "EEG-Video-Action", "video"),
    "brennan2019": ("Brennan2019", "Brennan2019", "audio"),
}

MODELS = ("shallow_fbcsp_net", "deep4net", "eegnet", "bdtcn", "atcnet",
          "eegconformer", "simpleconv_time_agg", "ctnet", "labram", "reve",
          "luna")

SEEDS = (33, 34, 35)


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    model: str
    seed: int
    directory: Path
    modality: str


def dataset_root(dataset: str) -> Path:
    if dataset not in DATASETS:
        raise KeyError(f"unknown retrieval dataset {dataset!r}; "
                       f"known: {sorted(DATASETS)}")
    return EEGBENCH_ROOT / DATASETS[dataset][0]


def runs_root(dataset: str) -> Path:
    root = dataset_root(dataset) / "runs"
    entry = DATASETS[dataset]
    if len(entry) > 3 and entry[3]:
        root = root / entry[3]
    return root


def run_dir(dataset: str, model: str, seed: int) -> Path:
    return runs_root(dataset) / model / f"seed-{seed}"


def resolve_run(dataset: str, model: str, seed: int) -> RunSpec:
    if dataset not in DATASETS:
        raise KeyError(f"unknown retrieval dataset {dataset!r}")
    if model not in MODELS:
        raise KeyError(f"unknown retrieval model {model!r}; known: {list(MODELS)}")
    directory = run_dir(dataset, model, seed)
    if not (directory / "predictions.npz").is_file():
        raise FileNotFoundError(f"no completed retrieval run at {directory}")
    return RunSpec(dataset=dataset, model=model, seed=seed,
                   directory=directory, modality=DATASETS[dataset][2])


def iter_runs(dataset: str | None = None) -> Iterator[RunSpec]:
    names = [dataset] if dataset else list(DATASETS)
    for name in names:
        root = runs_root(name)
        if not root.is_dir():
            continue
        for model_dir in sorted(root.iterdir()):
            if not model_dir.is_dir():
                continue
            for seed_dir in sorted(model_dir.glob("seed-*")):
                seed_text = seed_dir.name.split("-", 1)[-1]
                if not seed_text.isdigit():
                    continue
                if (seed_dir / "predictions.npz").is_file():
                    yield RunSpec(name, model_dir.name, int(seed_text), seed_dir,
                                  DATASETS[name][2])


def launch_task(dataset: str) -> str:
    return DATASETS[dataset][1]


__all__ = ["DATASETS", "EEGBENCH_CODE", "EEGBENCH_ROOT", "MODELS", "RunSpec",
           "SEEDS", "dataset_root", "iter_runs", "launch_task", "resolve_run",
           "run_dir", "runs_root"]
