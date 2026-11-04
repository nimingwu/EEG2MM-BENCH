"""EEG-To-Text（Wang & Ji, AAAI 2022）baseline：词级特征 + BART。

它不走 GLIM 的 encoder/aligner，而是把旁挂的 56x840 词级特征送进一个
Transformer 编码器，再交给冻结的 BART 解码。输入是“协议分支 H5 + 词级特征
npz”的组合，所以两个路径都必须存在。

只做推理：checkpoint 从文本仓库的干净轨道读，产物写到 v3 的 artifacts 下。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg2x.data import write_json
from eeg2x.datasets import DatasetSpec
from eeg2x.evaluation.api import evaluate_predictions
from eeg2x.paths import V2_TEXT_ROOT, V3_ROOT, require_v3_output
from eeg2x.protocol import SEED
from eeg2x.seeding import seed_everything, seed_worker

from .module.dataset import WordFeatureDataset
from .module.model import EEGToTextModule

log = logging.getLogger("eeg2x.baselines.eeg2text_orig")

TEXT_MODEL = "facebook/bart-large"

#: dataset -> (v2 run 目录, v2 h5 目录, 词级特征 npz)
_RUNS = {
    "zuco1": ("zuco1_eegtotext_seed2000", "zuco1_glim_seed2000_h5",
              "zuco1_eegtotext_words.npz"),
    "zuco2": ("zuco2_eegtotext_seed2000", "zuco2_glim_seed2000_h5",
              "zuco2_eegtotext_words.npz"),
}


def _h5_shape(h5_dir: Path) -> tuple[int, int]:
    files = sorted(h5_dir.glob("*_sentences.h5"))
    if not files:
        raise FileNotFoundError(f"no *_sentences.h5 under {h5_dir}")
    with h5py.File(files[0], "r") as h5:
        data = h5["data"] if "data" in h5 else h5["eeg"]
        return int(data.shape[1]), int(data.shape[2])


def _features_shape(npz: Path) -> tuple[int, int]:
    with np.load(npz) as data:
        shape = tuple(data["features"].shape)
    if len(shape) != 3:
        raise ValueError(f"{npz}: features shape is {shape}, expected 3-D")
    return int(shape[1]), int(shape[2])


def run_inference(dataset: DatasetSpec, *, device: str = "cuda:0",
                  noise: str = "none", noise_seed: int = SEED,
                  overwrite: bool = False, batch_size: int = 24,
                  limit: int | None = None) -> Path:
    if dataset.name not in _RUNS:
        raise KeyError(f"EEG-To-Text has no track for {dataset.name!r}; "
                       f"known: {sorted(_RUNS)}")
    run_name, h5_name, words_name = _RUNS[dataset.name]
    source = V2_TEXT_ROOT / "outputs" / run_name
    h5_dir = V2_TEXT_ROOT / h5_name
    word_features = V2_TEXT_ROOT / words_name
    checkpoint = source / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing EEG-To-Text checkpoint {checkpoint}")
    if not word_features.is_file():
        raise FileNotFoundError(f"missing word-level features {word_features}")

    output_root = require_v3_output(
        V3_ROOT / "artifacts" / dataset.name / "eeg2text_orig")
    if noise != "none":
        output_root = require_v3_output(output_root / "noise_control")
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions" / "preds_sentence_generation.json"
    if predictions_path.is_file() and not overwrite:
        log.info("reusing predictions: %s", predictions_path)
    else:
        seed_everything(SEED)
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        target = torch.device(device)
        n_words, n_features = _features_shape(word_features)
        model = EEGToTextModule(text_model=TEXT_MODEL, max_len=n_words)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model.to(target).eval()

        dataset_ = WordFeatureDataset(str(h5_dir), str(word_features), "test",
                                      expand_targets=False)
        loader = DataLoader(dataset_, batch_size=batch_size, shuffle=False,
                            num_workers=2, worker_init_fn=seed_worker)
        generator = (torch.Generator(device=target).manual_seed(noise_seed)
                     if noise == "gaussian" else None)
        predictions: Dict[str, Dict[str, str]] = {}
        with torch.no_grad():
            for batch in loader:
                eeg = batch["eeg"].to(target).float()
                mask = batch.get("mask")
                mask = None if mask is None else mask.to(target)
                if eeg.shape[1:] != (n_words, n_features):
                    raise ValueError(
                        f"word features shape {tuple(eeg.shape[1:])} does not "
                        f"match {(n_words, n_features)}")
                if generator is not None:
                    eeg = torch.randn(eeg.shape, generator=generator,
                                      device=target, dtype=eeg.dtype)
                    mask = None if mask is None else torch.ones_like(mask)
                with torch.autocast(device_type=target.type, dtype=torch.bfloat16,
                                    enabled=target.type == "cuda"):
                    ids = model.decode(eeg, mask)
                texts = model.tokenizer.batch_decode(ids, skip_special_tokens=True)
                for sid, text in zip(batch["sample id"], texts):
                    predictions[str(sid)] = {"generated_text": text}
                if limit is not None and len(predictions) >= limit:
                    break
        write_json(predictions_path, predictions)
        write_json(predictions_path.with_name("predictions.meta.json"), {
            "variant": "eegtotext", "dataset": dataset.name,
            "h5_dir": str(h5_dir), "word_features": str(word_features),
            "checkpoint": str(checkpoint), "text_model": TEXT_MODEL,
            "noise": noise, "noise_seed": int(noise_seed),
            "n_predictions": len(predictions),
        })

    evaluation = evaluate_predictions(
        task="sentence_generation", h5_dir=h5_dir,
        predictions_path=predictions_path,
        output_path=output_root / "evaluation" / "report_semkey.json",
        metrics=None)
    write_json(output_root / "run_config.json", {
        "experiment_id": output_root.name,
        "dataset": dataset.name, "task": "sentence_generation",
        "model": "eeg2text_orig", "pipeline": "word-level features -> "
        "Transformer encoder -> frozen BART decoder",
        "noise": noise, "noise_seed": int(noise_seed),
        "checkpoint": str(checkpoint), "text_model": TEXT_MODEL,
        "seed": SEED, "derived_inputs_used": False,
    })
    return evaluation


__all__ = ["run_inference"]
