"""GLIM 文本框架的推理编排：加载 checkpoint -> 生成 -> 评测。

迁自文本仓库的 ``generate_sentence_predictions.py``（glim / labram / luna /
reve 四个变体那一段）。训练不在这里；checkpoint 从文本仓库的干净输出目录读，
产物写到 v3 的 artifacts 下。

文本框架与图像、视频都不同：它自己带一个冻结的 encoder-decoder LM
（ZuCo2 用 flan-t5-large，Chisco 用 mt0-large），四个变体只替换 EEG 编码器。
所以 checkpoint 的 hyper_parameters 决定用哪个 LM 与是否启用 domain prompt，
这里不从命令行猜。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import torch
from torch.utils.data import DataLoader

from eeg2x.data import write_json
from eeg2x.datasets import DatasetSpec
from eeg2x.evaluation.api import evaluate_predictions
from eeg2x.paths import V2_TEXT_ROOT, V3_ROOT, require_v3_output
from eeg2x.protocol import SEED
from eeg2x.seeding import seed_everything, seed_worker

log = logging.getLogger("eeg2x.glim.pipeline")

VARIANTS = ("glim", "labram", "luna", "reve")

#: (dataset, variant) -> (v2 run 目录名, v2 h5 目录名)
_TEXT_RUNS = {
    ("zuco2", "glim"): ("zuco2_glim_full18_seed2000", "zuco2_glim_seed2000_h5"),
    ("zuco2", "labram"): ("zuco2_labram_full18_seed2000", "zuco2_labram_seed2000_h5"),
    ("zuco2", "luna"): ("zuco2_luna_full18_seed2000", "zuco2_luna_seed2000_h5"),
    ("zuco2", "reve"): ("zuco2_reve_l2s3_full18_lr1e5_seed2000",
                        "zuco2_reve_seed2000_h5"),
    ("chisco", "glim"): ("chisco_glim_full5_es_seed2000", "chisco_glim_seed2000_h5"),
    ("chisco", "labram"): ("chisco_labram_full5_es_seed2000",
                           "chisco_labram_seed2000_h5"),
    ("chisco", "luna"): ("chisco_luna_full5_es_seed2000", "chisco_luna_seed2000_h5"),
    ("chisco", "reve"): ("chisco_reve_full5_es_seed2000", "chisco_reve_seed2000_h5"),
    # GLIM Orig.：domain prompt 全开的原论文条件；只有 ZuCo2 有这条轨道。
    ("zuco2", "glim_orig"): ("zuco2_glim_full18_prompt_seed2000",
                              "zuco2_glim_seed2000_h5"),
}


@dataclass
class RunPaths:
    dataset: str
    variant: str
    source: Path       # v2 run 目录（checkpoint / predictions / evaluation）
    h5_dir: Path       # v2 HDF5 目录（全部被试、全部 split）
    output: Path       # v3 产物目录


def resolve_paths(dataset: DatasetSpec, variant: str) -> RunPaths:
    key = (dataset.name, variant)
    if key not in _TEXT_RUNS:
        raise KeyError(
            f"no GLIM text run for {key}; known: {sorted(_TEXT_RUNS)}")
    run_name, h5_name = _TEXT_RUNS[key]
    source = V2_TEXT_ROOT / "outputs" / run_name
    if variant == "glim_orig":
        output = V3_ROOT / "artifacts" / dataset.name / "glim_orig"
    else:
        output = V3_ROOT / "artifacts" / dataset.name / "glim" / variant
    return RunPaths(dataset=dataset.name, variant=variant, source=source,
                    h5_dir=V2_TEXT_ROOT / h5_name, output=output)


def resolve_checkpoint(run_dir: Path) -> Path:
    """优先 ``checkpoints_best/best.json``，其次 best 目录里的 ckpt，最后 last。"""
    best_dir = run_dir / "checkpoints_best"
    meta = best_dir / "best.json"
    if meta.is_file():
        try:
            recorded = json.loads(meta.read_text(encoding="utf-8")).get("best_model_path")
        except Exception:
            recorded = None
        if recorded and Path(recorded).is_file():
            return Path(recorded)
    if best_dir.is_dir():
        hits = sorted(best_dir.glob("*.ckpt"))
        if hits:
            return hits[0]
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.is_file():
        return last
    raise FileNotFoundError(
        f"no GLIM checkpoint under {run_dir}/checkpoints_best or checkpoints")


def h5_n_channels(h5_dir: Path) -> int:
    files = sorted(h5_dir.glob("*_sentences.h5"))
    if not files:
        raise FileNotFoundError(f"no *_sentences.h5 under {h5_dir}")
    counts = set()
    for path in files:
        with h5py.File(path, "r") as h5:
            counts.add(int(h5.attrs["n_channels"]))
    if len(counts) != 1:
        raise ValueError(f"{h5_dir}: channel counts differ across files: {sorted(counts)}")
    return counts.pop()


def _build_model(variant: str, n_ch: int, text_model_id: str,
                 use_null_prompt: bool, prompt_dropout_probs, hparams: Dict[str, Any],
                 source: Path):
    common = dict(
        input_text_len=96, tgt_text_len=64, hidden_eeg_len=96, input_dim=n_ch,
        hidden_dim=256, embed_dim=1024, text_model_id=text_model_id,
        model_cache_dir="/datas/hf-cache", local_files_only=True,
        use_null_prompt=use_null_prompt,
        prompt_nums=(3, 3, 31), prompt_dropout_probs=prompt_dropout_probs,
        n_in_blocks=6, n_out_blocks=6, in_temporal_modulate=True,
        out_is_causal=True, clip_loss_weight=0.5, commitment_loss_weight=0.0,
        bsz_val=24, full_val_interval=10,
    )
    if variant in ("glim", "glim_orig"):
        from .core.glim import GLIM
        return GLIM(input_eeg_len=1280, bsz_train=72, **common)
    if variant == "labram":
        from .labram import GLIMLaBraMText
        return GLIMLaBraMText(
            input_eeg_len=2000, bsz_train=8, **common,
            labram_pretrained=str(
                V2_TEXT_ROOT / "outputs/glim_labram_seed2000/pretrained/labram-base.pth"))
    if variant == "luna":
        from .foundation import GLIMLunaText
        return GLIMLunaText(
            input_eeg_len=1280, bsz_train=8, **common,
            luna_pretrained=str(
                V2_TEXT_ROOT / "outputs/luna_seed2000/pretrained/luna-base.safetensors"))
    if variant == "reve":
        from .foundation import GLIMReveText
        return GLIMReveText(
            input_eeg_len=2000, bsz_train=8, **common,
            reve_pretrained=str(
                V2_TEXT_ROOT / "outputs/reve_seed2000/pretrained/reve-base.safetensors"),
            reve_boundary_norm=hparams.get("reve_boundary_norm", "none"),
            reve_norm_scale=float(hparams.get("reve_norm_scale", 3.0)),
            reve_normalize=bool(hparams.get("reve_normalize", True)))
    raise KeyError(f"unknown GLIM variant {variant!r}")


def run_inference(paths: RunPaths, *, device: str = "cuda:0",
                  noise: str = "none", noise_seed: int = SEED,
                  batch_size: int = 24, overwrite: bool = False) -> Path:
    """跑一个 (数据集, 变体) 的文本生成与 15 项 SemKey 评测。"""
    checkpoint = resolve_checkpoint(paths.source)
    output_root = require_v3_output(paths.output)
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
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hparams = state.get("hyper_parameters", {}) or {}
        text_model_id = (hparams.get("text_model_id")
                         or hparams.get("text_model") or "google/flan-t5-large")
        use_null_prompt = bool(hparams.get("use_null_prompt", True))
        prompt_dropout_probs = tuple(hparams.get("prompt_dropout_probs", (0.0, 1.0, 1.0)))
        model = _build_model(paths.variant, h5_n_channels(paths.h5_dir),
                             text_model_id, use_null_prompt,
                             prompt_dropout_probs, hparams, paths.source)
        model.load_state_dict(state.get("state_dict", state), strict=False)
        target = torch.device(device)
        model.to(target)
        model.eval()
        model.setup("test")

        from .text import ZuCoH5Dataset
        from transformers.modeling_outputs import BaseModelOutput

        dataset = ZuCoH5Dataset(str(paths.h5_dir), "test", expand_targets=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=2, worker_init_fn=seed_worker)
        generator = (torch.Generator(device=target).manual_seed(noise_seed)
                     if noise == "gaussian" else None)
        predictions: Dict[str, Dict[str, str]] = {}
        with torch.no_grad():
            for batch in loader:
                eeg = batch["eeg"].to(target)
                mask = batch["mask"].to(target)
                if generator is not None:
                    eeg = torch.randn(eeg.shape, generator=generator,
                                      device=target, dtype=eeg.dtype)
                    mask = torch.ones_like(mask)
                if use_null_prompt:
                    pids = torch.zeros((eeg.shape[0], 3), dtype=torch.int,
                                       device=target)
                else:
                    pids = model.p_embedder.encode(batch["prompt"], device=target)
                pembed = model.p_embedder(pids, model.eval_pembed)
                hidden, _ = model.eeg_encoder(eeg, mask, pembed)
                embeds, _ = model.aligner.embed_eeg(hidden)
                with torch.autocast(device_type=target.type, dtype=torch.bfloat16,
                                    enabled=target.type == "cuda"):
                    ids = model.text_model.generate(
                        encoder_outputs=BaseModelOutput(embeds),
                        num_beams=2, min_length=0, max_length=model.tgt_text_len)
                texts = model.tokenizer.batch_decode(ids, skip_special_tokens=True)
                for sid, text in zip(batch["sample id"], texts):
                    predictions[str(sid)] = {"generated_text": text}
        write_json(predictions_path, predictions)
        write_json(output_root / "predictions" / "predictions.meta.json", {
            "variant": paths.variant, "dataset": paths.dataset,
            "h5_dir": str(paths.h5_dir), "checkpoint": str(checkpoint),
            "text_model": text_model_id, "noise": noise,
            "noise_seed": int(noise_seed), "n_predictions": len(predictions),
        })

    evaluation = evaluate_predictions(
        task="sentence_generation",
        h5_dir=paths.h5_dir,
        predictions_path=predictions_path,
        output_path=output_root / "evaluation" / "report_semkey.json",
        metrics=None)

    write_json(output_root / "run_config.json", {
        "experiment_id": output_root.name,
        "dataset": paths.dataset, "task": "sentence_generation",
        "model": "glim", "encoder": paths.variant,
        "pipeline": "EEG -> GLIM encoder/aligner -> frozen LM decoder",
        "noise": noise, "noise_seed": int(noise_seed),
        "checkpoint": str(checkpoint),
        "seed": SEED,
        "derived_inputs_used": False,
    })
    return evaluation


__all__ = ["RunPaths", "VARIANTS", "h5_n_channels", "resolve_checkpoint",
           "resolve_paths", "run_inference"]
