"""RAG baseline：EEG 编码 -> 检索库 top-k -> LLM 融合。

它没有解码器：EEG 编码器输出句向量，检索库给出 k 个候选句，最终文本由
``Qwen2.5-7B-Instruct`` 融合而成。推理用 retrieval index 与 refinement cache
都在文本仓库的 clean run 里，不重训。

只做推理；``--no-refine`` 是消融/无 LLM 的降级路径，不是论文条件。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from eeg2x.data import write_json
from eeg2x.datasets import DatasetSpec
from eeg2x.evaluation.api import evaluate_predictions
from eeg2x.paths import V2_TEXT_ROOT, V3_ROOT, require_v3_output
from eeg2x.protocol import SEED
from eeg2x.seeding import seed_everything, seed_worker

from .rag_text.dataset import RAGDataset
from .rag_text.module import RAGTextModule
from .rag_text.refine import LLMRefiner
from .rag_text.retrieval import RetrievalIndex, index_dir_of

log = logging.getLogger("eeg2x.baselines.rag_orig")

DEFAULT_LLM = "Qwen/Qwen2.5-7B-Instruct"

_RUNS = {
    "zuco1": ("zuco1_rag_seed2000", "zuco1_glim_seed2000_h5"),
    "zuco2": ("zuco2_rag_seed2000", "zuco2_glim_seed2000_h5"),
}

_HPARAM_KEYS = ("n_channels", "n_times", "out_dim", "width", "n_res_blocks",
                "dropout", "lr", "weight_decay", "max_epochs")


def run_inference(dataset: DatasetSpec, *, device: str = "cuda:0",
                  noise: str = "none", noise_seed: int = SEED,
                  overwrite: bool = False, batch_size: int = 24,
                  rag_k: int = 3, llm_batch_size: int = 8,
                  no_refine: bool = False,
                  limit: int | None = None) -> Path:
    if dataset.name not in _RUNS:
        raise KeyError(f"RAG has no track for {dataset.name!r}; "
                       f"known: {sorted(_RUNS)}")
    run_name, h5_name = _RUNS[dataset.name]
    source = V2_TEXT_ROOT / "outputs" / run_name
    h5_dir = V2_TEXT_ROOT / h5_name
    checkpoint = source / "checkpoints" / "last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing RAG checkpoint {checkpoint}")

    output_root = require_v3_output(
        V3_ROOT / "artifacts" / dataset.name / "rag_orig")
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
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        hparams = state.get("hyper_parameters", {})
        model = RAGTextModule(**{k: hparams[k] for k in _HPARAM_KEYS
                                 if k in hparams})
        model.load_state_dict(state.get("state_dict", state), strict=False)
        model.to(target).eval()

        index = RetrievalIndex.load(index_dir_of(source))
        dataset_ = RAGDataset(str(h5_dir), "test", None)
        loader = DataLoader(dataset_, batch_size=batch_size, shuffle=False,
                            num_workers=2, worker_init_fn=seed_worker)
        generator = (torch.Generator(device=target).manual_seed(noise_seed)
                     if noise == "gaussian" else None)
        refiner = None
        if not no_refine:
            refiner = LLMRefiner(
                DEFAULT_LLM,
                cache_path=source / "refinement_cache.jsonl")
        predictions: Dict[str, Dict[str, str]] = {}
        pending: list = []
        n_fallback = 0

        def drain(force: bool = False) -> None:
            nonlocal pending, n_fallback
            if refiner is None:
                return
            while pending and (force or len(pending) >= llm_batch_size):
                chunk, pending = (pending[:llm_batch_size],
                                  pending[llm_batch_size:])
                texts = refiner.refine([c[1] for c in chunk],
                                       batch_size=llm_batch_size)
                for (sid, _), (text, fallback) in zip(chunk, texts):
                    predictions[sid] = {"generated_text": text}
                    n_fallback += int(fallback)

        with torch.no_grad():
            for batch in loader:
                eeg = batch["eeg"].to(target).float()
                if generator is not None:
                    eeg = torch.randn(eeg.shape, generator=generator,
                                      device=target, dtype=eeg.dtype)
                with torch.autocast(device_type=target.type, dtype=torch.bfloat16,
                                    enabled=target.type == "cuda"):
                    z = model.decode(eeg)
                hits = index.search(z.float().cpu(), rag_k)
                for sid, row_hits in zip((str(s) for s in batch["sample id"]), hits):
                    texts_hit = [hit.text for hit in row_hits]
                    if refiner is None:
                        predictions[sid] = {"generated_text":
                                            texts_hit[0] if texts_hit else ""}
                    else:
                        pending.append((sid, texts_hit))
                drain()
                if limit is not None and len(predictions) + len(pending) >= limit:
                    break
        drain(force=True)
        if pending:
            raise RuntimeError(f"{len(pending)} samples were not refined")
        write_json(predictions_path, predictions)
        write_json(predictions_path.with_name("predictions.meta.json"), {
            "variant": "rag", "dataset": dataset.name,
            "h5_dir": str(h5_dir), "checkpoint": str(checkpoint),
            "llm_id": DEFAULT_LLM, "rag_k": rag_k, "refined": refiner is not None,
            "n_fallback": n_fallback,
            "fallback_rate": (n_fallback / len(predictions)) if predictions else 0.0,
            "n_index_items": len(index), "noise": noise,
            "noise_seed": int(noise_seed), "n_predictions": len(predictions),
        })

    evaluation = evaluate_predictions(
        task="sentence_generation", h5_dir=h5_dir,
        predictions_path=predictions_path,
        output_path=output_root / "evaluation" / "report_semkey.json",
        metrics=None)
    write_json(output_root / "run_config.json", {
        "experiment_id": output_root.name,
        "dataset": dataset.name, "task": "sentence_generation",
        "model": "rag_orig", "pipeline": "EEG encoder -> retrieval top-k -> "
        f"{DEFAULT_LLM} fusion",
        "noise": noise, "noise_seed": int(noise_seed),
        "checkpoint": str(checkpoint), "seed": SEED,
        "derived_inputs_used": False,
    })
    return evaluation


__all__ = ["run_inference"]
