"""训练目标：把每一行的 `sentence_text` 编码成 all-mpnet-base-v2 的 768 维句向量。

论文 arXiv:2605.17503 §II-D：目标表示是 `all-mpnet-base-v2`（Sentence Transformers）的
768 维嵌入，编码器学的是「余弦对齐」而不是「逐词生成」。**同一个模型正是本 benchmark 的
SemKey 指标在用的那个**（`metrics/semkey_metrics.py::_get_sbert`），所以论文的 headline
指标 `semkey_sbert_cosine` 与训练目标处在同一嵌入空间里，可以直接对照。

行序必须与 `ZuCoH5Dataset` 完全一致 —— 这里不重新实现文件发现规则，而是**直接实例化那个
dataset 并遍历它的 `records`**，再按 `(h5_path, h5_index)` 回读 `sentence_text`。
这样即使将来 dataset 的排序/过滤规则变了，目标数组也会跟着变，不会静默错位。
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np

log = logging.getLogger(__name__)

SBERT_MODEL = "all-mpnet-base-v2"


def _fingerprint(h5_dir: str, split: str, model_name: str, n_rows: int) -> str:
    """缓存键：目录 + 分片清单（名字+行数）+ 句向量模型 + 总行数。

    不引入 h5 内容 hash（几百 MB，太慢）；分片名与行数足以识别"换了一份数据"。
    """
    paths = sorted(p.name for p in Path(h5_dir).glob("*_sentences.h5")
                   if f"_{split}_" in p.name)
    blob = json.dumps({"dir": str(Path(h5_dir).resolve()), "split": split,
                       "files": paths, "model": model_name, "n_rows": n_rows},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _read_sentence_texts(records: List[Dict]) -> List[str]:
    """按 dataset records 的顺序回读 h5 的 `sentence_text`（= 评测用的 GT 字段）。"""
    handles: Dict[str, h5py.File] = {}
    out: List[str] = []
    try:
        for r in records:
            path = r["h5_path"]
            h = handles.get(path)
            if h is None:
                h = h5py.File(path, "r", swmr=True)
                handles[path] = h
            v = h["sentence_text"][r["h5_index"]]
            out.append(v.decode("utf-8") if isinstance(v, bytes) else str(v))
    finally:
        for h in handles.values():
            h.close()
    return out


def build_targets(h5_dir: str, split: str, cache_dir: str | Path,
                  model_name: str = SBERT_MODEL, device: str = "cpu",
                  force: bool = False) -> Dict[str, object]:
    """返回 {embeds (n,768) float32 L2 归一化, texts, sample_ids, fingerprint, cached}。

    结果缓存在 ``<cache_dir>/targets/{split}_{fingerprint}.npz``；命中即直接读回，
    不重新加载句向量模型（那个模型初始化要几十秒）。
    """
    from eeg2x.frameworks.glim.text import ZuCoH5Dataset

    ds = ZuCoH5Dataset(h5_dir, split, expand_targets=False)
    texts = _read_sentence_texts(ds.records)
    sids = [r["sample id"] for r in ds.records]
    if len(texts) != len(ds):
        raise SystemExit(
            f"✗ 目标行数 {len(texts)} 与 dataset 行数 {len(ds)} 不一致——"
            "目标数组会和 EEG 错位，拒绝继续")
    fp = _fingerprint(h5_dir, split, model_name, len(texts))

    cache_dir = Path(cache_dir) / "targets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 向量存 .npz、文本存 .json —— 不要把字符串列表塞进 object 数组：numpy 2.x
    # 默认 allow_pickle=False，那样的缓存写得出、**读不回来**（冒烟时踩到）。
    emb_path = cache_dir / f"{split}_{fp}.npz"
    meta_path = cache_dir / f"{split}_{fp}.json"
    if emb_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cached_texts = [str(x) for x in meta["texts"]]
        if cached_texts != texts:
            raise SystemExit(
                f"✗ 缓存 {meta_path} 的文本与当前数据不一致，用 --force-targets 重算")
        with np.load(emb_path, allow_pickle=False) as d:
            embeds = d["embeds"]
        log.info("Loaded cached SBERT targets from %s", emb_path)
        return {"embeds": embeds, "texts": texts, "sample_ids": sids,
                "fingerprint": fp, "cached": True}

    # 私有 API（同仓库内共用），缓存键含 device，不会和评测进程里的实例冲突
    from metrics.semkey_metrics import _get_sbert

    model = _get_sbert(model_name, device=device)
    embeds = model.encode(texts, batch_size=64, convert_to_tensor=True,
                          show_progress_bar=False, normalize_embeddings=True)
    embeds = embeds.detach().cpu().numpy().astype(np.float32)
    if embeds.shape != (len(texts), 768):
        raise SystemExit(f"✗ 句向量形状异常 {embeds.shape}，期望 ({len(texts)}, 768)")
    np.savez(emb_path, embeds=embeds)
    meta_path.write_text(json.dumps(
        {"texts": texts, "sample_ids": sids, "model": model_name,
         "fingerprint": fp, "split": split, "n": len(texts)}, ensure_ascii=False),
        encoding="utf-8")
    log.info("Built %d SBERT targets (%s) -> %s", len(texts), model_name, emb_path)
    return {"embeds": embeds, "texts": texts, "sample_ids": sids,
            "fingerprint": fp, "cached": False}


__all__ = ["build_targets", "SBERT_MODEL"]
