"""向量检索：论文 §II-F 的 "Vector Indexing and Retrieval"。

论文用 FAISS，这里用「L2 归一化 + 矩阵乘 + topk」——项目没有装 FAISS，而候选库只有
~9k 条 × 768 维（ZuCo1 训练集 8898 行，约 27 MB），稠密检索是精确的且足够快，
引入 ANN 只会带来近似误差与一个额外依赖。这一点记在 README 的偏差表里。

**协议约束（论文与 benchmark 都要求）**：索引**只由训练集构建**，测试句子一律不得进入。
这不只是论文的 "no data circularity" 声明，也是 benchmark 的泄漏边界；`build()` 因此
只接受训练集的行，且把来源指纹一并存下，供事后核对。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

INDEX_EMBEDS = "train_embeds.npy"
INDEX_ITEMS = "items.json"


@dataclass
class RetrievalHit:
    sample_id: str
    text: str
    score: float


class RetrievalIndex:
    """训练集的句向量库（余弦检索）。"""

    def __init__(self, embeds: np.ndarray, sample_ids: Sequence[str],
                 texts: Sequence[str], source_fingerprint: Optional[str] = None):
        if len(embeds) != len(sample_ids) or len(embeds) != len(texts):
            raise ValueError(
                f"索引长度不一致: embeds={len(embeds)} ids={len(sample_ids)} texts={len(texts)}")
        e = np.asarray(embeds, dtype=np.float32)
        # 允许调用方传未归一化的向量，这里统一归一化 —— 余弦检索的前提
        norms = np.linalg.norm(e, axis=1, keepdims=True)
        e = e / np.clip(norms, 1e-8, None)
        self.embeds = e
        self.sample_ids = [str(s) for s in sample_ids]
        self.texts = [str(t) for t in texts]
        self.source_fingerprint = source_fingerprint

    def __len__(self) -> int:
        return len(self.sample_ids)

    @classmethod
    def build(cls, embeds: np.ndarray, sample_ids: Sequence[str], texts: Sequence[str],
              source_fingerprint: Optional[str] = None) -> "RetrievalIndex":
        return cls(embeds, sample_ids, texts, source_fingerprint)

    # ---------- 检索 ----------
    def topk(self, queries, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """queries: (Q, D) tensor/ndarray → (scores (Q,k), indices (Q,k))，按分数降序。"""
        q = torch.as_tensor(queries, dtype=torch.float32)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if q.shape[1] != self.embeds.shape[1]:
            raise ValueError(f"查询维度 {q.shape[1]} != 索引维度 {self.embeds.shape[1]}")
        k = min(int(k), len(self))
        e = torch.from_numpy(self.embeds)
        sims = F.normalize(q, dim=-1) @ e.T
        scores, idx = sims.topk(k, dim=-1)
        return scores.numpy(), idx.numpy()

    def search(self, queries, k: int) -> List[List[RetrievalHit]]:
        scores, idx = self.topk(queries, k)
        return [[RetrievalHit(self.sample_ids[j], self.texts[j], float(s))
                 for s, j in zip(row_s, row_i)]
                for row_s, row_i in zip(scores, idx)]

    # ---------- 持久化 ----------
    def save(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / INDEX_EMBEDS, self.embeds)
        (d / INDEX_ITEMS).write_text(json.dumps(
            {"sample_ids": self.sample_ids, "texts": self.texts,
             "source_fingerprint": self.source_fingerprint,
             "n": len(self), "dim": int(self.embeds.shape[1])},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return d

    @classmethod
    def load(cls, directory: str | Path) -> "RetrievalIndex":
        d = Path(directory)
        meta_path, emb_path = d / INDEX_ITEMS, d / INDEX_EMBEDS
        if not meta_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                f"✗ {d} 里没有检索索引（需要 {INDEX_ITEMS} + {INDEX_EMBEDS}）。"
                "先跑 run_rag_text.py --mode index 建库")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        embeds = np.load(emb_path)
        if len(embeds) != meta["n"]:
            raise ValueError(f"✗ 索引损坏：embeds {len(embeds)} 行，meta 说 {meta['n']} 行")
        return cls(embeds, meta["sample_ids"], meta["texts"],
                   meta.get("source_fingerprint"))


def index_dir_of(run_dir: str | Path) -> Path:
    """索引固定放在 run 目录下的 `index/`，推理端由 checkpoint 的路径反推。"""
    return Path(run_dir) / "index"


__all__ = ["RetrievalIndex", "RetrievalHit", "index_dir_of", "INDEX_EMBEDS", "INDEX_ITEMS"]
