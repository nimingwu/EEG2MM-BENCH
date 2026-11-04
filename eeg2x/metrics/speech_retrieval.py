"""
Speech Retrieval Metrics — 两种粒度的检索指标，专门适配
EEG-to-Speech 词级检索任务（Brennan2019）。

=== 粒度 1：实例级（instance-exact，已有指标族 speech_topK / speech_rank / speech_map）===
  候选池含重复词（同一词文本出现多次）。GT = 词实例 id（word_order，每行唯一），
  命中 = 正确实例位于 top-k / 排名位置。按词文本字符串匹配会因重复词虚高。

=== 粒度 2：词类型级（word-type aggregated，新增指标族 *_instance-agg / *_subject-agg / *_size=250）===
  对齐 NeuralBench (Meta, 2025) 的 full-retrieval 指标口径：
    - instance-agg: 同一词文本的多次 EEG 预测先平均（per word-type），再对候选池
      中同一词文本的多次 audio embedding 也做平均，然后按词文本排序。GT = 词文本。
    - subject-agg: 在每个 subject 内先做 instance-agg，再跨 subject 宏平均。
    - _size=250:  仅保留测试集中最高频的 250 个词类型参与评测。

与图像检索指标（topk_accuracy / map / retrieval_rank）的区别：
  - 图像检索按"概念名"字符串匹配，候选池无重复概念 → 字符串即身份；
  - speech 候选池含**重复词**（同一词文本出现多次），词类型聚合指标按词文本
    匹配（同一词文本的多次出现视为同一检索目标）。
"""

from __future__ import annotations

import numpy as np
from typing import Any, List, Optional
from collections import defaultdict

from .base import Metric


# ===================================================================
# Helpers
# ===================================================================

def _group_by_subject(predictions, ground_truths, subject_ids):
    """按 subject_ids 分组，返回 {subject: (preds, gts)}。"""
    groups = defaultdict(lambda: ([], []))
    for p, g, s in zip(predictions, ground_truths, subject_ids):
        groups[s][0].append(p)
        groups[s][1].append(g)
    return groups


def _group_by_subject_list(items, subject_ids):
    """按 subject_ids 分组任意 list，返回 {subject: [items]}。"""
    groups = defaultdict(list)
    for item, s in zip(items, subject_ids):
        groups[s].append(item)
    return groups


def _top_frequent_word_types(word_texts, k=250):
    """返回测试集中最高频的 k 个词类型（词文本去重排序后取前 k）。

    Returns
    -------
    set of str  需要保留的高频词类型集合
    """
    from collections import Counter
    freq = Counter(word_texts)
    return {w for w, _ in freq.most_common(k)}


def _instance_aggregate(predictions, ground_truths, word_texts,
                        candidate_pool_word_texts):
    """将实例级预测聚合为词类型级排名。

    对每个 query（词类型），将其多次出现的 EEG 预测 embedding 平均；
    候选池中同一词类型的多次出现的 embedding 也平均；
    然后按词类型文本排序。

    Parameters
    ----------
    predictions : list[list]  每个样本的排序候选实例 id（word_order 串）列表
    ground_truths : list[str]  每个样本的正确实例 id（word_order 串）
    word_texts : list[str]     每个样本对应的词文本
    candidate_pool_word_texts : list[str]  候选池 id → 词文本的映射

    Returns
    -------
    preds_agg : list[list[str]]   去重词类型排序（词文本列表）
    gts_agg : list[str]           去重词类型的 ground truth（词文本）
    """
    # 对候选池做词类型→平均 rank 的映射
    # 思路：对每个 query 词类型 w，计算所有出现实例的"平均相似度排名"，
    # 然后按平均排名排序。
    # 简化：用 instance-agg 的标准做法——
    #   1. 将同一词类型的多次 prediction 的 ranked_indices 合并为"命中计数"
    #   2. 对候选池中每个词类型，统计被排在前面的平均频率
    #
    # 更直接的做法（对应 NeuralBench）：
    #   同一词文本 w_q 的多次出现：将其 ranked_indices（实例 id）映射为词文本，
    #   统计每个候选词类型在所有 instance 中的排名中位数/均值，然后排序。

    unique_word_types = sorted(set(word_texts))
    # 候选池词文本列表（按 candidate_pool 顺序）
    n_cand = len(candidate_pool_word_texts)

    agg_preds = []
    agg_gts = []

    for qtype in unique_word_types:
        # 该词类型的所有 instance 的预测
        idxs = [i for i, wt in enumerate(word_texts) if wt == qtype]
        if not idxs:
            continue

        # 对每个候选词类型，计算它在这些 instance 预测中的平均 rank
        cand_scores = np.zeros(n_cand)
        for i in idxs:
            ranked = predictions[i]
            for rank_pos, inst_id in enumerate(ranked):
                cand_idx = int(inst_id)
                if 0 <= cand_idx < n_cand:
                    # 使用倒数排名作为分数（越高越好）
                    cand_scores[cand_idx] += 1.0 / (rank_pos + 1)

        # 按分数降序排列候选词类型
        sorted_cand_indices = np.argsort(-cand_scores)
        agg_pred = [candidate_pool_word_texts[idx] for idx in sorted_cand_indices]
        agg_preds.append(agg_pred)
        agg_gts.append(qtype)

    return agg_preds, agg_gts


def _filter_size_250(predictions, ground_truths, word_texts,
                     candidate_pool_word_texts, top_250_set):
    """仅保留 query 和 candidate 在 top-250 高频词集合内的部分。

    candidate_pool 会被裁剪为仅含 top-250 词类型的子集；predictions 的
    ranked_indices 会被重新映射到裁剪后的池。

    Returns (preds_filtered, gts_filtered, cand_pool_filtered) 或
    若 ground_truth 不在 top-250 中返回 None。
    """
    # 候选池裁剪：只保留在 top_250_set 中的词类型
    keep_cand_mask = [w in top_250_set for w in candidate_pool_word_texts]
    old_to_new = {}
    new_cand_pool = []
    for i, (keep, w) in enumerate(zip(keep_cand_mask, candidate_pool_word_texts)):
        if keep:
            old_to_new[i] = len(new_cand_pool)
            new_cand_pool.append(w)

    new_preds = []
    new_gts = []
    for pred_ranked, gt, wt in zip(predictions, ground_truths, word_texts):
        if wt not in top_250_set:
            continue
        # 重映射 ranked_indices
        new_ranked = []
        for inst_id in pred_ranked:
            old_idx = int(inst_id)
            if old_idx in old_to_new:
                new_ranked.append(old_to_new[old_idx])
        if not new_ranked:
            continue
        new_preds.append(new_ranked)
        new_gts.append(wt)

    return new_preds, new_gts, new_cand_pool


# ===================================================================
# Embedding-level aggregation（对齐 NeuralBench full-retrieval）
# ===================================================================

def _instance_aggregate_emb(query_embs, query_word_texts,
                            cand_embs, cand_word_texts):
    """嵌入级词类型聚合（对齐 NeuralBench instance-agg）。

    完全镜像 NeuralBench agg_retrieval_preds + Rank._compute_sim 逻辑：
      1. query 侧：同一词文本的 EEG embedding 取逐维 mean()
      2. candidate 侧：同一词文本的候选 embedding 取逐维 mean()
      3. 对聚合后的 query/candidate 做 cosine similarity 排序
    （NeuralBench norm_kind="y" 仅归一化候选侧；query 不归一化）

    query_embs  : np.ndarray (N, D)   EEG embeddings（未 L2 归一化）
    cand_embs   : np.ndarray (M, D)   候选 embeddings（已 L2 归一化）
    query_word_texts : list[str]
    cand_word_texts  : list[str]

    Returns (agg_preds, agg_gts) where both are list[str]（词文本）
    """
    import pandas as pd
    from collections import defaultdict

    # --- query 侧：按词类型平均 embedding ---
    groups_q = defaultdict(list)
    for i, wt in enumerate(query_word_texts):
        groups_q[wt].append(i)
    unique_q = sorted(groups_q.keys())
    agg_q = np.zeros((len(unique_q), query_embs.shape[1]), dtype=np.float32)
    for j, wt in enumerate(unique_q):
        idxs = groups_q[wt]
        agg_q[j] = query_embs[idxs].mean(axis=0)

    # --- candidate 侧：按词类型平均 embedding ---
    groups_c = defaultdict(list)
    for i, wt in enumerate(cand_word_texts):
        groups_c[wt].append(i)
    # 保持原顺序但有重复词类型的做平均
    seen_c = {}
    agg_c_list = []
    agg_c_texts = []
    for wt in cand_word_texts:
        if wt in seen_c:
            continue
        idxs = groups_c[wt]
        agg_c_list.append(cand_embs[idxs].mean(axis=0))
        agg_c_texts.append(wt)
        seen_c[wt] = True
    agg_c = np.stack(agg_c_list, axis=0)

    # --- cosine similarity（NeuralBench: norm_kind="y"，候选已归一化）---
    # query 侧不归一化，只归一化候选（传入的 cand_embs 已归一化，meano保留了归一化后向量的方向）
    # 重新 L2 归一化以确保 mean 后仍然是单位向量
    norms_c = np.linalg.norm(agg_c, axis=1, keepdims=True)
    norms_c = np.where(norms_c < 1e-8, 1e-8, norms_c)
    agg_c_norm = agg_c / norms_c
    sim = agg_q @ agg_c_norm.T  # (U, K)

    # --- rank ---
    agg_preds = []
    for j in range(len(unique_q)):
        order = np.argsort(-sim[j])
        agg_preds.append([agg_c_texts[k] for k in order])

    return agg_preds, unique_q


# ===================================================================
# 粒度 1：实例级（已有，保留不变）
# ===================================================================

class SpeechTopKAccuracy(Metric):
    """Speech 检索 Top-K 精度（实例级）。

    predictions : list[list]  每行是候选实例 id（word_order 字符串）的排序
    ground_truths : list[str]  正确实例 id（word_order 字符串）
    """

    def __init__(self, k: int = 1, subject_agg: bool = False):
        self.k = k
        self.subject_agg = subject_agg
        self._name = f"speech_top{k}_accuracy"

    @property
    def name(self) -> str:
        return self._name

    @property
    def applicable(self) -> List[str]:
        return ["speech_retrieval"]

    def _batch_score(self, predictions, ground_truths) -> float:
        correct = sum(1 for p, g in zip(predictions, ground_truths)
                      if g in p[:self.k])
        return correct / max(len(ground_truths), 1)

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return 1.0 if ground_truths in predictions[:self.k] else 0.0

    def batch_compute(self, predictions, ground_truths, **kwargs) -> float:
        subject_ids = kwargs.get("subject_ids")
        if self.subject_agg and subject_ids is not None:
            groups = _group_by_subject(predictions, ground_truths, subject_ids)
            if not groups:
                return 0.0
            scores = [self._batch_score(ps, gs) for ps, gs in groups.values()]
            return float(np.mean(scores))
        return self._batch_score(predictions, ground_truths)

    def reset(self):
        pass


class SpeechRetrievalRank(Metric):
    """Speech 检索排名（实例级）：median / mean / std of 正确实例的 1-based 排名。

    未命中时 rank = pool_size（kwargs 传入；未提供则取 len(ranked)）。
    空排序显式抛错（原则 4：空预测应在评测入口被过滤为 invalid）。
    """

    _REDUCTIONS = ("median", "mean", "std")

    def __init__(self, reduction: str = "median", subject_agg: bool = False):
        if reduction not in self._REDUCTIONS:
            raise ValueError(f"reduction must be one of {self._REDUCTIONS}")
        self.reduction = reduction
        self.subject_agg = subject_agg
        self._name = f"speech_{reduction}_retrieval_rank"

    @property
    def name(self) -> str:
        return self._name

    @property
    def applicable(self) -> List[str]:
        return ["speech_retrieval"]

    def _rank(self, ranked: List[Any], correct: Any,
              pool_size: Optional[int] = None) -> int:
        if not ranked:
            raise ValueError(
                "empty ranked list — prediction should have been rejected "
                "as invalid (principle 4: no silent degradation)"
            )
        for k, item in enumerate(ranked, 1):
            if item == correct:
                return k
        return pool_size if pool_size is not None else len(ranked)

    def _reduce(self, ranks: List[int]) -> float:
        if not ranks:
            return 0.0
        if self.reduction == "median":
            return float(np.median(ranks))
        if self.reduction == "mean":
            return float(np.mean(ranks))
        return float(np.std(ranks))

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return float(self._rank(predictions, ground_truths,
                                pool_size=kwargs.get("pool_size")))

    def batch_compute(self, predictions, ground_truths, **kwargs) -> float:
        pool_size = kwargs.get("pool_size")
        ranks = [self._rank(p, g, pool_size=pool_size)
                 for p, g in zip(predictions, ground_truths)]
        subject_ids = kwargs.get("subject_ids")
        if self.subject_agg and subject_ids is not None:
            from collections import defaultdict
            groups = defaultdict(list)
            for r, s in zip(ranks, subject_ids):
                groups[s].append(r)
            if not groups:
                return 0.0
            return float(np.mean([self._reduce(v) for v in groups.values()]))
        return self._reduce(ranks)

    def reset(self):
        pass


class SpeechMeanAveragePrecision(Metric):
    """Speech 检索 mAP（实例级）：正确实例排名的倒数均值。"""

    @property
    def name(self) -> str:
        return "speech_map"

    @property
    def applicable(self) -> List[str]:
        return ["speech_retrieval"]

    def _ap(self, ranked: List[Any], correct: Any) -> float:
        for k, item in enumerate(ranked, 1):
            if item == correct:
                return 1.0 / k
        return 0.0

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return self._ap(predictions, ground_truths)

    def batch_compute(self, predictions, ground_truths, **kwargs) -> float:
        aps = [self._ap(p, g)
               for p, g in zip(predictions, ground_truths)]
        subject_ids = kwargs.get("subject_ids")
        if subject_ids is not None:
            groups = _group_by_subject(predictions, ground_truths, subject_ids)
            if not groups:
                return 0.0
            per_subj = [float(np.mean([self._ap(p, g) for p, g in zip(ps, gs)]))
                        for ps, gs in groups.values()]
            return float(np.mean(per_subj))
        return float(np.mean(aps)) if aps else 0.0

    def reset(self):
        pass


# ===================================================================
# 粒度 2：词类型级（word-type aggregated）— 对齐 NeuralBench full-retrieval
# ===================================================================

class _WordTypeTopKAccuracy(Metric):
    """词类型聚合 Top-K 精度（内部指标）。

    同一个词文本的多次 EEG 预测先聚合，对候选池同词文本也做聚合后排序。
    GT = 词文本字符串。
    """

    def __init__(self, k: int = 1, size_250: bool = False):
        self.k = k
        self.size_250 = size_250
        suffix = "_size250" if size_250 else ""
        self._name = f"speech_top{k}_accuracy_instance-agg{suffix}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def applicable(self) -> List[str]:
        return ["speech_retrieval"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return 1.0 if ground_truths in predictions[:self.k] else 0.0

    def batch_compute(self, predictions, ground_truths, **kwargs) -> float:
        word_texts = kwargs.get("word_texts")
        candidate_pool_word_texts = kwargs.get("candidate_pool_word_texts")

        # 嵌入级路径（对齐 NeuralBench instance-agg）
        query_embs = kwargs.get("query_embeddings")
        cand_embs = kwargs.get("candidate_embeddings")
        if query_embs is not None and cand_embs is not None and \
           word_texts is not None and candidate_pool_word_texts is not None:
            query = np.asarray(query_embs, dtype=np.float32)
            cand = np.asarray(cand_embs, dtype=np.float32)
            if self.size_250:
                top_set = _top_frequent_word_types(word_texts, k=250)
                mask_q = np.array([w in top_set for w in word_texts])
                mask_c = np.array([w in top_set for w in candidate_pool_word_texts])
                query = query[mask_q]
                cand = cand[mask_c]
                wts_q = [w for w, m in zip(word_texts, mask_q) if m]
                wts_c = [w for w, m in zip(candidate_pool_word_texts, mask_c) if m]
                if len(wts_q) == 0 or len(wts_c) == 0:
                    return 0.0
                agg_preds, agg_gts = _instance_aggregate_emb(
                    query, wts_q, cand, wts_c)
            else:
                agg_preds, agg_gts = _instance_aggregate_emb(
                    query, word_texts, cand, candidate_pool_word_texts)
        else:
            # 排名级路径（fallback，向后兼容）
            if not word_texts or not candidate_pool_word_texts:
                return 0.0
            if self.size_250:
                top_set = _top_frequent_word_types(word_texts, k=250)
                result = _filter_size_250(predictions, ground_truths, word_texts,
                                         candidate_pool_word_texts, top_set)
                if result is None:
                    return 0.0
                preds_f, gts_f, cand_pool_f = result
                if len(gts_f) == 0:
                    return 0.0
                agg_preds, agg_gts = _instance_aggregate(
                    preds_f, gts_f, gts_f, cand_pool_f)
            else:
                agg_preds, agg_gts = _instance_aggregate(
                    predictions, ground_truths, word_texts,
                    candidate_pool_word_texts)

        if not agg_gts:
            return 0.0
        correct = sum(1 for p, g in zip(agg_preds, agg_gts)
                    if g in p[:self.k])
        return correct / max(len(agg_gts), 1)

    def reset(self):
        pass


class _WordTypeRetrievalRank(Metric):
    """词类型聚合检索排名（内部指标）。

    median / mean / std of 正确词类型的 1-based 排名。
    """

    _REDUCTIONS = ("median", "mean", "std")

    def __init__(self, reduction: str = "median", size_250: bool = False):
        if reduction not in self._REDUCTIONS:
            raise ValueError(f"reduction must be one of {self._REDUCTIONS}")
        self.reduction = reduction
        self.size_250 = size_250
        suffix = "_size250" if size_250 else ""
        self._name = f"speech_{reduction}_retrieval_rank_instance-agg{suffix}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def applicable(self) -> List[str]:
        return ["speech_retrieval"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        for k, item in enumerate(predictions, 1):
            if item == ground_truths:
                return float(k)
        return float(len(predictions))

    def batch_compute(self, predictions, ground_truths, **kwargs) -> float:
        word_texts = kwargs.get("word_texts")
        candidate_pool_word_texts = kwargs.get("candidate_pool_word_texts")

        query_embs = kwargs.get("query_embeddings")
        cand_embs = kwargs.get("candidate_embeddings")
        if query_embs is not None and cand_embs is not None and \
           word_texts is not None and candidate_pool_word_texts is not None:
            query = np.asarray(query_embs, dtype=np.float32)
            cand = np.asarray(cand_embs, dtype=np.float32)
            if self.size_250:
                top_set = _top_frequent_word_types(word_texts, k=250)
                mask_q = np.array([w in top_set for w in word_texts])
                mask_c = np.array([w in top_set for w in candidate_pool_word_texts])
                query = query[mask_q]
                cand = cand[mask_c]
                wts_q = [w for w, m in zip(word_texts, mask_q) if m]
                wts_c = [w for w, m in zip(candidate_pool_word_texts, mask_c) if m]
                if len(wts_q) == 0 or len(wts_c) == 0:
                    return 0.0
                agg_preds, agg_gts = _instance_aggregate_emb(
                    query, wts_q, cand, wts_c)
                pool_len = len(wts_c)
            else:
                agg_preds, agg_gts = _instance_aggregate_emb(
                    query, word_texts, cand, candidate_pool_word_texts)
                pool_len = len(candidate_pool_word_texts)
        else:
            if not word_texts or not candidate_pool_word_texts:
                return 0.0
            if self.size_250:
                top_set = _top_frequent_word_types(word_texts, k=250)
                result = _filter_size_250(predictions, ground_truths, word_texts,
                                         candidate_pool_word_texts, top_set)
                if result is None:
                    return 0.0
                preds_f, gts_f, cand_pool_f = result
                if len(gts_f) == 0:
                    return 0.0
                agg_preds, agg_gts = _instance_aggregate(
                    preds_f, gts_f, gts_f, cand_pool_f)
                pool_len = len(cand_pool_f)
            else:
                agg_preds, agg_gts = _instance_aggregate(
                    predictions, ground_truths, word_texts,
                    candidate_pool_word_texts)
                pool_len = len(candidate_pool_word_texts)

        ranks = []
        for p, g in zip(agg_preds, agg_gts):
            try:
                r = p.index(g) + 1
            except ValueError:
                r = pool_len
            ranks.append(r)

        if not ranks:
            return 0.0
        if self.reduction == "median":
            return float(np.median(ranks))
        if self.reduction == "mean":
            return float(np.mean(ranks))
        return float(np.std(ranks))

    def reset(self):
        pass


class _WordTypeSubjectAggTopKAccuracy(Metric):
    """词类型 subject-agg Top-K 精度。

    在每个 subject 内先做 instance-agg，再跨 subject 宏平均。
    """

    def __init__(self, k: int = 1, size_250: bool = False):
        self.k = k
        self.size_250 = size_250
        suffix = "_size250" if size_250 else ""
        self._name = f"speech_top{k}_accuracy_subject-agg{suffix}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def applicable(self) -> List[str]:
        return ["speech_retrieval"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        return 1.0 if ground_truths in predictions[:self.k] else 0.0

    def batch_compute(self, predictions, ground_truths, **kwargs) -> float:
        word_texts = kwargs.get("word_texts")
        subject_ids = kwargs.get("subject_ids")
        candidate_pool_word_texts = kwargs.get("candidate_pool_word_texts")
        query_embs = kwargs.get("query_embeddings")
        cand_embs = kwargs.get("candidate_embeddings")
        use_emb = query_embs is not None and cand_embs is not None

        if not word_texts or not subject_ids or not candidate_pool_word_texts:
            return 0.0

        # Pre-convert embeddings once
        _query = np.asarray(query_embs, dtype=np.float32) if use_emb else None
        _cand = np.asarray(cand_embs, dtype=np.float32) if use_emb else None

        # 按 subject 分组
        if use_emb:
            groups = defaultdict(list)
            for i, (wt, s) in enumerate(zip(word_texts, subject_ids)):
                groups[s].append(i)
        else:
            groups_raw = _group_by_subject_list(
                list(zip(predictions, ground_truths, word_texts)), subject_ids)
            groups = {s: list(range(len(items))) for s, items in groups_raw.items()}

        per_subj_scores = []
        for subj in sorted(set(subject_ids)):
            idxs = groups.get(subj, [])
            if not idxs:
                continue

            if use_emb:
                q = _query[idxs]
                wts = [word_texts[i] for i in idxs]
                if self.size_250:
                    top_set = _top_frequent_word_types(wts, k=250)
                    mask_q = np.array([w in top_set for w in wts])
                    mask_c = np.array([w in top_set for w in candidate_pool_word_texts])
                    if mask_q.sum() == 0 or mask_c.sum() == 0:
                        continue
                    agg_preds, agg_gts = _instance_aggregate_emb(
                        q[mask_q], [w for w, m in zip(wts, mask_q) if m],
                        _cand[mask_c],
                        [w for w, m in zip(candidate_pool_word_texts, mask_c) if m])
                else:
                    agg_preds, agg_gts = _instance_aggregate_emb(
                        q, wts, _cand, candidate_pool_word_texts)
            else:
                items = groups_raw[subj]
                preds = [it[0] for it in items]
                gts = [it[1] for it in items]
                wts = [it[2] for it in items]
                if self.size_250:
                    top_set = _top_frequent_word_types(wts, k=250)
                    result = _filter_size_250(preds, gts, wts,
                                             candidate_pool_word_texts, top_set)
                    if result is None:
                        continue
                    preds_f, gts_f, cand_pool_f = result
                    if len(gts_f) == 0:
                        continue
                    agg_preds, agg_gts = _instance_aggregate(
                        preds_f, gts_f, gts_f, cand_pool_f)
                else:
                    agg_preds, agg_gts = _instance_aggregate(
                        preds, gts, wts, candidate_pool_word_texts)

            if not agg_gts:
                continue
            correct = sum(1 for p, g in zip(agg_preds, agg_gts)
                        if g in p[:self.k])
            per_subj_scores.append(correct / max(len(agg_gts), 1))

        if not per_subj_scores:
            return 0.0
        return float(np.mean(per_subj_scores))

    def reset(self):
        pass


class _WordTypeSubjectAggRetrievalRank(Metric):
    """词类型 subject-agg 检索排名。

    在每个 subject 内先做 instance-agg 排名计算，再跨 subject 宏平均。
    """

    _REDUCTIONS = ("median", "mean", "std")

    def __init__(self, reduction: str = "median", size_250: bool = False):
        if reduction not in self._REDUCTIONS:
            raise ValueError(f"reduction must be one of {self._REDUCTIONS}")
        self.reduction = reduction
        self.size_250 = size_250
        suffix = "_size250" if size_250 else ""
        self._name = f"speech_{reduction}_retrieval_rank_subject-agg{suffix}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def applicable(self) -> List[str]:
        return ["speech_retrieval"]

    def compute(self, predictions: Any, ground_truths: Any, **kwargs) -> float:
        for k, item in enumerate(predictions, 1):
            if item == ground_truths:
                return float(k)
        return float(len(predictions))

    def batch_compute(self, predictions, ground_truths, **kwargs) -> float:
        word_texts = kwargs.get("word_texts")
        subject_ids = kwargs.get("subject_ids")
        candidate_pool_word_texts = kwargs.get("candidate_pool_word_texts")
        query_embs = kwargs.get("query_embeddings")
        cand_embs = kwargs.get("candidate_embeddings")
        use_emb = query_embs is not None and cand_embs is not None

        if not word_texts or not subject_ids or not candidate_pool_word_texts:
            return 0.0

        _query = np.asarray(query_embs, dtype=np.float32) if use_emb else None
        _cand = np.asarray(cand_embs, dtype=np.float32) if use_emb else None

        def _reduce(ranks):
            if not ranks:
                return 0.0
            if self.reduction == "median":
                return float(np.median(ranks))
            if self.reduction == "mean":
                return float(np.mean(ranks))
            return float(np.std(ranks))

        if use_emb:
            groups = defaultdict(list)
            for i, (wt, s) in enumerate(zip(word_texts, subject_ids)):
                groups[s].append(i)
        else:
            groups_raw = _group_by_subject_list(
                list(zip(predictions, ground_truths, word_texts)), subject_ids)

        per_subj_reduced = []
        for subj in sorted(set(subject_ids)):
            if use_emb:
                idxs = groups.get(subj, [])
                if not idxs:
                    continue
                q = _query[idxs]
                wts = [word_texts[i] for i in idxs]
                if self.size_250:
                    top_set = _top_frequent_word_types(wts, k=250)
                    mask_q = np.array([w in top_set for w in wts])
                    mask_c = np.array([w in top_set for w in candidate_pool_word_texts])
                    if mask_q.sum() == 0 or mask_c.sum() == 0:
                        continue
                    agg_preds, agg_gts = _instance_aggregate_emb(
                        q[mask_q], [w for w, m in zip(wts, mask_q) if m],
                        _cand[mask_c],
                        [w for w, m in zip(candidate_pool_word_texts, mask_c) if m])
                    pool_len = mask_c.sum()
                else:
                    agg_preds, agg_gts = _instance_aggregate_emb(
                        q, wts, _cand, candidate_pool_word_texts)
                    pool_len = len(candidate_pool_word_texts)
            else:
                items = groups_raw.get(subj, [])
                if not items:
                    continue
                preds = [it[0] for it in items]
                gts = [it[1] for it in items]
                wts = [it[2] for it in items]
                if self.size_250:
                    top_set = _top_frequent_word_types(wts, k=250)
                    result = _filter_size_250(preds, gts, wts,
                                             candidate_pool_word_texts, top_set)
                    if result is None:
                        continue
                    preds_f, gts_f, cand_pool_f = result
                    if len(gts_f) == 0:
                        continue
                    agg_preds, agg_gts = _instance_aggregate(
                        preds_f, gts_f, gts_f, cand_pool_f)
                    pool_len = len(cand_pool_f)
                else:
                    agg_preds, agg_gts = _instance_aggregate(
                        preds, gts, wts, candidate_pool_word_texts)
                    pool_len = len(candidate_pool_word_texts)

            ranks = []
            for p, g in zip(agg_preds, agg_gts):
                try:
                    ranks.append(p.index(g) + 1)
                except ValueError:
                    ranks.append(pool_len)
            if ranks:
                per_subj_reduced.append(_reduce(ranks))

        if not per_subj_reduced:
            return 0.0
        return float(np.mean(per_subj_reduced))

    def reset(self):
        pass
