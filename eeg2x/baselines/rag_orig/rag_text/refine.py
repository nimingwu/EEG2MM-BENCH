"""RAG 的最后一步：把检索到的 k 句交给 LLM，融合成一句。

论文 §II-F 给的是 `meta-llama/Meta-Llama-3-8B`（**基座模型，不是 Instruct**）与一段固定
指令；论文只给了指令原文、没给 prompt 模板，所以下面 `build_prompt()` 的模板是**我们的
具体化**（记在 README 偏差表里）。

**基座模型不会"听话"** —— 它没见过 instruction tuning，最可能的行为是把 prompt 续写下去，
包括原样抄检索句、或一口气列好几条。所以这里配了一个 `parse_output()`：

    去掉回显的 prompt → 取第一个非空行 → 剥掉 "Corrected sentence:" 之类的标签与引号
    → 若为空 / 退化复读 / 原样抄了两句以上 → **回落到 top-1 检索句**，并把它记成 fallback

回落率会写进预测的 `.meta.json`；如果高得离谱，就该换 Instruct 版（只需改一个 model id）。
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# 论文 §II-F 的指令原文，逐字照抄，不要"润色"
PAPER_INSTRUCTION = (
    "You are given three similar sentences. Your task is to generate one single, "
    "fluent, grammatically correct sentence that captures the same meaning, without "
    "adding or removing information. Only output the corrected sentence."
)
ANSWER_CUE = "Corrected sentence:"

# 解析器要剥掉的前缀标签（基座模型可能自己加，也可能把 cue 抄回来）
_LABEL_RE = re.compile(
    r"^\s*(?:corrected\s+sentence|answer|output|sentence|response)\s*[:：\-]\s*",
    re.IGNORECASE)


def build_prompt(sentences: Sequence[str]) -> str:
    """论文指令 + 编号列出的检索句 + 收尾 cue（基座模型靠续写完成）。"""
    lines = [PAPER_INSTRUCTION, ""]
    for i, s in enumerate(sentences, 1):
        lines.append(f"Sentence {i}: {s}")
    lines += ["", ANSWER_CUE]
    return "\n".join(lines)


def _is_degenerate(text: str) -> bool:
    """复读机判定：词数不少但去重后极少（例如 "the the the the …"）。"""
    words = text.split()
    if len(words) < 8:
        return False
    return len(set(w.lower() for w in words)) / len(words) < 0.3


def parse_output(raw: str, prompt: str, fallback: str,
                 retrieved: Sequence[str] = ()) -> Tuple[str, bool]:
    """把 LLM 的原始输出变成一句可用文本。返回 (text, used_fallback)。"""
    text = raw or ""

    # 1) 基座模型常把 prompt 原样续写出来 —— 只保留 cue 之后的部分
    if prompt and prompt in text:
        text = text.rsplit(prompt, 1)[-1]
    if ANSWER_CUE.lower() in text.lower():
        idx = text.lower().rindex(ANSWER_CUE.lower())
        text = text[idx + len(ANSWER_CUE):]

    # 2) 取第一个非空行，剥标签与引号
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return fallback, True
    out = _LABEL_RE.sub("", lines[0]).strip().strip('"“”\'').strip()

    # 3) 退化判定：空 / 太短 / 复读 / 把检索句抄了两句以上
    copied = sum(1 for s in retrieved if s and s.strip() and s.strip() in out)
    if (len(out) < 2 or _is_degenerate(out) or copied >= 2
            or out.lower().startswith("sentence 1")):
        return fallback, True
    return out, False


class LLMRefiner:
    """批量调用一个 causal LM 做融合，带磁盘缓存（clean/noisy 两遍不重复烧算力）。"""

    def __init__(self, model_id: str, device: Optional[str] = None,
                 cache_path: Optional[str | Path] = None, max_new_tokens: int = 64,
                 offline: Optional[bool] = None, hf_cache: Optional[str] = None):
        import os
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.device = device or ("cuda" if _torch().cuda.is_available() else "cpu")
        self.hf_cache = hf_cache or os.environ.get("EEG2X_HF_CACHE", "/datas/hf-cache")
        self.offline = (os.environ.get("EEG2X_HF_OFFLINE", "1") == "1"
                        if offline is None else bool(offline))
        self._tok = None
        self._model = None
        self.n_calls = 0
        self.n_cache_hits = 0
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: Dict[str, dict] = {}
        if self.cache_path and self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec

    # ---------- 模型 ----------
    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        common = dict(cache_dir=self.hf_cache, local_files_only=self.offline)
        self._tok = AutoTokenizer.from_pretrained(self.model_id, **common)
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, **common).to(self.device).eval()
        log.info("Loaded refinement LLM %s on %s", self.model_id, self.device)

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha1(prompt.encode("utf-8")).hexdigest()

    # ---------- 调用 ----------
    def refine(self, batches: Sequence[Sequence[str]], batch_size: int = 8,
               fallbacks: Optional[Sequence[str]] = None) -> List[Tuple[str, bool]]:
        """batches: 每个元素是该样本检索到的 k 句（top-1 在首位）。

        返回等长的 (text, used_fallback) 列表。fallbacks 默认取每条的第一个检索句。
        """
        if fallbacks is None:
            fallbacks = [b[0] if b else "" for b in batches]
        prompts = [build_prompt(b) for b in batches]
        results: List[Optional[Tuple[str, bool]]] = [None] * len(prompts)

        todo = []
        for i, p in enumerate(prompts):
            rec = self._cache.get(self._key(p))
            if rec is not None:
                results[i] = (rec["text"], bool(rec.get("fallback", False)))
                self.n_cache_hits += 1
            else:
                todo.append(i)
        if todo:
            self._ensure_model()
            for start in range(0, len(todo), batch_size):
                chunk = todo[start:start + batch_size]
                raws = self._generate([prompts[i] for i in chunk])
                for i, raw in zip(chunk, raws):
                    text, fb = parse_output(raw, prompts[i], fallbacks[i], batches[i])
                    results[i] = (text, fb)
                    self._record(prompts[i], text, fb)
        return [r if r is not None else ("", True) for r in results]

    def _generate(self, prompts: List[str]) -> List[str]:
        import torch
        self.n_calls += len(prompts)
        enc = self._tok(list(prompts), return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            out = self._model.generate(
                **enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                num_beams=1, pad_token_id=self._tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        return self._tok.batch_decode(gen, skip_special_tokens=True)

    def _record(self, prompt: str, text: str, fallback: bool) -> None:
        key = self._key(prompt)
        self._cache[key] = {"key": key, "text": text, "fallback": bool(fallback)}
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(self._cache[key], ensure_ascii=False) + "\n")

    def stats(self) -> Dict[str, object]:
        return {"model_id": self.model_id, "llm_calls": self.n_calls,
                "llm_cache_hits": self.n_cache_hits,
                "max_new_tokens": self.max_new_tokens}


def _torch():
    import torch
    return torch


__all__ = ["LLMRefiner", "build_prompt", "parse_output", "PAPER_INSTRUCTION", "ANSWER_CUE"]
