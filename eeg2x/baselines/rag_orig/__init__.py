"""RAG baseline：EEG 编码 -> 检索 top-k -> LLM 融合。"""
from .runner import run_inference

__all__ = ["run_inference"]
