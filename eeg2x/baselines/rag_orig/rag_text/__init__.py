"""RAG-based EEG-to-Text（arXiv:2605.17503）—— 检索 + LLM 融合的独立 baseline。

与其它 baseline 的根本区别：**它不生成文本**。EEG 编码器被训练成把一句话的 EEG 映射到
`all-mpnet-base-v2` 的句向量空间（余弦对齐），推理时在**训练集**的句向量库里做 top-k
余弦检索，再把 k 句交给一个 LLM 融合成最终输出。所以：

* 没有词表、没有 tokenizer、没有自回归解码 —— 「生成」发生在检索之后，由外部 LLM 完成；
* 训练目标是句向量（`targets.py`），不是某一句改写文本；
* 检索库**只放训练集**（论文 §II-F 明确要求，也是 benchmark 的泄漏边界）。

论文没有开源代码（arXiv 全文无仓库链接），这是照 6 页正文的重写；论文没给全的卷积细节、
prompt 模板与超参见 README 的偏差表。

入口：`run_rag_text.py`（训练 / 建索引）、
`generate_sentence_predictions.py --variant rag`（检索 + 融合 + 预测）。
"""
from .dataset import RAGDataModule, RAGDataset
from .encoder import EEGChannelNetEncoder
from .module import RAGText, RAGTextModule
from .refine import LLMRefiner, build_prompt, parse_output
from .retrieval import RetrievalIndex
from .targets import SBERT_MODEL, build_targets

__all__ = [
    "RAGText", "RAGTextModule", "EEGChannelNetEncoder",
    "RAGDataset", "RAGDataModule",
    "RetrievalIndex", "LLMRefiner", "build_prompt", "parse_output",
    "build_targets", "SBERT_MODEL",
]
