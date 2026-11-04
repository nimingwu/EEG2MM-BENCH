"""EEG-To-Text（Wang & Ji, AAAI 2022, arXiv:2112.02690）—— 词级输入 baseline。

与四变体框架、与 `eeg2text` 都不同的一族：输入是**词级** EEG 特征，每条句子 ≤56 个词，
每词 840 维 = 105 通道 × 8 个频带（GD 特征），来自原作者用眼动注视对齐算出来的 pickle。

模型接线逐层照抄 `EEG-To-Text-N/model_decoding.py::BrainTranslator`（Yang & Jo 修正版）：

    6 × TransformerEncoderLayer(d=840, nhead=8, ff=2048)   ← src_key_padding_mask = 词槽掩码
    ReLU(Linear(840, 1024))                                ← 投到 BART 的 d_model
    BART(inputs_embeds=上面这个, labels=目标 token)         ← **注意是解码器的 inputs_embeds**

最后一行是这个 baseline 的关键：EEG embedding 不是喂给 BART 的 encoder，而是当作
**解码器前缀**，目标序列被 tokenize 成同样 56 个位置、逐位置算交叉熵（`inputs_embeds`
给定时 HF 不会右移 labels）。这不是笔误，是原实现的行为，迁移时不要"顺手修正"——
修正之后就不再是这个 baseline 了。`tests/test_eegtotext.py` 把这个接线钉住。
"""
from __future__ import annotations

import json
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

from eeg2x.frameworks.eeg2video.core.model_base import BaseModel

N_BANDS = 8
N_CHANNELS = 105
N_FEATURES = N_CHANNELS * N_BANDS          # 840
MAX_WORDS = 56                             # 官方 max_len，输入与目标等长


def _hf_cache() -> str:
    import os
    return os.environ.get("EEG2X_HF_CACHE", "/datas/hf-cache")


def _offline() -> bool:
    import os
    return os.environ.get("EEG2X_HF_OFFLINE", "1") == "1"


class WordLevelEncoder(nn.Module):
    """词级 EEG 编码器（官方 BrainTranslator.addin_forward 的等价写法）。"""

    def __init__(self, in_features: int = N_FEATURES, nhead: int = 8,
                 dim_feedforward: int = 2048, n_layers: int = 6,
                 decoder_embedding_size: int = 1024):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=in_features, nhead=nhead,
                                           dim_feedforward=dim_feedforward,
                                           batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.fc1 = nn.Linear(in_features, decoder_embedding_size)

    def forward(self, word_embeds: torch.Tensor, pad_mask: torch.Tensor | None = None):
        """word_embeds: (B, 56, 840)；pad_mask: (B, 56) bool，True = 该词槽是 padding。"""
        h = self.additional_encoder(word_embeds, src_key_padding_mask=pad_mask)
        return F.relu(self.fc1(h))          # (B, 56, d_model)


class EEGToTextModule(L.LightningModule):
    """训练用的 LightningModule，超参照 EEG-To-Text-N 的 one-step 配置。

    官方那次（实验报告2，dev loss 3.5198）用的是：`--one_step` ⇒ 跳过 step1，
    只跑 step2 = SGD(lr=5e-7, momentum=0.9) + StepLR(step_size=30, gamma=0.1)，
    batch 32，30 个 epoch，全部参数可训练。
    """

    def __init__(self, text_model: str = "facebook/bart-large",
                 max_len: int = MAX_WORDS, lr: float = 5e-7, momentum: float = 0.9,
                 lr_step: int = 30, lr_gamma: float = 0.1, nhead: int = 8,
                 dim_feedforward: int = 2048, n_layers: int = 6):
        super().__init__()
        self.save_hyperparameters()
        self.max_len = int(max_len)
        self.lr, self.momentum = lr, momentum
        self.lr_step, self.lr_gamma = lr_step, lr_gamma
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_model, cache_dir=_hf_cache(), local_files_only=_offline())
        self.text_model = AutoModelForSeq2SeqLM.from_pretrained(
            text_model, cache_dir=_hf_cache(), local_files_only=_offline())
        self.encoder = WordLevelEncoder(
            nhead=nhead, dim_feedforward=dim_feedforward, n_layers=n_layers,
            decoder_embedding_size=int(self.text_model.config.d_model))

    # ---------- 前向 ----------
    def encode(self, word_embeds: torch.Tensor, pad_mask: torch.Tensor | None = None):
        return self.encoder(word_embeds, pad_mask)

    def _labels(self, texts) -> torch.Tensor:
        enc = self.tokenizer(list(texts), padding="max_length", max_length=self.max_len,
                             truncation=True, return_tensors="pt")
        return enc.input_ids

    def _forward(self, batch):
        eeg, mask = batch["eeg"].float(), batch["mask"]
        # mask: 1 = 有效词槽（协议约定）；BART/pytorch 要的是"忽略哪些"
        pad_mask = None if mask is None else (mask == 0)
        embeds = self.encode(eeg, pad_mask)
        labels = self._labels(batch["target text"]).to(embeds.device)
        # 官方接线：EEG embedding 当**解码器**输入，labels 与它逐位置对齐（不右移）
        return self.text_model(inputs_embeds=embeds, attention_mask=mask.long(),
                               labels=labels)

    # ---------- 训练/验证 ----------
    def training_step(self, batch, batch_idx):
        out = self._forward(batch)
        self.log("loss", out.loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self._forward(batch)
        self.log("val_loss", out.loss, on_step=False, on_epoch=True, sync_dist=True)
        return out.loss

    @torch.no_grad()
    def decode(self, word_embeds: torch.Tensor, mask: torch.Tensor | None = None,
               **gen_kwargs) -> torch.Tensor:
        """预测用：官方 generate() 走的就是 inputs_embeds（解码器前缀）。"""
        pad_mask = None if mask is None else (mask == 0)
        embeds = self.encode(word_embeds, pad_mask)
        kwargs = dict(num_beams=2, min_length=0, max_length=self.max_len)
        kwargs.update(gen_kwargs)
        return self.text_model.generate(
            inputs_embeds=embeds,
            attention_mask=None if mask is None else mask.long()[:, :embeds.shape[1]],
            **kwargs)

    def configure_optimizers(self):
        opt = torch.optim.SGD((p for p in self.parameters() if p.requires_grad),
                              lr=self.lr, momentum=self.momentum)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=self.lr_step,
                                                gamma=self.lr_gamma)
        return {"optimizer": opt, "lr_scheduler": sched}


class EEGToText(BaseModel):
    """注册表外壳（与 glim_full / eeg2text 同样的处理）：真正的训练与推理在
    `run_eegtotext.py` 与 `generate_sentence_predictions.py --variant eegtotext`。"""

    def load(self):
        raise RuntimeError("Use run_eegtotext.py --mode train for EEG-To-Text.")

    def generate(self, inputs):
        raise RuntimeError("Use generate_sentence_predictions.py --variant eegtotext.")

    def train(self, config=None):
        raise RuntimeError("Use run_eegtotext.py --mode train.")
