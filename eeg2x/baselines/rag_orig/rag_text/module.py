"""RAG baseline 的训练模块：EEG 编码器 + 余弦对齐损失。

论文 §II-E 的损失（式 1）：

    L = 1 - z_EEG · z_TXT / (‖z_EEG‖ ‖z_TXT‖)

就是「1 - 余弦相似度」，对 batch 取均值。目标 `z_TXT` 由 `targets.py` 预先算好
（all-mpnet-base-v2，已 L2 归一化），所以这里只需把两侧都归一化后做内积。

除了损失，还记录一个**批内 top-1 检索准确率**（`batch_top1`）：它是这套 pipeline 最直接
的健康指标 —— 编码器有没有把不同试次映射到可区分的嵌入上。随机水平是 1/batch，训练不动
就说明编码器没学到东西（这正是我们诊断 REVE 时用的那类量，见 eval_logs/diag_reve_wiring.py）。
"""
from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from eeg2x.frameworks.eeg2video.core.model_base import BaseModel
from .encoder import EEGChannelNetEncoder


class RAGTextModule(L.LightningModule):
    def __init__(self, n_channels: int, n_times: int, out_dim: int = 768,
                 width: int = 64, n_res_blocks: int = 3, dropout: float = 0.3,
                 lr: float = 1e-3, weight_decay: float = 1e-4, max_epochs: int = 200):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = EEGChannelNetEncoder(
            n_channels=n_channels, n_times=n_times, out_dim=out_dim,
            width=width, n_res_blocks=n_res_blocks, dropout=dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self._max_epochs = int(max_epochs)

    # ---------- 前向 ----------
    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        """eeg: (B, C, T) → (B, out_dim) 的 L2 归一化嵌入。"""
        return self.encoder(eeg)

    @torch.no_grad()
    def decode(self, eeg: torch.Tensor) -> torch.Tensor:
        """推理用（与 forward 同义；显式分开是为了让预测端的意图清楚）。"""
        return self.encoder(eeg)

    def _loss(self, eeg: torch.Tensor, target: torch.Tensor):
        z = self(eeg)
        t = F.normalize(target.float(), dim=-1)
        cos = (z * t).sum(dim=-1)                     # (B,)
        loss = (1.0 - cos).mean()
        with torch.no_grad():
            # 批内 top-1：随机水平 = 1/B，用它判断编码器是否真的在学
            pred = (z @ t.T).argmax(dim=-1)
            acc = (pred == torch.arange(z.shape[0], device=z.device)).float().mean()
        return loss, cos.mean(), acc

    # ---------- 训练/验证 ----------
    def training_step(self, batch, batch_idx):
        loss, cos, acc = self._loss(batch["eeg"].float(), batch["target_embed"])
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/cos", cos, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/batch_top1", acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, cos, acc = self._loss(batch["eeg"].float(), batch["target_embed"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/cos", cos, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/batch_top1", acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        # 论文 §II-E 说做过 dropout / L2 / LeakyReLU / cosine annealing 的网格搜索，
        # 但没给最终超参；这里取 AdamW + cosine annealing（论文提到用了它）
        opt = torch.optim.AdamW((p for p in self.parameters() if p.requires_grad),
                                lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self._max_epochs))
        return {"optimizer": opt, "lr_scheduler": sched}


class RAGText(BaseModel):
    """注册表外壳（与 glim_full / eeg2text / eegtotext 同样的处理）。

    真正的训练在 `run_rag_text.py`，推理在
    `generate_sentence_predictions.py --variant rag`；这里存在只是为了让 registry
    这个「单一事实源」是完整的。
    """

    def load(self):
        raise RuntimeError("Use run_rag_text.py --mode train for the RAG baseline.")

    def generate(self, inputs):
        raise RuntimeError("Use generate_sentence_predictions.py --variant rag.")

    def train(self, config=None):
        raise RuntimeError("Use run_rag_text.py --mode train.")


__all__ = ["RAGTextModule", "RAGText"]
