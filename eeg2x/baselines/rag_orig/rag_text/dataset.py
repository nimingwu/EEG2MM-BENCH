"""RAG baseline 的数据层：协议 h5 + 旁挂的句向量目标。

直接复用 `model.glim_text.ZuCoH5Dataset`（划分、行序、`sample id` 全由它决定），
只做两件事：

1. **转置**：协议 dataset 交付 `(T, C)`，本 baseline 的编码器要 `(C, T)`
   （`EEGChannelNetEncoder` 的契约），在 `__getitem__` 里转掉，让 collate 出来就是 (B,C,T)。
2. **挂目标**：按行号把 `build_targets()` 的 768 维句向量挂上，键名 `target_embed`。

`expand_targets` 固定为 False:本 baseline 的目标是**整句的句向量**，不是 8 个 MTV 改写
变体（那 8 个变体是四变体框架训练时用的）。这一点写进 run config 的 deviations。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from eeg2x.frameworks.glim.text import UniqueTextBatchSampler, ZuCoH5Dataset
from eeg2x.protocol import SEED
from eeg2x.seeding import seed_worker


class RAGDataset(Dataset):
    """协议 h5 的一层薄封装：eeg 转成 (C,T)，可选地挂上句向量目标。"""

    def __init__(self, h5_dir: str, split: str,
                 targets: Optional[Dict[str, Any]] = None):
        self.split = split
        self.base = ZuCoH5Dataset(h5_dir, split, expand_targets=False)
        self.embeds = None if targets is None else targets["embeds"]
        if self.embeds is not None and len(self.embeds) != len(self.base):
            raise SystemExit(
                f"✗ {split}: 目标 {len(self.embeds)} 行 != dataset {len(self.base)} 行；"
                "两者错位会静默训出垃圾，拒绝继续")

    # UniqueTextBatchSampler 需要读它，转发给底层
    @property
    def records(self):
        return self.base.records

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = dict(self.base[idx])
        item["eeg"] = item["eeg"].transpose(0, 1).contiguous()   # (T,C) -> (C,T)
        if self.embeds is not None:
            item["target_embed"] = torch.from_numpy(self.embeds[idx])
        item["__row__"] = idx
        return item


class RAGDataModule:
    """与 `ZuCoH5DataModule` 同构，但训练集是「一行一条」且带句向量目标。

    故意不继承 LightningDataModule：本 baseline 只需要 train/val 两个 loader，
    `run_rag_text.py` 直接调 `train_dataloader()` / `val_dataloader()`，
    走 `Trainer.fit(model, train_dataloaders=…, val_dataloaders=…)`，
    避免为一个两方法的类引入多余的生命周期约定。
    """

    def __init__(self, h5_dir: str, train_targets: Optional[Dict[str, Any]] = None,
                 val_targets: Optional[Dict[str, Any]] = None,
                 bsz_train: int = 32, bsz_val: int = 24, bsz_test: int = 24,
                 num_workers: int = 2):
        self.h5_dir = h5_dir
        self.bsz_train, self.bsz_val, self.bsz_test = bsz_train, bsz_val, bsz_test
        self.num_workers = num_workers
        self._train = None
        self._val = None
        self._test = None
        self._targets = train_targets
        # 验证集也要目标向量：`validation_step` 算的就是同一个余弦损失，
        # 没挂目标就会 KeyError（第一版就是这样，冒烟时才暴露）。
        self._val_targets = val_targets

    def setup(self, stage: str = "fit") -> None:
        if stage in ("fit", None):
            self._train = RAGDataset(self.h5_dir, "train", self._targets)
            self._val = RAGDataset(self.h5_dir, "val", self._val_targets)
        if stage in ("test", None):
            self._test = RAGDataset(self.h5_dir, "test", None)

    def _loader(self, ds: RAGDataset, bsz: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, batch_sampler=UniqueTextBatchSampler(ds, bsz, shuffle, seed=SEED),
            num_workers=self.num_workers, pin_memory=True, worker_init_fn=seed_worker)

    def train_dataloader(self) -> DataLoader:
        if self._train is None:
            self.setup("fit")
        return self._loader(self._train, self.bsz_train, True)

    def val_dataloader(self) -> DataLoader:
        if self._val is None:
            self.setup("fit")
        return self._loader(self._val, self.bsz_val, False)

    def test_dataloader(self) -> DataLoader:
        if self._test is None:
            self.setup("test")
        return self._loader(self._test, self.bsz_test, False)


__all__ = ["RAGDataset", "RAGDataModule"]
