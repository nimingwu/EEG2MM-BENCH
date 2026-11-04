"""词级特征的 Dataset / DataModule：在协议 h5 之上叠一层旁挂特征。

为什么要叠而不是换成新 h5 分支：EEG-To-Text 的输入是**词级频带特征**，不是协议里的
句子级时域信号。协议 h5（`zuco*_glim_seed2000_h5/`）继续做**行序 / 划分 / 文本 / 样本 id**
的权威，词级特征以 `prepare_eegtotext_words.py` 产出的 `.npz` 旁挂进来——这样：
  * 样本 id 与评测端天然一致（不依赖任何重命名约定）；
  * "这份输入不是协议信号"这件事在结构上就写清楚了，不会被误当成 v2 分支。

行序对齐：`prepare_eegtotext_words.py` 按「文件名排序 + 行号」写特征，这里用同一条规则
把 `(h5 文件, 行号)` 映射到全局行号，因此对得上；不依赖 record 的顺序，也不解析 sample id。
"""
from __future__ import annotations

from pathlib import Path

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from eeg2x.frameworks.glim.text import ZuCoH5Dataset, UniqueTextBatchSampler
from eeg2x.seeding import seed_worker
from eeg2x.protocol import SEED


class WordFeatureDataset(Dataset):
    """协议 h5 的行 + 旁挂的词级特征。"""

    def __init__(self, h5_dir: str, features_npz: str, split: str,
                 expand_targets: bool = True):
        self.base = ZuCoH5Dataset(h5_dir, split, expand_targets=expand_targets)
        data = np.load(features_npz)
        self.features, self.fmask = data["features"], data["mask"]

        # 与 prepare_eegtotext_words.py **逐行一致**的行序：整个分支（所有 split）
        # 按文件名排序、再按行号。注意不能只数本 split 的文件——npz 覆盖的是全分支，
        # 否则行号会整体错位（踩过：train 8898 行 vs npz 11113 行）。
        row_of, n = {}, 0
        for path in sorted(Path(h5_dir).glob("*_sentences.h5")):
            with h5py.File(path, "r") as h:
                cnt = int(h.attrs["n_epochs"])
            for i in range(cnt):
                row_of[(str(path), i)] = n + i
            n += cnt
        self.row_of = row_of
        if n != self.features.shape[0]:
            raise SystemExit(
                f"✗ 词级特征有 {self.features.shape[0]} 行，而 {h5_dir} 共 {n} 行；"
                f"确认 npz 是用同一个源分支生成的（prepare_eegtotext_words.py）")
        # 逐条 record 绑好行号（train 展开 8 个变体时，同一条 EEG 会被多条 record 共用）
        self.row_index = [row_of[(r["h5_path"], r["h5_index"])] for r in self.base.records]
        if len(set(self.row_index)) > n:
            raise SystemExit("✗ 行号映射异常（超出特征行数）")

        # 少数行确实拿不到词级特征（他们的 pickle 里就没有，manifest 的 gap_rows 会列出来）。
        # 这些行的 mask 全 0：训练时不该拿"没有输入"去学文本，预测时更不该凭空生成——
        # 一律跳过，评测端会如实体现为"有 GT 但没有预测"。
        keep = [i for i, r in enumerate(self.row_index) if int(self.fmask[r].sum()) > 0]
        self.dropped_rows = len(self.row_index) - len(keep)
        if self.dropped_rows:
            print(f"[WordFeatureDataset] {split}: 跳过 {self.dropped_rows} 条没有词级特征的记录"
                  f"（占 {self.dropped_rows / max(1, len(self.row_index)):.2%}）", flush=True)
        if self.dropped_rows:
            self.base.records = [self.base.records[i] for i in keep]
            self.row_index = [self.row_index[i] for i in keep]

    @property
    def records(self):
        """`UniqueTextBatchSampler` 直接读 `dataset.records[i]["text uid"]`（用于保证
        一个 batch 内文本唯一），所以必须转发——而且必须是**过滤后**的那份。"""
        return self.base.records

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        out = dict(self.base[idx])
        row = self.row_index[idx]
        out["eeg"] = torch.from_numpy(self.features[row])          # (56, 840)
        out["mask"] = torch.from_numpy(self.fmask[row].astype(np.int8))
        out["__row__"] = row
        return out


class WordFeatureDataModule(L.LightningDataModule):
    """与 ZuCoH5DataModule 同构，只是数据集换成词级。"""

    def __init__(self, h5_dir: str, features_npz: str, bsz_train: int = 32,
                 bsz_val: int = 24, bsz_test: int = 24, num_workers: int = 0):
        super().__init__()
        self.h5_dir, self.features_npz = h5_dir, features_npz
        self.bsz_train, self.bsz_val, self.bsz_test = bsz_train, bsz_val, bsz_test
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_set = WordFeatureDataset(self.h5_dir, self.features_npz, "train")
            self.val_set = WordFeatureDataset(self.h5_dir, self.features_npz, "val",
                                              expand_targets=False)
        if stage in (None, "test"):
            self.test_set = WordFeatureDataset(self.h5_dir, self.features_npz, "test",
                                               expand_targets=False)

    def _loader(self, ds, bsz, shuffle):
        return DataLoader(ds, batch_sampler=UniqueTextBatchSampler(ds, bsz, shuffle,
                                                                   seed=SEED),
                          num_workers=self.num_workers, pin_memory=True,
                          worker_init_fn=seed_worker)

    def train_dataloader(self):
        return self._loader(self.train_set, self.bsz_train, True)

    def val_dataloader(self):
        return self._loader(self.val_set, self.bsz_val, False)

    def test_dataloader(self):
        return self._loader(self.test_set, self.bsz_test, False)
