"""GLIM v2 adapter for the EEG2X benchmark.

This module keeps the GLIM v2 network unchanged and only supplies a benchmark
protocol HDF5 data interface.  The benchmark HDF5 stores EEG as
``(channels, time)`` while GLIM consumes ``(time, channels)``; the conversion
is performed in :class:`ZuCoH5Dataset` and is recorded in the benchmark
preprocessing profile.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import h5py
import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader, Sampler

from eeg2x.frameworks.eeg2video.core.model_base import BaseModel
from eeg2x.protocol import SEED
from eeg2x.seeding import seed_worker

log = logging.getLogger(__name__)


def _s(x: Any) -> str:
    return x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)


def _dataset_prompt(dataset_id: str) -> str:
    """Map an HDF5 ``dataset_id`` to the prompt vocabulary's dataset key.

    The original GLIM passed ``df['dataset']`` straight through, i.e. the
    literal strings 'ZuCo1' / 'ZuCo2'.  The prompt vocabulary is
    ``['<UNK>', 'ZuCo1', 'ZuCo2']``, so a dataset outside it (Chisco) must fall
    back to '<UNK>' rather than invent a key that PromptEmbedder cannot index.

    This used to be hard-coded to 'ZuCo1', which silently mislabelled every
    ZuCo2 sample.  It only mattered once domain prompts were switched on
    (``use_null_prompt=False``), because in null mode the prompt is never read.
    """
    ds = (dataset_id or "").lower()
    if ds.startswith("zuco1"):
        return "ZuCo1"
    if ds.startswith("zuco2"):
        return "ZuCo2"
    return "<UNK>"


def _input_prefix(dataset_id: str) -> str:
    """Encoder-input prefix for a dataset.

    GLIM phrases the text-model input as a translation instruction,
    ``"To English: <sentence>"``.  That is meaningful for the English ZuCo
    corpora but must NOT be carried onto a non-English dataset: for Chisco it
    would instruct the (frozen) decoder to translate while the target text stays
    Chinese, which is incoherent.

    The choice is keyed off the ``dataset_id`` declared in the HDF5 attrs, so it
    travels with the data rather than being inferred from a path.
    """
    ds = (dataset_id or "").lower()
    if ds.startswith("zuco"):
        return "To English: "
    return ""


class ZuCoH5Dataset(Dataset):
    """Load protocol-v2 ZuCo1 sentence files.

    For training, each source row is exposed once for each of the eight target
    variants, matching the original GLIM ``GLIMDataModule`` semantics.  For
    validation/test, one row is exposed and all variants are retained for
    evaluation.  No metadata is used to construct the prompt when the model
    is in null-prompt mode.
    """

    def __init__(self, h5_dir: str, split: str, expand_targets: bool = True):
        self.split = split
        self.records: List[Dict[str, Any]] = []
        self._h5_handles: Dict[str, h5py.File] = {}
        paths = sorted(Path(h5_dir).glob("*_sentences.h5"))
        paths = [p for p in paths if self._split_from_name(p.name) == split]
        if not paths:
            raise FileNotFoundError(f"No {split} sentence HDF5 files in {h5_dir}")
        target_keys = (
            "lexical simplification (v0)", "lexical simplification (v1)",
            "semantic clarity (v0)", "semantic clarity (v1)",
            "syntax simplification (v0)", "syntax simplification (v1)",
            "naive rewritten", "naive simplified",
        )
        for path in paths:
            with h5py.File(path, "r") as h:
                n = int(h.attrs["n_epochs"])
                data_shape = tuple(h["data"].shape)
                # Prefix travels with the dataset (see _input_prefix).
                _ds_id = _s(h.attrs.get("dataset_id", h.attrs.get("dataset", "")))
                prefix = _input_prefix(_ds_id)
                ds_prompt = _dataset_prompt(_ds_id)
                sentence = [_s(v) for v in h["sentence_text"][:]]
                raw_input = [_s(v) for v in h.get("raw_input_text", h["sentence_text"])[:]]
                input_text = [_s(v) for v in h.get("input_text", h["sentence_text"])[:]]
                uid = np.asarray(h.get("text_uid", np.arange(n))[:], dtype=np.int64)
                # ``valid_eeg_samples`` is the canonical protocol field.  An
                # early writer also emitted an empty compatibility
                # ``valid_samples`` dataset, so prefer the canonical values
                # whenever they are present.
                valid_ds = h.get("valid_eeg_samples", h.get("valid_samples", None))
                valid = np.asarray(valid_ds[:] if valid_ds is not None else np.full(n, data_shape[-1]), dtype=np.int64)
                tasks = [_s(v) for v in h.get("task", np.array(["task1"] * n, dtype="S8"))[:]]
                subjects = [_s(v) for v in h.get("subject", np.array([_s(h.attrs["subject_id"])] * n, dtype="S8"))[:]]
                if "target_texts_json" in h:
                    targets = [tuple(json.loads(_s(v))) for v in h["target_texts_json"][:]]
                else:
                    targets = [(sentence[i],) for i in range(n)]
                for i in range(n):
                    all_targets = tuple(str(x) for x in targets[i])
                    variants = all_targets if (split == "train" and expand_targets) else (all_targets[:1] or (sentence[i],))
                    for target_idx, target in enumerate(variants):
                        self.records.append({
                            "h5_path": str(path), "h5_index": int(i),
                            "n_channels": int(data_shape[1]), "n_times": int(data_shape[2]),
                            "valid": int(max(0, min(valid[i], data_shape[-1]))),
                            "input text": f"{prefix}{input_text[i]}",
                            "raw input text": raw_input[i],
                            "target text": str(target),
                            "all target texts": all_targets or (sentence[i],),
                            "text uid": int(uid[i]), "subject": subjects[i], "task": tasks[i],
                            # (task, dataset, subject); the original GLIM built
                            # these from the dataframe's own columns.
                            "prompt": ("<TSR>" if tasks[i] == "task3" else "<NR>",
                                       ds_prompt, subjects[i]),
                            "sentiment label": "nan", "relation label": "nan", "raw task key": tasks[i],
                            "target index": target_idx, "source file": path.name, "source index": i,
                            "sample id": f"{path.stem}_sent-{i:04d}",
                        })
        log.info("Loaded %s %s records from %d HDF5 files", len(self.records), split, len(paths))

    @staticmethod
    def _split_from_name(name: str) -> str:
        if "_train_" in name: return "train"
        if "_val_" in name: return "val"
        if "_test_" in name: return "test"
        return "unspecified"

    def __len__(self) -> int: return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        # Keep one read-only HDF5 handle per worker process.  This avoids
        # materialising the 8 target variants (roughly 20+ GB for ZuCo1) in
        # RAM while still returning the exact float32 source values.
        path = r["h5_path"]
        h = self._h5_handles.get(path)
        if h is None:
            h = h5py.File(path, "r", swmr=True)
            self._h5_handles[path] = h
        eeg = np.asarray(h["data"][r["h5_index"]].T, dtype=np.float32)
        l = eeg.shape[0]
        m = np.zeros(l, dtype=np.int8); m[: r["valid"]] = 1
        return {"eeg": torch.from_numpy(eeg), "mask": torch.from_numpy(m),
                "sample id": r["sample id"],
                "prompt": r["prompt"], "text uid": r["text uid"],
                "input text": r["input text"], "target text": r["target text"],
                "sentiment label": r["sentiment label"], "relation label": r["relation label"],
                "raw task key": r["raw task key"], "raw input text": r["raw input text"],
                "all target texts": r["all target texts"]}


class UniqueTextBatchSampler(Sampler[List[int]]):
    """Deterministic GLIM-style batches with unique text IDs per batch."""
    def __init__(self, dataset: ZuCoH5Dataset, batch_size: int, shuffle: bool, drop_last: bool = True, seed: int = SEED):
        self.dataset, self.batch_size, self.shuffle, self.drop_last, self.seed = dataset, int(batch_size), shuffle, drop_last, int(seed)
        self.epoch = 0; self._n = self._estimate_len()

    def _make(self, epoch: int) -> List[List[int]]:
        g = torch.Generator().manual_seed(self.seed + epoch)
        idx = torch.randperm(len(self.dataset), generator=g).tolist() if self.shuffle else list(range(len(self.dataset)))
        batches: List[List[int]] = []
        # Bucket by text uid, then take at most one member from each bucket per
        # batch.  This is equivalent to GLIM's repeated non-overlapping scan
        # but O(N) instead of O(N * repetitions_per_uid).
        buckets: Dict[int, List[int]] = {}
        order: List[int] = []
        for i in idx:
            uid = self.dataset.records[i]["text uid"]
            if uid not in buckets: buckets[uid] = []; order.append(uid)
            buckets[uid].append(i)
        positions = {u: 0 for u in order}
        active = order
        while active:
            take = active[: self.batch_size]
            if len(take) < self.batch_size and self.drop_last: break
            batches.append([buckets[u][positions[u]] for u in take])
            next_active = active[self.batch_size:]
            for u in take:
                positions[u] += 1
                if positions[u] < len(buckets[u]): next_active.append(u)
            active = next_active
        return batches

    def _estimate_len(self) -> int:
        return max(1, len(self._make(0)))

    def _rank_world(self) -> tuple:
        """(rank, world_size), or (0, 1) outside distributed training.

        Needed because `use_distributed_sampler=False` is set on the Trainer, so
        Lightning does NOT wrap this sampler for DDP.  Without explicit
        sharding every rank would iterate the identical batch list and the
        GPUs would compute the same gradients on the same data.
        """
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def __len__(self) -> int:
        rank, world = self._rank_world()
        if world <= 1:
            return self._n
        # 与 __iter__ 的补齐一致：每个 rank 都跑 ceil(_n/world) 批
        return (self._n + world - 1) // world

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._make(self.epoch); self.epoch += 1
        if len(batches) < self._n and batches:
            batches = (batches * ((self._n + len(batches) - 1) // len(batches)))[: self._n]
        batches = batches[: self._n]
        # Shard AFTER the cycle-to-fill above, so `batches[rank::world]` over
        # all ranks still covers every batch exactly once -- no rank re-runs
        # another's data and none is dropped.  Every rank computes the same
        # list because `_make` is seeded by epoch alone, which is what makes
        # the strided split consistent across processes.
        rank, world = self._rank_world()
        if world > 1:
            # 每个 rank 的批数必须**完全相等**，否则 rank 之间集合通信的形状对不上，
            # 直接 NCCL 死等：已踩过——TMNRED sub12 的 val 一共 11 批，4 卡分片成
            # 3/3/3/2，rank 3 的 all_gather NumelIn 只有别人的 2/3，rank 0 在
            # ALLGATHER 上等了 30 分钟被看门狗杀掉（训练本身 914 步是齐的，所以
            # epoch 0 能跑完，死在 val 收尾）。训练侧不等也会以同样方式死。
            # 补齐方式同 Lightning 的 DistributedSampler：从头循环取。
            # 注意这意味着不能整除时，多出来的批会在某些 rank 上被**重复使用**
            # （现在只有 val 会触发：11 批 -> 3,3,3,3，多出来的那批被多算一次，
            # 影响的是训练期 full_val 诊断指标，不是最终 test 指标——评测走
            # 单卡 generate_sentence_predictions.py）。
            shard = batches[rank::world]
            per_rank = (len(batches) + world - 1) // world
            if shard and len(shard) < per_rank:
                shard = (shard * ((per_rank + len(shard) - 1) // len(shard)))[:per_rank]
            batches = shard
        yield from batches


class ZuCoH5DataModule(L.LightningDataModule):
    """LightningDataModule-compatible wrapper around protocol HDF5."""
    def __init__(self, h5_dir: str, bsz_train: int = 72, bsz_val: int = 24, bsz_test: int = 24,
                 num_workers: int = 0, expand_targets_train: bool = True):
        super().__init__()
        self.h5_dir, self.bsz_train, self.bsz_val, self.bsz_test, self.num_workers = h5_dir, bsz_train, bsz_val, bsz_test, num_workers
        # 训练集默认按 8 个 target 变体展开（benchmark 的 MTV 协议）。只有**不用文本**
        # 的训练才该关掉：EEG2Text 的掩码重建预训练只看 EEG，展开等于把同一条 EEG
        # 复制 8 遍 × 重新掩码，白跑 8 倍（实测 72624 -> 9078 条）。
        self.expand_targets_train = bool(expand_targets_train)

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_set = ZuCoH5Dataset(self.h5_dir, "train",
                                           expand_targets=self.expand_targets_train)
            self.val_set = ZuCoH5Dataset(self.h5_dir, "val", expand_targets=False); self.n_target_text = 8
        if stage in (None, "test"):
            self.test_set = ZuCoH5Dataset(self.h5_dir, "test", expand_targets=False); self.n_target_text = 8

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_sampler=UniqueTextBatchSampler(self.train_set, self.bsz_train, True, seed=SEED), num_workers=self.num_workers, pin_memory=True, worker_init_fn=seed_worker)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_sampler=UniqueTextBatchSampler(self.val_set, self.bsz_val, False, seed=SEED), num_workers=self.num_workers, pin_memory=True, worker_init_fn=seed_worker)
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_sampler=UniqueTextBatchSampler(self.test_set, self.bsz_test, False, seed=SEED), num_workers=self.num_workers, pin_memory=True, worker_init_fn=seed_worker)


class GLIMOnlyEEGV2(BaseModel):
    """Benchmark-facing GLIM v2 null-prompt model."""
    def __init__(self, **kwargs): self.config = dict(kwargs); self.net = None

    def _make_net(self):
        from .core.glim import GLIM
        cfg = {"input_eeg_len": 1280, "hidden_eeg_len": 96, "input_text_len": 96, "tgt_text_len": 64,
               "input_dim": 128, "hidden_dim": 256, "embed_dim": 1024, "text_model_id": "google/flan-t5-large",
               "model_cache_dir": "/datas/hf-cache", "local_files_only": True, "use_null_prompt": True,
               "results_dir": "./outputs/glim_only_eeg_v2", "prompt_nums": (3, 3, 31), "prompt_dropout_probs": (0., 1., 1.),
               "evaluate_prompt_embed": "src", "n_in_blocks": 6, "n_out_blocks": 6, "in_temporal_modulate": True,
               "out_is_causal": True, "prompt_tuning_len": 0, "num_heads": 8, "mlp_ratio": 4, "dropout": 0.,
               "clip_loss_weight": .5, "commitment_loss_weight": 0., "commitment_loss_key": "mse", "use_y_mask": False,
               "bsz_train": 72, "bsz_val": 24, "lr": 1e-4, "weight_decay": 0, "full_val_interval": 10, "bs_retrieval": 24}
        cfg.update(self.config); return GLIM(**cfg)

    def load(self, config=None):
        if isinstance(config, str): config = {"checkpoint": config}
        if config: self.config.update(config)
        self.net = self._make_net(); ckpt = self.config.get("checkpoint")
        if ckpt:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            missing, unexpected = self.net.load_state_dict(state.get("state_dict", state), strict=False)
            log.info("Loaded GLIM checkpoint %s (missing=%d unexpected=%d)", ckpt, len(missing), len(unexpected))
        self.net.eval()

    def generate(self, inputs: List[Any]) -> List[Dict[str, Any]]:
        raise RuntimeError("Use run_glim_benchmark.py for batched generation; it preserves GLIM masks and metrics.")
    def train(self, config=None):
        raise RuntimeError("Use run_glim_benchmark.py --mode train for the complete Lightning protocol.")
    def unload(self):
        self.net = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()


class GLIMFull(BaseModel):
    """Benchmark-facing *full* GLIM — domain prompts ENABLED.

    The difference from `GLIMOnlyEEGV2` is exactly two config values:

        use_null_prompt      True  -> False
        prompt_dropout_probs (0,1,1) -> (0,0,0)

    i.e. task, dataset and subject prompts are all read from the sample and fed
    through PromptEmbedder, instead of every sample being forced to the reserved
    <UNK> id.  That is the condition the GLIM paper actually reports, and it is
    the only thing that separates this model from the null-prompt skeleton the
    foundation-encoder variants (glim/labram/luna/reve) share.

    Everything else — EEG encoder, Q-aligner, frozen LM, loss weights, batch
    sizes — is byte-identical to `GLIMOnlyEEGV2`, so a comparison between the two
    isolates the domain-prompt contribution.

    Requires the HDF5 to carry per-sample `task` / `subject` and a `dataset_id`
    attr the prompt vocabulary can resolve; `ZuCoH5Dataset` builds the
    (task, dataset, subject) tuple from those.
    """

    def __init__(self, **kwargs): self.config = dict(kwargs); self.net = None

    def _make_net(self):
        from .core.glim import GLIM
        cfg = {"input_eeg_len": 1280, "hidden_eeg_len": 96, "input_text_len": 96, "tgt_text_len": 64,
               "input_dim": 128, "hidden_dim": 256, "embed_dim": 1024, "text_model_id": "google/flan-t5-large",
               "model_cache_dir": "/datas/hf-cache", "local_files_only": True,
               # --- the two lines that make this the full model ---
               "use_null_prompt": False,
               "prompt_dropout_probs": (0., 0., 0.),
               # ---------------------------------------------------
               "results_dir": "./outputs/glim_full", "prompt_nums": (3, 3, 31),
               "evaluate_prompt_embed": "src", "n_in_blocks": 6, "n_out_blocks": 6, "in_temporal_modulate": True,
               "out_is_causal": True, "prompt_tuning_len": 0, "num_heads": 8, "mlp_ratio": 4, "dropout": 0.,
               "clip_loss_weight": .5, "commitment_loss_weight": 0., "commitment_loss_key": "mse", "use_y_mask": False,
               "bsz_train": 72, "bsz_val": 24, "lr": 1e-4, "weight_decay": 0, "full_val_interval": 10, "bs_retrieval": 24}
        cfg.update(self.config); return GLIM(**cfg)

    def load(self, config=None):
        if isinstance(config, str): config = {"checkpoint": config}
        if config: self.config.update(config)
        self.net = self._make_net(); ckpt = self.config.get("checkpoint")
        if ckpt:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            missing, unexpected = self.net.load_state_dict(state.get("state_dict", state), strict=False)
            log.info("Loaded GLIM checkpoint %s (missing=%d unexpected=%d)", ckpt, len(missing), len(unexpected))
        self.net.eval()

    def generate(self, inputs: List[Any]) -> List[Dict[str, Any]]:
        raise RuntimeError("Use run_glim_full.py for batched generation; it preserves GLIM masks and metrics.")
    def train(self, config=None):
        raise RuntimeError("Use run_glim_full.py --mode train for the complete Lightning protocol.")
    def unload(self):
        self.net = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()


__all__ = ["GLIMOnlyEEGV2", "GLIMFull", "ZuCoH5Dataset", "ZuCoH5DataModule", "UniqueTextBatchSampler"]
