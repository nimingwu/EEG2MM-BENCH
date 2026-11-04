"""ENIGMA dataset backed by the benchmark's protocol-v2 HDF5 artifacts.

This replaces upstream ``source/dataset.py``, which read ENIGMA's own
``preprocessed_eeg_*_flat.npy`` arrays plus a parquet metadata table.  The
``DataLoaderItem`` contract seen by ``training.py`` is unchanged.

Where the data now comes from
-----------------------------
ENIGMA's own preprocessing (paper Appendix A.1) is *the same contract* the
benchmark already publishes for the ATM image-generation profiles: MNE,
epoch ``[-0.2, 1.0]`` s, baseline-corrected to the pre-stimulus window,
resampled to 250 Hz "to match the ATM-S benchmark", and a whitening matrix
estimated on the training partition only.  The benchmark therefore needs no
new profile: ENIGMA consumes the existing ``*-image-atm-*`` HDF5.

One deliberate difference from upstream: ENIGMA averages the repetitions of
each training image before training ("we average together these multiple
trials for each image presentation"), so this adapter groups the benchmark's
per-trial rows by ``stimulus_id`` and averages them.  The benchmark's test
file is already one concept-averaged row per stimulus.

Text features
-------------
Upstream keys CLIP text features by the stimulus' category name.  The
benchmark manifest carries ``event_label`` instead, so the same open_clip
``ViT-H-14/laion2b_s32b_b79k`` encoder is run over those labels here.  Image
features come from the benchmark's frozen ``clip_vit_h`` store, which is the
identical un-normalised ``encode_image`` output.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class DataLoaderItem(NamedTuple):
    eeg: torch.Tensor
    class_id: int
    subject: str
    category: str
    img_path: str
    txt_vector: torch.Tensor
    img_vector: torch.Tensor
    txt_vector_norm: torch.Tensor
    img_vector_norm: torch.Tensor


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _normalise(values: torch.Tensor) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp(min=1e-12)


class EEGDataset(Dataset):
    """One subject's benchmark split, in ENIGMA's item format."""

    def __init__(
        self,
        *,
        subject: str,
        h5_dir: str,
        profile_id: str,
        feature_store: str,
        split: str = "train",
        text_features: Dict[str, torch.Tensor] | None = None,
    ):
        import h5py
        from eeg2x.rawdata import require_benchmark_output
        from eeg2x.data import FeatureStore, discover_subject_h5

        if split not in ("train", "test"):
            raise ValueError(f"Unsupported split: {split!r}")
        self.subject = subject
        self.split = split
        self.profile_id = profile_id
        paths = discover_subject_h5(
            require_benchmark_output(h5_dir), subject, profile_id=profile_id,
            split=split)

        store = FeatureStore(require_benchmark_output(feature_store), ("clip_vit_h",))
        eeg_rows: List[np.ndarray] = []
        image_rows: List[torch.Tensor] = []
        labels: List[str] = []
        stimulus_ids: List[str] = []
        for path in paths:
            with h5py.File(path, "r") as h5:
                data = np.asarray(h5["data"][:], dtype=np.float32)
                ids = [_text(value) for value in h5["stimulus_id"][:]]
                lab = [_text(value) for value in h5["event_label"][:]]
            eeg_rows.append(data)
            labels.extend(lab)
            stimulus_ids.extend(ids)
            image_rows.extend(store.tensor("clip_vit_h", store.row(sid))
                              for sid in ids)
        store.close()
        eeg = np.concatenate(eeg_rows, axis=0)
        images = torch.stack(image_rows)

        # ENIGMA averages the repetitions of an image before training; the
        # benchmark test file is already one averaged row per stimulus.
        groups: Dict[str, List[int]] = {}
        for row, sid in enumerate(stimulus_ids):
            groups.setdefault(sid, []).append(row)
        self.stimulus_ids = sorted(
            groups, key=lambda sid: (labels[groups[sid][0]], sid))
        self.eeg_data = torch.from_numpy(np.stack([
            eeg[groups[sid]].mean(axis=0) for sid in self.stimulus_ids]))
        self.img_paths = [labels[groups[sid][0]] for sid in self.stimulus_ids]
        self.class_ids = {name: index for index, name in enumerate(
            sorted(set(self.img_paths)))}
        self.img_vectors = torch.stack([
            images[groups[sid]].mean(axis=0) for sid in self.stimulus_ids])
        self.txt_vectors = self._text_vectors(self.img_paths, text_features)
        print(f"ENIGMA dataset [{split}] {subject}: "
              f"eeg={tuple(self.eeg_data.shape)} stimuli={len(self.stimulus_ids)} "
              f"classes={len(self.class_ids)} profile={profile_id}")

    @staticmethod
    def _text_vectors(categories: Sequence[str],
                      cache: Dict[str, torch.Tensor] | None) -> torch.Tensor:
        """CLIP text features of the stimulus labels, keyed as upstream does."""
        if cache is None:
            cache = {}
        missing = sorted({name for name in categories if name not in cache})
        if missing:
            import open_clip
            from eeg2x.targets.clip import HF_CACHE
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, _ = open_clip.create_model_and_transforms(
                "ViT-H-14", pretrained="laion2b_s32b_b79k", precision="fp32",
                device=device, cache_dir=str(HF_CACHE))
            model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-H-14")
            with torch.no_grad():
                features = model.encode_text(tokenizer(missing).to(device))
            for name, vector in zip(missing, features.float().cpu()):
                cache[name] = vector
            del model
            torch.cuda.empty_cache()
        return torch.stack([cache[name] for name in categories])

    def __getitem__(self, index: int) -> DataLoaderItem:
        img_vector = self.img_vectors[index]
        txt_vector = self.txt_vectors[index]
        label = self.img_paths[index]
        return DataLoaderItem(
            eeg=self.eeg_data[index],
            class_id=self.class_ids[label],
            subject=self.subject,
            category=label,
            img_path=label,
            txt_vector=txt_vector,
            img_vector=img_vector,
            txt_vector_norm=_normalise(txt_vector.unsqueeze(0)).squeeze(0),
            img_vector_norm=_normalise(img_vector.unsqueeze(0)).squeeze(0),
        )

    def __len__(self) -> int:
        return len(self.stimulus_ids)
