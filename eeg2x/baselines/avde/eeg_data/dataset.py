"""AVDE EEG dataset backed by the benchmark's protocol-v2 HDF5 artifacts.

This replaces upstream ``eeg_data/dataset.py``, which read pickled
``preprocessed_eeg_*.npy`` files produced by AVDE's own ``eeg_data/preprocess.py``.
Everything else about the dataset -- the six-element ``__getitem__`` contract,
``prepare_image``, the whitened-signal ``clamp(-5, 5)``, the ``rearrange`` into
``(channels, 1, time)``, the CLIP text/image feature roles -- is kept identical,
so ``engine_for_finetuning``, ``trainer_eeg`` and both entry scripts work
unchanged.

Where the data now comes from
-----------------------------
The benchmark already writes a whitened, 200 Hz, 200-sample, train-only-MNN HDF5
for every encoder that needs the one-second LaBraM contract
(``*-image-labram-atm-*`` profiles).  Those files are exactly AVDE's own
preprocessing result: the paper epochs ``[-0.2, 1.0]`` s, subtracts the
``[-0.2, 0)`` s channel mean, resamples to 200 Hz and keeps ``[0, 1.0]`` s.
Reusing them keeps one preprocessing contract for every 200 Hz LaBraM consumer
instead of introducing a second, near-identical one.

The one deliberate difference from upstream: the benchmark HDF5 keeps *all*
available training repetitions rather than resampling a fixed ``max_rep``, so a
dataset with fewer repetitions is not silently duplicated and one with more is
not silently discarded.  Test data is still the per-concept average across
sessions, as the paper specifies.

Channel identity
----------------
Upstream hardcoded the 63-channel ThingsEEG montage for LaBraM's spatial
positional embedding.  The benchmark runs AVDE on datasets with 62, 63 and 32
channels, so the montage is read from the HDF5 ``channel_names`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode, transforms
from torch.utils.data import Dataset
from einops import rearrange
import open_clip
import torch.nn.functional as F

import dist
from eeg2x.rawdata import require_benchmark_output
from eeg2x.data import (
    PROFILE_CONTRACTS, discover_subject_h5, h5_fingerprint, FeatureStore,
)
from eeg2x.targets.clip import HF_CACHE

#: OpenCLIP identity used by the released AVDE code and by the benchmark's
#: ``clip_vit_h`` feature store (same model, same training transform).
CLIP_MODEL_TYPE = "ViT-H-14"
CLIP_PRETRAINED = "laion2b_s32b_b79k"


#: Set by :func:`configure` before any dataset is built.
_CONFIG: Dict[str, object] = {}
#: Set by :func:`configure` from the HDF5 manifest, consumed by the trainers.
_INPUT_CHANS: List[int] = []
_CHANNEL_NAMES: List[str] = []


@dataclass(frozen=True)
class _Row:
    """One EEG trial (train) or one averaged concept (test)."""

    file_index: int
    h5_row: int
    label: int
    stimulus_index: int
    description: str
    image_uri: str
    stimulus_id: str


def configure(
    *,
    subject: str,
    h5_dir: str | Path,
    profile_id: str,
    feature_store: str | Path,
) -> None:
    """Bind the dataset to one subject's benchmark artifacts."""
    if profile_id not in PROFILE_CONTRACTS:
        raise ValueError(f"Unregistered AVDE profile: {profile_id!r}")
    contract = PROFILE_CONTRACTS[profile_id]
    if int(contract["sampling_rate"]) != 200 or int(contract["n_samples"]) != 200:
        raise ValueError(
            "AVDE's LaBraM encoder requires the 200 Hz / 200-sample contract, "
            f"but {profile_id!r} is {contract['sampling_rate']} Hz / "
            f"{contract['n_samples']} samples")
    _CONFIG.clear()
    _CONFIG.update({
        "subject": subject,
        "h5_dir": require_benchmark_output(h5_dir),
        "profile_id": profile_id,
        "feature_store": require_benchmark_output(feature_store),
    })
    _INPUT_CHANS.clear()
    _CHANNEL_NAMES.clear()


def _require_config() -> Dict[str, object]:
    if not _CONFIG:
        raise RuntimeError(
            "AVDE dataset is not configured; call "
            "model.avde.eeg_data.dataset.configure(...) first")
    return _CONFIG


def current_input_chans() -> List[int]:
    """LaBraM ``input_chans`` for the montage of the configured dataset."""
    if not _INPUT_CHANS:
        raise RuntimeError("AVDE dataset has no channel montage yet")
    return list(_INPUT_CHANS)


def current_channel_names() -> List[str]:
    """Electrode names in the capitalisation LaBraM's ``standard_1020`` uses.

    The HDF5 stores each dataset's native spelling (``Fp1``, ``Fz``); upstream
    ``labram_utils.get_input_chans`` indexes its uppercase database directly, so
    the montage is upper-cased here rather than at every call site.
    """
    if not _CHANNEL_NAMES:
        raise RuntimeError("AVDE dataset has no channel montage yet")
    return [name.upper() for name in _CHANNEL_NAMES]


def _text(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _description(label: str) -> str:
    """Upstream derives the CLIP text from the stimulus folder name.

    The benchmark manifest stores the stimulus file name, so the same
    "drop the leading numeric id" rule is applied to ``<id>_<concept>``, with the
    trailing image index, speed tag and extension removed.
    """
    stem = os.path.splitext(os.path.basename(label))[0]
    head, sep, tail = stem.partition("_")
    text = tail if (sep and head.isdigit()) else stem
    # ``..._2`` / ``..._1b`` style presentation indices.
    body, sep2, last = text.rpartition("_")
    if sep2 and last[:1].isdigit() and len(last) <= 2:
        text = body
    return text.replace("_", " ").strip() or stem


def _montage(channel_names: Sequence[str]) -> List[int]:
    from models.labram import standard_1020

    upper = [name.upper() for name in channel_names]
    unknown = [name for name in upper if name not in standard_1020]
    if unknown:
        raise ValueError(
            "LaBraM needs every electrode in its 10-20 database; unmapped: "
            f"{unknown}")
    return [0] + [standard_1020.index(name) + 1 for name in upper]


def _read_manifest(paths: Sequence[Path]) -> Tuple[np.ndarray, List[dict]]:
    """Concatenate the EEG of every split file and collect its manifest."""
    arrays: List[np.ndarray] = []
    records: List[dict] = []
    for index, path in enumerate(paths):
        with h5py.File(path, "r") as h5:
            data = np.asarray(h5["data"][:], dtype=np.float32)
            rows = int(data.shape[0])
            ids = [_text(value) for value in h5["stimulus_id"][:]]
            uris = [_text(value) for value in h5["stimulus_uri"][:]]
            labels = [_text(value) for value in h5["event_label"][:]]
            objects = [int(value) for value in h5["event_object_number"][:]]
            stimuli = [int(value) for value in h5["event_stim_number"][:]]
            channels = [_text(value) for value in h5["channel_names"][:]]
        if data.ndim != 3:
            raise ValueError(f"{path} is not [N, C, T]: {data.shape}")
        arrays.append(data)
        records.extend({
            "file_index": index, "h5_row": row, "stimulus_id": ids[row],
            "image_uri": uris[row], "label_text": labels[row],
            "object_number": objects[row], "stimulus_number": stimuli[row],
            "channel_names": channels,
        } for row in range(rows))
    return np.concatenate(arrays, axis=0), records


def _text_features(descriptions: Sequence[str]) -> torch.Tensor:
    """Upstream ``Textencoder``: L2-normalized CLIP text features."""
    model_type = "ViT-H-14"
    device = dist.get_device()
    vlmodel, _, _ = open_clip.create_model_and_transforms(
        model_type, pretrained=CLIP_PRETRAINED, precision="fp32", device=device,
        cache_dir=str(HF_CACHE))
    tokenizer = open_clip.get_tokenizer(model_type)
    text_inputs = torch.cat([tokenizer(t) for t in descriptions]).to(device)
    with torch.no_grad():
        features = vlmodel.encode_text(text_inputs)
    return F.normalize(features, dim=-1).detach().cpu()


class EEGDataset(Dataset):
    """The upstream dataset, reading benchmark HDF5 instead of pickles."""

    def __init__(self, subjects, classes=None, pictures=None, train=True,
                 exclude_subject=None):
        if classes is not None or pictures is not None:
            raise NotImplementedError(
                "The benchmark AVDE baseline always uses the full split")
        if exclude_subject is not None:
            raise NotImplementedError("Cross-subject AVDE is out of scope")
        config = _require_config()
        subject = list(subjects)[0] if subjects else str(config["subject"])
        if subject != config["subject"]:
            raise ValueError(
                f"Dataset is configured for {config['subject']}, got {subject}")

        self.train = train
        split = "train" if train else "test"
        self.paths = discover_subject_h5(
            config["h5_dir"], subject, profile_id=str(config["profile_id"]),
            split=split)
        self.fingerprints = [h5_fingerprint(path) for path in self.paths]
        data, records = _read_manifest(self.paths)

        # Concept identity: contiguous label ids over the split's own concepts.
        objects = sorted({row["object_number"] for row in records})
        label_of = {value: index for index, value in enumerate(objects)}
        # Stimulus identity: one feature row per distinct ``stimulus_id`` -- the
        # benchmark's stable stimulus key -- ordered by concept so the row order
        # matches the label order.  ``event_stim_number`` is deliberately not
        # part of the key: on ds005106 it counts presentations within a block and
        # therefore varies across trials of the same stimulus.
        canonical: Dict[str, dict] = {}
        for row in records:
            canonical.setdefault(row["stimulus_id"], row)
        stimuli = sorted(
            canonical.values(),
            key=lambda row: (row["object_number"], row["stimulus_id"]))
        stimulus_of = {row["stimulus_id"]: index
                       for index, row in enumerate(stimuli)}
        self.stimulus_ids = [row["stimulus_id"] for row in stimuli]
        self.stimulus_uris = [row["image_uri"] for row in stimuli]

        self.rows = [_Row(
            file_index=row["file_index"], h5_row=row["h5_row"],
            label=label_of[row["object_number"]],
            stimulus_index=stimulus_of[row["stimulus_id"]],
            description=_description(row["label_text"]),
            image_uri=row["image_uri"], stimulus_id=row["stimulus_id"],
        ) for row in records]

        self.data = torch.from_numpy(data).float()
        self.ch_names = list(records[0]["channel_names"])
        self.labels = torch.tensor([row.label for row in self.rows], dtype=torch.long)
        self.n_cls = len(objects)
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject

        global _INPUT_CHANS, _CHANNEL_NAMES
        _CHANNEL_NAMES = list(self.ch_names)
        _INPUT_CHANS = _montage(self.ch_names)

        store = FeatureStore(config["feature_store"], ("clip_vit_h",))
        feature_rows = [store.row(stimulus_id) for stimulus_id in self.stimulus_ids]
        self.img_features = torch.stack(
            [store.tensor("clip_vit_h", row) for row in feature_rows])
        store.close()

        # Upstream keeps one text embedding per *concept*, indexed by label.
        concept_texts: List[str] = [""] * self.n_cls
        for row in self.rows:
            concept_texts[row.label] = row.description
        self.text = concept_texts
        self.text_features = _text_features(concept_texts)

        print(f"AVDE dataset [{split}] {subject}: eeg={tuple(self.data.shape)} "
              f"concepts={self.n_cls} stimuli={len(self.stimulus_ids)} "
              f"channels={len(self.ch_names)} profile={config['profile_id']}")

    # -- upstream helpers, unchanged -------------------------------------
    def Textencoder(self, text):
        return _text_features(text)

    def ImageEncoder(self, images):
        model_type = "ViT-H-14"
        device = dist.get_device()
        vlmodel, preprocess_train, _ = open_clip.create_model_and_transforms(
            model_type, pretrained=CLIP_PRETRAINED, precision="fp32",
            device=device, cache_dir=str(HF_CACHE))
        batch_size = 1024
        chunks = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            image_inputs = torch.stack([
                preprocess_train(Image.open(path).convert("RGB")) for path in batch
            ]).to(device)
            with torch.no_grad():
                # Not normalized: upstream removes the L2 step for reconstruction.
                chunks.append(vlmodel.encode_image(image_inputs))
        features = torch.cat(chunks, dim=0)
        print("image_features shape", features.shape)
        return features.cpu()

    def prepare_image(self, img_path: str, final_reso: int, hflip=False,
                      mid_reso=1.125):
        with open(img_path, "rb") as handle:
            img = Image.open(handle).convert("RGB")
        mid_reso = round(mid_reso * final_reso)
        train_aug, val_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(), normalize_01_into_pm1,
        ], [
            transforms.Resize(final_reso, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(), normalize_01_into_pm1,
        ]
        if hflip:
            train_aug.insert(0, transforms.RandomHorizontalFlip())
        train_aug = transforms.Compose(train_aug)
        val_aug = transforms.Compose(val_aug)
        return train_aug(img) if self.train else val_aug(img)

    # -- torch Dataset ---------------------------------------------------
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Upstream selects on `'whitened' in data_path` for both splits: the
        # benchmark HDF5 stores train-only MNN output for train *and* test, so
        # the whitened branch (clamp, never /100) always applies here.
        x = self.data[index].float().clamp(-5, 5)
        x = rearrange(x, 'N (A T) -> N A T', T=200)
        return (
            x,
            self.prepare_image(row.image_uri, final_reso=256),
            torch.tensor(row.label, dtype=torch.long),
            self.text[row.label],
            self.text_features[row.label],
            self.img_features[row.stimulus_index],
        )


def load_data(subjects, classes=None, pictures=None, train=True,
              exclude_subject=None):
    """Retained for parity with upstream call sites."""
    dataset = EEGDataset(subjects, classes, pictures, train, exclude_subject)
    return dataset.data, dataset.labels, dataset.text, dataset.stimulus_uris


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)
