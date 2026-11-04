"""Train EEGVid's encoder with the paper's triplet objective.

The paper's Appendix D, verbatim:

    "the NICE [52] architecture is used as the backbone for EEG feature
     extraction (psi in R^1024), which is unit normalized.  To train the
     network, we have used the triplet loss [46] with a multi-similarity
     miner, which helps the network learn from hard and semi-hard triplets.
     We use Adam optimizer with learning rate 3e-4 and weight decay 1e-4.
     The network is trained for 300 epochs."

and Section 3.1 on how pairs are formed:

    "a) EEG segments from the same video clip form a positive pair for video
     classification... b) EEG segments from different videos ... form negative
     pairs.  This encourages the network to learn stimulus-related features
     while reducing subject-specific noise."

Both components are the ones the authors' own EEGStyleGAN-ADA release uses
(``pytorch_metric_learning``'s ``MultiSimilarityMiner`` + ``TripletMarginLoss``),
so the objective is not a guess -- only the hyperparameters above are taken from
the EEGVid paper rather than from that release.

Scope note: the paper trains this on the encoder's own dataset split.  Here it
runs on the benchmark's train split, and the *test* split is never touched, so
the encoder is held out from the clips it will later be asked to condition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .encoder import DEFAULT_FEATURE_DIM, EEGVidEncoder

LOG = logging.getLogger("eeg2video_eegvid.train_encoder")

#: The paper's stated optimisation settings.
PAPER_LR = 3e-4
PAPER_WEIGHT_DECAY = 1e-4
PAPER_EPOCHS = 300


def load_eeg_and_labels(h5_paths: Sequence[Path]):
    """``(N, C, T)`` float32 EEG and ``(N,)`` int64 video-class labels.

    The label is the trial's concept id: on SEED-DV the five repeats of a
    concept show the *same stimulus video*, which is exactly the paper's
    "EEG segments from the same video clip form a positive pair".
    """

    import h5py

    eeg, labels = [], []
    for path in h5_paths:
        with h5py.File(Path(path), "r") as h5:
            eeg.append(np.asarray(h5["data"][:], dtype=np.float32))
            concept = np.asarray(h5["event_concept"][:], dtype=np.int64)
        labels.append(concept)
    return np.concatenate(eeg, axis=0), np.concatenate(labels, axis=0)


def train_encoder(
    h5_paths: Sequence[Path],
    out_path: Path,
    *,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    epochs: int = PAPER_EPOCHS,
    batch_size: int = 256,
    lr: float = PAPER_LR,
    weight_decay: float = PAPER_WEIGHT_DECAY,
    device: str = "cuda",
    seed: int = 0,
    log_every: int = 25,
    force: bool = False,
) -> Path:
    """Train the encoder and write a checkpoint.  Returns the checkpoint path."""

    import random

    from pytorch_metric_learning import losses, miners

    out_path = Path(out_path)
    if out_path.is_file() and not force:
        LOG.info("encoder already trained: %s", out_path)
        return out_path

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    eeg, labels = load_eeg_and_labels(h5_paths)
    n, n_chans, n_samples = eeg.shape
    LOG.info("triplet training on %d segments (%d channels x %d samples), "
             "%d distinct video classes", n, n_chans, n_samples,
             len(np.unique(labels)))
    if len(np.unique(labels)) < 2:
        raise ValueError("triplet learning needs at least two classes")

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = EEGVidEncoder(n_chans=n_chans, n_samples=n_samples,
                          feature_dim=feature_dim).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    miner = miners.MultiSimilarityMiner()
    criterion = losses.TripletMarginLoss()

    # A fixed permutation per epoch, drawn from the seeded generator: the paper
    # does not describe batching, and a deterministic order keeps the run
    # reproducible under the benchmark's seed protocol.
    rng = np.random.RandomState(seed)
    history: List[float] = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(n)
        epoch_loss = 0.0
        batches = 0
        for begin in range(0, n, batch_size):
            idx = order[begin:begin + batch_size]
            if len(idx) < 4:
                continue                      # a triplet needs a few pairs
            x = torch.from_numpy(eeg[idx]).to(dev, dtype=torch.float32)
            y = torch.from_numpy(labels[idx]).to(dev, dtype=torch.long)
            optimizer.zero_grad()
            embeddings = model(x)
            hard_pairs = miner(embeddings, y)
            loss = criterion(embeddings, y, hard_pairs)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            batches += 1
        history.append(epoch_loss / max(batches, 1))
        if epoch % log_every == 0 or epoch == epochs - 1:
            LOG.info("epoch %d/%d loss=%.4f", epoch, epochs, history[-1])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "geometry": {"n_chans": n_chans, "n_samples": n_samples,
                     "feature_dim": feature_dim},
        "labels": {"n_classes": int(len(np.unique(labels)))},
        "epochs": epochs,
        "final_loss": history[-1] if history else None,
        "loss_history": history,
        "objective": "triplet + multi-similarity miner (EEGVid arXiv:2505.21385 App. D)",
    }, out_path)
    LOG.info("wrote %s", out_path)
    return out_path


def load_encoder(path: Path, device: str = "cuda") -> EEGVidEncoder:
    """Load a checkpoint written by :func:`train_encoder`, geometry checked."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    geometry = payload["geometry"]
    model = EEGVidEncoder(n_chans=geometry["n_chans"],
                          n_samples=geometry["n_samples"],
                          feature_dim=geometry["feature_dim"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
