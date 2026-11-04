"""REVE electrode position bank, vendored from ``brain-bzh/reve-positions``.

Copied from ``reve_eeg/hf/reve-positions/position_bank.py`` (byte-identical to
the file in the released checkpoint snapshot) and adapted the same way as
:mod:`reve_model`: the ``transformers.PreTrainedModel`` base class is dropped
and the weight is loaded directly from ``model.safetensors``.

The bank is a 543 x 3 table of 3-D electrode coordinates.  Its coordinates are
**mne ``standard_1005`` metres** -- checked against ``mne`` for Fp1/Fpz/Cz/O1
and identical to the last digit, which is why :func:`channel_positions` can
fall back to mne for the few SEED montage names the bank does not list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import torch

#: Names that are the same electrode under a different recording system's
#: spelling.  Each is a naming fix rather than an approximation: the bank has
#: no coordinate for the alias, but it has one for the name it maps to, and
#: that name is the same physical position.
#:
#: ``cb1``/``cb2``  SEED-DV's 62-channel montage lists the inferior-occipital
#:                  pair under these names; the bank (and mne's standard_1005,
#:                  the coordinate source the bank was built from) calls them
#:                  ``POO7``/``POO8``.  The LUNA project makes the same
#:                  mapping.  Without it, two of SEED-DV's channels cannot be
#:                  placed.
#: ``o9``/``o10``   ds008257's extended 10-05 montage names the same pair
#:                  ``O9``/``O10`` -- the BrainVision/EGI spelling -- while the
#:                  standard 10-05 name for that position is ``I1``/``I2``.
#:                  Without it, two of EMD's 127 channels cannot be placed.
POSITION_ALIASES: Dict[str, str] = {
    "CB1": "POO7", "CB2": "POO8",
    "O9": "I1", "O10": "I2",
}

BANK_DIR = Path(__file__).resolve().parent
POSITIONS_JSON = BANK_DIR / "reve_positions.json"
POSITIONS_WEIGHTS = (
    Path("/datas/hf-cache/models--brain-bzh--reve-positions")
    / "snapshots" / "befa5b57a455b77cf302daf610c2e9ed8140bace" / "model.safetensors"
)


def _alias(name: str) -> str:
    return POSITION_ALIASES.get(str(name).upper(), name)


class RevePositionBank:
    """Name -> (x, y, z) lookup, equivalent to the released position bank."""

    def __init__(self, embedding: torch.Tensor, position_names: Sequence[str]):
        if embedding.shape != (len(position_names), 3):
            raise ValueError(
                f"position bank embedding {tuple(embedding.shape)} does not "
                f"match the {len(position_names)} declared names"
            )
        self.embedding = embedding.float()
        self.position_names = list(position_names)
        self.mapping = {name: i for i, name in enumerate(self.position_names)}

    @classmethod
    def from_release(cls, positions_json: Path = POSITIONS_JSON,
                     weights: Path = POSITIONS_WEIGHTS) -> "RevePositionBank":
        from safetensors.torch import load_file

        if not positions_json.is_file():
            raise FileNotFoundError(f"missing {positions_json}")
        if not weights.is_file():
            raise FileNotFoundError(f"missing {weights}")
        names = json.loads(positions_json.read_text(encoding="utf-8"))
        embedding = load_file(str(weights))["embedding"]
        return cls(embedding, names)

    def forward(self, channel_names: Sequence[str]) -> torch.Tensor:
        return self.embedding[self._indices(channel_names)]

    def _indices(self, channel_names: Sequence[str]) -> List[int]:
        return [self.mapping[_alias(name)] for name in channel_names]

    def channel_positions(self, channel_names: Sequence[str]) -> torch.Tensor:
        """Coordinates for ``channel_names``, in the order given.

        Unlike the released ``forward`` -- which *silently drops* unknown names
        and then returns a shorter tensor than there are EEG channels -- this
        raises, and only for a name that is missing from both the bank and the
        mne fallback.
        """

        missing = [n for n in channel_names if _alias(n) not in self.mapping]
        extra = None
        if missing:
            extra = _mne_positions(missing)
        rows = []
        for index, name in enumerate(channel_names):
            key = _alias(name)
            if key in self.mapping:
                rows.append(self.embedding[self.mapping[key]])
            else:
                rows.append(extra[name])
        return torch.stack(rows) if rows else torch.zeros(0, 3)

    def has_all(self, channel_names: Sequence[str]) -> bool:
        return all(_alias(n) in self.mapping for n in channel_names)


def _mne_positions(channel_names: Sequence[str]) -> Dict[str, torch.Tensor]:
    """Coordinates for names absent from the bank, from mne standard_1005."""

    import mne

    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]
    out: Dict[str, torch.Tensor] = {}
    still_missing = []
    for name in channel_names:
        key = _alias(name)
        if key not in ch_pos:
            still_missing.append(name)
            continue
        out[name] = torch.tensor(ch_pos[key], dtype=torch.float32)
    if still_missing:
        raise KeyError(
            f"no 3-D position for {still_missing}: absent from the REVE "
            "position bank and from mne standard_1005"
        )
    return out
