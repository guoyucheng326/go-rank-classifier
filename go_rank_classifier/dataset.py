from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import board_features, transform_moves


@dataclass(frozen=True)
class SplitData:
    moves: np.ndarray
    lengths: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return int(self.labels.shape[0])


def load_split(path: str | Path) -> SplitData:
    requested = Path(path)
    if requested.is_file():
        with np.load(requested, allow_pickle=False) as data:
            return SplitData(
                moves=data["moves"],
                lengths=data["lengths"],
                labels=data["labels"],
            )

    prefix = requested.with_suffix("") if requested.suffix == ".npz" else requested
    files = {
        name: prefix.with_name(f"{prefix.name}_{name}.npy")
        for name in ("moves", "lengths", "labels")
    }
    missing = [str(file) for file in files.values() if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prepared arrays: {', '.join(missing)}")
    return SplitData(
        moves=np.load(files["moves"], mmap_mode="r", allow_pickle=False),
        lengths=np.load(files["lengths"], mmap_mode="r", allow_pickle=False),
        labels=np.load(files["labels"], mmap_mode="r", allow_pickle=False),
    )


class SGFGameDataset(Dataset):
    """Build board features from compact move sequences on demand."""

    def __init__(
        self,
        data: SplitData,
        *,
        training: bool,
        seed: int = 0,
        limit: int | None = None,
    ) -> None:
        self.data = data
        self.training = training
        self.seed = seed
        self.size = len(data) if limit is None else min(len(data), limit)
        self.epoch = 0

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic augmentation stream between epochs."""
        self.epoch = epoch

    def augmentation_for(self, index: int) -> tuple[int, int]:
        length = int(self.data.lengths[index])
        if not self.training:
            return min(300, length), 0
        mixed_seed = (
            self.seed * 1_000_003
            + self.epoch * 97_409
            + int(index) * 2_654_435_761
        ) & ((1 << 63) - 1)
        rng = random.Random(mixed_seed)
        cutoff = rng.randint(100, min(300, length))
        symmetry = rng.randrange(8)
        return cutoff, symmetry

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        length = int(self.data.lengths[index])
        game_moves = self.data.moves[index, :length]
        cutoff, symmetry = self.augmentation_for(index)

        game_moves = transform_moves(game_moves[:cutoff], symmetry)
        features = board_features(game_moves.tolist(), len(game_moves))
        label = int(self.data.labels[index])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

