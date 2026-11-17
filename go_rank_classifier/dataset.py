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
    with np.load(path, allow_pickle=False) as data:
        return SplitData(
            moves=data["moves"],
            lengths=data["lengths"],
            labels=data["labels"],
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

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + int(index))
        length = int(self.data.lengths[index])
        game_moves = self.data.moves[index, :length]

        if self.training:
            cutoff = rng.randint(100, min(300, length))
            symmetry = rng.randrange(8)
        else:
            cutoff = min(300, length)
            symmetry = 0

        game_moves = transform_moves(game_moves[:cutoff], symmetry)
        features = board_features(game_moves.tolist(), len(game_moves))
        label = int(self.data.labels[index])
        return torch.from_numpy(features), torch.tensor(label, dtype=torch.long)

