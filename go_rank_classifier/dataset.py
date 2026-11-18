from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .features import board_features, transform_moves


@dataclass(frozen=True)
class SplitData:
    moves: np.ndarray
    lengths: np.ndarray
    black_labels: np.ndarray
    white_labels: np.ndarray
    shuffled: bool = False

    def __len__(self) -> int:
        return int(self.black_labels.shape[0])


@dataclass(frozen=True)
class KataGoFeatureData:
    black: np.ndarray
    white: np.ndarray
    mask: np.ndarray

    def __len__(self) -> int:
        return int(self.black.shape[0])

    @staticmethod
    def load(prefix: str | Path) -> "KataGoFeatureData":
        prefix = Path(prefix)
        black = np.load(
            prefix.with_name(f"{prefix.name}_katago_black.npy"), mmap_mode="r"
        )
        white = np.load(
            prefix.with_name(f"{prefix.name}_katago_white.npy"), mmap_mode="r"
        )
        mask_path = prefix.with_name(f"{prefix.name}_katago_mask.npy")
        mask = (
            np.load(mask_path, mmap_mode="r")
            if mask_path.is_file()
            else np.ones(black.shape[0], dtype=bool)
        )
        if black.shape[0] != white.shape[0]:
            raise ValueError("Black and white KataGo feature counts differ")
        if mask.shape[0] != black.shape[0]:
            raise ValueError("KataGo mask does not match feature rows")
        return KataGoFeatureData(black=black, white=white, mask=mask)


def load_split(path: str | Path) -> SplitData:
    requested = Path(path)
    if requested.is_file():
        with np.load(requested, allow_pickle=False) as data:
            return SplitData(
                moves=data["moves"],
                lengths=data["lengths"],
                black_labels=data["black_labels"],
                white_labels=data["white_labels"],
                shuffled=False,
            )

    prefix = requested.with_suffix("") if requested.suffix == ".npz" else requested
    files = {
        name: prefix.with_name(f"{prefix.name}_{name}.npy")
        for name in ("moves", "lengths", "black_labels", "white_labels")
    }
    missing = [str(file) for file in files.values() if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing prepared arrays: {', '.join(missing)}")
    return SplitData(
        moves=np.load(files["moves"], mmap_mode="r", allow_pickle=False),
        lengths=np.load(files["lengths"], mmap_mode="r", allow_pickle=False),
        black_labels=np.load(files["black_labels"], mmap_mode="r", allow_pickle=False),
        white_labels=np.load(files["white_labels"], mmap_mode="r", allow_pickle=False),
        shuffled=prefix.with_name(f"{prefix.name}.shuffled").is_file(),
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
        fixed_cutoff: int | None = None,
        katago: KataGoFeatureData | None = None,
    ) -> None:
        self.data = data
        self.training = training
        self.seed = seed
        self.size = len(data) if limit is None else min(len(data), limit)
        self.epoch = 0
        self.fixed_cutoff = fixed_cutoff
        self.katago = katago
        self.indices = None
        if katago is not None:
            self.indices = np.flatnonzero(np.asarray(katago.mask))
            self.size = min(self.size, len(self.indices))
        if katago is not None and len(katago) != len(data):
            raise ValueError("KataGo feature rows do not match the SGF split")

    def __len__(self) -> int:
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Change the deterministic augmentation stream between epochs."""
        self.epoch = epoch

    def augmentation_for(self, index: int) -> tuple[int, int]:
        length = int(self.data.lengths[index])
        if not self.training:
            cutoff = min(self.fixed_cutoff or 300, length)
            if cutoff < 100:
                raise ValueError(f"Game {index} is shorter than 100 moves")
            return cutoff, 0
        mixed_seed = (
            self.seed * 1_000_003
            + self.epoch * 97_409
            + int(index) * 2_654_435_761
        ) & ((1 << 63) - 1)
        rng = random.Random(mixed_seed)
        cutoff = rng.randint(100, min(300, length))
        symmetry = rng.randrange(8)
        return cutoff, symmetry

    def __getitem__(self, index: int):
        data_index = int(self.indices[index]) if self.indices is not None else index
        length = int(self.data.lengths[data_index])
        game_moves = self.data.moves[data_index, :length]
        cutoff, symmetry = self.augmentation_for(data_index)

        game_moves = transform_moves(game_moves[:cutoff], symmetry)
        features = board_features(game_moves.tolist(), len(game_moves))
        black_label = torch.tensor(
            int(self.data.black_labels[data_index]), dtype=torch.long
        )
        white_label = torch.tensor(
            int(self.data.white_labels[data_index]), dtype=torch.long
        )
        if self.katago is None:
            return torch.from_numpy(features), black_label, white_label
        black = np.asarray(self.katago.black[data_index], dtype=np.float32)
        white = np.asarray(self.katago.white[data_index], dtype=np.float32)
        tabular = np.concatenate((black, white, black - white))
        return (
            torch.from_numpy(features),
            torch.from_numpy(tabular),
            black_label,
            white_label,
        )


class DeterministicBatchSampler(Sampler[list[int]]):
    """Produce a reproducible shuffled epoch that can resume at a batch offset."""

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        epoch: int,
        seed: int,
        *,
        start_batch: int = 0,
        drop_last: bool = True,
    ) -> None:
        if data_size <= 0 or batch_size <= 0:
            raise ValueError("data_size and batch_size must be positive")
        self.data_size = data_size
        self.batch_size = batch_size
        self.start_batch = max(0, start_batch)
        rng = np.random.default_rng(
            (seed * 1_000_003 + epoch * 97_409) & ((1 << 63) - 1)
        )
        full_size = (data_size // batch_size) * batch_size if drop_last else data_size
        self.order = rng.permutation(data_size)[:full_size]
        self.num_batches = (full_size + batch_size - 1) // batch_size

    def __iter__(self):
        for batch_index in range(self.start_batch, self.num_batches):
            start = batch_index * self.batch_size
            yield self.order[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return max(0, self.num_batches - self.start_batch)


class PreShuffledBatchSampler(Sampler[list[int]]):
    """Read an already-shuffled cache sequentially and rotate each epoch."""

    def __init__(
        self,
        data_size: int,
        batch_size: int,
        epoch: int,
        seed: int,
        *,
        start_batch: int = 0,
        drop_last: bool = True,
    ) -> None:
        if data_size <= 0 or batch_size <= 0:
            raise ValueError("data_size and batch_size must be positive")
        usable = (data_size // batch_size) * batch_size if drop_last else data_size
        self.num_batches = (usable + batch_size - 1) // batch_size
        rng = np.random.default_rng(
            (seed * 1_000_003 + epoch * 97_409) & ((1 << 63) - 1)
        )
        self.rotation = int(rng.integers(0, max(self.num_batches, 1)))
        self.usable = usable
        self.batch_size = batch_size
        self.start_batch = max(0, start_batch)

    def __iter__(self):
        for offset in range(self.start_batch, self.num_batches):
            batch_index = (self.rotation + offset) % self.num_batches
            start = batch_index * self.batch_size
            yield range(start, min(start + self.batch_size, self.usable))

    def __len__(self) -> int:
        return max(0, self.num_batches - self.start_batch)

