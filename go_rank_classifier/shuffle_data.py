from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def shuffle_split(prefix: Path, seed: int, block_size: int) -> None:
    moves_path = prefix.with_name(f"{prefix.name}_moves.npy")
    lengths_path = prefix.with_name(f"{prefix.name}_lengths.npy")
    labels_path = prefix.with_name(f"{prefix.name}_labels.npy")
    marker_path = prefix.with_name(f"{prefix.name}.shuffled")
    if not all(path.is_file() for path in (moves_path, lengths_path, labels_path)):
        raise FileNotFoundError(f"Missing prepared arrays for {prefix}")

    moves = np.load(moves_path, mmap_mode="r", allow_pickle=False)
    lengths = np.load(lengths_path, mmap_mode="r", allow_pickle=False)
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    count = len(labels)
    if len(moves) != count or len(lengths) != count:
        raise ValueError(f"Prepared arrays have inconsistent sizes for {prefix}")

    permutation = np.random.default_rng(seed).permutation(count)
    temporary = {
        name: prefix.with_name(f".{prefix.name}_{name}.shuffle-tmp.npy")
        for name in ("moves", "lengths", "labels")
    }
    destination_moves = np.lib.format.open_memmap(
        temporary["moves"], mode="w+", dtype=moves.dtype, shape=moves.shape
    )
    destination_lengths = np.lib.format.open_memmap(
        temporary["lengths"], mode="w+", dtype=lengths.dtype, shape=lengths.shape
    )
    destination_labels = np.lib.format.open_memmap(
        temporary["labels"], mode="w+", dtype=labels.dtype, shape=labels.shape
    )

    started = time.time()
    for start in range(0, count, block_size):
        indices = permutation[start : start + block_size]
        destination_moves[start : start + len(indices)] = moves[indices]
        destination_lengths[start : start + len(indices)] = lengths[indices]
        destination_labels[start : start + len(indices)] = labels[indices]
        print(
            f"{prefix.name}: shuffled {start + len(indices):,}/{count:,}",
            flush=True,
        )
    for array in (destination_moves, destination_lengths, destination_labels):
        array.flush()
    del destination_moves, destination_lengths, destination_labels

    os.replace(temporary["moves"], moves_path)
    os.replace(temporary["lengths"], lengths_path)
    os.replace(temporary["labels"], labels_path)
    marker_path.write_text(
        f"seed={seed}\ncount={count}\nseconds={time.time() - started:.1f}\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    for split in args.splits:
        prefix = data_dir / split
        marker = prefix.with_name(f"{prefix.name}.shuffled")
        if marker.is_file() and not args.force:
            print(f"{split}: already shuffled; skipping", flush=True)
            continue
        shuffle_split(prefix, args.seed, args.block_size)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create an in-place, globally shuffled memory-mapped cache"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=("train", "val", "test"))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--block-size", type=int, default=131_072)
    parser.add_argument("--force", action="store_true")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
