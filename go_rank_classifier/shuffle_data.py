from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def shuffle_split(prefix: Path, seed: int, block_size: int) -> None:
    moves_path = prefix.with_name(f"{prefix.name}_moves.npy")
    lengths_path = prefix.with_name(f"{prefix.name}_lengths.npy")
    marker_path = prefix.with_name(f"{prefix.name}.shuffled")
    optional_paths = [
        prefix.with_name(f"{prefix.name}_{name}.npy")
        for name in ("black_labels", "white_labels", "hashes", "pair_hashes")
    ]
    optional_paths = [path for path in optional_paths if path.is_file()]
    if not all(path.is_file() for path in (moves_path, lengths_path)):
        raise FileNotFoundError(f"Missing prepared arrays for {prefix}")

    moves = np.load(moves_path, mmap_mode="r", allow_pickle=False)
    lengths = np.load(lengths_path, mmap_mode="r", allow_pickle=False)
    optionals = {
        path: np.load(path, mmap_mode="r", allow_pickle=False)
        for path in optional_paths
    }
    count = len(lengths)
    if len(moves) != count or any(len(array) != count for array in optionals.values()):
        raise ValueError(f"Prepared arrays have inconsistent sizes for {prefix}")

    permutation = np.random.default_rng(seed).permutation(count)
    sources = {
        moves_path: moves,
        lengths_path: lengths,
        **optionals,
    }
    temporary = {
        path: path.with_name(f".{path.name}.shuffle-tmp.npy")
        for path in sources
    }
    destination_moves = np.lib.format.open_memmap(
        temporary[moves_path], mode="w+", dtype=moves.dtype, shape=moves.shape
    )
    destinations = {
        path: np.lib.format.open_memmap(
            temporary[path], mode="w+", dtype=array.dtype, shape=array.shape
        )
        for path, array in sources.items()
    }

    started = time.time()
    for start in range(0, count, block_size):
        indices = permutation[start : start + block_size]
        destination_moves = destinations[moves_path]
        destination_moves[start : start + len(indices)] = moves[indices]
        destinations[lengths_path][start : start + len(indices)] = lengths[indices]
        for path, source in optionals.items():
            destinations[path][start : start + len(indices)] = source[indices]
        print(
            f"{prefix.name}: shuffled {start + len(indices):,}/{count:,}",
            flush=True,
        )
    for array in destinations.values():
        array.flush()
    del destination_moves, destinations

    for path in sources:
        os.replace(temporary[path], path)
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
