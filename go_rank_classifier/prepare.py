from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .ranks import RANKS, RANK_TO_ID
from .sgf import MAX_VALID_MOVES, MIN_MOVES, TRUNCATE_MOVES, iter_sgf_bytes, parse_sgf_bytes
from .shuffle_data import shuffle_split

_FIRST_VOLUME_PATTERN = re.compile(r"\.7z(?:\.001)?$", re.IGNORECASE)
_SPLITS = ("train", "val", "test")


def archive_paths(rank_dir: Path) -> list[Path]:
    """Return each logical archive once, ignoring follow-on .7z.002 volumes."""
    return [
        path
        for path in sorted(rank_dir.iterdir())
        if path.is_file() and _FIRST_VOLUME_PATTERN.search(path.name)
    ]


def _iter_archive_sgfs(path: Path):
    command = ["7z", "x", "-so", "-bd", str(path)]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )
    assert process.stdout is not None
    try:
        yield from iter_sgf_bytes(iter(lambda: process.stdout.read(1024 * 1024), b""))
    finally:
        return_code = process.wait()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        if return_code != 0:
            raise RuntimeError(f"7z failed for {path}: {stderr.strip()}")


def prepare_rank(
    archive_root: str,
    rank: str,
    per_class: int,
    seed: int,
    shard_root: str,
) -> tuple[str, dict[str, int], dict[str, int | float]]:
    """Stream one rank and write mmap-friendly train/val/test shards."""
    rank_dir = Path(archive_root) / rank
    paths = archive_paths(rank_dir)
    if not paths:
        raise FileNotFoundError(f"No 7z archives found in {rank_dir}")

    rng = random.Random(seed)
    reservoir: list[np.ndarray] = []
    seen_accepted = 0
    seen_games = 0
    rejected_length = 0
    rejected_board = 0
    started = time.time()

    for archive in paths:
        for raw in _iter_archive_sgfs(archive):
            seen_games += 1
            game = parse_sgf_bytes(raw)
            if game.board_size != 19:
                rejected_board += 1
                continue
            if not (MIN_MOVES <= game.move_count <= MAX_VALID_MOVES):
                rejected_length += 1
                continue

            seen_accepted += 1
            packed = np.full(TRUNCATE_MOVES, -1, dtype=np.int16)
            moves = game.truncated_moves()
            packed[: len(moves)] = moves

            if per_class <= 0:
                reservoir.append(packed)
            elif len(reservoir) < per_class:
                reservoir.append(packed)
            else:
                replacement = rng.randrange(seen_accepted)
                if replacement < per_class:
                    reservoir[replacement] = packed

    moves = np.stack(reservoir) if reservoir else np.empty((0, TRUNCATE_MOVES), np.int16)
    lengths = np.count_nonzero(moves != -1, axis=1).astype(np.int16)
    labels = np.full(len(moves), RANK_TO_ID[rank], dtype=np.int16)

    split_rng = np.random.default_rng(seed + RANK_TO_ID[rank])
    order = split_rng.permutation(len(moves))
    train_end = int(len(order) * 0.8)
    val_end = int(len(order) * 0.9)
    portions = {
        "train": order[:train_end],
        "val": order[train_end:val_end],
        "test": order[val_end:],
    }

    counts: dict[str, int] = {}
    rank_shard = Path(shard_root) / rank
    for split, indices in portions.items():
        destination = rank_shard / split
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "moves.npy", moves[indices], allow_pickle=False)
        np.save(destination / "lengths.npy", lengths[indices], allow_pickle=False)
        np.save(destination / "labels.npy", labels[indices], allow_pickle=False)
        counts[split] = int(len(indices))

    stats: dict[str, int | float] = {
        "seen_games": seen_games,
        "accepted_games": seen_accepted,
        "sampled_games": len(moves),
        "rejected_length": rejected_length,
        "rejected_board": rejected_board,
        "seconds": round(time.time() - started, 3),
    }
    print(
        f"{rank:>3}: accepted={seen_accepted:,} stored={len(moves):,} "
        f"seen={seen_games:,} time={stats['seconds']:.1f}s",
        flush=True,
    )
    return rank, counts, stats


def _combine_shards(
    output: Path,
    shard_root: Path,
    counts: dict[str, dict[str, int]],
    ranks: list[str],
) -> None:
    """Combine per-rank shards without loading the full corpus into memory."""
    for split in _SPLITS:
        total = sum(counts.get(rank, {}).get(split, 0) for rank in ranks)
        arrays = {
            "moves": np.lib.format.open_memmap(
                output / f"{split}_moves.npy",
                mode="w+",
                dtype=np.int16,
                shape=(total, TRUNCATE_MOVES),
            ),
            "lengths": np.lib.format.open_memmap(
                output / f"{split}_lengths.npy",
                mode="w+",
                dtype=np.int16,
                shape=(total,),
            ),
            "labels": np.lib.format.open_memmap(
                output / f"{split}_labels.npy",
                mode="w+",
                dtype=np.int16,
                shape=(total,),
            ),
        }
        cursor = 0
        for rank in ranks:
            count = counts.get(rank, {}).get(split, 0)
            if count == 0:
                continue
            shard = shard_root / rank / split
            for name, destination in arrays.items():
                source = np.load(shard / f"{name}.npy", mmap_mode="r", allow_pickle=False)
                destination[cursor : cursor + count] = source
                del source
            cursor += count
        for array in arrays.values():
            array.flush()
        del arrays


def run(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shard_root = output / "_rank_shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True)

    for legacy in (
        "labels.json",
        "manifest.json",
        "train.npz",
        "val.npz",
        "test.npz",
        "train.shuffled",
        "val.shuffled",
        "test.shuffled",
        "train_moves.npy",
        "train_lengths.npy",
        "train_labels.npy",
        "val_moves.npy",
        "val_lengths.npy",
        "val_labels.npy",
        "test_moves.npy",
        "test_lengths.npy",
        "test_labels.npy",
    ):
        (output / legacy).unlink(missing_ok=True)

    selected_ranks = list(args.ranks or RANKS)
    unknown = sorted(set(selected_ranks) - set(RANKS))
    if unknown:
        raise ValueError(f"Unknown ranks: {', '.join(unknown)}")

    counts: dict[str, dict[str, int]] = {}
    stats: dict[str, dict[str, int | float]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                prepare_rank,
                str(Path(args.archive_root).resolve()),
                rank,
                args.per_class,
                args.seed,
                str(shard_root),
            ): rank
            for rank in selected_ranks
        }
        for future in as_completed(futures):
            rank, rank_counts, rank_stats = future.result()
            counts[rank] = rank_counts
            stats[rank] = rank_stats

    _combine_shards(output, shard_root, counts, selected_ranks)
    shutil.rmtree(shard_root)

    if args.pre_shuffle:
        for offset, split in enumerate(_SPLITS):
            shuffle_split(
                output / split,
                args.seed + offset,
                args.shuffle_block_size,
            )

    split_counts = {
        split: sum(counts.get(rank, {}).get(split, 0) for rank in selected_ranks)
        for split in _SPLITS
    }
    class_counts = {rank: counts[rank] for rank in RANKS if rank in counts}
    mapping = {str(index): rank for index, rank in enumerate(RANKS)}
    (output / "labels.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source": "https://github.com/featurecat/go-dataset",
        "format": "npy-memmap",
        "rules": {
            "min_moves": MIN_MOVES,
            "max_valid_moves": MAX_VALID_MOVES,
            "truncate_moves": TRUNCATE_MOVES,
            "per_class_cap": args.per_class,
            "board_size": 19,
        },
        "splits": split_counts,
        "class_counts": class_counts,
        "ranks": stats,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"splits": split_counts, "class_counts": class_counts}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare SGF rank training data")
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--per-class",
        type=int,
        default=0,
        help="Maximum games per rank; 0 keeps every accepted game (default).",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pre-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create a globally shuffled cache for sequential batch reads.",
    )
    parser.add_argument("--shuffle-block-size", type=int, default=131_072)
    parser.add_argument(
        "--ranks",
        nargs="+",
        help="Optional subset such as: --ranks Pro 9d 1k 18k",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
