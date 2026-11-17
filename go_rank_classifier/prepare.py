from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .ranks import RANKS
from .sgf import MAX_VALID_MOVES, MIN_MOVES, TRUNCATE_MOVES, iter_sgf_bytes, parse_sgf_bytes

_FIRST_VOLUME_PATTERN = re.compile(r"\.7z(?:\.001)?$", re.IGNORECASE)


def archive_paths(rank_dir: Path) -> list[Path]:
    paths = [
        path
        for path in sorted(rank_dir.iterdir())
        if path.is_file() and _FIRST_VOLUME_PATTERN.search(path.name)
    ]
    return paths


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


def sample_rank(
    archive_root: str,
    rank: str,
    per_class: int,
    seed: int,
) -> tuple[str, np.ndarray, np.ndarray, dict[str, int]]:
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
            if len(reservoir) < per_class:
                reservoir.append(packed)
            else:
                replacement = rng.randrange(seen_accepted)
                if replacement < per_class:
                    reservoir[replacement] = packed

    moves_array = np.stack(reservoir) if reservoir else np.empty((0, TRUNCATE_MOVES), np.int16)
    lengths = np.array(
        [int(np.count_nonzero(row != -1)) for row in moves_array], dtype=np.int16
    )

    stats = {
        "seen_games": seen_games,
        "accepted_games": seen_accepted,
        "rejected_length": rejected_length,
        "rejected_board": rejected_board,
        "sample_size": len(reservoir),
        "seconds": round(time.time() - started, 3),
    }
    print(
        f"{rank:>3}: accepted={seen_accepted:,} sampled={len(reservoir):,} "
        f"seen={seen_games:,} time={stats['seconds']:.1f}s",
        flush=True,
    )
    return rank, moves_array, lengths, stats


def _split_records(
    rank: str,
    moves: np.ndarray,
    lengths: np.ndarray,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    from .ranks import RANK_TO_ID

    rng = np.random.default_rng(seed + RANK_TO_ID[rank])
    order = rng.permutation(len(moves))
    train_end = int(len(order) * 0.8)
    val_end = int(len(order) * 0.9)
    portions = (order[:train_end], order[train_end:val_end], order[val_end:])
    return (
        [moves[index] for index in portions],
        [lengths[index] for index in portions],
        [np.full(len(index), RANK_TO_ID[rank], dtype=np.int16) for index in portions],
        [len(index) for index in portions],
    )


def _concat(parts: list[np.ndarray], dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    if not parts:
        return np.empty((0, *shape), dtype=dtype)
    return np.stack(parts).astype(dtype, copy=False).reshape((-1, *shape))


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    split_moves: list[list[np.ndarray]] = [[], [], []]
    split_lengths: list[list[np.ndarray]] = [[], [], []]
    split_labels: list[list[np.ndarray]] = [[], [], []]
    stats: dict[str, dict[str, int | float]] = {}

    selected_ranks = args.ranks or RANKS
    unknown = sorted(set(selected_ranks) - set(RANKS))
    if unknown:
        raise ValueError(f"Unknown ranks: {', '.join(unknown)}")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                sample_rank,
                str(Path(args.archive_root).resolve()),
                rank,
                args.per_class,
                args.seed,
            ): rank
            for rank in selected_ranks
        }
        for future in as_completed(futures):
            rank, moves, lengths, rank_stats = future.result()
            stats[rank] = rank_stats
            local_moves, local_lengths, labels, counts = _split_records(
                rank, moves, lengths, args.seed
            )
            for split in range(3):
                split_moves[split].extend(local_moves[split])
                split_lengths[split].extend(local_lengths[split])
                split_labels[split].extend(labels[split])

    filenames = ("train.npz", "val.npz", "test.npz")
    counts: dict[str, int] = {}
    for split, filename in enumerate(filenames):
        all_moves = _concat(split_moves[split], np.dtype("int16"), (TRUNCATE_MOVES,))
        all_lengths = _concat(split_lengths[split], np.dtype("int16"), ())
        all_labels = _concat(split_labels[split], np.dtype("int16"), ())
        order = np.random.default_rng(args.seed + split).permutation(len(all_labels))
        np.savez(
            output / filename,
            moves=all_moves[order],
            lengths=all_lengths[order],
            labels=all_labels[order],
        )
        counts[filename] = int(len(all_labels))

    mapping = {str(index): rank for index, rank in enumerate(RANKS)}
    (output / "labels.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "source": "https://github.com/featurecat/go-dataset",
        "rules": {
            "min_moves": MIN_MOVES,
            "max_valid_moves": MAX_VALID_MOVES,
            "truncate_moves": TRUNCATE_MOVES,
            "per_class_cap": args.per_class,
            "board_size": 19,
        },
        "splits": counts,
        "ranks": stats,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-class", type=int, default=20_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--ranks",
        nargs="+",
        help="Optional subset such as: --ranks Pro 9d 1k 18k",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

