from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .ranks import RANKS
from .sgf import MAX_VALID_MOVES, MIN_MOVES, TRUNCATE_MOVES, iter_sgf_bytes, parse_sgf_bytes
from .shuffle_data import shuffle_split

_FIRST_VOLUME_PATTERN = re.compile(r"\.7z(?:\.001)?$", re.IGNORECASE)
_SPLITS = ("train", "val", "test")


def archive_paths(rank_dir: Path) -> list[Path]:
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
    seed: int,
    shard_root: str,
) -> tuple[str, int, dict[str, int | float]]:
    """Extract games with valid BR/WR labels and no handicap."""
    rank_dir = Path(archive_root) / rank
    paths = archive_paths(rank_dir)
    if not paths:
        raise FileNotFoundError(f"No 7z archives found in {rank_dir}")

    moves_rows: list[np.ndarray] = []
    lengths: list[int] = []
    black_labels: list[int] = []
    white_labels: list[int] = []
    hashes: list[int] = []
    seen_games = 0
    rejected_length = 0
    rejected_board = 0
    rejected_labels = 0
    rejected_handicap = 0
    rejected_setup = 0
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
            if game.black_rank is None or game.white_rank is None:
                rejected_labels += 1
                continue
            if game.handicap != 0:
                rejected_handicap += 1
                continue
            if game.has_setup_stones:
                rejected_setup += 1
                continue

            truncated = np.asarray(game.truncated_moves(), dtype=np.int16)
            packed = np.full(TRUNCATE_MOVES, -1, dtype=np.int16)
            packed[: len(truncated)] = truncated
            moves_rows.append(packed)
            lengths.append(len(truncated))
            black_labels.append(int(game.black_rank))
            white_labels.append(int(game.white_rank))
            digest = hashlib.blake2b(truncated.tobytes(), digest_size=8).digest()
            hashes.append(int.from_bytes(digest, "little"))

    shard = Path(shard_root) / rank
    shard.mkdir(parents=True, exist_ok=True)
    np.save(
        shard / "moves.npy",
        np.stack(moves_rows) if moves_rows else np.empty((0, TRUNCATE_MOVES), np.int16),
    )
    np.save(shard / "lengths.npy", np.asarray(lengths, dtype=np.int16))
    np.save(shard / "black_labels.npy", np.asarray(black_labels, dtype=np.int16))
    np.save(shard / "white_labels.npy", np.asarray(white_labels, dtype=np.int16))
    np.save(shard / "hashes.npy", np.asarray(hashes, dtype=np.uint64))

    stats: dict[str, int | float] = {
        "seen_games": seen_games,
        "accepted_games": len(moves_rows),
        "rejected_length": rejected_length,
        "rejected_board": rejected_board,
        "rejected_labels": rejected_labels,
        "rejected_handicap": rejected_handicap,
        "rejected_setup": rejected_setup,
        "seconds": round(time.time() - started, 3),
    }
    print(
        f"{rank:>3}: accepted={len(moves_rows):,} seen={seen_games:,} "
        f"time={stats['seconds']:.1f}s",
        flush=True,
    )
    return rank, len(moves_rows), stats


def _concat_rank_shards(
    shard_root: Path,
    ranks: list[str],
    output: Path,
) -> dict[str, int]:
    totals = {
        rank: int(np.load(shard_root / rank / "lengths.npy", mmap_mode="r").shape[0])
        for rank in ranks
    }
    total = sum(totals.values())
    arrays = {
        "moves": np.lib.format.open_memmap(
            output / "_all_moves.npy", mode="w+", dtype=np.int16, shape=(total, TRUNCATE_MOVES)
        ),
        "lengths": np.lib.format.open_memmap(
            output / "_all_lengths.npy", mode="w+", dtype=np.int16, shape=(total,)
        ),
        "black_labels": np.lib.format.open_memmap(
            output / "_all_black_labels.npy", mode="w+", dtype=np.int16, shape=(total,)
        ),
        "white_labels": np.lib.format.open_memmap(
            output / "_all_white_labels.npy", mode="w+", dtype=np.int16, shape=(total,)
        ),
        "hashes": np.lib.format.open_memmap(
            output / "_all_hashes.npy", mode="w+", dtype=np.uint64, shape=(total,)
        ),
    }
    cursor = 0
    for rank in ranks:
        count = totals[rank]
        for name, destination in arrays.items():
            source = np.load(shard_root / rank / f"{name}.npy", mmap_mode="r")
            destination[cursor : cursor + count] = source
            del source
        cursor += count
    for array in arrays.values():
        array.flush()
    return {"total": total, **totals}


def _write_dual_splits(
    output: Path,
    seed: int,
    pre_shuffle: bool,
    shuffle_block_size: int,
) -> tuple[dict[str, int], dict[str, dict[str, int]], str]:
    moves = np.load(output / "_all_moves.npy", mmap_mode="r")
    lengths = np.load(output / "_all_lengths.npy", mmap_mode="r")
    black = np.load(output / "_all_black_labels.npy", mmap_mode="r")
    white = np.load(output / "_all_white_labels.npy", mmap_mode="r")
    hashes = np.load(output / "_all_hashes.npy", mmap_mode="r")

    _, first_indices = np.unique(hashes, return_index=True)
    first_indices.sort()
    rng = np.random.default_rng(seed)
    order = first_indices[rng.permutation(first_indices.size)]
    train_end = int(order.size * 0.8)
    val_end = int(order.size * 0.9)
    portions = {
        "train": order[:train_end],
        "val": order[train_end:val_end],
        "test": order[val_end:],
    }

    split_counts: dict[str, int] = {}
    class_counts: dict[str, dict[str, int]] = {"black": {}, "white": {}}
    fingerprint = hashlib.sha256()
    for split, indices in portions.items():
        count = len(indices)
        split_counts[split] = count
        targets = {
            "moves": np.lib.format.open_memmap(
                output / f"{split}_moves.npy",
                mode="w+",
                dtype=np.int16,
                shape=(count, TRUNCATE_MOVES),
            ),
            "lengths": np.lib.format.open_memmap(
                output / f"{split}_lengths.npy", mode="w+", dtype=np.int16, shape=(count,)
            ),
            "black_labels": np.lib.format.open_memmap(
                output / f"{split}_black_labels.npy",
                mode="w+",
                dtype=np.int16,
                shape=(count,),
            ),
            "white_labels": np.lib.format.open_memmap(
                output / f"{split}_white_labels.npy",
                mode="w+",
                dtype=np.int16,
                shape=(count,),
            ),
            "hashes": np.lib.format.open_memmap(
                output / f"{split}_hashes.npy", mode="w+", dtype=np.uint64, shape=(count,)
            ),
        }
        sources = {
            "moves": moves,
            "lengths": lengths,
            "black_labels": black,
            "white_labels": white,
            "hashes": hashes,
        }
        block = 131_072
        for start in range(0, count, block):
            chosen = indices[start : start + block]
            for name, destination in targets.items():
                values = sources[name][chosen]
                destination[start : start + len(chosen)] = values
                fingerprint.update(np.asarray(values).tobytes())
        for array in targets.values():
            array.flush()
        del targets

        split_black = np.bincount(np.asarray(black[indices]), minlength=len(RANKS))
        split_white = np.bincount(np.asarray(white[indices]), minlength=len(RANKS))
        class_counts["black"][split] = split_black.astype(int).tolist()
        class_counts["white"][split] = split_white.astype(int).tolist()
        if pre_shuffle:
            shuffle_split(output / split, seed + _SPLITS.index(split), shuffle_block_size)

    return split_counts, class_counts, fingerprint.hexdigest()


def run(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    shard_root = output / "_rank_shards"
    if shard_root.exists():
        shutil.rmtree(shard_root)
    shard_root.mkdir(parents=True)

    for path in output.glob("*.npy"):
        path.unlink()
    for name in ("manifest.json", "labels.json", "train.shuffled", "val.shuffled", "test.shuffled"):
        (output / name).unlink(missing_ok=True)

    selected_ranks = list(args.ranks or RANKS)
    unknown = sorted(set(selected_ranks) - set(RANKS))
    if unknown:
        raise ValueError(f"Unknown ranks: {', '.join(unknown)}")

    stats: dict[str, dict[str, int | float]] = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                prepare_rank,
                str(Path(args.archive_root).resolve()),
                rank,
                args.seed,
                str(shard_root),
            ): rank
            for rank in selected_ranks
        }
        for future in as_completed(futures):
            rank, _, rank_stats = future.result()
            stats[rank] = rank_stats

    combined = _concat_rank_shards(shard_root, selected_ranks, output)
    split_counts, class_counts, fingerprint = _write_dual_splits(
        output, args.seed, args.pre_shuffle, args.shuffle_block_size
    )
    shutil.rmtree(shard_root)
    for path in output.glob("_all_*.npy"):
        path.unlink()

    labels = {str(index): rank for index, rank in enumerate(RANKS)}
    (output / "labels.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "source": "https://github.com/featurecat/go-dataset",
        "format": "dual-label-npy-memmap-v2",
        "rules": {
            "min_moves": MIN_MOVES,
            "max_valid_moves": MAX_VALID_MOVES,
            "truncate_moves": TRUNCATE_MOVES,
            "both_br_wr_required": True,
            "allow_handicap": False,
            "allow_setup_stones": False,
            "pro_strength": 10.0,
            "board_size": 19,
        },
        "splits": split_counts,
        "class_counts": class_counts,
        "fingerprint": fingerprint,
        "source_counts": combined,
        "source_stats": stats,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"splits": split_counts, "fingerprint": fingerprint}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare dual-label SGF rank data")
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--pre-shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--shuffle-block-size", type=int, default=131_072)
    parser.add_argument("--ranks", nargs="+")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
