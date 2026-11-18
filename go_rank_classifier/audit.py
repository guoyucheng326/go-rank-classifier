from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .dataset import load_split
from .ranks import ID_TO_RANK


def _split_hashes(
    path: Path, block_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = load_split(path)
    hashes = np.empty(len(split), dtype=np.uint64)
    for start in range(0, len(split), block_size):
        stop = min(start + block_size, len(split))
        for offset in range(start, stop):
            length = int(split.lengths[offset])
            digest = hashlib.blake2b(
                np.asarray(split.moves[offset, :length]).tobytes(),
                digest_size=8,
            ).digest()
            hashes[offset] = int.from_bytes(digest, "little")
        print(f"{path.name}: hashed {stop:,}/{len(split):,}", flush=True)
    return hashes, np.asarray(split.black_labels), np.asarray(split.white_labels)


def audit(data_dir: str | Path, block_size: int = 131_072) -> dict:
    data_dir = Path(data_dir)
    split_names = ("train", "val", "test")
    hashes: dict[str, np.ndarray] = {}
    black_labels: dict[str, np.ndarray] = {}
    white_labels: dict[str, np.ndarray] = {}
    for split in split_names:
        hashes[split], black_labels[split], white_labels[split] = _split_hashes(
            data_dir / split, block_size
        )

    report: dict = {"splits": {}, "cross_split_duplicates": {}}
    for split in split_names:
        unique = np.unique(hashes[split])
        report["splits"][split] = {
            "samples": int(len(hashes[split])),
            "unique_move_sequences": int(len(unique)),
            "duplicate_rows": int(len(hashes[split]) - len(unique)),
        }

    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            overlap = np.intersect1d(
                hashes[left], hashes[right], assume_unique=False
            )
            report["cross_split_duplicates"][f"{left}__{right}"] = int(
                len(overlap)
            )

    combined_hashes = np.concatenate([hashes[split] for split in split_names])
    combined_black = np.concatenate([black_labels[split] for split in split_names])
    combined_white = np.concatenate([white_labels[split] for split in split_names])
    combined_labels = (
        combined_black.astype(np.int32) * len(ID_TO_RANK) + combined_white
    )
    order = np.argsort(combined_hashes, kind="stable")
    sorted_hashes = combined_hashes[order]
    starts = np.r_[
        0,
        np.flatnonzero(np.diff(sorted_hashes)) + 1,
    ]
    ends = np.r_[starts[1:], len(sorted_hashes)]
    conflicting_groups = 0
    for start, end in zip(starts, ends):
        if end - start > 1 and np.unique(combined_labels[order[start:end]]).size > 1:
            conflicting_groups += 1
    report["duplicate_groups_with_conflicting_ranks"] = int(conflicting_groups)
    report["rank_names"] = {
        str(index): rank for index, rank in ID_TO_RANK.items()
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact duplicate SGF main lines across dataset splits"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--block-size", type=int, default=131_072)
    args = parser.parse_args()
    report = audit(args.data_dir, args.block_size)
    output = Path(args.output) if args.output else Path(args.data_dir) / "audit.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
