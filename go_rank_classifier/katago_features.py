from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import load_split
from .ranks import RANKS

FEATURE_NAMES: tuple[str, ...] = (
    "samples",
    "candidate_coverage",
    "top1_rate",
    "top3_rate",
    "top5_rate",
    "mean_played_rank",
    "mean_visit_ratio",
    "mean_played_visit_share",
    "mean_winrate_loss",
    "median_winrate_loss",
    "p90_winrate_loss",
    "mean_score_loss",
    "median_score_loss",
    "p90_score_loss",
    "blunder_rate",
    "severe_blunder_rate",
    "mean_policy_entropy",
    "mean_root_winrate",
    "mean_root_score_lead",
    "early_winrate_loss",
    "middle_winrate_loss",
    "late_winrate_loss",
    "early_score_loss",
    "middle_score_loss",
    "late_score_loss",
)

GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"


def move_to_gtp(move: int) -> str:
    if move == 361:
        return "pass"
    row, column = divmod(int(move), 19)
    return f"{GTP_COLUMNS[column]}{19 - row}"


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_positions(length: int, color: str, count: int) -> list[int]:
    parity = 0 if color == "B" else 1
    end = min(300, length)
    positions = list(range(parity, end, 2))
    if count <= 0 or len(positions) <= count:
        return positions
    picks = np.linspace(0, len(positions) - 1, count).round().astype(int)
    return sorted({positions[int(index)] for index in picks})


def make_query(
    game_id: str,
    moves: np.ndarray,
    turns: list[int],
    *,
    rules: str,
    komi: float,
    max_visits: int,
) -> dict:
    gtp_moves = [
        ["B" if index % 2 == 0 else "W", move_to_gtp(int(move))]
        for index, move in enumerate(moves)
        if int(move) >= 0
    ]
    return {
        "id": game_id,
        "moves": gtp_moves,
        "initialStones": [],
        "rules": rules,
        "komi": komi,
        "boardXSize": 19,
        "boardYSize": 19,
        "analyzeTurns": turns,
        "maxVisits": max_visits,
        "includeOwnership": False,
    }


def _candidate_by_move(move_infos: list[dict], move: str) -> tuple[dict | None, int]:
    for index, info in enumerate(move_infos):
        if str(info.get("move", "")).upper() == move.upper():
            return info, index
    return None, len(move_infos)


def parse_move_sample(
    response: dict,
    *,
    played_move: int,
    max_move: int,
) -> dict:
    move_infos = list(response.get("moveInfos") or [])
    best = move_infos[0] if move_infos else {}
    played_gtp = move_to_gtp(played_move)
    played, rank = _candidate_by_move(move_infos, played_gtp)
    played = played or {}
    played_found = bool(played)
    best_visits = float(best.get("visits", 0.0))
    played_visits = float(played.get("visits", 0.0))
    best_winrate = float(best.get("winrate", 0.0))
    played_winrate = float(played.get("winrate", 0.0))
    best_score = float(best.get("scoreLead", 0.0))
    played_score = float(played.get("scoreLead", 0.0))
    visits = [max(0.0, float(info.get("visits", 0.0))) for info in move_infos]
    total_visits = sum(visits)
    probabilities = [value / total_visits for value in visits] if total_visits else []
    entropy = (
        -sum(value * math.log(value) for value in probabilities if value > 0)
        / math.log(len(probabilities))
        if len(probabilities) > 1
        else 0.0
    )
    move_number = int(response.get("turnNumber", 0))
    progress = move_number / max(max_move - 1, 1)
    phase = "early" if progress < 1 / 3 else "middle" if progress < 2 / 3 else "late"
    return {
        "move_number": move_number,
        "color": response.get("_color", "B"),
        "played_move_found": played_found,
        "played_rank": int(rank),
        "top1": bool(rank == 0),
        "top3": bool(rank < 3),
        "top5": bool(rank < 5),
        "best_visits": best_visits,
        "played_visits": played_visits,
        "visit_ratio": min(1.0, played_visits / best_visits) if best_visits else 0.0,
        "played_visit_share": played_visits / total_visits if total_visits else 0.0,
        "winrate_loss": max(0.0, best_winrate - played_winrate),
        "score_loss": max(0.0, best_score - played_score),
        "root_winrate": float((response.get("rootInfo") or {}).get("winrate", 0.0)),
        "root_score_lead": float(
            (response.get("rootInfo") or {}).get("scoreLead", 0.0)
        ),
        "policy_entropy": float(entropy),
        "phase": phase,
    }


def aggregate_player(samples: list[dict]) -> np.ndarray:
    if not samples:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    found = [item for item in samples if item["played_move_found"]]
    metric_samples = found if found else []
    winrate_loss = np.asarray([item["winrate_loss"] for item in metric_samples])
    score_loss = np.asarray([item["score_loss"] for item in metric_samples])

    def phase_mean(phase: str, metric: str) -> float:
        values = [
            item[metric] for item in metric_samples if item["phase"] == phase
        ]
        return float(np.mean(values)) if values else 0.0

    values = np.asarray(
        [
            len(samples),
            len(found) / len(samples) if samples else 0.0,
            np.mean([item["top1"] for item in metric_samples])
            if metric_samples
            else 0.0,
            np.mean([item["top3"] for item in metric_samples])
            if metric_samples
            else 0.0,
            np.mean([item["top5"] for item in metric_samples])
            if metric_samples
            else 0.0,
            np.mean([item["played_rank"] for item in metric_samples])
            if metric_samples
            else 0.0,
            np.mean([item["visit_ratio"] for item in metric_samples])
            if metric_samples
            else 0.0,
            np.mean([item["played_visit_share"] for item in metric_samples])
            if metric_samples
            else 0.0,
            winrate_loss.mean() if winrate_loss.size else 0.0,
            np.median(winrate_loss) if winrate_loss.size else 0.0,
            np.percentile(winrate_loss, 90) if winrate_loss.size else 0.0,
            score_loss.mean() if score_loss.size else 0.0,
            np.median(score_loss) if score_loss.size else 0.0,
            np.percentile(score_loss, 90) if score_loss.size else 0.0,
            np.mean(winrate_loss >= 0.10) if winrate_loss.size else 0.0,
            np.mean(winrate_loss >= 0.25) if winrate_loss.size else 0.0,
            np.mean([item["policy_entropy"] for item in metric_samples])
            if metric_samples
            else 0.0,
            np.mean([item["root_winrate"] for item in metric_samples])
            if metric_samples
            else 0.0,
            np.mean([item["root_score_lead"] for item in metric_samples])
            if metric_samples
            else 0.0,
            phase_mean("early", "winrate_loss"),
            phase_mean("middle", "winrate_loss"),
            phase_mean("late", "winrate_loss"),
            phase_mean("early", "score_loss"),
            phase_mean("middle", "score_loss"),
            phase_mean("late", "score_loss"),
        ],
        dtype=np.float32,
    )
    return values


def _select_games(
    black_labels: np.ndarray,
    white_labels: np.ndarray,
    games_per_class: int,
    seed: int,
) -> np.ndarray:
    if games_per_class <= 0:
        return np.arange(len(black_labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(len(RANKS)):
        indices = np.flatnonzero(
            (black_labels == class_id) | (white_labels == class_id)
        )
        if indices.size:
            count = min(games_per_class, indices.size)
            selected.append(rng.choice(indices, size=count, replace=False))
    return np.sort(np.concatenate(selected)) if selected else np.empty(0, np.int64)


def generate(
    data_dir: str | Path,
    split: str,
    output_dir: str | Path,
    *,
    katago_bin: str | None,
    model: str | None,
    config: str | None,
    rules: str,
    komi: float,
    max_visits: int,
    positions_per_player: int,
    games_per_class: int,
    job_size: int,
    seed: int,
    responses: str | Path | None = None,
    keep_queries: bool = False,
) -> dict:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_split(data_dir / split)
    selected = _select_games(
        data.black_labels,
        data.white_labels,
        games_per_class,
        seed,
    )
    if responses and len(selected) > job_size:
        raise ValueError("--responses requires all selected games to fit in one job")
    moves_all = np.asarray(data.moves)
    lengths = np.asarray(data.lengths)

    black_features = np.zeros((len(data), len(FEATURE_NAMES)), dtype=np.float32)
    white_features = np.zeros_like(black_features)
    mask = np.zeros(len(data), dtype=bool)
    per_move_dir = output_dir / f"{split}_per_move"
    per_move_dir.mkdir(exist_ok=True)

    query_paths: list[Path] = []
    response_paths: list[Path] = []
    for job_index, start in enumerate(range(0, len(selected), job_size)):
        job_indices = selected[start : start + job_size]
        query_path = output_dir / f"{split}_queries_{job_index:05d}.jsonl"
        response_path = (
            Path(responses)
            if responses and len(selected) <= job_size
            else output_dir / f"{split}_responses_{job_index:05d}.jsonl"
        )
        expected_turns: dict[str, list[int]] = {}
        with query_path.open("w", encoding="utf-8") as handle:
            for game_index in job_indices:
                length = min(300, int(lengths[game_index]))
                turns = sorted(
                    set(select_positions(length, "B", positions_per_player))
                    | set(select_positions(length, "W", positions_per_player))
                )
                game_id = f"{split}:{game_index}"
                expected_turns[game_id] = turns
                query = make_query(
                    game_id,
                    moves_all[game_index, :length],
                    turns,
                    rules=rules,
                    komi=komi,
                    max_visits=max_visits,
                )
                handle.write(json.dumps(query, separators=(",", ":")) + "\n")
        query_paths.append(query_path)

        if responses is None:
            if not all((katago_bin, model, config)):
                raise ValueError("--katago-bin, --model, and --config are required")
            command = [
                str(katago_bin),
                "analysis",
                "-model",
                str(model),
                "-config",
                str(config),
                "-override-config",
                (
                    f"maxVisits={max_visits},analysisPVLen=15,"
                    "includeOwnership=false,allowResignation=false,"
                    "reportAnalysisWinratesAs=SIDETOMOVE,"
                    "rootNoiseEnabled=false,wideRootNoise=0"
                ),
            ]
            with query_path.open("rb") as query_handle, response_path.open(
                "wb"
            ) as response_handle:
                subprocess.run(
                    command,
                    stdin=query_handle,
                    stdout=response_handle,
                    check=True,
                )
        response_paths.append(response_path)

        samples: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: {"B": [], "W": []}
        )
        with response_path.open(encoding="utf-8") as handle:
            for line in handle:
                response = json.loads(line)
                if "error" in response:
                    raise RuntimeError(
                        f"KataGo analysis error for {response.get('id', 'unknown')}: "
                        f"{response['error']} ({response.get('field', 'unknown field')})"
                    )
                game_id = str(response["id"])
                if game_id not in expected_turns:
                    continue
                turn = int(response.get("turnNumber", response.get("turn", -1)))
                if turn < 0:
                    raise RuntimeError(
                        f"KataGo response for {game_id} has no turnNumber"
                    )
                game_index = int(game_id.split(":")[-1])
                played = int(moves_all[game_index, turn])
                color = "B" if turn % 2 == 0 else "W"
                response["_color"] = color
                sample = parse_move_sample(
                    response,
                    played_move=played,
                    max_move=min(300, int(lengths[game_index])),
                )
                samples[game_id][color].append(sample)

        move_rows = []
        for game_id, turns in expected_turns.items():
            game_index = int(game_id.split(":")[-1])
            complete = all(
                any(sample["move_number"] == turn for sample in samples[game_id][color])
                for turn in turns
                for color in ("B" if turn % 2 == 0 else "W",)
            )
            black_features[game_index] = aggregate_player(samples[game_id]["B"])
            white_features[game_index] = aggregate_player(samples[game_id]["W"])
            mask[game_index] = complete
            for color in ("B", "W"):
                for sample in samples[game_id][color]:
                    move_rows.append((game_index, color, sample))

        move_dtypes = {
            "game_index": np.int32,
            "move_number": np.uint16,
            "color": np.uint8,
            "played_rank": np.uint16,
            "top1": np.bool_,
            "top3": np.bool_,
            "top5": np.bool_,
            "best_visits": np.uint32,
            "played_visits": np.uint32,
            "visit_ratio": np.float16,
            "played_visit_share": np.float16,
            "winrate_loss": np.float16,
            "score_loss": np.float16,
            "root_winrate": np.float16,
            "root_score_lead": np.float16,
            "policy_entropy": np.float16,
        }
        arrays = {}
        for name, dtype in move_dtypes.items():
            if name == "game_index":
                arrays[name] = np.asarray([row[0] for row in move_rows], dtype=dtype)
            elif name == "color":
                arrays[name] = np.asarray(
                    [0 if row[1] == "B" else 1 for row in move_rows], dtype=dtype
                )
            else:
                arrays[name] = np.asarray(
                    [row[2][name] for row in move_rows], dtype=dtype
                )
        np.savez_compressed(per_move_dir / f"part-{job_index:05d}.npz", **arrays)
        if not keep_queries:
            query_path.unlink(missing_ok=True)
            if responses is None:
                response_path.unlink(missing_ok=True)

    np.save(output_dir / f"{split}_katago_black.npy", black_features)
    np.save(output_dir / f"{split}_katago_white.npy", white_features)
    np.save(output_dir / f"{split}_katago_mask.npy", mask)
    (output_dir / f"{split}_katago_feature_names.json").write_text(
        json.dumps(list(FEATURE_NAMES), indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "split": split,
        "feature_names": list(FEATURE_NAMES),
        "selected_games": int(len(selected)),
        "complete_games": int(mask.sum()),
        "positions_per_player": positions_per_player,
        "max_visits": max_visits,
        "rules": rules,
        "komi": komi,
        "katago_sha256": _file_sha256(katago_bin),
        "model_sha256": _file_sha256(model),
        "config_sha256": _file_sha256(config),
        "seed": seed,
        "data_fingerprint": json.loads(
            (data_dir / "manifest.json").read_text(encoding="utf-8")
        ).get("fingerprint")
        if (data_dir / "manifest.json").is_file()
        else None,
    }
    (output_dir / f"{split}_katago_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def generate_resumable(
    data_dir: str | Path,
    split: str,
    output_dir: str | Path,
    *,
    katago_bin: str | None,
    model: str | None,
    config: str | None,
    rules: str,
    komi: float,
    max_visits: int,
    positions_per_player: int,
    games_per_class: int,
    job_size: int,
    seed: int,
    cutoffs: tuple[int, ...],
    responses: str | Path | None = None,
    keep_queries: bool = False,
) -> dict:
    """Generate resumable, hash-checked, multi-cutoff KataGo feature caches."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_split(data_dir / split)
    if data.hashes is None:
        raise ValueError("Prepared SGF data must include per-game hashes")
    selected = _select_games(
        data.black_labels, data.white_labels, games_per_class, seed
    )
    if responses and len(selected) > job_size:
        raise ValueError("--responses requires all selected games to fit in one job")
    cutoffs = tuple(sorted(set(int(value) for value in cutoffs)))
    if not cutoffs:
        raise ValueError("At least one cutoff is required")

    moves_all = np.asarray(data.moves)
    lengths = np.asarray(data.lengths)
    data_hashes = np.asarray(data.hashes)
    per_move_dir = output_dir / f"{split}_per_move"
    per_move_dir.mkdir(exist_ok=True)
    data_fingerprint = None
    data_manifest_path = data_dir / "manifest.json"
    if data_manifest_path.is_file():
        data_fingerprint = json.loads(
            data_manifest_path.read_text(encoding="utf-8")
        ).get("fingerprint")
    engine_identity = {
        "katago_sha256": _file_sha256(katago_bin),
        "model_sha256": _file_sha256(model),
        "config_sha256": _file_sha256(config),
    }

    job_records: list[Path] = []
    completed_jobs = 0
    for job_index, start in enumerate(range(0, len(selected), job_size)):
        job_indices = selected[start : start + job_size]
        job_path = output_dir / f"{split}_job_features_{job_index:05d}.npz"
        job_meta_path = output_dir / f"{split}_job_features_{job_index:05d}.json"
        move_path = per_move_dir / f"part-{job_index:05d}.npz"
        signature = hashlib.sha256(
            json.dumps(
                {
                    "data_fingerprint": data_fingerprint,
                    "indices_sha256": hashlib.sha256(job_indices.tobytes()).hexdigest(),
                    "feature_names": FEATURE_NAMES,
                    "engine": engine_identity,
                    "rules": rules,
                    "komi": komi,
                    "max_visits": max_visits,
                    "positions_per_player": positions_per_player,
                    "cutoffs": cutoffs,
                    "seed": seed,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if (
            job_path.is_file()
            and job_meta_path.is_file()
            and move_path.is_file()
        ):
            metadata = json.loads(job_meta_path.read_text(encoding="utf-8"))
            if metadata.get("signature") == signature:
                job_records.append(job_path)
                completed_jobs += 1
                print(
                    f"{split}: job {job_index} already complete; skipping",
                    flush=True,
                )
                continue

        query_path = output_dir / f"{split}_queries_{job_index:05d}.jsonl"
        response_path = (
            Path(responses)
            if responses and len(selected) <= job_size
            else output_dir / f"{split}_responses_{job_index:05d}.jsonl"
        )
        expected_turns: dict[str, list[int]] = {}
        with query_path.open("w", encoding="utf-8") as handle:
            for game_index in job_indices:
                length = min(300, int(lengths[game_index]))
                turns = sorted(
                    set(select_positions(length, "B", positions_per_player))
                    | set(select_positions(length, "W", positions_per_player))
                )
                game_id = f"{split}:{game_index}"
                expected_turns[game_id] = turns
                handle.write(
                    json.dumps(
                        make_query(
                            game_id,
                            moves_all[game_index, :length],
                            turns,
                            rules=rules,
                            komi=komi,
                            max_visits=max_visits,
                        ),
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        if responses is None:
            if not all((katago_bin, model, config)):
                raise ValueError("--katago-bin, --model, and --config are required")
            command = [
                str(katago_bin),
                "analysis",
                "-model",
                str(model),
                "-config",
                str(config),
                "-override-config",
                (
                    f"maxVisits={max_visits},analysisPVLen=15,"
                    "includeOwnership=false,allowResignation=false,"
                    "reportAnalysisWinratesAs=SIDETOMOVE,"
                    "rootNoiseEnabled=false,wideRootNoise=0"
                ),
            ]
            with query_path.open("rb") as query_handle, response_path.open(
                "wb"
            ) as response_handle:
                subprocess.run(
                    command,
                    stdin=query_handle,
                    stdout=response_handle,
                    check=True,
                )

        samples: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: {"B": [], "W": []}
        )
        with response_path.open(encoding="utf-8") as handle:
            for line in handle:
                response = json.loads(line)
                game_id = str(response["id"])
                if game_id not in expected_turns:
                    continue
                turn = int(response["turnNumber"])
                game_index = int(game_id.split(":")[-1])
                color = "B" if turn % 2 == 0 else "W"
                response["_color"] = color
                samples[game_id][color].append(
                    parse_move_sample(
                        response,
                        played_move=int(moves_all[game_index, turn]),
                        max_move=min(300, int(lengths[game_index])),
                    )
                )

        black_job = np.zeros(
            (len(job_indices), len(cutoffs), len(FEATURE_NAMES)),
            dtype=np.float32,
        )
        white_job = np.zeros_like(black_job)
        complete_job = np.zeros(len(job_indices), dtype=bool)
        move_rows: list[tuple[int, str, dict]] = []
        for local_index, game_index in enumerate(job_indices):
            game_id = f"{split}:{game_index}"
            turns = expected_turns[game_id]
            complete_job[local_index] = all(
                any(
                    sample["move_number"] == turn
                    for sample in samples[game_id]["B" if turn % 2 == 0 else "W"]
                )
                for turn in turns
            )
            for cutoff_index, cutoff in enumerate(cutoffs):
                black_job[local_index, cutoff_index] = aggregate_player(
                    [
                        sample
                        for sample in samples[game_id]["B"]
                        if sample["move_number"] < cutoff
                    ]
                )
                white_job[local_index, cutoff_index] = aggregate_player(
                    [
                        sample
                        for sample in samples[game_id]["W"]
                        if sample["move_number"] < cutoff
                    ]
                )
            for color in ("B", "W"):
                for sample in samples[game_id][color]:
                    move_rows.append((int(game_index), color, sample))

        move_dtypes = {
            "game_index": np.int32,
            "move_number": np.uint16,
            "color": np.uint8,
            "played_move_found": np.bool_,
            "played_rank": np.uint16,
            "top1": np.bool_,
            "top3": np.bool_,
            "top5": np.bool_,
            "best_visits": np.uint32,
            "played_visits": np.uint32,
            "visit_ratio": np.float16,
            "played_visit_share": np.float16,
            "winrate_loss": np.float16,
            "score_loss": np.float16,
            "root_winrate": np.float16,
            "root_score_lead": np.float16,
            "policy_entropy": np.float16,
        }
        move_arrays = {}
        for name, dtype in move_dtypes.items():
            if name == "game_index":
                values = [row[0] for row in move_rows]
            elif name == "color":
                values = [0 if row[1] == "B" else 1 for row in move_rows]
            else:
                values = [row[2][name] for row in move_rows]
            move_arrays[name] = np.asarray(values, dtype=dtype)
        temporary_move = move_path.with_name(f".{move_path.name}.tmp")
        with temporary_move.open("wb") as handle:
            np.savez_compressed(handle, **move_arrays)
        os.replace(temporary_move, move_path)

        temporary_job = job_path.with_name(f".{job_path.name}.tmp")
        with temporary_job.open("wb") as handle:
            np.savez(
                handle,
                indices=job_indices.astype(np.int64),
                black=black_job,
                white=white_job,
                complete=complete_job,
            )
        os.replace(temporary_job, job_path)
        job_meta_path.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "job_index": job_index,
                    "games": int(len(job_indices)),
                    "max_visits": max_visits,
                    "cutoffs": list(cutoffs),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        job_records.append(job_path)
        completed_jobs += 1
        if not keep_queries:
            query_path.unlink(missing_ok=True)
            if responses is None:
                response_path.unlink(missing_ok=True)
        print(
            f"{split}: completed job {job_index + 1}/{max((len(selected) + job_size - 1) // job_size, 1)}",
            flush=True,
        )

    black_features = np.zeros(
        (len(data), len(cutoffs), len(FEATURE_NAMES)), dtype=np.float32
    )
    white_features = np.zeros_like(black_features)
    mask = np.zeros(len(data), dtype=bool)
    for job_path in job_records:
        with np.load(job_path) as job:
            indices = job["indices"]
            black_features[indices] = job["black"]
            white_features[indices] = job["white"]
            mask[indices] = job["complete"]

    np.save(output_dir / f"{split}_katago_black.npy", black_features)
    np.save(output_dir / f"{split}_katago_white.npy", white_features)
    np.save(output_dir / f"{split}_katago_mask.npy", mask)
    np.save(output_dir / f"{split}_katago_hashes.npy", data_hashes)
    (output_dir / f"{split}_katago_cutoffs.json").write_text(
        json.dumps(list(cutoffs)) + "\n", encoding="utf-8"
    )
    (output_dir / f"{split}_katago_feature_names.json").write_text(
        json.dumps(list(FEATURE_NAMES), indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "split": split,
        "feature_names": list(FEATURE_NAMES),
        "selected_games": int(len(selected)),
        "complete_games": int(mask.sum()),
        "completed_jobs": completed_jobs,
        "job_size": job_size,
        "positions_per_player": positions_per_player,
        "max_visits": max_visits,
        "rules": rules,
        "komi": komi,
        "cutoffs": list(cutoffs),
        "seed": seed,
        "data_fingerprint": data_fingerprint,
        **engine_identity,
    }
    (output_dir / f"{split}_katago_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate cached KataGo training features")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--katago-bin")
    parser.add_argument("--model")
    parser.add_argument("--config")
    parser.add_argument("--responses")
    parser.add_argument("--rules", default="chinese")
    parser.add_argument("--komi", type=float, default=7.5)
    parser.add_argument("--max-visits", type=int, default=500)
    parser.add_argument(
        "--cutoffs",
        nargs="+",
        type=int,
        default=(100, 150, 200, 250, 300),
    )
    parser.add_argument(
        "--positions-per-player",
        type=int,
        default=16,
        help="0 analyzes every move for each player",
    )
    parser.add_argument(
        "--games-per-class",
        type=int,
        default=0,
        help="0 analyzes every selected game",
    )
    parser.add_argument("--job-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--keep-queries", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = generate_resumable(
        args.data_dir,
        args.split,
        args.output,
        katago_bin=args.katago_bin,
        model=args.model,
        config=args.config,
        rules=args.rules,
        komi=args.komi,
        max_visits=args.max_visits,
        cutoffs=tuple(args.cutoffs),
        positions_per_player=args.positions_per_player,
        games_per_class=args.games_per_class,
        job_size=args.job_size,
        seed=args.seed,
        responses=args.responses,
        keep_queries=args.keep_queries,
    )
    print(json.dumps(result, indent=2))
