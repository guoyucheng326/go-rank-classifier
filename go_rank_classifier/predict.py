from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .features import board_features
from .metrics import DECODERS, decode_probabilities, expected_strength, transform_features
from .model import DualGoRankCNN
from .ranks import RANKS
from .sgf import parse_sgf_file


class RankPredictor:
    def __init__(
        self,
        checkpoint: str | Path,
        player: str,
        device: str | None = None,
        *,
        symmetries: int = 8,
        cutoffs: tuple[int, ...] = (200, 250, 300),
        decoder: str = "expected",
        katago_black: str | Path | None = None,
        katago_white: str | Path | None = None,
        katago_cutoffs: tuple[int, ...] = (100, 150, 200, 250, 300),
        katago_bin: str | Path | None = None,
        katago_model: str | Path | None = None,
        katago_config: str | Path | None = None,
        katago_visits: int = 500,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.player = player.lower()
        if self.player not in {"black", "white"}:
            raise ValueError("player must be black or white")
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if not state.get("dual_player"):
            raise ValueError("This predictor requires a dual-player checkpoint")
        self.ranks = tuple(state.get("ranks", RANKS))
        self.tabular_features = int(state.get("tabular_features", 0))
        self.model = DualGoRankCNN(
            num_classes=len(self.ranks),
            tabular_features=self.tabular_features,
        )
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()
        self.symmetries = tuple(range(symmetries))
        self.cutoffs = tuple(sorted(set(cutoffs)))
        self.decoder = decoder
        self.katago_black = self._load_tabular(katago_black)
        self.katago_white = self._load_tabular(katago_white)
        self.online_features = katago_black is None and katago_white is None
        if self.tabular_features and not self.online_features and (
            self.katago_black is None or self.katago_white is None
        ):
            raise ValueError("Provide both KataGo feature files or neither")
        self.katago_cutoffs = tuple(sorted(set(katago_cutoffs)))
        self.katago_bin = katago_bin
        self.katago_model = katago_model
        self.katago_config = katago_config
        self.katago_visits = katago_visits
        if self.tabular_features and (
            self.katago_black is not None
            and self.katago_black.shape[0] != len(self.katago_cutoffs)
        ):
            raise ValueError("KataGo feature rows must match --katago-cutoffs")
        if self.tabular_features and (
            self.katago_white is not None
            and self.katago_white.shape[0] != len(self.katago_cutoffs)
        ):
            raise ValueError("KataGo feature rows must match --katago-cutoffs")

    def analyze_game(self, game) -> None:
        if game.handicap != 0 or game.has_setup_stones:
            raise ValueError("Online analysis supports only HA=0 games without setup stones")
        if not all((self.katago_bin, self.katago_model, self.katago_config)):
            raise ValueError(
                "Provide --katago-black/--katago-white or all KataGo engine paths"
            )
        from .katago_features import generate_resumable

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            features_dir = root / "features"
            data_dir.mkdir()
            moves = np.full((1, 300), -1, dtype=np.int16)
            truncated = np.asarray(game.truncated_moves(), dtype=np.int16)
            moves[0, : len(truncated)] = truncated
            digest = hashlib.blake2b(truncated.tobytes(), digest_size=8).digest()
            np.save(data_dir / "train_moves.npy", moves)
            np.save(data_dir / "train_lengths.npy", np.array([len(truncated)], np.int16))
            np.save(data_dir / "train_black_labels.npy", np.array([0], np.int16))
            np.save(data_dir / "train_white_labels.npy", np.array([0], np.int16))
            np.save(
                data_dir / "train_hashes.npy",
                np.array([int.from_bytes(digest, "little")], np.uint64),
            )
            fingerprint = int.from_bytes(digest, "little")
            (data_dir / "manifest.json").write_text(
                json.dumps({"fingerprint": f"{fingerprint:016x}"})
            )
            generate_resumable(
                data_dir,
                "train",
                features_dir,
                katago_bin=str(self.katago_bin),
                model=str(self.katago_model),
                config=str(self.katago_config),
                rules="chinese",
                komi=7.5,
                max_visits=self.katago_visits,
                positions_per_player=0,
                games_per_class=0,
                job_size=1,
                seed=0,
                cutoffs=self.katago_cutoffs,
            )
            self.katago_black = np.load(
                features_dir / "train_katago_black.npy"
            )[0]
            self.katago_white = np.load(
                features_dir / "train_katago_white.npy"
            )[0]

    @staticmethod
    def _load_tabular(path: str | Path | None) -> np.ndarray | None:
        if path is None:
            return None
        values = np.asarray(np.load(path))
        if values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if values.ndim != 2:
            raise ValueError("KataGo inference features must have shape (cutoffs, features)")
        return values

    def predict_moves(self, moves: list[int]) -> np.ndarray:
        if not 100 <= len(moves) <= 350:
            raise ValueError("SGF main line must contain 100 through 350 moves")
        valid_cutoffs = sorted(
            {
                min(cutoff, len(moves))
                for cutoff in self.cutoffs
                if min(cutoff, len(moves)) >= 100
            }
        )
        if not valid_cutoffs:
            valid_cutoffs = [min(300, len(moves))]
        all_logits: list[torch.Tensor] = []
        for cutoff in valid_cutoffs:
            selected = moves[:cutoff]
            features = board_features(selected, len(selected))
            tensor = torch.from_numpy(features).unsqueeze(0)
            views = [
                transform_features(tensor, symmetry) for symmetry in self.symmetries
            ]
            batch = torch.cat(views, dim=0).to(self.device)
            tabular = None
            if self.tabular_features:
                eligible = [
                    index
                    for index, value in enumerate(self.katago_cutoffs)
                    if value <= cutoff
                ]
                feature_index = eligible[-1] if eligible else 0
                black = self.katago_black[feature_index]
                white = self.katago_white[feature_index]
                tabular = torch.from_numpy(
                    np.concatenate((black, white, black - white)).astype(np.float32)
                ).repeat(batch.shape[0], 1).to(self.device)
            with torch.inference_mode():
                logits = self.model(batch, tabular)
                all_logits.append(
                    torch.softmax(
                        logits[0] if self.player == "black" else logits[1],
                        dim=1,
                    ).mean(dim=0)
                )
        with torch.inference_mode():
            probability = torch.stack(all_logits).mean(dim=0).cpu().numpy()
        return probability

    def predict_file(self, path: str | Path) -> tuple[int, str, float]:
        game = parse_sgf_file(path)
        if game.board_size != 19:
            raise ValueError(f"Expected a 19x19 SGF, got {game.board_size}x{game.board_size}")
        if self.tabular_features and self.online_features:
            self.analyze_game(game)
        probability = self.predict_moves(list(game.moves))
        rank_id = int(decode_probabilities(probability[None, :], self.decoder)[0])
        strength = float(expected_strength(probability[None, :])[0])
        return rank_id, self.ranks[rank_id], strength


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one player's rank from an SGF")
    parser.add_argument("sgf", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--player", required=True, choices=("black", "white"))
    parser.add_argument("--device")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--symmetries", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=(200, 250, 300))
    parser.add_argument("--decode", choices=DECODERS, default="expected")
    parser.add_argument("--katago-black")
    parser.add_argument("--katago-white")
    parser.add_argument("--katago-cutoffs", nargs="+", type=int, default=(100, 150, 200, 250, 300))
    parser.add_argument("--katago-bin")
    parser.add_argument("--katago-model")
    parser.add_argument("--katago-config")
    parser.add_argument("--katago-visits", type=int, default=500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = RankPredictor(
        args.checkpoint,
        args.player,
        args.device,
        symmetries=args.symmetries,
        cutoffs=tuple(args.cutoffs),
        decoder=args.decode,
        katago_black=args.katago_black,
        katago_white=args.katago_white,
        katago_cutoffs=tuple(args.katago_cutoffs),
        katago_bin=args.katago_bin,
        katago_model=args.katago_model,
        katago_config=args.katago_config,
        katago_visits=args.katago_visits,
    )
    for path in args.sgf:
        game = parse_sgf_file(path)
        if predictor.tabular_features and predictor.online_features:
            predictor.analyze_game(game)
        probability = predictor.predict_moves(list(game.moves))
        rank_id = int(decode_probabilities(probability[None, :], args.decode)[0])
        strength = float(expected_strength(probability[None, :])[0])
        if not args.json:
            print(rank_id)
            continue
        top_indices = np.argsort(probability)[::-1][: args.top_k]
        print(
            json.dumps(
                {
                    "player": args.player,
                    "rank_id": rank_id,
                    "rank": predictor.ranks[rank_id],
                    "estimated_strength": strength,
                    "moves": game.move_count,
                    "decoder": args.decode,
                    "symmetries": args.symmetries,
                    "cutoffs": sorted(set(args.cutoffs)),
                    "top_probabilities": [
                        {
                            "rank_id": int(index),
                            "rank": predictor.ranks[int(index)],
                            "probability": float(probability[index]),
                        }
                        for index in top_indices
                    ],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
