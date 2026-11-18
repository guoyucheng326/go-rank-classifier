from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .features import board_features
from .metrics import DECODERS, decode_probabilities, transform_features
from .model import GoRankCNN
from .ranks import RANKS
from .sgf import parse_sgf_file


class RankPredictor:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        *,
        symmetries: int = 8,
        cutoffs: tuple[int, ...] = (200, 250, 300),
        decoder: str = "expected",
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.ranks = tuple(state.get("ranks", RANKS))
        self.model = GoRankCNN(num_classes=len(self.ranks))
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()
        self.symmetries = tuple(range(symmetries))
        self.cutoffs = tuple(sorted(set(cutoffs)))
        self.decoder = decoder

    def predict_moves(self, moves: list[int]) -> np.ndarray:
        if not 100 <= len(moves) <= 350:
            raise ValueError("SGF main line must contain 100 through 350 moves")
        views: list[torch.Tensor] = []
        valid_cutoffs = sorted(
            {
                min(cutoff, len(moves))
                for cutoff in self.cutoffs
                if min(cutoff, len(moves)) >= 100
            }
        )
        if not valid_cutoffs:
            valid_cutoffs = [min(300, len(moves))]
        for cutoff in valid_cutoffs:
            selected = moves[:cutoff]
            features = board_features(selected, len(selected))
            tensor = torch.from_numpy(features).unsqueeze(0)
            views.extend(
                transform_features(tensor, symmetry) for symmetry in self.symmetries
            )
        batch = torch.cat(views, dim=0).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            probability = (
                torch.softmax(logits, dim=1)
                .mean(dim=0)
                .cpu()
                .numpy()
            )
        return probability

    def predict_file(self, path: str | Path) -> tuple[int, str]:
        game = parse_sgf_file(path)
        if game.board_size != 19:
            raise ValueError(f"Expected a 19x19 SGF, got {game.board_size}x{game.board_size}")
        probability = self.predict_moves(list(game.moves))
        rank_id = int(decode_probabilities(probability[None, :], self.decoder)[0])
        return rank_id, self.ranks[rank_id]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a Go rank ID from an SGF file")
    parser.add_argument("sgf", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--symmetries", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=(200, 250, 300))
    parser.add_argument("--decode", choices=DECODERS, default="expected")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = RankPredictor(
        args.checkpoint,
        args.device,
        symmetries=args.symmetries,
        cutoffs=tuple(args.cutoffs),
        decoder=args.decode,
    )
    for path in args.sgf:
        game = parse_sgf_file(path)
        if game.board_size != 19:
            raise ValueError(f"{path}: expected 19x19, got {game.board_size}x{game.board_size}")
        probability = predictor.predict_moves(list(game.moves))
        rank_id = int(decode_probabilities(probability[None, :], args.decode)[0])
        if not args.json:
            print(rank_id)
            continue
        top_indices = np.argsort(probability)[::-1][: args.top_k]
        print(
            json.dumps(
                {
                    "rank_id": rank_id,
                    "rank": predictor.ranks[rank_id],
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

