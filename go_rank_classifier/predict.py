from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .features import board_features
from .model import GoRankCNN
from .ranks import RANKS
from .sgf import parse_sgf_file


class RankPredictor:
    def __init__(self, checkpoint: str | Path, device: str | None = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.ranks = tuple(state.get("ranks", RANKS))
        self.model = GoRankCNN(num_classes=len(self.ranks))
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.device).eval()

    def predict_moves(self, moves: list[int]) -> np.ndarray:
        if not 100 <= len(moves) <= 350:
            raise ValueError("SGF main line must contain 100 through 350 moves")
        selected = moves[:300]
        features = board_features(selected, len(selected))
        tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probability = torch.softmax(self.model(tensor), dim=1)[0].cpu().numpy()
        return probability

    def predict_file(self, path: str | Path) -> tuple[int, str]:
        game = parse_sgf_file(path)
        if game.board_size != 19:
            raise ValueError(f"Expected a 19x19 SGF, got {game.board_size}x{game.board_size}")
        probability = self.predict_moves(list(game.moves))
        rank_id = int(probability.argmax())
        return rank_id, self.ranks[rank_id]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict a Go rank ID from an SGF file")
    parser.add_argument("sgf", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    predictor = RankPredictor(args.checkpoint, args.device)
    for path in args.sgf:
        game = parse_sgf_file(path)
        if game.board_size != 19:
            raise ValueError(f"{path}: expected 19x19, got {game.board_size}x{game.board_size}")
        probability = predictor.predict_moves(list(game.moves))
        rank_id = int(probability.argmax())
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

