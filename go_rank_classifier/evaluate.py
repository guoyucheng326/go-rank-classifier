from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import KataGoFeatureData, SGFGameDataset, load_split
from .metrics import (
    DECODERS,
    decode_probabilities,
    json_safe,
    predict_dual_loader,
    probability_metrics,
    save_evaluation,
)
from .model import DualGoRankCNN


def run(args: argparse.Namespace) -> dict:
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tabular_features = int(state.get("tabular_features", 0))
    kata_dir = Path(args.katago_data) if args.katago_data else None
    if tabular_features and kata_dir is None:
        raise ValueError("This checkpoint requires --katago-data")
    kata = KataGoFeatureData.load(kata_dir / args.split) if kata_dir else None
    model = DualGoRankCNN(
        num_classes=len(state["ranks"]),
        tabular_features=tabular_features,
    ).to(device)
    model.load_state_dict(state["model_state"])

    data = load_split(Path(args.data) / args.split)
    labels_black = labels_white = None
    probabilities_black = probabilities_white = None
    for cutoff in sorted(set(args.cutoffs)):
        dataset = SGFGameDataset(
            data,
            training=False,
            fixed_cutoff=cutoff,
            katago=kata,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        current_black, current_white, prob_black, prob_white = predict_dual_loader(
            model,
            loader,
            device,
            symmetries=tuple(range(args.symmetries)),
        )
        if labels_black is None:
            labels_black, labels_white = current_black, current_white
            probabilities_black, probabilities_white = prob_black, prob_white
        else:
            if not (current_black == labels_black).all() or not (
                current_white == labels_white
            ).all():
                raise RuntimeError("Dataset order changed between cutoff passes")
            probabilities_black += prob_black
            probabilities_white += prob_white
    assert (
        labels_black is not None
        and labels_white is not None
        and probabilities_black is not None
        and probabilities_white is not None
    )
    divisor = len(set(args.cutoffs))
    probabilities_black /= divisor
    probabilities_white /= divisor
    black_metrics = probability_metrics(
        labels_black, probabilities_black, args.decode
    )
    white_metrics = probability_metrics(
        labels_white, probabilities_white, args.decode
    )
    result = {
        "split": args.split,
        "decoder": args.decode,
        "symmetries": args.symmetries,
        "cutoffs": sorted(set(args.cutoffs)),
        "black": black_metrics,
        "white": white_metrics,
        "average": {
            key: 0.5 * (black_metrics[key] + white_metrics[key])
            for key in (
                "accuracy",
                "top3_accuracy",
                "macro_f1",
                "mae",
                "expected_strength_mae",
                "expected_within_0_5",
                "expected_within_1_0",
            )
        },
    }
    if args.output:
        destination = Path(args.output)
    else:
        destination = Path(args.checkpoint).parent / f"evaluation_{args.split}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json_safe(result), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_evaluation(
        labels_black,
        decode_probabilities(probabilities_black, args.decode),
        destination.with_name(destination.stem + "_black.json"),
        top3_accuracy=black_metrics["top3_accuracy"],
    )
    save_evaluation(
        labels_white,
        decode_probabilities(probabilities_white, args.decode),
        destination.with_name(destination.stem + "_white.json"),
        top3_accuracy=white_metrics["top3_accuracy"],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a dual-head rank classifier")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--katago-data")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--device")
    parser.add_argument("--symmetries", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=(200, 250, 300))
    parser.add_argument("--decode", choices=DECODERS, default="expected")
    parser.add_argument("--output")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result["average"], ensure_ascii=False))


if __name__ == "__main__":
    main()
