from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SGFGameDataset, load_split
from .metrics import DECODERS, decode_probabilities, predict_loader, save_evaluation
from .model import GoRankCNN


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved rank classifier")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--symmetries", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=(200, 250, 300))
    parser.add_argument("--decode", choices=DECODERS, default="expected")
    parser.add_argument("--output")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GoRankCNN(num_classes=len(state["ranks"])).to(device)
    model.load_state_dict(state["model_state"])
    split_data = load_split(Path(args.data) / args.split)
    probabilities = None
    labels = None
    for cutoff in sorted(set(args.cutoffs)):
        dataset = SGFGameDataset(
            split_data,
            training=False,
            fixed_cutoff=cutoff,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
        )
        cutoff_labels, cutoff_probabilities = predict_loader(
            model,
            loader,
            device,
            symmetries=tuple(range(args.symmetries)),
        )
        if labels is None:
            labels = cutoff_labels
            probabilities = cutoff_probabilities
        else:
            if not (cutoff_labels == labels).all():
                raise RuntimeError("Dataset order changed between cutoff passes")
            probabilities += cutoff_probabilities
    if labels is None or probabilities is None:
        raise RuntimeError("No cutoff was evaluated")
    probabilities /= len(set(args.cutoffs))
    predictions = decode_probabilities(probabilities, args.decode)
    top3 = np.argpartition(probabilities, -3, axis=1)[:, -3:]
    top3_hits = (top3 == labels[:, None]).any(axis=1)
    destination = args.output or str(
        Path(args.checkpoint).parent / f"evaluation_{args.split}.json"
    )
    result = save_evaluation(
        labels,
        predictions,
        destination,
        top3_accuracy=float(top3_hits.mean()),
    )
    result["decoder"] = args.decode
    result["symmetries"] = args.symmetries
    result["cutoffs"] = sorted(set(args.cutoffs))
    Path(destination).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "split": args.split,
                "accuracy": result["accuracy"],
                "top3_accuracy": result["top3_accuracy"],
                "macro_f1": result["macro_f1"],
                "adjacent_accuracy": result["adjacent_accuracy"],
                "decoder": args.decode,
                "symmetries": args.symmetries,
                "cutoffs": sorted(set(args.cutoffs)),
                "output": destination,
            }
        )
    )


if __name__ == "__main__":
    main()

