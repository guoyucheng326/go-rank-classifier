from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import SGFGameDataset, load_split
from .model import GoRankCNN
from .train import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved rank classifier")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = GoRankCNN(num_classes=len(state["ranks"])).to(device)
    model.load_state_dict(state["model_state"])
    dataset = SGFGameDataset(
        load_split(Path(args.data) / args.split), training=False
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    accuracy, top3 = evaluate(model, loader, device)
    print(json.dumps({"split": args.split, "accuracy": accuracy, "top3_accuracy": top3}))


if __name__ == "__main__":
    main()

