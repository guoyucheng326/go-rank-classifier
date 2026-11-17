from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .dataset import SGFGameDataset, load_split
from .model import GoRankCNN
from .ranks import RANKS


def class_weights(labels: np.ndarray, num_classes: int, power: float) -> np.ndarray:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float64)
    present = counts > 0
    if power > 0:
        weights[present] = (counts[present].sum() / counts[present]) ** power
        weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    top3_correct = 0
    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            prediction = logits.argmax(dim=1)
            correct += int((prediction == labels).sum())
            top3 = logits.topk(min(3, logits.shape[1]), dim=1).indices
            top3_correct += int(top3.eq(labels[:, None]).any(dim=1).sum())
            total += int(labels.numel())
    return correct / max(total, 1), top3_correct / max(total, 1)


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_data = load_split(Path(args.data) / "train")
    val_data = load_split(Path(args.data) / "val")
    train_dataset = SGFGameDataset(
        train_data, training=True, seed=args.seed, limit=args.limit_train
    )
    val_dataset = SGFGameDataset(
        val_data, training=False, seed=args.seed, limit=args.limit_val
    )
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
    }
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=args.workers > 0,
        **loader_options,
    )

    model = GoRankCNN(num_classes=len(RANKS)).to(device)
    weights = class_weights(
        np.asarray(train_data.labels),
        len(RANKS),
        args.class_weight_power,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device),
        label_smoothing=args.label_smoothing,
    )
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    print(
        json.dumps(
            {
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "class_counts": np.bincount(
                    np.asarray(train_data.labels), minlength=len(RANKS)
                ).tolist(),
                "class_weights": weights.tolist(),
            }
        ),
        flush=True,
    )

    best_accuracy = -1.0
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    if args.resume and Path(args.resume).is_file():
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_accuracy = float(
            checkpoint.get("best_accuracy", checkpoint.get("val_accuracy", -1.0))
        )
        history_path = output / "history.csv"
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = [
                    {
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "val_accuracy": float(row["val_accuracy"]),
                        "val_top3_accuracy": float(row["val_top3_accuracy"]),
                        "lr": float(row["lr"]),
                        "seconds": float(row["seconds"]),
                    }
                    for row in csv.DictReader(handle)
                ]
        print(
            json.dumps({"resumed_from": args.resume, "start_epoch": start_epoch}),
            flush=True,
        )

    started = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        # Recreate workers so every worker receives the current epoch value.
        train_dataset.set_epoch(epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_options,
        )
        model.train()
        running_loss = 0.0
        seen = 0
        stopped_early = False
        epoch_started = time.time()
        for step, (features, labels) in enumerate(train_loader):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                logits = model(features)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * labels.numel()
            seen += int(labels.numel())
            if (
                args.max_seconds is not None
                and step % 50 == 0
                and time.time() - started >= args.max_seconds
            ):
                stopped_early = True
                break

        scheduler.step()
        val_accuracy, val_top3 = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_accuracy": val_accuracy,
            "val_top3_accuracy": val_top3,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_started, 1),
            "partial": stopped_early,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "ranks": list(RANKS),
            "feature_channels": 22,
            "board_size": 19,
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "best_accuracy": best_accuracy,
            "args": vars(args),
        }
        torch.save(checkpoint, output / "last.pt")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            checkpoint["best_accuracy"] = best_accuracy
            torch.save(checkpoint, output / "best.pt")
            (output / "metrics.json").write_text(
                json.dumps(
                    {
                        "best_epoch": epoch,
                        "best_val_accuracy": best_accuracy,
                        "best_val_top3_accuracy": val_top3,
                        "training_seconds": round(time.time() - started, 1),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        if stopped_early:
            print(
                json.dumps(
                    {
                        "stopped": "max_seconds",
                        "elapsed_seconds": round(time.time() - started, 1),
                        "epochs_recorded": len(history),
                    }
                ),
                flush=True,
            )
            break

    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Fox Go rank classifier")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/go_rank")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-seconds",
        type=float,
        help="Wall-clock budget for training; stops at the next 50-step check once exceeded.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())

