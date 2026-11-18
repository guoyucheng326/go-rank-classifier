from __future__ import annotations

import argparse
import csv
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .dataset import (
    DeterministicBatchSampler,
    KataGoFeatureData,
    PreShuffledBatchSampler,
    SGFGameDataset,
    load_split,
)
from .losses import OrdinalSoftCrossEntropy
from .metrics import (
    decode_probabilities,
    json_safe,
    predict_dual_loader,
    probability_metrics,
    save_evaluation,
)
from .model import DualGoRankCNN
from .ranks import RANKS


def class_weights(labels: np.ndarray, num_classes: int, power: float) -> np.ndarray:
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    weights = np.ones(num_classes, dtype=np.float64)
    present = counts > 0
    if power > 0:
        weights[present] = (counts[present].sum() / counts[present]) ** power
        weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def dataset_labels(labels: np.ndarray, dataset: SGFGameDataset) -> np.ndarray:
    if dataset.indices is None:
        return np.asarray(labels)[: dataset.size]
    return np.asarray(labels)[dataset.indices[: dataset.size]]


def verify_feature_fingerprint(data_dir: Path, kata_dir: Path, split: str) -> None:
    data_manifest = data_dir / "manifest.json"
    feature_manifest = kata_dir / f"{split}_katago_manifest.json"
    if not data_manifest.is_file() or not feature_manifest.is_file():
        raise FileNotFoundError("Missing SGF or KataGo feature manifest")
    expected = json.loads(data_manifest.read_text(encoding="utf-8")).get("fingerprint")
    actual = json.loads(feature_manifest.read_text(encoding="utf-8")).get(
        "data_fingerprint"
    )
    if expected is None or actual != expected:
        raise ValueError(
            f"KataGo features for {split} do not match the current SGF data fingerprint"
        )


def verify_feature_hashes(data, features, split: str) -> None:
    if data.hashes is None or features.hashes is None:
        raise ValueError(f"Missing per-row hashes for {split}")
    selected = np.flatnonzero(np.asarray(features.mask))
    if not np.array_equal(
        np.asarray(data.hashes[selected]),
        np.asarray(features.hashes[selected]),
    ):
        raise ValueError(f"KataGo rows are misaligned with the SGF split: {split}")


def make_criterion(weights: np.ndarray, args: argparse.Namespace, device: torch.device):
    weight_tensor = torch.from_numpy(weights).to(device)
    if args.loss == "ordinal-soft":
        return OrdinalSoftCrossEntropy(
            weight_tensor,
            sigma=args.ordinal_sigma,
            soft_weight=args.ordinal_loss_weight,
            distance_weight=args.distance_loss_weight,
            label_smoothing=args.label_smoothing,
        )
    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=args.label_smoothing,
    )


def evaluate_dual(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    symmetries: tuple[int, ...] = (0,),
    decoder: str = "expected",
) -> tuple[dict, dict]:
    black_labels, white_labels, black_probability, white_probability = predict_dual_loader(
        model, loader, device, symmetries=symmetries
    )
    return (
        probability_metrics(black_labels, black_probability, decoder),
        probability_metrics(white_labels, white_probability, decoder),
    )


def evaluate_dual_tta(
    model: nn.Module,
    data,
    *,
    device: torch.device,
    cutoffs: tuple[int, ...],
    symmetries: tuple[int, ...],
    decoder: str,
    batch_size: int,
    workers: int,
    limit: int | None,
    katago,
) -> tuple[dict, dict]:
    labels_black = labels_white = None
    probabilities_black = probabilities_white = None
    for cutoff in sorted(set(cutoffs)):
        dataset = SGFGameDataset(
            data,
            training=False,
            fixed_cutoff=cutoff,
            limit=limit,
            katago=katago,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=device.type == "cuda",
        )
        current_black, current_white, prob_black, prob_white = predict_dual_loader(
            model, loader, device, symmetries=symmetries
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
    divisor = len(set(cutoffs))
    probabilities_black /= divisor
    probabilities_white /= divisor
    return (
        probability_metrics(labels_black, probabilities_black, decoder),
        probability_metrics(labels_white, probabilities_white, decoder),
    )


def composite_score(black: dict, white: dict) -> float:
    return 0.5 * (black["composite"] + white["composite"])


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA GPU is required for training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data)
    train_data = load_split(data_dir / "train")
    val_data = load_split(data_dir / "val")
    kata_dir = Path(args.katago_data) if args.katago_data else None
    if kata_dir is not None:
        for split in ("train", "val"):
            verify_feature_fingerprint(data_dir, kata_dir, split)
    train_kata = KataGoFeatureData.load(kata_dir / "train") if kata_dir else None
    val_kata = KataGoFeatureData.load(kata_dir / "val") if kata_dir else None
    if train_kata is not None and val_kata is not None:
        verify_feature_hashes(train_data, train_kata, "train")
        verify_feature_hashes(val_data, val_kata, "val")
    feature_count = train_kata.black.shape[2] if train_kata is not None else 0
    tabular_features = feature_count * 3

    train_dataset = SGFGameDataset(
        train_data,
        training=True,
        seed=args.seed,
        limit=args.limit_train,
        katago=train_kata,
    )
    loader_options = {"num_workers": args.workers, "pin_memory": True}

    model = DualGoRankCNN(
        num_classes=len(RANKS),
        tabular_features=tabular_features,
    ).to(device)
    selected_black = dataset_labels(train_data.black_labels, train_dataset)
    selected_white = dataset_labels(train_data.white_labels, train_dataset)
    black_weights = class_weights(selected_black, len(RANKS), args.class_weight_power)
    white_weights = class_weights(selected_white, len(RANKS), args.class_weight_power)
    black_criterion = make_criterion(black_weights, args, device)
    white_criterion = make_criterion(white_weights, args, device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    print(
        json.dumps(
            {
                "train_games": len(train_data),
                "val_games": len(val_data),
                "katago_features_per_player": feature_count,
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "black_class_counts": np.bincount(
                    selected_black, minlength=len(RANKS)
                ).tolist(),
                "white_class_counts": np.bincount(
                    selected_white, minlength=len(RANKS)
                ).tolist(),
            }
        ),
        flush=True,
    )

    history: list[dict] = []
    start_epoch = 1
    start_batch = 0
    resumed = False
    best_accuracy = -1.0
    best_monitor_score = -float("inf")
    epochs_without_improvement = 0
    if args.resume and Path(args.resume).is_file():
        resumed = True
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        progress = checkpoint.get("progress", {})
        start_epoch = int(progress.get("next_epoch", checkpoint.get("epoch", 0) + 1))
        start_batch = int(progress.get("next_batch", 0))
        best_accuracy = float(checkpoint.get("best_accuracy", -1.0))
        best_monitor_score = float(checkpoint.get("monitor_score", -float("inf")))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        if history_path.is_file():
            with history_path.open(newline="", encoding="utf-8") as handle:
                history = list(csv.DictReader(handle))
        print(
            json.dumps(
                {
                    "resumed_from": args.resume,
                    "start_epoch": start_epoch,
                    "start_batch": start_batch,
                }
            ),
            flush=True,
        )
    elif args.resume:
        raise FileNotFoundError(args.resume)

    for group in optimizer.param_groups:
        group["lr"] = args.lr
    scheduler = None
    if args.scheduler == "cosine":
        if args.warmup_epochs > 0 and not resumed and args.epochs > args.warmup_epochs:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(
                        optimizer,
                        start_factor=0.1,
                        total_iters=args.warmup_epochs,
                    ),
                    CosineAnnealingLR(
                        optimizer,
                        T_max=args.epochs - args.warmup_epochs,
                        eta_min=args.lr * 0.02,
                    ),
                ],
                milestones=[args.warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.02
            )

    stop_requested = False
    previous_handlers: dict[int, object] = {}

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(json.dumps({"stop_requested": signal.Signals(signum).name}), flush=True)

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)

    started = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_dataset.set_epoch(epoch)
        sampler_class = (
            PreShuffledBatchSampler if train_data.shuffled else DeterministicBatchSampler
        )
        batch_sampler = sampler_class(
            len(train_dataset),
            args.batch_size,
            epoch,
            args.seed,
            start_batch=start_batch if epoch == start_epoch else 0,
            drop_last=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            **loader_options,
        )
        model.train()
        running_loss = 0.0
        seen = 0
        stopped_early = False
        stop_reason = ""
        epoch_started = time.time()
        for step, batch in enumerate(train_loader):
            if stop_requested:
                stopped_early = True
                stop_reason = "signal"
                break
            absolute_step = batch_sampler.start_batch + step
            if len(batch) == 4:
                features, tabular, black_labels, white_labels = batch
                tabular = tabular.to(device, non_blocking=True)
            else:
                features, black_labels, white_labels = batch
                tabular = None
            features = features.to(device, non_blocking=True)
            black_labels = black_labels.to(device, non_blocking=True)
            white_labels = white_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                black_logits, white_logits = model(features, tabular)
                loss = 0.5 * (
                    black_criterion(black_logits, black_labels)
                    + white_criterion(white_logits, white_labels)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * features.shape[0]
            seen += int(features.shape[0])

            if (
                args.checkpoint_every_batches > 0
                and (absolute_step + 1) % args.checkpoint_every_batches == 0
            ):
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict() if scheduler else None,
                        "ranks": list(RANKS),
                        "dual_player": True,
                        "tabular_features": tabular_features,
                        "katago_features_per_player": feature_count,
                        "epoch": epoch,
                        "best_accuracy": best_accuracy,
                        "monitor_score": best_monitor_score,
                        "epochs_without_improvement": epochs_without_improvement,
                        "progress": {
                            "next_epoch": epoch,
                            "next_batch": absolute_step + 1,
                        },
                        "args": vars(args),
                    },
                    output / "last.pt",
                )
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "batch": absolute_step + 1,
                            "batches": batch_sampler.num_batches,
                            "loss": running_loss / max(seen, 1),
                            "elapsed_seconds": round(time.time() - started, 1),
                        }
                    ),
                    flush=True,
                )
            if (
                args.max_seconds is not None
                and (step + 1) % 50 == 0
                and time.time() - started >= args.max_seconds
            ):
                stopped_early = True
                stop_reason = "max_seconds"
                break

        if not stopped_early and scheduler is not None:
            scheduler.step()
        black_metrics, white_metrics = evaluate_dual_tta(
            model,
            val_data,
            device=device,
            cutoffs=tuple(args.val_cutoffs),
            symmetries=tuple(range(args.val_symmetries)),
            decoder=args.decode,
            batch_size=args.batch_size,
            workers=args.workers,
            limit=args.limit_val,
            katago=val_kata,
        )
        monitor_score = 0.5 * (
            black_metrics[args.early_stop_metric]
            + white_metrics[args.early_stop_metric]
        )
        improved = monitor_score > best_monitor_score + args.min_delta
        if improved:
            best_monitor_score = monitor_score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        early_stop = (
            not stopped_early
            and args.patience > 0
            and epochs_without_improvement >= args.patience
        )
        if early_stop:
            stop_reason = "early_stopping"

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "black_accuracy": black_metrics["accuracy"],
            "white_accuracy": white_metrics["accuracy"],
            "black_top3_accuracy": black_metrics["top3_accuracy"],
            "white_top3_accuracy": white_metrics["top3_accuracy"],
            "black_macro_f1": black_metrics["macro_f1"],
            "white_macro_f1": white_metrics["macro_f1"],
            "black_within_1": black_metrics["expected_within_1_0"],
            "white_within_1": white_metrics["expected_within_1_0"],
            "monitor_score": monitor_score,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - epoch_started, 1),
            "partial": stopped_early,
            "stop_reason": stop_reason,
        }
        history.append(row)
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "ranks": list(RANKS),
            "dual_player": True,
            "tabular_features": tabular_features,
            "katago_features_per_player": feature_count,
            "epoch": epoch,
            "val_accuracy": 0.5 * (
                black_metrics["accuracy"] + white_metrics["accuracy"]
            ),
            "best_accuracy": best_accuracy,
            "monitor_metric": args.early_stop_metric,
            "monitor_score": best_monitor_score,
            "epochs_without_improvement": epochs_without_improvement,
            "progress": {
                "next_epoch": epoch if stopped_early else epoch + 1,
                "next_batch": (
                    batch_sampler.start_batch + step + 1 if stopped_early else 0
                ),
            },
            "args": vars(args),
        }
        torch.save(checkpoint, output / "last.pt")
        if not stopped_early and improved:
            best_accuracy = max(
                best_accuracy,
                0.5 * (black_metrics["accuracy"] + white_metrics["accuracy"]),
            )
            checkpoint["best_accuracy"] = best_accuracy
            checkpoint["monitor_score"] = best_monitor_score
            torch.save(checkpoint, output / "best.pt")
            (output / "metrics.json").write_text(
                json.dumps(
                    json_safe(
                    {
                        "best_epoch": epoch,
                        "best_val_accuracy": best_accuracy,
                        "best_monitor_score": best_monitor_score,
                        "monitor_metric": args.early_stop_metric,
                        "black": black_metrics,
                        "white": white_metrics,
                    }
                    ),
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        if early_stop or stopped_early:
            print(
                json.dumps(
                    {
                        "stopped": stop_reason,
                        "epoch": epoch,
                        "elapsed_seconds": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
            break

    for signum, handler in previous_handlers.items():
        signal.signal(signum, handler)

    if args.evaluate_test and not stop_requested and (output / "best.pt").is_file():
        best = torch.load(output / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best["model_state"])
        test_data = load_split(data_dir / "test")
        if kata_dir is not None:
            verify_feature_fingerprint(data_dir, kata_dir, "test")
        test_kata = KataGoFeatureData.load(kata_dir / "test") if kata_dir else None
        if test_kata is not None:
            verify_feature_hashes(test_data, test_kata, "test")
        labels_black = labels_white = None
        probabilities_black = probabilities_white = None
        for cutoff in sorted(set(args.test_cutoffs)):
            dataset = SGFGameDataset(
                test_data,
                training=False,
                fixed_cutoff=cutoff,
                limit=args.limit_test,
                katago=test_kata,
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                **loader_options,
            )
            current_black, current_white, prob_black, prob_white = predict_dual_loader(
                model,
                loader,
                device,
                symmetries=tuple(range(args.test_symmetries)),
            )
            if labels_black is None:
                labels_black, labels_white = current_black, current_white
                probabilities_black, probabilities_white = prob_black, prob_white
            else:
                probabilities_black += prob_black
                probabilities_white += prob_white
        assert labels_black is not None and probabilities_black is not None
        assert labels_white is not None and probabilities_white is not None
        divisor = len(set(args.test_cutoffs))
        probabilities_black /= divisor
        probabilities_white /= divisor
        result = {
            "black": probability_metrics(labels_black, probabilities_black, args.decode),
            "white": probability_metrics(labels_white, probabilities_white, args.decode),
            "decoder": args.decode,
            "symmetries": args.test_symmetries,
            "cutoffs": sorted(set(args.test_cutoffs)),
        }
        result["average"] = {
            key: 0.5 * (result["black"][key] + result["white"][key])
            for key in ("accuracy", "top3_accuracy", "macro_f1", "mae", "expected_strength_mae")
        }
        (output / "test_metrics.json").write_text(
            json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8"
        )
        save_evaluation(
            labels_black,
            decode_probabilities(probabilities_black, args.decode),
            output / "test_black_metrics.json",
            top3_accuracy=result["black"]["top3_accuracy"],
        )
        save_evaluation(
            labels_white,
            decode_probabilities(probabilities_white, args.decode),
            output / "test_white_metrics.json",
            top3_accuracy=result["white"]["top3_accuracy"],
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a dual-label Go rank classifier")
    parser.add_argument("--data", required=True)
    parser.add_argument("--katago-data")
    parser.add_argument("--output", default="runs/go_rank_v2")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--class-weight-power", type=float, default=0.75)
    parser.add_argument("--loss", choices=("ce", "ordinal-soft"), default="ordinal-soft")
    parser.add_argument("--ordinal-sigma", type=float, default=1.0)
    parser.add_argument("--ordinal-loss-weight", type=float, default=0.3)
    parser.add_argument("--distance-loss-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-val", type=int)
    parser.add_argument("--limit-test", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--scheduler", choices=("cosine", "fixed"), default="cosine")
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument(
        "--early-stop-metric",
        choices=("composite", "macro_f1", "adjacent_accuracy", "accuracy"),
        default="composite",
    )
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--checkpoint-every-batches", type=int, default=500)
    parser.add_argument("--val-symmetries", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--val-cutoffs", nargs="+", type=int, default=(300,))
    parser.add_argument("--test-symmetries", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--test-cutoffs", nargs="+", type=int, default=(200, 250, 300))
    parser.add_argument(
        "--decode",
        choices=("argmax", "expected", "median", "top3_expected"),
        default="expected",
    )
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
