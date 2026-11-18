from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .ranks import RANKS


DECODERS = ("argmax", "expected", "median", "top3_expected")


def transform_features(features: torch.Tensor, symmetry: int) -> torch.Tensor:
    """Apply a dihedral transform to a batch of board feature tensors."""
    if symmetry not in range(8):
        raise ValueError("symmetry must be in [0, 7]")
    transformed = torch.flip(features, dims=(3,)) if symmetry >= 4 else features
    return torch.rot90(transformed, k=symmetry % 4, dims=(2, 3))


def decode_probabilities(probabilities: np.ndarray, method: str) -> np.ndarray:
    probabilities = np.asarray(probabilities)
    if method == "argmax":
        return probabilities.argmax(axis=1).astype(np.int16)
    positions = np.arange(probabilities.shape[1], dtype=np.float64)
    if method == "expected":
        return np.rint(probabilities @ positions).astype(np.int16)
    if method == "median":
        cumulative = np.cumsum(probabilities, axis=1)
        return (cumulative < 0.5).sum(axis=1).astype(np.int16)
    if method == "top3_expected":
        top = np.argpartition(probabilities, -3, axis=1)[:, -3:]
        masked = np.zeros_like(probabilities)
        rows = np.arange(probabilities.shape[0])[:, None]
        masked[rows, top] = probabilities[rows, top]
        totals = masked.sum(axis=1, keepdims=True)
        normalized = masked / np.maximum(totals, 1e-12)
        return np.rint(normalized @ positions).astype(np.int16)
    raise ValueError(f"Unknown decoder: {method}")


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    symmetries: tuple[int, ...] = (0,),
) -> tuple[np.ndarray, np.ndarray]:
    """Return labels and mean class probabilities over board symmetries."""
    model.eval()
    true_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                features, tabular, labels = batch
                tabular = tabular.to(device, non_blocking=True)
            else:
                features, labels = batch
                tabular = None
            features = features.to(device, non_blocking=True)
            probability = torch.zeros(
                (features.shape[0], len(RANKS)),
                dtype=torch.float32,
                device=device,
            )
            for symmetry in symmetries:
                logits = model(transform_features(features, symmetry), tabular)
                probability += torch.softmax(logits, dim=1)
            probability /= len(symmetries)
            probability_batches.append(probability.cpu().numpy())
            true_batches.append(labels.numpy())
    return np.concatenate(true_batches), np.concatenate(probability_batches)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    symmetries: tuple[int, ...] = (0,),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, top-1 predictions, and top-3 correctness."""
    labels, probabilities = predict_loader(
        model,
        loader,
        device,
        symmetries=symmetries,
    )
    predictions = decode_probabilities(probabilities, "argmax")
    top3 = np.argpartition(probabilities, -3, axis=1)[:, -3:]
    top3_hits = (top3 == labels[:, None]).any(axis=1)
    return labels, predictions, top3_hits


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: tuple[str, ...] = RANKS,
) -> dict:
    labels = np.asarray(labels, dtype=np.int16)
    predictions = np.asarray(predictions, dtype=np.int16)
    num_classes = len(class_names)
    confusion = np.bincount(
        labels.astype(np.int64) * num_classes + predictions.astype(np.int64),
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    precision = np.divide(
        diagonal,
        predicted,
        out=np.zeros(num_classes, dtype=np.float64),
        where=predicted > 0,
    )
    recall = np.divide(
        diagonal,
        support,
        out=np.zeros(num_classes, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(num_classes, dtype=np.float64),
        where=(precision + recall) > 0,
    )
    present = support > 0
    per_class = {}
    for index, rank in enumerate(class_names):
        per_class[rank] = {
            "id": index,
            "support": int(support[index]),
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
        }
    return {
        "samples": int(labels.size),
        "accuracy": float(diagonal.sum() / max(labels.size, 1)),
        "adjacent_accuracy": float(
            np.mean(np.abs(labels.astype(np.int16) - predictions.astype(np.int16)) <= 1)
        ),
        "macro_precision": float(precision[present].mean()),
        "macro_recall": float(recall[present].mean()),
        "macro_f1": float(f1[present].mean()),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def save_evaluation(
    labels: np.ndarray,
    predictions: np.ndarray,
    destination: str | Path,
    top3_accuracy: float | None = None,
) -> dict:
    import csv
    import json

    result = classification_metrics(labels, predictions)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    confusion = result.pop("confusion_matrix")
    per_class = result.pop("per_class")
    serializable = {
        **result,
        "classes": list(RANKS),
        "per_class": per_class,
    }
    if top3_accuracy is not None:
        serializable["top3_accuracy"] = float(top3_accuracy)
    destination.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(destination.with_name(destination.stem + "_confusion.npy"), confusion)
    with destination.with_name(destination.stem + "_per_class.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("rank", "id", "support", "precision", "recall", "f1"),
        )
        writer.writeheader()
        for rank, values in per_class.items():
            writer.writerow({"rank": rank, **values})
    return serializable

