from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .ranks import RANKS


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, top-1 predictions, and top-3 correctness."""
    model.eval()
    true_batches: list[np.ndarray] = []
    prediction_batches: list[np.ndarray] = []
    top3_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device, non_blocking=True)
            logits = model(features)
            prediction_batches.append(logits.argmax(dim=1).cpu().numpy())
            true_batches.append(labels.numpy())
            top3 = logits.topk(min(3, logits.shape[1]), dim=1).indices
            top3_batches.append(
                top3.cpu().eq(labels[:, None]).any(dim=1).numpy()
            )
    return (
        np.concatenate(true_batches),
        np.concatenate(prediction_batches),
        np.concatenate(top3_batches),
    )


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

