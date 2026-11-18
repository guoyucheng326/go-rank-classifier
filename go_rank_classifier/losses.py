from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class OrdinalSoftCrossEntropy(nn.Module):
    """Combine hard classification, adjacent-rank soft labels, and distance cost."""

    def __init__(
        self,
        class_weights: torch.Tensor,
        *,
        sigma: float = 1.0,
        soft_weight: float = 0.3,
        distance_weight: float = 0.05,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        self.register_buffer("class_weights", class_weights.float())
        self.sigma = sigma
        self.soft_weight = soft_weight
        self.distance_weight = distance_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        class_ids = torch.arange(logits.shape[1], device=logits.device)
        hard = F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        distance = class_ids[None, :] - targets[:, None]
        soft_targets = torch.exp(-(distance.float() ** 2) / (2 * self.sigma**2))
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True)
        log_probabilities = F.log_softmax(logits.float(), dim=1)
        sample_weights = self.class_weights[targets]
        soft = -(soft_targets * log_probabilities).sum(dim=1)
        soft = (soft * sample_weights).sum() / sample_weights.sum().clamp_min(1e-12)
        expected_distance = (
            torch.softmax(logits.float(), dim=1) * distance.float().abs()
        ).sum(dim=1)
        expected_distance = (
            expected_distance * sample_weights
        ).sum() / sample_weights.sum().clamp_min(1e-12)
        expected_distance /= max(logits.shape[1] - 1, 1)
        return (
            hard
            + self.soft_weight * soft
            + self.distance_weight * expected_distance
        )

