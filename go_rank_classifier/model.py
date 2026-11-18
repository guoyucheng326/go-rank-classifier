from __future__ import annotations

import torch
from torch import nn

from .features import FEATURE_CHANNELS
from .ranks import RANKS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.body(inputs))


class GoRankCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = len(RANKS),
        tabular_features: int = 0,
    ) -> None:
        super().__init__()
        self.tabular_features = tabular_features
        self.features = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.SiLU(inplace=True),
            ResidualBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 256),
            nn.SiLU(inplace=True),
            ResidualBlock(256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.tabular_encoder = (
            nn.Sequential(
                nn.LayerNorm(tabular_features),
                nn.Linear(tabular_features, 64),
                nn.SiLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(64, 64),
                nn.SiLU(inplace=True),
            )
            if tabular_features > 0
            else None
        )
        classifier_inputs = 256 + (64 if tabular_features > 0 else 0)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(classifier_inputs, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        tabular: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = self.pool(self.features(inputs)).flatten(1)
        if self.tabular_encoder is not None:
            if tabular is None:
                raise ValueError("KataGo tabular features are required by this checkpoint")
            pooled = torch.cat((pooled, self.tabular_encoder(tabular)), dim=1)
        return self.classifier(pooled)

