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


class DualGoRankCNN(nn.Module):
    """Wider shared trunk with independent black and white rank heads."""

    def __init__(
        self,
        num_classes: int = len(RANKS),
        tabular_features: int = 0,
    ) -> None:
        super().__init__()
        self.tabular_features = tabular_features
        self.features = nn.Sequential(
            nn.Conv2d(FEATURE_CHANNELS, 96, 3, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.SiLU(inplace=True),
            ResidualBlock(96),
            ResidualBlock(96),
            nn.Conv2d(96, 192, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 192),
            nn.SiLU(inplace=True),
            ResidualBlock(192),
            nn.Conv2d(192, 384, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 384),
            nn.SiLU(inplace=True),
            ResidualBlock(384),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.tabular_encoder = (
            nn.Sequential(
                nn.BatchNorm1d(tabular_features),
                nn.Linear(tabular_features, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(0.15),
                nn.Linear(128, 128),
                nn.SiLU(inplace=True),
            )
            if tabular_features > 0
            else None
        )
        fusion_inputs = 384 + (128 if tabular_features > 0 else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_inputs, 384),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
        )

        def rank_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(384, 128),
                nn.SiLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, num_classes),
            )

        self.black_head = rank_head()
        self.white_head = rank_head()

    def forward(
        self,
        inputs: torch.Tensor,
        tabular: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.pool(self.features(inputs)).flatten(1)
        if self.tabular_encoder is not None:
            if tabular is None:
                raise ValueError("KataGo tabular features are required by this checkpoint")
            shared = torch.cat((shared, self.tabular_encoder(tabular)), dim=1)
        shared = self.fusion(shared)
        return self.black_head(shared), self.white_head(shared)

