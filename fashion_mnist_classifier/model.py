from __future__ import annotations

import torch
from torch import nn


class SimpleCnn(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 14x14
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(images)
        logits = self.classifier(features)
        return logits


def build_model(model_name: str, dropout: float) -> nn.Module:
    normalized_name = model_name.strip().lower()
    if normalized_name == "simple_cnn":
        return SimpleCnn(dropout=dropout)
    raise ValueError(f"Unknown model name: {model_name!r}")
