"""Simple model that is fast on GPU — the bottleneck should be the DataLoader."""
from __future__ import annotations

import torch
import torch.nn as nn


class SmallClassifier(nn.Module):
    """2-layer MLP with a leading conv to consume 3x200x200 images."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=4, padding=3),  # -> 16x50x50
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(5),                               # -> 16x5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
