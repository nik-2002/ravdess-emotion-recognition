"""Model definition for the RAVDESS mel-spectrogram CNN."""

from __future__ import annotations

import torch
from torch import nn


class SimpleMelCNN(nn.Module):
    """CNN architecture used for mel-spectrogram emotion classification."""

    def __init__(self, n_mels: int, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        dummy = torch.zeros(1, 1, n_mels, 200)
        with torch.no_grad():
            feat = self.features(dummy)
        flattened = feat.view(1, -1).shape[1]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class ImprovedMelCNN(nn.Module):
    """Deep CNN architecture for mel-spectrogram emotion classification.
    
    Improvements over SimpleMelCNN:
    - one extra conv block (128 filters)
    - larger dense layer (256 units)
    - slightly higher dropout for regularization
    """

    def __init__(self, n_mels: int, n_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Block 4 (New)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        # Compute flattened size dynamically
        dummy = torch.zeros(1, 1, n_mels, 200)
        with torch.no_grad():
            feat = self.features(dummy)
        flattened = feat.view(1, -1).shape[1]
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 256),  # Increased size
            nn.ReLU(),
            nn.Dropout(0.4),            # Increased dropout
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
