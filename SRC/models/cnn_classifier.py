"""
Compact CNN for music genre classification using mel-spectrogram inputs.

Input shape: (batch, 1, n_mels, time_frames)
Designed to be small (fast to train) but expressive enough for demonstration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNGenreClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Feature extractor: conv blocks with batchnorm + relu + pooling
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (B,16,H,W)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # reduce H,W by 2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
        )

        # Adaptive pooling to guarantee fixed-size feature map regardless of input time length
        self.adapt_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: torch.Tensor, shape (B, 1, H=n_mels, W=time_frames)

        Returns:
            logits (B, num_classes)
        """
        x = self.features(x)
        x = self.adapt_pool(x)
        x = self.classifier(x)
        return x
