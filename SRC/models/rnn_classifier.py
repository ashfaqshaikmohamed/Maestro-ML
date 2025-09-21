"""
A hybrid CNN + RNN model to capture both local spectrogram features and temporal dynamics.

This model:
- Uses convolutional layers to obtain local features across frequency bins
- Collapses frequency dimension and uses a GRU over time frames
"""

import torch
import torch.nn as nn


class CNNRNNGenreClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, cnn_channels: int = 32, rnn_hidden: int = 128):
        super().__init__()
        # Small CNN to reduce frequency dimension but preserve time frames
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # downsample frequency but keep time resolution
        )

        # After CNN, we'll reshape to (B, time_steps, features) for RNN
        self.rnn = nn.GRU(input_size=cnn_channels * 64, hidden_size=rnn_hidden, batch_first=True, bidirectional=False)

        # Final linear
        self.fc = nn.Linear(rnn_hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, n_mels, time_frames)
        Returns:
            logits: (B, num_classes)
        """
        b = x.size(0)
        out = self.cnn(x)  # (B, C, H', T)
        c, h, t = out.size(1), out.size(2), out.size(3)

        # Prepare sequence: (B, T, C * H)
        out = out.permute(0, 3, 1, 2).contiguous()  # (B, T, C, H)
        out = out.view(b, t, c * h)

        rnn_out, _ = self.rnn(out)  # (B, T, rnn_hidden)
        last = rnn_out[:, -1, :]  # take last time-step
        logits = self.fc(last)
        return logits
