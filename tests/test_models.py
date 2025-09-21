"""
Unit tests for CNN and RNN models.
Run with: pytest tests/test_models.py
"""

import torch
from src.models.cnn_classifier import CNNClassifier
from src.models.rnn_classifier import RNNClassifier

def test_cnn_forward_pass():
    model = CNNClassifier(num_classes=10)
    x = torch.randn(4, 1, 128, 44)  # batch, channels, features, time
    out = model(x)
    assert out.shape == (4, 10), "CNN output shape mismatch"

def test_rnn_forward_pass():
    model = RNNClassifier(num_classes=10)
    x = torch.randn(4, 128, 44)  # batch, features, time
    out = model(x)
    assert out.shape == (4, 10), "RNN output shape mismatch"
