"""
Unit tests for training loop utilities.
Run with: pytest tests/test_training.py
"""

import torch
import torch.nn as nn
from src.training.utils import accuracy

def test_accuracy_function():
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    targets = torch.tensor([1, 0])
    acc = accuracy(preds, targets)
    assert 0.0 <= acc <= 1.0, "Accuracy should be between 0 and 1"
    assert acc == 1.0, "Both predictions should be correct"
