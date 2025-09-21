"""
Utility functions for training: checkpointing, metrics persistence, and simple logger.
"""

import json
from pathlib import Path
import torch
import time


def save_checkpoint(state: dict, filepath: str):
    """
    Save a training checkpoint.

    Args:
        state: dict containing model_state, optimizer_state, epoch, etc.
        filepath: where to save .pth
    """
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, map_location=None):
    """
    Load checkpoint state dict into model (and optimizer if provided).
    Returns epoch number (if stored) or None.
    """
    ckpt = torch.load(filepath, map_location=map_location)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", None)


def write_metrics(metrics: dict, filepath: str):
    """
    Save metrics (dict) to JSON file for later inspection / README.
    """
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


class SimpleTimer:
    """Context manager for timing code blocks."""

    def __init__(self, name="block"):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.time() - self.start
        print(f"[TIMER] {self.name} took {elapsed:.2f}s")
