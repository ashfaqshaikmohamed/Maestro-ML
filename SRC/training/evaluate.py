"""
Evaluation utilities: compute accuracy, F1, confusion matrix, and helper to run evaluation loop.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch


def evaluate_model(model: torch.nn.Module, dataloader, device="cpu"):
    """
    Run model on dataloader and compute accuracy, weighted F1, confusion matrix.

    Returns:
        dict with keys: accuracy, f1, confusion_matrix (numpy array)
    """
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for batch in dataloader:
            # Accept either (x, y) or dict with 'audio'/'label' depending on dataset
            if isinstance(batch, (list, tuple)):
                x, y = batch
            elif isinstance(batch, dict):
                x, y = batch["input"], batch["label"]
            else:
                # fallback: assume (x,y)
                try:
                    x, y = batch
                except Exception as e:
                    raise ValueError("Unsupported batch type in evaluate_model") from e

            x = x.to(device)
            out = model(x)
            p = out.argmax(dim=1).cpu().numpy()
            preds.extend(p.tolist())
            trues.extend(y.cpu().numpy().tolist())

    preds = np.array(preds)
    trues = np.array(trues)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average="weighted")
    cm = confusion_matrix(trues, preds)
    return {"accuracy": float(acc), "f1": float(f1), "confusion_matrix": cm}
