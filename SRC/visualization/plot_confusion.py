"""
Plot and save confusion matrix heatmap (matplotlib + seaborn).
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def plot_confusion_matrix(cm: np.ndarray, labels: list, out_path: str, figsize=(10, 8)):
    """
    Save a confusion matrix heatmap image.

    Args:
        cm: square numpy array of integers
        labels: list of label names
        out_path: file path to save PNG
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
