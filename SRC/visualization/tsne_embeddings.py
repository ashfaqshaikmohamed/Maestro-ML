"""
Compute and plot t-SNE visualization of embeddings grouped by labels.
"""

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray, label_names: list, out_path: str, perplexity: int = 30):
    """
    Fit a t-SNE projection and save a scatter plot.

    Args:
        embeddings: (N, D) numpy array
        labels: (N,) int array of label indices
        label_names: list mapping index -> string
        out_path: file path to save PNG
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="random", random_state=42)
    z = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(label_names):
        idxs = (labels == i)
        if idxs.sum() == 0:
            continue
        plt.scatter(z[idxs, 0], z[idxs, 1], label=name, alpha=0.7)
    plt.legend()
    plt.title("t-SNE of Track Embeddings")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
