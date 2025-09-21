"""
Simple content-based recommender using cosine similarity on embeddings.

Expectations:
- An embeddings matrix (N, D) where N is #tracks and D is embedding dimension.
- A parallel list of track identifiers (filenames or IDs).
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from pathlib import Path


class Recommender:
    def __init__(self, embeddings: np.ndarray, track_ids: list):
        assert embeddings.shape[0] == len(track_ids)
        self.embeddings = embeddings
        self.track_ids = track_ids
        # normalize embeddings for faster cosine similarity if desired
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        self._normed = self.embeddings / norms

    def recommend(self, seed_idx: int, top_k: int = 5):
        """
        Return top_k track_ids most similar to seed_idx (excluding itself).
        """
        q = self._normed[seed_idx:seed_idx + 1]  # (1, D)
        sims = (q @ self._normed.T).flatten()  # cosine similarities
        order = np.argsort(-sims)
        # filter out the seed
        recs = [self.track_ids[i] for i in order if i != seed_idx]
        return recs[:top_k]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"embeddings": self.embeddings, "track_ids": self.track_ids}, path)

    @classmethod
    def load(cls, path: str):
        data = joblib.load(path)
        return cls(data["embeddings"], data["track_ids"])
