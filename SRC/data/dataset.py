"""
Dataset utilities for music genre classification.

This module handles:
- Traversing the dataset folder structure
- Loading audio files
- Splitting into train/test sets
- Wrapping data in PyTorch-friendly Dataset objects
"""

import os
import librosa
import torch
from torch.utils.data import Dataset

class GenreDataset(Dataset):
    """
    Custom PyTorch Dataset for music genre classification.
    Each sample is an audio file, transformed into a feature vector.
    """

    def __init__(self, file_paths, labels, transform=None):
        """
        Args:
            file_paths (list[str]): Paths to audio files.
            labels (list[int]): Numeric labels for each file.
            transform (callable, optional): Optional transform to apply.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Loads one audio file and returns (features, label).
        """
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # Load audio
        y, sr = librosa.load(file_path, duration=30)

        # By default, return raw waveform; feature extraction can be done later
        sample = {"audio": y, "sr": sr, "label": label}

        # Apply optional transform (e.g., MFCC extraction)
        if self.transform:
            sample = self.transform(sample)

        return sample
