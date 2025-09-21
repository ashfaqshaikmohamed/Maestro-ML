"""
Unit tests for dataset loading and feature extraction.
Run with: pytest tests/test_dataset.py
"""

import pytest
import numpy as np
from src.data.dataset import load_dataset
from src.data.feature_extraction import extract_mfcc

def test_load_dataset_structure(tmp_path):
    # Using tmp_path to simulate fake dataset folder
    d = tmp_path / "data"
    d.mkdir()
    (d / "track1.wav").write_text("fake-audio")
    
    data = load_dataset(str(d))
    assert isinstance(data, list), "Dataset should return a list of file paths"
    assert len(data) == 1, "Should load exactly one file"
    assert str(d / "track1.wav") in data

def test_feature_extraction_shape():
    # Simulate 1D random signal
    signal = np.random.randn(22050)
    mfccs = extract_mfcc(signal, sr=22050)
    assert mfccs.shape[0] > 0, "MFCCs should have at least one coefficient"
    assert mfccs.shape[1] > 0, "MFCCs should have multiple frames"
