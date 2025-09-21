"""
Feature extraction utilities for music genre classification.

This module provides reusable functions to convert raw audio
into numerical feature vectors suitable for machine learning.
"""

import numpy as np
import librosa

def extract_mfcc(y, sr, n_mfcc=20):
    """Compute MFCC features."""
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)

def extract_chroma(y, sr):
    """Compute Chroma features (pitch class energy)."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)

def extract_contrast(y, sr):
    """Compute Spectral Contrast features."""
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    return np.mean(contrast, axis=1)

def extract_features(y, sr):
    """
    Extract a full feature vector from an audio signal.
    Combines MFCC, Chroma, and Spectral Contrast.
    """
    mfccs = extract_mfcc(y, sr)
    chroma = extract_chroma(y, sr)
    contrast = extract_contrast(y, sr)

    # Concatenate into one feature vector
    return np.hstack([mfccs, chroma, contrast])
