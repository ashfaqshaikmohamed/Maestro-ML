"""
Helpers to plot waveform and mel-spectrogram for demo / README visuals.
"""

import matplotlib.pyplot as plt
import librosa.display
import numpy as np
from pathlib import Path


def plot_waveform(y: np.ndarray, sr: int, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_mel_spectrogram(mel: np.ndarray, sr: int, out_path: str, x_axis="time"):
    """
    mel: 2D numpy array (n_mels, time_frames)
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel, sr=sr, x_axis=x_axis, y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel spectrogram")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
