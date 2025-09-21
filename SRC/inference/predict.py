"""
Inference helper to predict a single audio file's genre using a trained model.

Functions:
- audio_path_to_mel: quick conversion using librosa (must match training preprocessing)
- load_model: loads checkpoint and returns model in eval mode
- predict_file: returns predicted label index (or class string)
"""

import numpy as np
import torch
import librosa
from pathlib import Path
from src.models.cnn_classifier import CNNGenreClassifier

# default genre list for GTZAN; replace if your dataset differs
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]


def audio_path_to_mel(path, sr=22050, n_mels=128, duration=30):
    """
    Load audio file and convert to normalized mel-spectrogram (float32).
    Returns array shaped (1, n_mels, time_frames)
    """
    y, _ = librosa.load(path, sr=sr, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-9)
    return mel_db.astype("float32")[None, ...]


def load_model(checkpoint_path: str, device=None, num_classes: int = 10):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = CNNGenreClassifier(num_classes=num_classes)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def predict_file(model, file_path: str, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    x = audio_path_to_mel(file_path)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)  # (1,1,H,W)
    with torch.no_grad():
        logits = model(x_tensor)
        idx = int(torch.argmax(logits, dim=1).cpu().item())
    return idx, GENRES[idx]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="Audio file to classify")
    parser.add_argument("--model", type=str, default="results/models/best.pth", help="Model checkpoint")
    args = parser.parse_args()

    model = load_model(args.model)
    idx, label = predict_file(model, args.file)
    print(f"Predicted index: {idx}, label: {label}")
