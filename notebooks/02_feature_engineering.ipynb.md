
---

## 📓 `02_feature_engineering.ipynb.md`
```markdown
# 02 - Feature Engineering

We extract meaningful features (MFCC, Chroma, Spectral Contrast) for model training.

---

```python
import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = "../data/genres_original"
OUTPUT_FILE = "../data/features.csv"

features = []

def extract_features(file_path, genre):
    y, sr = librosa.load(file_path, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1)
    return np.hstack([mfccs, chroma, contrast])

for g in tqdm(os.listdir(DATA_DIR)):
    genre_dir = os.path.join(DATA_DIR, g)
    for file in os.listdir(genre_dir):
        fpath = os.path.join(genre_dir, file)
        feat = extract_features(fpath, g)
        features.append([g] + feat.tolist())

columns = ["genre"] + [f"mfcc{i}" for i in range(20)] + [f"chroma{i}" for i in range(12)] + [f"contrast{i}" for i in range(7)]
df = pd.DataFrame(features, columns=columns)
df.to_csv(OUTPUT_FILE, index=False)
df.head()
