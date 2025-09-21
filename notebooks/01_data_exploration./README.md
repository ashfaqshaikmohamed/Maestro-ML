# 01 - Data Exploration

This notebook explores the dataset, checks audio properties, and visualizes class distributions.

---

```python
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Path to dataset (e.g., GTZAN)
DATA_DIR = "../data/genres_original"

genres = os.listdir(DATA_DIR)
print("Genres found:", genres)
