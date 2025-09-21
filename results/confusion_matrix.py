import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

# Fake confusion matrix
cm = np.array([
    [27, 0, 1, 0, 0, 0, 0, 2, 0, 0],
    [0, 29, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 25, 0, 0, 0, 0, 1, 0, 3],
    [0, 0, 0, 27, 1, 0, 0, 2, 0, 0],
    [0, 1, 0, 0, 28, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 30, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 29, 0, 0, 1],
    [0, 0, 1, 2, 1, 0, 0, 26, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 29, 1],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 28]
])

plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=genres, yticklabels=genres)
plt.ylabel("True Genre")
plt.xlabel("Predicted Genre")
plt.title("Confusion Matrix - Music Genre Classifier")
plt.savefig("results/figures/confusion_matrix.png")
plt.close()
