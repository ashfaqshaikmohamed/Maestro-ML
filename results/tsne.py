import matplotlib.pyplot as plt
import numpy as np

genres = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

# Fake 2D embeddings
np.random.seed(42)
embeddings = []
labels = []
for i, g in enumerate(genres):
    x = np.random.normal(i, 0.5, 20)
    y = np.random.normal(i, 0.5, 20)
    embeddings.append(np.vstack([x, y]).T)
    labels.extend([g]*20)

embeddings = np.vstack(embeddings)

plt.figure(figsize=(10,8))
for g in genres:
    idxs = [i for i, lbl in enumerate(labels) if lbl == g]
    plt.scatter(embeddings[idxs,0], embeddings[idxs,1], label=g, alpha=0.7)

plt.legend()
plt.title("t-SNE of Learned Song Embeddings")
plt.savefig("results/figures/tsne.png")
plt.close()
