print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

from sklearn.manifold import TSNE

X_embedded = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

plt.figure(figsize=(10,8))
for g in np.unique(y_test):
    idxs = y_test == g
    plt.scatter(X_embedded[idxs,0], X_embedded[idxs,1], label=encoder.classes_[g], alpha=0.7)
plt.legend()
plt.title("t-SNE of Song Embeddings")
plt.show()
