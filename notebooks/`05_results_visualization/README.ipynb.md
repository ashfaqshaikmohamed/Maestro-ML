
---

## 📓 `05_results_visualization.ipynb.md`
```markdown
# 05 - Results Visualization

Visualize performance using confusion matrices, accuracy, and t-SNE embeddings.

---

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Predictions
y_pred = model(X_test).argmax(dim=1).numpy()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
