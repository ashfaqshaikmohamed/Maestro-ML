
---

## 📓 `04_recommender_system.ipynb.md`
```markdown
# 04 - Recommender System

Build a simple content-based recommender using learned embeddings.

---

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Extract embeddings from trained model
model.eval()
embeddings = model(X_test).detach().numpy()

# Compute similarity matrix
sim_matrix = cosine_similarity(embeddings)

def recommend(idx, top_k=5):
    sims = sim_matrix[idx]
    recs = np.argsort(-sims)[1:top_k+1]
    return recs

# Example: recommend songs similar to the first test sample
print("Original Genre:", encoder.classes_[y_test[0]])
for rec in recommend(0, top_k=5):
    print("Recommended:", encoder.classes_[y_test[rec]])
