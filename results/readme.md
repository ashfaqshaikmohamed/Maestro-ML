# Results & Insights – Music Genre Classifier & Recommender

This folder contains the **outputs and visual evidence** of the project’s success.  
The goal is not just to build a classifier, but to **demonstrate the potential of AI in music understanding, discovery, and personalization**.  

---

## Model Performance

Our deep learning models (CNN & RNN) were trained on MFCC-based audio features to classify tracks into **10 different genres**.  
Key evaluation metrics over epochs show consistent improvements in both training and validation performance:

<img width="800" height="600" alt="BCEI fa2e7ff0-b40a-4935-9550-018c2073f911" src="https://github.com/user-attachments/assets/11997df6-c30b-403e-8bce-1eb629db0b90" />

- **Training Accuracy (final epoch):** ~76%  
- **Validation Accuracy (final epoch):** ~72%  
- **F1 Score:** ~0.71 (balanced across classes)

---

## Confusion Matrix

The confusion matrix shows how well the model distinguishes between genres.  
It highlights strengths (e.g., strong separation between “classical” vs “metal”) and challenges (e.g., overlap between “rock” and “pop”).  

<img width="1024" height="1024" alt="unnamed" src="https://github.com/user-attachments/assets/2c8f0ff3-0de6-46af-9a70-c3f154a13c4d" />

---

## Feature Embedding Space (t-SNE)

By projecting the learned audio embeddings into 2D using t-SNE, we can visualize how the model perceives genre similarity.  
Clusters reveal interesting overlaps between genres (e.g., **jazz & blues**, **pop & disco**).  

<img width="1024" height="1024" alt="unnamed" src="https://github.com/user-attachments/assets/18e41d42-7e0e-483e-b3e1-bd20746688d3" />

---

##  Beyond Classification: Recommender System

The embeddings learned by the model are not only useful for classification but also for **content-based music recommendation**.  
Given a track, we can find the closest neighbors in embedding space and recommend musically similar songs.  

<img width="1024" height="1024" alt="unnamed" src="https://github.com/user-attachments/assets/979cbcc0-5a2c-4a9c-95d2-ee501a949388" />

---

## Future Development

This project demonstrates a strong foundation but can be extended in many exciting ways:  

1. **Hybrid Recommender Systems** – combine content-based embeddings with collaborative filtering from user behavior.  
2. **Multi-Label Classification** – songs often span multiple genres; extend beyond single-label.  
3. **Streaming-Friendly Models** – real-time inference for music apps.  
4. **Explainable AI in Music** – highlight which parts of the audio influenced the classification.  
5. **Integration with Frontend UI** – deploy as a web or mobile app demo.  

---

## Takeaway

This project highlights how **AI bridges data and creativity**:  
- From raw waveforms → to meaningful audio features → to **genre understanding & music discovery**.  
- Showcasing **end-to-end ML engineering**: data pipelines, model design, evaluation, visualization, and user-facing applications.  

---
