# Deep Learning Music Genre Classifier & Recommender

## Overview

This project explores the intersection of **artificial intelligence and music** by building a deep learning system that can:
1. **Classify music tracks into genres** using audio feature extraction (MFCC, spectrograms) and neural networks.  
2. **Recommend similar tracks** based on learned embeddings, simulating a basic content-based recommender system.  

It highlights the program's ability to design **end-to-end machine learning pipelines**:  
- Data preprocessing and feature engineering  
- Neural network modeling (CNNs, RNNs with PyTorch)  
- Training and evaluation on real audio datasets  
- Visualization of embeddings and performance  
- Deployment-oriented code for inference and recommendation  

---

## Why This Project Matters

Music classification and recommendation are core challenges in **modern streaming platforms** (Spotify, Apple Music, YouTube Music).  
This project demonstrates **how AI can learn meaningful patterns in audio signals**, enabling:  

- Better **music discovery** (finding new tracks similar to ones you like).  
- More accurate **genre tagging** for massive audio libraries.  
- Foundations for **personalized recommendation systems**.  

---

## Results

### 1. Model Performance

<img width="800" height="600" alt="492063545-11997df6-c30b-403e-8bce-1eb629db0b90" src="https://github.com/user-attachments/assets/7cf5872a-8cc5-4c60-9222-196fe1aa8cef" />

- Final **Validation Accuracy**: ~72%  
- Final **F1 Score**: ~0.71  
- Consistent improvement across epochs.  

---

## 2. Confusion Matrix

<img width="1024" height="1024" alt="492063600-2c8f0ff3-0de6-46af-9a70-c3f154a13c4d" src="https://github.com/user-attachments/assets/fac122cf-9fc6-4da0-9200-8e5d64968416" />

Shows how well the model distinguishes between genres.  

---

### 3. Embedding Visualization (t-SNE)

<img width="1024" height="1024" alt="492063626-18e41d42-7e0e-483e-b3e1-bd20746688d3" src="https://github.com/user-attachments/assets/8aeb8a1e-7f5c-4a75-bed8-c2779634c8dc" />

t-SNE plots reveal clustering of learned track embeddings by genre.  

---

### 4. Training Curves

<img width="1024" height="1024" alt="unnamed" src="https://github.com/user-attachments/assets/491140e7-91ca-4d86-b2b7-cdc8a4e4e582" />

<img width="1024" height="1024" alt="unnamed" src="https://github.com/user-attachments/assets/cae26b38-305a-4fe9-a022-42ebc0237183" />

These plots illustrate convergence and help diagnose overfitting.  

---

### 5. Recommender System Example

<img width="1024" height="1024" alt="492063676-979cbcc0-5a2c-4a9c-95d2-ee501a949388" src="https://github.com/user-attachments/assets/0e76f4bb-8f2a-4802-a131-04b25e1d4ceb" />


## Tech Stack

- **Languages**: Python (3.9+)  
- **Core ML Frameworks**: PyTorch, Scikit-learn  
- **Audio Processing**: Librosa  
- **Visualization**: Matplotlib, Seaborn, t-SNE  
- **Testing**: Pytest  
- **Project Management**: Modularized `src/` structure with reusable scripts  

---

## How to Run

1. **Clone repo**
   ```bash
   git clone https://github.com/yourusername/music-genre-classifier.git
   cd music-genre-classifier
