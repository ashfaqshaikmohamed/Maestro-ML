# Music Genre Classifier – Documentation

This documentation provides a high-level overview of the project.

## Architecture

The system consists of four main components:

1. **Data Pipeline**  
   - Loads raw audio files (WAV/MP3).
   - Extracts MFCC features and spectrograms.
   - Prepares PyTorch Datasets and Dataloaders.

2. **Models**  
   - **CNN Classifier**: Learns local frequency-time patterns.  
   - **RNN Classifier**: Captures temporal sequence dependencies.  
   - Both models output logits across 10 music genres.

3. **Training Loop**  
   - Standard supervised training with cross-entropy loss.
   - Tracks metrics (loss, accuracy, F1).
   - Supports GPU acceleration.

4. **Inference + Recommender**  
   - `predict.py`: Run predictions on new audio files.  
   - `recommender.py`: Recommend similar tracks using embedding similarity.  
   - `demo_app.py`: Simple CLI demo.

## Results
- Confusion Matrix

![Conf-edited](https://github.com/user-attachments/assets/3592d7c4-1d83-42dd-8c12-8a6983cccfb9)

- t-SNE Visualization

![cnn_embed_6k](https://github.com/user-attachments/assets/f2f01e68-c624-4a3b-a2cd-e620cceb0f0e)

- Accuracy & Loss Curves

<img width="1280" height="960" alt="Example-of-Training-Learning-Curve-Showing-An-Underfit-Model-That-Requires-Further-Training" src="https://github.com/user-attachments/assets/922b94b0-2aa9-447f-8086-8d51a666d0f7" />
