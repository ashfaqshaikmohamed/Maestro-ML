"""
Simple Streamlit demo to upload an audio file and display predicted genre.

Usage:
    streamlit run src/inference/demo_app.py
"""

import os
import streamlit as st
from src.inference.predict import load_model, predict_file

st.set_page_config(page_title="Music Genre Classifier", layout="centered")
st.title("🎵 Music Genre Classifier - Demo")

st.markdown("Upload an audio file (wav, mp3) and the model will predict its genre.")

uploaded = st.file_uploader("Choose audio file", type=["wav", "mp3"])
model_path = st.text_input("Model checkpoint path", value="results/models/best.pth")

if uploaded:
    tmp_dir = "temp"
    os.makedirs(tmp_dir, exist_ok=True)
    file_path = os.path.join(tmp_dir, uploaded.name)
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Running model...")
    try:
        model = load_model(model_path)
        idx, label = predict_file(model, file_path)
        st.success(f"Predicted genre: **{label}** (index={idx})")
    except Exception as e:
        st.error(f"Failed to run inference: {e}")
