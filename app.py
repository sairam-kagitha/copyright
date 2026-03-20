# ==========================================
# AI-Based Semantic Copyright Detection
# Production Version (Professional UI)
# ==========================================

import streamlit as st
import whisper
import numpy as np
import faiss
import joblib
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from utils.audio_utils import extract_audio_from_video
# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------

st.set_page_config(
    page_title="Semantic Copyright Detection",
    layout="wide"
)

st.title("🎧 AI-Based Semantic Copyright Detection")
st.markdown("Upload an audio or video file to detect semantic copyright similarity.")

# ------------------------------------------
# LOAD MODELS (CACHED FOR PERFORMANCE)
# ------------------------------------------


@st.cache_resource
def load_resources():
    whisper_model = whisper.load_model("base")

    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    index = faiss.read_index("database/faiss_index.bin")

    clf = joblib.load("saved_models/copyright_model.pkl")
    scaler = joblib.load("saved_models/feature_scaler.pkl")

    transcripts = np.load("database/transcripts.npy", allow_pickle=True)

    return whisper_model, sbert_model, index, clf, scaler, transcripts

whisper_model, sbert_model, index, clf, scaler, transcripts = load_resources()

TOP_K = 6

# ------------------------------------------
# FILE UPLOAD
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Audio / Video File",
    type=["wav", "mp3", "m4a", "mp4"]
)

if uploaded_file is not None:

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    file_extension = uploaded_file.name.split(".")[-1].lower()

    # If video → extract audio
    if file_extension == "mp4":
        st.video(temp_path)
        audio_path = extract_audio_from_video(temp_path)
    else:
        st.audio(temp_path)
        audio_path = temp_path

    # --------------------------------------
    # TRANSCRIPTION
    # --------------------------------------

    with st.spinner("Transcribing audio..."):
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]

    st.subheader("📝 Transcription")
    st.write(transcript)

    # --------------------------------------
    # EMBEDDING + FAISS SEARCH
    # --------------------------------------

    with st.spinner("Computing similarity..."):
        query_embedding = sbert_model.encode(
            [transcript],
            normalize_embeddings=True
        )

        D, I = index.search(query_embedding, TOP_K)

        sims = D[0][0:5]
        indices = I[0][0:5]

        mean = np.mean(sims)
        std = np.std(sims)
        gap = sims[0] - sims[1]

        feature_vector = list(sims) + [mean, std, gap]

        X_input = scaler.transform([feature_vector])

        prediction = clf.predict(X_input)[0]
        confidence = clf.predict_proba(X_input)[0][1]

    # --------------------------------------
    # RESULT SECTION
    # --------------------------------------

    st.subheader("🔎 Detection Result")

    col1, col2, col3 = st.columns(3)

    if prediction == 1:
        col1.error("🚨 Copyright Detected")
    else:
        col1.success("✅ No Copyright Detected")

    col2.metric("Confidence Score", f"{confidence:.4f}")
    col3.metric("Top Similarity", f"{sims[0]:.4f}")

    # Risk Interpretation
    if confidence > 0.9:
        risk = "Very High Risk"
    elif confidence > 0.7:
        risk = "High Risk"
    elif confidence > 0.5:
        risk = "Moderate Risk"
    else:
        risk = "Low Risk"

    st.info(f"📊 Risk Level: **{risk}**")

    # --------------------------------------
    # MATCHED DATASET SENTENCES
    # --------------------------------------

    st.subheader("📚 Top Matched Dataset Sentences")

    for rank in range(5):
        st.markdown(f"**Rank {rank+1} — Similarity: {sims[rank]:.4f}**")
        st.write(transcripts[indices[rank]])
        st.divider()

    # --------------------------------------
    # EXPLAINABLE FEATURE VISUALIZATION
    # --------------------------------------

    st.subheader("📈 Decision Feature Visualization")

    feature_names = ["s1", "s2", "s3", "s4", "s5", "mean", "std", "gap"]

    fig, ax = plt.subplots()
    ax.bar(feature_names, feature_vector)
    ax.set_title("Feature Contribution Pattern")
    plt.xticks(rotation=45)

    st.pyplot(fig)

    # --------------------------------------
    # CLEAN TEMP FILE
    # --------------------------------------

    os.remove(temp_path)

    if file_extension == "mp4":
        os.remove(audio_path)