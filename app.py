# ==========================================
# AI-Based Semantic Copyright Detection
# Clean Professional UI Version
# ==========================================

import os
import tempfile

import faiss
import joblib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import whisper
from sentence_transformers import SentenceTransformer

from utils.audio_utils import extract_audio_from_video

# ------------------------------------------
# PAGE CONFIG
# ------------------------------------------

st.set_page_config(page_title="Semantic Copyright Detection", layout="wide")

# ------------------------------------------
# UI STYLING
# ------------------------------------------

st.markdown(
    """
<style>

/* Background */
[data-testid="stAppViewContainer"]{
background: linear-gradient(135deg,#0f2027,#203a43,#2c5364);
color:white;
}

/* Title */
h1{
text-align:center;
font-size:40px;
color:#00FFD1;
}

/* Section headers */
h2,h3{
color:#FFD369;
}

/* Uploader */
[data-testid="stFileUploader"]{
border:2px dashed #00FFD1;
padding:20px;
border-radius:10px;
background-color:rgba(255,255,255,0.05);
}

/* Metric values */
[data-testid="stMetricValue"]{
color:white;
font-size:30px;
font-weight:bold;
}

[data-testid="stMetricLabel"]{
color:#FFD369;
}

/* Result boxes */
.success-box{
background:#1d5b3a;
padding:20px;
border-radius:10px;
border:2px solid #00ff7f;
text-align:center;
font-size:22px;
font-weight:bold;
}

.error-box{
background:#5b1d1d;
padding:20px;
border-radius:10px;
border:2px solid #ff4c4c;
text-align:center;
font-size:22px;
font-weight:bold;
}

</style>
""",
    unsafe_allow_html=True,
)


st.title("🎧 AI-Based Semantic Copyright Detection")
st.markdown("Upload audio or video to detect semantic copyright similarity.")

# ------------------------------------------
# LOAD MODELS
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
    "Upload Audio / Video File", type=["wav", "mp3", "m4a", "mp4"]
)

# ------------------------------------------
# MAIN PIPELINE
# ------------------------------------------

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    file_extension = uploaded_file.name.split(".")[-1].lower()

    # --------------------------------------
    # VIDEO OR AUDIO PREVIEW
    # --------------------------------------

    if file_extension == "mp4":
        st.video(temp_path)
        audio_path = extract_audio_from_video(temp_path)
    else:
        st.audio(temp_path)
        audio_path = temp_path

    # --------------------------------------
    # TRANSCRIPTION
    # --------------------------------------

    with st.spinner("🧠 AI analyzing speech..."):
        result = whisper_model.transcribe(audio_path)
        transcript = result["text"]

    st.subheader("📝 Transcription")
    st.write(transcript)

    # --------------------------------------
    # EMBEDDING + FAISS SEARCH
    # --------------------------------------

    with st.spinner("🔍 Computing similarity..."):
        query_embedding = sbert_model.encode([transcript], normalize_embeddings=True)

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
    # METRICS
    # --------------------------------------

    st.subheader("🔎 Detection Result")

    col1, col2 = st.columns(2)

    col1.metric("🧠 Confidence Score", f"{confidence:.4f}")
    col2.metric("🔗 Top Similarity", f"{sims[0]:.4f}")

    # --------------------------------------
    # RESULT BANNER
    # --------------------------------------

    if prediction == 1:
        st.markdown(
            '<div class="error-box">🚨 COPYRIGHT DETECTED</div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="success-box">✅ NO COPYRIGHT DETECTED</div>',
            unsafe_allow_html=True,
        )

    # --------------------------------------
    # RISK LEVEL
    # --------------------------------------

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
    # MATCHED SENTENCES
    # --------------------------------------

    st.subheader("📚 Top Matched Dataset Sentences")

    for rank in range(5):
        st.markdown(
            f"""
        <div style="
        background-color:rgba(255,255,255,0.05);
        padding:15px;
        border-radius:8px;
        margin-bottom:10px">

        <b>Rank {rank + 1} — Similarity: {sims[rank]:.4f}</b><br>
        {transcripts[indices[rank]]}

        </div>
        """,
            unsafe_allow_html=True,
        )

    # --------------------------------------
    # FEATURE VISUALIZATION
    # --------------------------------------

    st.subheader("📈 Feature Contribution")

    feature_names = ["s1", "s2", "s3", "s4", "s5", "mean", "std", "gap"]

    fig, ax = plt.subplots(figsize=(6, 3))

    ax.bar(feature_names, feature_vector, color="#00FFD1")

    ax.set_facecolor("#1f3b4d")
    fig.patch.set_facecolor("#1f3b4d")

    ax.tick_params(colors="white")
    ax.set_title("Model Feature Pattern", color="white")

    plt.xticks(rotation=30)

    st.pyplot(fig)

    # --------------------------------------
    # CLEAN TEMP FILE
    # --------------------------------------

    os.remove(temp_path)

    if file_extension == "mp4":
        os.remove(audio_path)
