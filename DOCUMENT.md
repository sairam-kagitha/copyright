# DOCUMENT.md — AI-Based Semantic Copyright Detection

---

## 1. Project Overview

This project is an **AI-powered semantic copyright detection system** that determines whether an audio or video file contains content semantically similar to a known copyright-protected corpus (LibriSpeech dataset).

Unlike traditional fingerprint-based detection, this system operates at the **meaning/semantic level** — it transcribes speech to text, converts it into a dense vector embedding, and searches a prebuilt FAISS index for similarity matches. A trained ML classifier then makes the final copyright decision based on the retrieved similarity scores.

**Key capabilities:**
- Accepts `.wav`, `.mp3`, `.m4a` (audio) and `.mp4` (video) uploads
- Transcribes speech using OpenAI Whisper
- Embeds transcripts using Sentence-BERT (`all-MiniLM-L6-v2`)
- Searches a FAISS vector index built from LibriSpeech transcripts
- Classifies copyright infringement using a trained Logistic Regression (or SVM) model
- Delivers results through a Streamlit web UI with confidence score, risk level, top matches, and feature visualization

---

## 2. How to Run the Project

### Prerequisites

- **Python**: 3.8 or higher (inferred from dependencies)
- **System dependency**: `ffmpeg` must be installed and accessible in `PATH`

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

---

### Environment Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install Python dependencies
pip install -r requirements.txt
```

> **Note:** Installing `torch` and `openai-whisper` may take several minutes. GPU support is optional but speeds up Whisper transcription significantly.

---

### One-Time Setup (Build the Pipeline)

Run these scripts **once** in order before launching the app. Pre-built artifacts (`database/`, `embeddings/`, `saved_models/`) may already exist in the repo — skip any step whose output already exists.

#### Step 1 — Inspect the dataset (optional)
```bash
cd scripts
python load_dataset.py
```
Loads and previews the LibriSpeech parquet files from `../dataset/`.

#### Step 2 — Generate SBERT embeddings
```bash
cd scripts
python generate_embeddings.py
```
Encodes all dataset transcripts using `all-MiniLM-L6-v2`. Output: `embeddings/embeddings.npy`

#### Step 3 — Build the FAISS index
```bash
cd scripts
python build_faiss_db.py
```
Creates a FAISS `IndexFlatIP` (inner product / cosine similarity) index. Output: `database/faiss_index.bin`, `database/transcripts.npy`

#### Step 4 — Prepare negative training samples (TED-LIUM)
```bash
# From project root
python data.py
```
Streams the TED-LIUM Release 2 dataset via HuggingFace and saves cleaned sentence chunks. Output: `embeddings/ted_transcripts.npy`

> Alternatively, `process.py` can be used if a local `transcript_data.csv` is available.

#### Step 5 — Train the ML classifier
```bash
cd scripts
python train_ml.py
```
Generates positive/negative feature samples, trains the model, evaluates it, and saves artifacts. Output: `saved_models/copyright_model.pkl`, `saved_models/feature_scaler.pkl`, `saved_models/training_config.pkl`

---

### Launch the Application

```bash
# From project root
streamlit run app.py
```

The app will open at `http://localhost:8501` by default.

---

### Environment Variables / Config

There are no required environment variables. All paths are relative to the project root. The following constants are hardcoded in `app.py` and `scripts/train_ml.py` and can be adjusted directly:

| Constant | Location | Default | Purpose |
|---|---|---|---|
| `TOP_K` | `app.py`, `train_ml.py` | `6` | Number of FAISS neighbors retrieved |
| `NUM_POS` / `NUM_NEG` | `train_ml.py` | `8000` each | Training sample counts |
| `MODEL_TYPE` | `train_ml.py` | `"logistic"` | Classifier type (`"logistic"` or `"svm"`) |
| `BATCH_SIZE` | `train_ml.py` | `64` | Embedding batch size during training |

---

## 3. Project Structure Explanation

```
Project/
│
├── app.py                    # Main Streamlit application — entry point for inference
├── data.py                   # Streams TED-LIUM via HuggingFace; builds negative corpus
├── process.py                # Alternate negative corpus builder from local transcript_data.csv
├── text.py                   # Dev utility: inspects saved transcripts.npy; commented-out PCA visualizer
├── test_vid_aud.py           # Standalone test script for video-to-audio extraction via ffmpeg
├── transcript_data.csv       # Local CSV of transcripts used by process.py as negative samples
├── extracted_audio.wav       # Temporary output from test_vid_aud.py (not used by the app)
├── requirements.txt          # Python package dependencies
│
├── scripts/
│   ├── load_dataset.py       # Loads and previews LibriSpeech parquet dataset; one-time utility
│   ├── generate_embeddings.py# Encodes dataset transcripts with SBERT → embeddings/embeddings.npy
│   ├── build_faiss_db.py     # Builds FAISS index from embeddings; saves index + transcript map
│   ├── speech_to_text.py     # Standalone CLI: transcribes a single audio file with Whisper
│   ├── match_audio.py        # CLI prototype: transcribes + FAISS-matches audio (threshold-based, pre-ML)
│   └── train_ml.py           # Full ML training pipeline: feature engineering, training, evaluation, saving
│
├── utils/
│   ├── audio_utils.py        # Helper: extract_audio_from_video() wraps ffmpeg subprocess call
│   └── text_utils.py         # Currently empty; reserved for text preprocessing helpers
│
├── dataset/
│   └── 0000–0013.parquet     # LibriSpeech dataset shards (14 files); used as the copyright reference corpus
│
├── embeddings/
│   ├── embeddings.npy        # SBERT embeddings for all dataset transcripts (float32, shape: [N, 384])
│   └── ted_transcripts.npy   # Cleaned TED-LIUM sentence chunks used as negative training samples
│
├── database/
│   ├── faiss_index.bin       # Serialized FAISS IndexFlatIP; loaded at inference time
│   └── transcripts.npy       # Raw transcript strings parallel to FAISS index rows; used for result display
│
├── saved_models/
│   ├── copyright_model.pkl   # Trained classifier (Logistic Regression or SVM) via joblib
│   ├── feature_scaler.pkl    # Fitted StandardScaler applied to 8-feature input vectors
│   └── training_config.pkl   # Saved training hyperparameters/config dict for reproducibility
│
├── plots/
│   └── Figure_1/2.png        # Training evaluation plots (ROC curve, confusion matrix — inferred)
│
├── audio_input/              # Directory for storing input audio files (currently empty)
├── video_input/              # Directory for storing input video files; used in test_vid_aud.py
└── results/                  # Reserved output directory (currently empty)
```

---

## 4. Execution Flow

### Phase A — One-Time Pipeline Setup

```
dataset/*.parquet
       │
       ▼
scripts/generate_embeddings.py
  → Loads all transcripts from parquet shards
  → Encodes each with SBERT (all-MiniLM-L6-v2, 384-dim, L2-normalized)
  → Saves: embeddings/embeddings.npy  [shape: N × 384]
       │
       ▼
scripts/build_faiss_db.py
  → Loads embeddings.npy
  → Creates FAISS IndexFlatIP (cosine similarity via normalized inner product)
  → Adds all N vectors to the index
  → Saves: database/faiss_index.bin
  → Saves: database/transcripts.npy  (parallel text array for result lookup)
       │
       ▼
data.py  (negative corpus)
  → Streams TED-LIUM Release 2 from HuggingFace (up to 50,000 examples)
  → Cleans and filters text (lowercase, strip punctuation, min length 20)
  → Saves: embeddings/ted_transcripts.npy
       │
       ▼
scripts/train_ml.py
  → Loads embeddings.npy + faiss_index.bin + ted_transcripts.npy
  → Generates POSITIVE samples (8,000):
      For each LibriSpeech embedding[i]:
        Query FAISS → retrieve top-6 neighbors
        Extract 8 features: [s1, s2, s3, s4, s5, mean, std, gap]  → label = 1
  → Generates NEGATIVE samples (8,000):
      For each TED sentence (batches of 64):
        Encode with SBERT → Query FAISS → extract same 8 features  → label = 0
  → Splits 80/20 train/test with stratification
  → Fits StandardScaler on training features
  → Trains Logistic Regression (or SVM) classifier
  → Evaluates: classification report, confusion matrix, ROC-AUC
  → Saves: saved_models/copyright_model.pkl
           saved_models/feature_scaler.pkl
           saved_models/training_config.pkl
```

---

### Phase B — Runtime Inference (Streamlit App)

```
User opens http://localhost:8501
       │
       ▼
app.py  →  load_resources()  [cached via @st.cache_resource]
  → whisper.load_model("base")
  → SentenceTransformer("all-MiniLM-L6-v2")
  → faiss.read_index("database/faiss_index.bin")
  → joblib.load("saved_models/copyright_model.pkl")
  → joblib.load("saved_models/feature_scaler.pkl")
  → np.load("database/transcripts.npy")
       │
       ▼
User uploads file (.wav / .mp3 / .m4a / .mp4)
  → Saved to a temp file
  → If .mp4:  utils/audio_utils.extract_audio_from_video()
                → ffmpeg subprocess: extracts mono 16kHz PCM WAV
  → If audio: used directly
       │
       ▼
Whisper transcription
  → whisper_model.transcribe(audio_path)
  → Produces plain-text transcript string
  → Displayed in UI under "Transcription"
       │
       ▼
SBERT embedding
  → sbert_model.encode([transcript], normalize_embeddings=True)
  → Produces query vector of shape [1, 384]
       │
       ▼
FAISS search
  → index.search(query_embedding, TOP_K=6)
  → Returns distances D [1×6] and indices I [1×6]
  → Top-5 similarity scores extracted: [s1, s2, s3, s4, s5]
       │
       ▼
Feature engineering
  → mean  = mean(s1..s5)
  → std   = std(s1..s5)
  → gap   = s1 − s2
  → feature_vector = [s1, s2, s3, s4, s5, mean, std, gap]  (8 values)
       │
       ▼
ML classification
  → scaler.transform([feature_vector])
  → clf.predict()        → binary label (0 = no copyright, 1 = copyright)
  → clf.predict_proba()  → confidence score (probability of class 1)
       │
       ▼
Risk level assignment
  → confidence > 0.9  →  "Very High Risk"
  → confidence > 0.7  →  "High Risk"
  → confidence > 0.5  →  "Moderate Risk"
  → else              →  "Low Risk"
       │
       ▼
UI Output
  → Copyright detected / not detected banner
  → Confidence score + Top similarity score (metrics)
  → Risk level info box
  → Top 5 matched dataset sentences with similarity scores
  → Bar chart of 8-feature decision vector
       │
       ▼
Cleanup
  → Temp file deleted
  → Extracted audio WAV deleted (if input was .mp4)
```

---

*All model paths, index paths, and dataset paths are relative to the project root directory. The app must be launched from the project root (`streamlit run app.py`) for relative imports and file paths to resolve correctly.*