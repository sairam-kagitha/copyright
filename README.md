# 🎧 AI-Based Copyright Detection — Beginner's Guide

This guide will walk you through everything you need to know about this project —
what it does, how to set it up, and how it works — in plain, simple English.

---

## 1. What This Project Does

Have you ever wondered how YouTube detects if a video contains copyrighted music?
This project does something similar — but for **spoken audio and video files**.

Here is what it does in simple words:

1. You upload an audio or video file (like a `.wav`, `.mp3`, or `.mp4`)
2. The app **listens to it** and converts the speech into text (like a live transcript)
3. It then checks if the **meaning** of that text is too similar to known copyrighted content
4. Finally, it tells you: **"Copyright Detected"** or **"No Copyright Detected"**

> 💡 **Why meaning and not exact words?**
> Someone could slightly change the words but still copy the idea.
> This project is smart enough to catch that too.

### What you will see in the app

- ✅ or 🚨 — whether copyright was found
- A **confidence score** (how sure the AI is, from 0 to 1)
- A **risk level** — Low / Moderate / High / Very High
- The top 5 most similar sentences from the dataset
- A bar chart showing what influenced the decision

---

## 2. How to Run the Project

> ⚠️ **These instructions are written for Windows users.**
> Follow each step in order. Do not skip steps.

---

### Step 1 — Make sure Python is installed

Open **Command Prompt** (`Win + R` → type `cmd` → press Enter) and run:

```
python --version
```

You should see something like `Python 3.10.x`.
If you see an error, download Python from [https://www.python.org/downloads/](https://www.python.org/downloads/)

> ✅ Make sure to check **"Add Python to PATH"** during installation.

---

### Step 2 — Install ffmpeg (required for video files)

`ffmpeg` is a free tool that extracts audio from video files. The project needs it.

1. Go to: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Download the **Windows build** (look for a "Windows" section with a pre-built zip)
3. Extract the zip file somewhere safe (e.g., `C:\ffmpeg`)
4. Add it to your system PATH:
   - Search for **"Edit the system environment variables"** in the Start menu
   - Click **Environment Variables**
   - Under **System Variables**, find `Path` → click **Edit**
   - Click **New** → paste the path to the `bin` folder (e.g., `C:\ffmpeg\bin`)
   - Click OK on everything

5. Verify it works by opening a new Command Prompt and running:

```
ffmpeg -version
```

You should see version info printed. If so, you are good to go.

---

### Step 3 — Open the project folder

In Command Prompt, navigate to the project folder:

```
cd path\to\Project
```

Replace `path\to\Project` with the actual path on your computer.
For example: `cd C:\Users\YourName\Downloads\Project`

---

### Step 4 — Create a virtual environment

A virtual environment is like a **separate clean room** for this project's packages.
It keeps things organized and avoids conflicts with other Python projects.

```
python -m venv venv
```

Now activate it:

```
venv\Scripts\activate
```

> ✅ You will see `(venv)` appear at the start of your command line. That means it worked.

---

### Step 5 — Install required packages

```
pip install -r requirements.txt
```

This installs everything the project needs — AI models, data tools, the web interface, etc.

> ⏳ This will take a few minutes. Some packages like `torch` and `openai-whisper` are large.
> Just let it finish. Do not close the window.

---

### Step 6 — One-time setup (build the brain of the project)

Before you can use the app, you need to run a few scripts **once** to prepare everything.
Think of this like setting up a library before you can search through it.

> ⚠️ If the folders `database/`, `embeddings/`, and `saved_models/` already have files in them,
> you can **skip this entire section** and jump to Step 7.

#### 6a — Generate embeddings from the dataset

```
cd scripts
python generate_embeddings.py
```

This reads all the text from the dataset and converts each sentence into a
**list of numbers that represents its meaning** (called an "embedding").
Saved to: `embeddings/embeddings.npy`

---

#### 6b — Build the search database

```
python build_faiss_db.py
```

This takes all those number-lists and organizes them into a fast **search index**
(think of it like building a Google search index, but for sentences).
Saved to: `database/faiss_index.bin` and `database/transcripts.npy`

---

#### 6c — Download negative examples (non-copyrighted speech)

Go back to the project root first:

```
cd ..
python data.py
```

This downloads a large collection of TED Talk transcripts from the internet.
These are used as examples of **non-copyrighted** speech to teach the AI.
Saved to: `embeddings/ted_transcripts.npy`

> ⏳ This may take a while depending on your internet speed.

---

#### 6d — Train the AI decision model

```
cd scripts
python train_ml.py
```

This is where the AI actually **learns** to tell the difference between
copyrighted and non-copyrighted speech.
Saved to: `saved_models/` folder (3 files)

---

### Step 7 — Launch the app 🚀

Go back to the project root and run:

```
cd ..
streamlit run app.py
```

Your browser should automatically open at:

```
http://localhost:8501
```

Upload an audio or video file and the app will analyze it for you!

---

### Common Mistakes to Avoid

| Mistake | Fix |
|---|---|
| `python` not recognized | Make sure Python is added to PATH during install |
| `ffmpeg` not recognized | Make sure you added `C:\ffmpeg\bin` to system PATH and opened a **new** terminal |
| `(venv)` not showing | You forgot to activate the virtual environment (Step 4) |
| App gives an error on startup | Make sure you completed all of Step 6 before running Step 7 |
| `pip install` fails on `torch` | Try running: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

---

## 3. Project Structure — What Each File Does

You do not need to understand every file. Here is a simple guide to what matters.

---

### 🔑 The Most Important File

| File | What it does |
|---|---|
| `app.py` | **This is the main app.** It runs the website interface. You will interact with this most. |

---

### ⚙️ One-Time Setup Scripts (inside `scripts/` folder)

You run these **once** to prepare the project. After that, you can forget about them.

| File | What it does |
|---|---|
| `scripts/generate_embeddings.py` | Converts dataset text into numbers (embeddings) |
| `scripts/build_faiss_db.py` | Builds the searchable index from those numbers |
| `scripts/train_ml.py` | Trains the AI model to detect copyright |
| `scripts/load_dataset.py` | Just previews the dataset — optional, for curiosity |

---

### 📦 Data Preparation Scripts (project root)

| File | What it does |
|---|---|
| `data.py` | Downloads TED Talk transcripts as "non-copyright" examples for training |
| `process.py` | Alternative to `data.py` — uses a local CSV file instead of downloading |

---

### 🗂️ Folders — What They Store

| Folder | What is inside |
|---|---|
| `dataset/` | The original dataset files (`.parquet` format — like Excel sheets for big data) |
| `embeddings/` | The converted number-versions of all transcripts |
| `database/` | The search index that the app uses to find similar text quickly |
| `saved_models/` | The trained AI model files — the "brain" of the app |
| `plots/` | Charts generated during training (for evaluation) |
| `audio_input/` | Empty folder — you can put your audio files here |
| `video_input/` | Empty folder — you can put your video files here |
| `results/` | Empty folder — reserved for saving output results |

---

### 🔧 Helper Files

| File | What it does |
|---|---|
| `utils/audio_utils.py` | Contains the function that extracts audio from video using ffmpeg |
| `utils/text_utils.py` | Empty for now — reserved for future text helpers |
| `requirements.txt` | The list of all Python packages needed — used by `pip install` |

---

### 🧪 Testing / Dev Files (you can ignore these)

| File | What it does |
|---|---|
| `test_vid_aud.py` | A small test to check if video-to-audio extraction works |
| `text.py` | A small debug script to peek at saved transcript data |
| `scripts/speech_to_text.py` | A simple script to test transcribing one audio file manually |
| `scripts/match_audio.py` | An older version of the detection — works without the ML model |
| `transcript_data.csv` | A local CSV of transcripts used by `process.py` |
| `extracted_audio.wav` | A leftover audio file from testing — safe to ignore |

---

## 4. How the Project Works — Step by Step

Here is the full story of what happens when you use the app.

---

### Part 1 — Before the App (Setup, Done Once)

Think of this like building a reference library before you can search it.

**Step 1 — The dataset is read**
The project has a collection of known transcripts (spoken text from the LibriSpeech dataset).
These represent the "copyright-protected" content we are checking against.

**Step 2 — Text is converted to meaning-numbers (Embeddings)**
Each sentence from the dataset is passed through an AI model called **Sentence-BERT**.
It converts each sentence into a list of 384 numbers that captures the **meaning** of that sentence.
Think of it like a fingerprint — similar-sounding sentences get similar fingerprints.

**Step 3 — A fast search index is built (FAISS)**
All those number-lists are stored in a special search system called **FAISS**.
It is like a super-fast search engine — given a new sentence's number-list,
it can instantly find the most similar ones from the entire dataset.

**Step 4 — The AI is trained to decide**
The system generates examples of:
- **"This IS copyright"** → by taking sentences already in the dataset and searching for them
- **"This is NOT copyright"** → by using TED Talk sentences (clearly different content)

For each example, it records **how similar** the search results were.
Then it trains a simple AI model (Logistic Regression) to learn the pattern:
"When similarity scores look like THIS → it's copyright. When they look like THAT → it's not."

---

### Part 2 — When You Use the App (Every Upload)

**Step 1 — You upload a file**
You drag and drop a `.wav`, `.mp3`, `.m4a`, or `.mp4` file into the app.

**Step 2 — If it's a video, audio is extracted**
If you uploaded a `.mp4`, the app uses `ffmpeg` to pull out just the audio track.
It saves it as a temporary `.wav` file.

**Step 3 — The audio is transcribed**
The app uses **OpenAI Whisper** (a speech-to-text AI) to listen to the audio
and type out everything that was said. This becomes the transcript.
You can see it on screen under "Transcription".

**Step 4 — The transcript is converted to a meaning-fingerprint**
The transcript text is passed through Sentence-BERT, the same AI used during setup.
It gives back a list of 384 numbers representing the **meaning** of what was said.

**Step 5 — The search index is queried**
That number-list is sent to FAISS, which searches through the entire dataset
and returns the **top 6 most similar** sentences it could find.
Each result comes with a **similarity score** (0 = completely different, 1 = identical meaning).

**Step 6 — Features are calculated**
From those 6 similarity scores, 8 numbers are calculated:
- The top 5 individual similarity scores
- The **average** of those scores
- How much they **vary** (standard deviation)
- The **gap** between the 1st and 2nd score (a big gap means one clear match)

**Step 7 — The AI makes a decision**
Those 8 numbers are passed to the trained AI model (the one saved in `saved_models/`).
The model outputs:
- A **prediction**: 1 = Copyright Detected, 0 = No Copyright
- A **confidence score**: how sure it is (e.g., 0.94 means 94% confident)

**Step 8 — Risk level is assigned**
Based on the confidence score:
- Above 0.9 → 🔴 Very High Risk
- Above 0.7 → 🟠 High Risk
- Above 0.5 → 🟡 Moderate Risk
- Below 0.5 → 🟢 Low Risk

**Step 9 — Results are shown**
The app displays everything on screen:
- Whether copyright was detected
- Confidence score and similarity score
- Risk level
- The top 5 most similar sentences from the dataset (so you can see what matched)
- A bar chart of the 8 numbers that influenced the decision

**Step 10 — Cleanup**
The temporary files created during processing are automatically deleted.

---

> 💬 **Have questions?**
> Start by reading `app.py` — it is well-commented and is the heart of the whole system.
> Then look at `scripts/train_ml.py` to understand how the AI was trained.