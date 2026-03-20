#processing the dtaset for labelling 0 for ml prediction 
import pandas as pd
import numpy as np
import re
import os
from utils.audio_utils import extract_audio_from_video

# Load dataset
df = pd.read_csv("transcript_data.csv")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text, max_words=60):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        if len(chunk) > 40:
            chunks.append(chunk)
    return chunks

negative_sentences = []

for transcript in df["transcript"]:
    cleaned = clean_text(transcript)
    chunks = chunk_text(cleaned, max_words=60)
    negative_sentences.extend(chunks)
    
def prepare_audio(file_path):

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".mp4":
        print("Video detected → extracting audio")
        audio_path = extract_audio_from_video(file_path)
    else:
        audio_path = file_path

    return audio_path

print("Total negative chunks:", len(negative_sentences))

np.save("ted_transcripts.npy", np.array(negative_sentences))

print("Saved ted_transcripts.npy")