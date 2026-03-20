import whisper
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Loading models...")

whisper_model = whisper.load_model("base")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("../database/faiss_index.bin")
transcripts = np.load("../database/transcripts.npy", allow_pickle=True)

print("Models loaded!")

audio_path = input("Enter audio file path: ")

print("Transcribing audio...")
result = whisper_model.transcribe(audio_path)
query_text = result["text"]

print("\nTranscript:")
print(query_text)

query_embedding = embed_model.encode(
    [query_text],
    normalize_embeddings=True
)

query_embedding = np.array(query_embedding).astype("float32")

D, I = index.search(query_embedding, k=1)

similarity = float(D[0][0])
matched_text = transcripts[I[0][0]]

print("\nMost Similar Transcript Found:")
print(matched_text)

print("\nCosine Similarity Score:", round(similarity, 4))
# Decision logic
if similarity > 0.60:
    print("\n⚠ HIGH similarity → Copyright Detected")
elif similarity > 0.45:
    print("\n⚠ MEDIUM similarity → Possible Copyright")
else:
    print("\n✓ LOW similarity → No Copyright")
