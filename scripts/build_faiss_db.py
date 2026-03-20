import numpy as np
import faiss
import os
from datasets import load_dataset

print("Loading embeddings...")

embeddings = np.load("../embeddings/embeddings.npy")

print("Embedding shape:", embeddings.shape)

dimension = embeddings.shape[1]

print("Creating FAISS index...")
index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("Total vectors stored:", index.ntotal)

# Save FAISS index
os.makedirs("../database", exist_ok=True)
faiss.write_index(index, "../database/faiss_index.bin")

print("FAISS index saved!")

# Save transcripts for mapping results
print("Saving transcripts...")

dataset = load_dataset(
    "parquet",
    data_files="../dataset/*.parquet"
)

data = dataset["train"]
data = data.remove_columns(["audio"])

texts = [x["text"] for x in data]

np.save("../database/transcripts.npy", texts)

print("Transcripts saved!")
