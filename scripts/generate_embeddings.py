from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import numpy as np
import os

print("Loading dataset...")

dataset = load_dataset(
    "parquet",
    data_files="../dataset/*.parquet"
)

data = dataset["train"]
data = data.remove_columns(["audio"])

texts = [x["text"] for x in data]

print("Total texts:", len(texts))

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings (this may take some time)...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Create folder if not exists
os.makedirs("../embeddings", exist_ok=True)

np.save("../embeddings/embeddings.npy", embeddings)

print("Embeddings saved successfully!")
