'''from datasets import load_dataset

dataset = load_dataset(
    "parquet",
    data_files="dataset/*.parquet"
)

print("Loaded successfully!")
print("Total samples:", len(dataset["train"]))
'''

from datasets import load_dataset
import numpy as np
import re

print("Loading TED-LIUM Release 2 in streaming mode...")

ted_stream = load_dataset(
    "tedlium",
    name="release2",
    split="train",
    streaming=True
)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

ted_sentences = []

for i, example in enumerate(ted_stream):
    cleaned = clean_text(example["text"])
    if len(cleaned) > 20:
        ted_sentences.append(cleaned)

    if i > 50000:   # limit to first 50k for now
        break

print("Collected:", len(ted_sentences))

np.save("ted_transcripts.npy", np.array(ted_sentences))
print("Saved ted_transcripts.npy")