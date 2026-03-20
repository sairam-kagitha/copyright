from datasets import load_dataset

print("Loading dataset (TEXT ONLY)...")

dataset = load_dataset(
    "parquet",
    data_files=r"C:\Users\yaswanth\Downloads\*.parquet"
)

data = dataset["train"]

# Remove audio column completely
data = data.remove_columns(["audio"])

print("Dataset loaded successfully!")
print("Total samples:", len(data))

print("\nShowing 5 sample transcripts:\n")

for i in range(5):
    print(data[i]["text"])
