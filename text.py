import numpy as np

texts = np.load("database/transcripts.npy", allow_pickle=True)

for i in range(8):
    print(texts[i])


'''import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load embeddings you already created
embeddings = np.load("embeddings/embeddings.npy")

# Take only first 500 for visualization (faster)
sample = embeddings[:500]

# Reduce 384D → 2D
pca = PCA(n_components=2)
reduced = pca.fit_transform(sample)

plt.figure(figsize=(6,5))
plt.scatter(reduced[:,0], reduced[:,1], s=10)
plt.title("2D Visualization of Text Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

plt.savefig("embedding_visualization.png")
plt.show()'''

'''import numpy as np
import matplotlib.pyplot as plt

# Create distance values
distances = np.linspace(0, 2, 50)

# Convert to similarity
similarities = 1 / (1 + distances)

plt.figure(figsize=(6,4))
plt.plot(distances, similarities)
plt.xlabel("FAISS Distance")
plt.ylabel("Similarity Score")
plt.title("Distance vs Similarity Relationship")

plt.savefig("graph1_distance_vs_similarity.png")
plt.show()
'''
