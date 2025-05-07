import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# -------------------------------
# 1. Load Weights and Vocabulary
# -------------------------------
VOCAB_JSON = "vocab.json"
WEIGHTS_FILE = "weights"
EMBEDDING_DIM = 300  # Change if you used a different dimension

# Load vocabulary
with open(VOCAB_JSON, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Load embedding weights
embedding_weights = np.fromfile(WEIGHTS_FILE, dtype=np.float32)
embedding_weights = embedding_weights.reshape(len(vocab), EMBEDDING_DIM)

UNK_TOKEN = "UNK"

# -------------------------------
# 2. Select Words for Visualization
# -------------------------------
N = 50
# Pick N words from the middle of the vocab (skip UNK)
words = list(vocab.keys())[1010:1010+N]
words = [w for w in words if w != UNK_TOKEN]
indices = [vocab[word] for word in words]
embeddings_subset = embedding_weights[indices]

# -------------------------------
# 3. t-SNE Visualization
# -------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=20, init='pca')
embeddings_2d = tsne.fit_transform(embeddings_subset)

plt.figure(figsize=(16, 12))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=140)
for i, word in enumerate(words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=28, alpha=0.7)
plt.title("Visualization of Word Embeddings", fontsize=32)
plt.xlabel("Dimension 1", fontsize=28)
plt.ylabel("Dimension 2", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.savefig("Visualization-Embeddings-script.png")
print("Diagram saved as Visualization-Embeddings.png")
