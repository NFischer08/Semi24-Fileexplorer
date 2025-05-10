import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nearest_neighbors import nearest_neighbor

try:
    from adjustText import adjust_text
    USE_ADJUST_TEXT = True
except ImportError:
    USE_ADJUST_TEXT = False

# Parameters and constants for visualization and embedding
INPUT_WORDS = ["ozean", "harmonie", "laptop", "deutsch"]
ANNOTATE_NEIGHBORS = 6

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

VOCAB_JSON = "deu_vocab.json"
WEIGHTS_FILE = "deu_weights_D300"
EMBEDDING_DIM = 300
number_nearest_words = 6

# Load vocabulary from file
with open(VOCAB_JSON, "r", encoding="utf-8") as f:
    vocab = json.load(f)

# Load embedding weights from file and reshape
embedding_weights = np.fromfile(WEIGHTS_FILE, dtype=np.float32)
embedding_weights = embedding_weights.reshape(len(vocab), EMBEDDING_DIM)

UNK_TOKEN = "UNK"

all_words = set()
word_to_neighbors = {}
word_to_cluster = {}
annotate_words = set(INPUT_WORDS)

# Find nearest neighbors for each input word, assign clusters, prepare annotation list
for idx, word in enumerate(INPUT_WORDS):
    if word in vocab and word != UNK_TOKEN:
        all_words.add(word)
        neighbors = nearest_neighbor(word, embedding_weights, vocab, top_k=number_nearest_words, exclude_unk=True)
        neighbor_words = [w for w, sim in neighbors if w != UNK_TOKEN]
        word_to_neighbors[word] = neighbor_words
        all_words.update(neighbor_words)
        word_to_cluster[word] = idx
        for n in neighbor_words:
            word_to_cluster[n] = idx
        annotate_words.update(neighbor_words[:ANNOTATE_NEIGHBORS])
    else:
        print(f"Warning: '{word}' not found in vocabulary or is UNK.")

WORDS_FOR_VIS = list(all_words)

# Prepare embeddings and indices for visualization
words = [w for w in WORDS_FOR_VIS if w in vocab and w != UNK_TOKEN]
indices = [vocab[word] for word in words]
embeddings_subset = embedding_weights[indices]

# Print cosine similarities for each input word against all selected words
for input_word in INPUT_WORDS:
    if input_word not in vocab or input_word == UNK_TOKEN:
        print(f"Warning: '{input_word}' not found in vocabulary or is UNK.")
        continue
    input_idx = vocab[input_word]
    input_vec = embedding_weights[input_idx].reshape(1, -1)
    similarities = []
    for word in words:
        word_idx = vocab[word]
        word_vec = embedding_weights[word_idx].reshape(1, -1)
        sim = float((np.dot(input_vec, word_vec.T) / (np.linalg.norm(input_vec) * np.linalg.norm(word_vec))).item())
        similarities.append((word, sim))
    similarities.sort(key=lambda x: -x[1])  # Sort by similarity
    print(f"\nSimilarities for '{input_word}':")
    for word, sim in similarities:
        print(f"  {word:15s} (cosine similarity: {sim:.4f})")

# Plotting setup and TSNE dimensionality reduction
plt.rcParams['svg.fonttype'] = 'path'
plt.rcParams['font.family'] = 'Sans serif'

plt.figure(figsize=(16, 12), dpi=300)
tsne = TSNE(n_components=2, random_state=42, perplexity=5, init='pca')
embeddings_2d = tsne.fit_transform(embeddings_subset)

texts = []
for i, word in enumerate(words):
    cluster_idx = word_to_cluster.get(word, -1)
    color = CLUSTER_COLORS[cluster_idx % len(CLUSTER_COLORS)]
    marker = 'o'
    if word in INPUT_WORDS:
        size = 280
        alpha = 1.0
        fontweight = 'bold'
        fontsize = 24
        zorder = 3
    else:
        size = 120
        alpha = 0.7
        fontweight = 'normal'
        fontsize = 20
        zorder = 2
    plt.scatter(
        embeddings_2d[i, 0], embeddings_2d[i, 1],
        s=size, c=color, alpha=alpha, marker=marker, edgecolor='k', linewidth=1.2, zorder=zorder
    )
    if word in annotate_words:
        t = plt.text(
            embeddings_2d[i, 0], embeddings_2d[i, 1], word,
            fontsize=fontsize, color=color, fontweight=fontweight,
            alpha=1,
            zorder=zorder+1
        )
        texts.append(t)

# Optionally adjust text labels to avoid overlap
if USE_ADJUST_TEXT and texts:
    adjust_text(texts)

# Final plot formatting and saving
plt.title("Gruppierte Worteinbettungen", fontsize=28, weight='bold', pad=20)
plt.xlabel("Dimension 1", fontsize=26)
plt.ylabel("Dimension 2", fontsize=26)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(False)
plt.tight_layout()
plt.savefig("Visualization-Embeddings.svg", format="svg")
plt.savefig("Visualization-Embeddings.png", dpi=300)
print("Diagrame saved as Visualization-Embeddings")

# Prints sorted cosine similarities for given input words and comparison set
def print_sorted_similarities(input_words, words_to_compare, embedding_weights, vocab):
    for input_word in input_words:
        if input_word not in vocab or input_word == "UNK":
            print(f"Warning: '{input_word}' not found in vocabulary or is UNK.")
            continue
        input_idx = vocab[input_word]
        input_vec = embedding_weights[input_idx].reshape(1, -1)
        similarities = []
        for word in words_to_compare:
            word_idx = vocab[word]
            word_vec = embedding_weights[word_idx].reshape(1, -1)
            sim = float(np.dot(input_vec, word_vec.T) / (np.linalg.norm(input_vec) * np.linalg.norm(word_vec)))
            similarities.append((word, sim))
        similarities.sort(key=lambda x: -x[1])  # Sort by similarity
        print(f"\nSimilarities for '{input_word}':")
        for word, sim in similarities:
            print(f"  {word:15s} (cosine similarity: {sim:.4f})")
