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

# -------------------------------
# 0. Input: Your words and annotation settings
# -------------------------------
INPUT_WORDS = ["ocean", "harmony", "laptop", "galaxy"]
ANNOTATE_NEIGHBORS = 6  # How many neighbors per input word to annotate

CLUSTER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

VOCAB_JSON = "eng_vocab.json"
WEIGHTS_FILE = "eng_weights_D300"
EMBEDDING_DIM = 300

with open(VOCAB_JSON, "r", encoding="utf-8") as f:
    vocab = json.load(f)

embedding_weights = np.fromfile(WEIGHTS_FILE, dtype=np.float32)
embedding_weights = embedding_weights.reshape(len(vocab), EMBEDDING_DIM)

UNK_TOKEN = "UNK"

# -------------------------------
# 1. Find neighbors and build word list
# -------------------------------
all_words = set()
word_to_neighbors = {}
word_to_cluster = {}
annotate_words = set(INPUT_WORDS)  # Always annotate input words

for idx, word in enumerate(INPUT_WORDS):
    if word in vocab and word != UNK_TOKEN:
        all_words.add(word)
        neighbors = nearest_neighbor(word, embedding_weights, vocab, top_k=6, exclude_unk=True)
        neighbor_words = [w for w, sim in neighbors if w != UNK_TOKEN]
        word_to_neighbors[word] = neighbor_words
        all_words.update(neighbor_words)
        word_to_cluster[word] = idx
        for n in neighbor_words:
            word_to_cluster[n] = idx
        # Add only the specified number of neighbors to annotate
        annotate_words.update(neighbor_words[:ANNOTATE_NEIGHBORS])
    else:
        print(f"Warning: '{word}' not found in vocabulary or is UNK.")

WORDS_FOR_VIS = list(all_words)

# -------------------------------
# 2. Prepare Embeddings
# -------------------------------
words = [w for w in WORDS_FOR_VIS if w in vocab and w != UNK_TOKEN]
if not words:
    raise ValueError("None of the selected words were found in the vocabulary.")

indices = [vocab[word] for word in words]
embeddings_subset = embedding_weights[indices]

print("\nCosine similarities between each INPUT_WORD and all plotted words:\n")
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
    # Sort by similarity descending
    similarities.sort(key=lambda x: -x[1])
    print(f"\nSimilarities for '{input_word}':")
    for word, sim in similarities:
        print(f"  {word:15s} (cosine similarity: {sim:.4f})")

# -------------------------------
# 3. t-SNE Visualization
# -------------------------------
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
    # Annotate only if word is in annotate_words
    if word in annotate_words:
        t = plt.text(
            embeddings_2d[i, 0], embeddings_2d[i, 1], word,
            fontsize=fontsize, color=color, fontweight=fontweight,
            alpha=1,  # <-- removed path_effects
            zorder=zorder+1
        )
        texts.append(t)

# --- Add this block to avoid overlapping labels ---
if USE_ADJUST_TEXT and texts:
    adjust_text(texts)



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
print("Diagram saved as Visualization-Embeddings")

def print_sorted_similarities(input_words, words_to_compare, embedding_weights, vocab):
    """
    For each input word, print cosine similarities to all words in words_to_compare,
    sorted from highest to lowest.
    """
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
        # Sort by similarity, descending
        similarities.sort(key=lambda x: -x[1])
        print(f"\nSimilarities for '{input_word}':")
        for word, sim in similarities:
            print(f"  {word:15s} (cosine similarity: {sim:.4f})")
