import numpy as np
import json
from scipy.linalg import orthogonal_procrustes

# ---------------------------
# 1. Load English and German Embeddings & Vocabs
# ---------------------------
EMBEDDING_DIM = 300  # Set to your embedding dimension

# English
with open('eng_vocab.json', 'r', encoding='utf-8') as f:
    eng_vocab = json.load(f)
eng_weights = np.fromfile('eng_weights', dtype=np.float32).reshape(len(eng_vocab), EMBEDDING_DIM)

# German
with open('deu_vocab.json', 'r', encoding='utf-8') as f:
    deu_vocab = json.load(f)
deu_weights = np.fromfile('deu_weights', dtype=np.float32).reshape(len(deu_vocab), EMBEDDING_DIM)

# ---------------------------
# 2. Load Dictionary
# ---------------------------
# Format: one pair per line, separated by tab
dict_path = 'deu-eng_dict.txt'
eng_words = []
deu_words = []
with open(dict_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        deu, eng = parts
        if eng in eng_vocab and deu in deu_vocab:
            eng_words.append(eng)
            deu_words.append(deu)

print(f"Dictionary pairs used: {len(eng_words)}")

# ---------------------------
# 3. Build Matrices for Alignment
# ---------------------------
eng_indices = [eng_vocab[w] for w in eng_words]
deu_indices = [deu_vocab[w] for w in deu_words]
X = eng_weights[eng_indices]  # English embeddings
Y = deu_weights[deu_indices]  # German embeddings

# ---------------------------
# 4. Learn Orthogonal Mapping
# ---------------------------
W, _ = orthogonal_procrustes(X, Y)
eng_weights_aligned = eng_weights @ W  # Map all English embeddings

# ---------------------------
# 5. Save Aligned Embeddings
# ---------------------------
eng_weights_aligned.tofile('eng_weights_aligned')

print("Alignment complete. Aligned English embeddings saved as 'eng_weights_aligned'.")

# ---------------------------
# 6. Example: Cross-lingual Nearest Neighbors
# ---------------------------
from sklearn.metrics.pairwise import cosine_similarity

