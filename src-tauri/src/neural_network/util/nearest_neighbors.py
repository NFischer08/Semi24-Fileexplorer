import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Parameters
EMBEDDING_DIM = 300
sample_words = ["test", "first", "king"]
vocap_path = "../eng_vocab.json"
weights_path = "eng_weights_D300"
number_nearest_words = 10


def nearest_neighbor(word, embedding_weights, vocab, top_k=number_nearest_words, exclude_unk=True):
    if word not in vocab:
        print(f"Word '{word}' not in vocabulary.")
        return []
    idx = vocab[word]
    word_vec = embedding_weights[idx].reshape(1, -1)
    similarities = cosine_similarity(word_vec, embedding_weights)[0]
    sorted_indices = np.argsort(-similarities)
    neighbors = []
    for i in sorted_indices:
        if i == idx:
            continue
        if exclude_unk and i == vocab.get("UNK", -1):
            continue
        neighbors.append((i, similarities[i]))
        if len(neighbors) >= top_k:
            break
    # Map indices back to words
    inv_vocab = {i: w for w, i in vocab.items()}
    return [(inv_vocab[i], sim) for i, sim in neighbors]

def usage_nearest_neighbor(embedding_weights, vocab, sample_words):
    for word in sample_words:
        print(f"\nNearest neighbors for '{word}':")
        neighbors = nearest_neighbor(word, embedding_weights, vocab, top_k=number_nearest_words)
        for neighbor, sim in neighbors:
            print(f"  {neighbor:15s} (similarity: {sim:.4f})")
def main():
    with open(vocap_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    embedding_weights = np.fromfile(weights_path, dtype=np.float32)
    embedding_weights = embedding_weights.reshape((vocab_size, EMBEDDING_DIM))

    usage_nearest_neighbor(embedding_weights, vocab, sample_words)


if __name__ == "__main__":
    main()
