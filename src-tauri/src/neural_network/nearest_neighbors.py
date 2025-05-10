import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_DIM = 300  # Set this to your embedding dimension used in training

def nearest_neighbor(word, embedding_weights, vocab, top_k=10, exclude_unk=True):
    """
    Returns the top_k nearest neighbors for a given word based on cosine similarity.
    """
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
            continue  # skip the word itself
        if exclude_unk and i == vocab.get("UNK", -1):
            continue  # skip UNK token if requested
        neighbors.append((i, similarities[i]))
        if len(neighbors) >= top_k:
            break
    # Map indices back to words
    inv_vocab = {i: w for w, i in vocab.items()}
    return [(inv_vocab[i], sim) for i, sim in neighbors]

def usage_nearest_neighbor(embedding_weights, vocab):
    """
    Example usage of nearest_neighbor: prints neighbors for a list of sample words.
    """
    sample_words = ["test", "first", "king"]
    for word in sample_words:
        print(f"\nNearest neighbors for '{word}':")
        neighbors = nearest_neighbor(word, embedding_weights, vocab, top_k=5)
        for neighbor, sim in neighbors:
            print(f"  {neighbor:15s} (similarity: {sim:.4f})")
def main():
        # --- 1. Load vocab ---
    with open('eng_vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)

    # --- 2. Load weights ---
    embedding_weights = np.fromfile('eng_weights_D300', dtype=np.float32)
    embedding_weights = embedding_weights.reshape((vocab_size, EMBEDDING_DIM))

    # --- 3. Run monolingual nearest neighbor example ---
    usage_nearest_neighbor(embedding_weights, vocab)


if __name__ == "__main__":
    main()
