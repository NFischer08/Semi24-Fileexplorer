import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from neural_network.post_hoc_alignment import eng_vocab, deu_weights, deu_vocab

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
    sample_words = ["en:test", "en:first", "de:könig", "en:king"]
    for word in sample_words:
        print(f"\nNearest neighbors for '{word}':")
        neighbors = nearest_neighbor(word, embedding_weights, vocab, top_k=5)
        for neighbor, sim in neighbors:
            print(f"  {neighbor:15s} (similarity: {sim:.4f})")

def nearest_neighbor_bilingual(query_word, query_lang, eng_weights, deu_weights, eng_vocab, deu_vocab, top_k=5):
    """
    Finds cross-lingual nearest neighbors for a word in English or German.
    """
    if query_lang == 'en':
        if query_word not in eng_vocab:
            print(f"{query_word} not in English vocab.")
            return []
        idx = eng_vocab[query_word]
        vec = eng_weights[idx].reshape(1, -1)
        sims = cosine_similarity(vec, deu_weights)[0]
        inv_deu_vocab = {i: w for w, i in deu_vocab.items()}
        top_indices = np.argsort(-sims)[:top_k]
        return [(inv_deu_vocab[i], sims[i]) for i in top_indices]
    elif query_lang == 'de':
        if query_word not in deu_vocab:
            print(f"{query_word} not in German vocab.")
            return []
        idx = deu_vocab[query_word]
        vec = deu_weights[idx].reshape(1, -1)
        sims = cosine_similarity(vec, eng_weights)[0]
        inv_eng_vocab = {i: w for w, i in eng_vocab.items()}
        top_indices = np.argsort(-sims)[:top_k]
        return [(inv_eng_vocab[i], sims[i]) for i in top_indices]
    else:
        print("query_lang must be 'en' or 'de'.")
        return []

def usage_nearest_neighbor_bilingual(eng_weights_aligned, deu_weights, eng_vocab, deu_vocab):
    """
    Example usage of nearest_neighbor_bilingual: prints cross-lingual neighbors for sample words.
    """
    print("\nSample cross-lingual neighbors:")
    for word, lang in [('dog', 'en'), ('Hund', 'de')]:
        neighbors = nearest_neighbor_bilingual(
            word, lang, eng_weights_aligned, deu_weights, eng_vocab, deu_vocab, top_k=5)
        if neighbors:
            if lang == 'en':
                print(f"German neighbors for English word '{word}':")
            else:
                print(f"English neighbors for German word '{word}':")
            for neighbor, sim in neighbors:
                print(f"  {neighbor:15s} (similarity: {sim:.4f})")

def main():
    """

        # --- 1. Load vocab ---
    with open('vocab.json', 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    VOCAB_SIZE = len(vocab)

    # --- 2. Load weights ---
    embedding_weights = np.fromfile('weights', dtype=np.float32)
    embedding_weights = embedding_weights.reshape((VOCAB_SIZE, EMBEDDING_DIM))

    # --- 3. Run monolingual nearest neighbor example ---
    usage_nearest_neighbor(embedding_weights, vocab)


    :return:
    """

    # --- 4. Load aligned English weights for cross-lingual search ---
    eng_weights_aligned = np.fromfile('eng_weights_aligned', dtype=np.float32).reshape(len(eng_vocab), EMBEDDING_DIM)

    # --- 5. Run bilingual nearest neighbor example ---
    usage_nearest_neighbor_bilingual(eng_weights_aligned, deu_weights, eng_vocab, deu_vocab)

if __name__ == "__main__":
    main()
