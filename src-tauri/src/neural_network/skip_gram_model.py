import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from collections import Counter
from tqdm import tqdm
import json

# --- Step 1: Preprocessing the Corpus ---
class Corpus:
    def __init__(self, file_path, min_freq=5):
        self.file_path = file_path
        self.min_freq = min_freq
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}

    def preprocess(self):
        # Read and tokenize text
        with open(self.file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            tokens = text.split()

        # Build vocabulary with minimum frequency threshold
        word_counts = Counter(tokens)
        self.vocab = {word for word, freq in word_counts.items() if freq >= self.min_freq}
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        return tokens

# --- Step 2: Create Skip-gram Dataset ---
class SkipGramDataset(Dataset):
    def __init__(self, tokens, vocab, word_to_idx, window_size=5):
        self.data = []
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.window_size = window_size

        for i, word in enumerate(tokens):
            if word not in vocab:
                continue
            center_idx = word_to_idx[word]
            context_indices = [
                word_to_idx[tokens[j]]
                for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1))
                if j != i and tokens[j] in vocab
            ]
            for context_idx in context_indices:
                self.data.append((center_idx, context_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# --- Step 3: Define the Skip-gram Model ---
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_word_idx, context_word_idx):
        center_embeds = self.center_embeddings(center_word_idx)  # [batch_size, embedding_dim]
        context_embeds = self.context_embeddings(context_word_idx)  # [batch_size, embedding_dim]

        scores = torch.sum(center_embeds * context_embeds, dim=1)  # Dot product similarity
        return scores

# --- Step 4: Training Function ---
def train_skipgram_model(corpus_file, embedding_dim=100, window_size=5, batch_size=1024,
                         num_epochs=10, learning_rate=0.01):
    # Preprocess corpus and create dataset
    print("Preprocessing corpus...")
    corpus = Corpus(corpus_file)
    tokens = corpus.preprocess()

    print(f"Vocabulary size: {len(corpus.vocab)}")

    dataset = SkipGramDataset(tokens=tokens,
                              vocab=corpus.vocab,
                              word_to_idx=corpus.word_to_idx,
                              window_size=window_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = SkipGramModel(vocab_size=len(corpus.vocab), embedding_dim=embedding_dim)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Training model...")
    for epoch in range(num_epochs):
        total_loss = 0
        for center_words, context_words in tqdm(dataloader):
            # Positive samples (label=1)
            positive_labels = torch.ones(center_words.shape[0])

            # Negative samples (label=0)
            num_negatives = 5  # Number of negative samples per positive sample
            negative_contexts = torch.randint(0, len(corpus.vocab), (center_words.shape[0], num_negatives))
            negative_labels = torch.zeros(center_words.shape[0] * num_negatives)

            # Forward pass for positive samples
            positive_scores = model(center_words, context_words)

            # Forward pass for negative samples
            center_words_repeated = center_words.repeat_interleave(num_negatives)  # Repeat center words for negatives
            negative_scores = model(center_words_repeated, negative_contexts.flatten())

            # Combine losses
            loss_positive = loss_fn(positive_scores.squeeze(), positive_labels)
            loss_negative = loss_fn(negative_scores.squeeze(), negative_labels)

            loss = loss_positive + loss_negative

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save the trained embeddings as a .pth file
    print("Saving embeddings...")
    torch.save(model.state_dict(), "skipgram_model.pth")

    # Convert the model to TorchScript and save as .pt file
    print("Converting model to TorchScript...")
    example_input_center = torch.tensor([0])  # Example input tensor (word index for center word)
    example_input_context = torch.tensor([1])  # Example input tensor (word index for context word)

    scripted_model = torch.jit.trace(model, (example_input_center, example_input_context))
    scripted_model.save("skipgram_model.pt")

    print("TorchScript model saved as skipgram_model.pt")

    with open("vocab.json", "w") as f:
        json.dump(corpus.word_to_idx, f)

    return scripted_model

# --- Step 5: Run Training ---
if __name__ == "__main__":
    corpus_file_path = "deu_wikipedia_2021_100K/deu_wikipedia_2021_100K-sentences.txt"

    trained_model = train_skipgram_model(
        corpus_file=corpus_file_path,
        embedding_dim=128,         # Reduced embedding dimension
        window_size=5,
        batch_size=512,            # Reduced batch size
        num_epochs=1,
        learning_rate=0.01         # Lower learning rate for stability
    )


