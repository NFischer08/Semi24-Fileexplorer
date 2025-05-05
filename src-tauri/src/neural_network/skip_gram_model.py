import re
from typing import TextIO
import numpy
import os
import GPUtil

if not GPUtil.getGPUs():
    os.putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.putenv("AMD_SERIALIZE_KERNEL", "3")

import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import PyQt5
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

VOCAB_SIZE = 5000

print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define SkipGram Model
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output.weight = self.embeddings.weight

    def forward(self, target):
        embed = self.embeddings(target)
        return torch.matmul(embed, self.embeddings.weight.t())

    @torch.jit.export
    def get_embedding(self, target):
        embed = self.embeddings(target)
        return torch.matmul(embed, self.embeddings.weight.t())

# Read file names from a .txt file
file_path = "eng-simple_wikipedia_2021_10K/eng-simple_wikipedia_2021_10K-sentences.txt"

with open(file_path, "r", encoding="utf-8") as f:
    file_names = [line.strip() for line in f.readlines()]

# Tokenization
def normalize_token(token):
    # Match YYYY-MM-DD, YYYY:MM:DD, or YYYY.MM.DD
    if re.fullmatch(r"\d{4}([-:.])\d{2}\1\d{2}", token):
        return "DATE"
    # Match years (four consecutive digits)
    elif re.fullmatch(r"\d{4}", token):
        return "YEAR"
    else:
        return token

tokens = []
for name in file_names:
    name = name.lower()
    # Match words, and all three date formats, and years
    raw_tokens = re.findall(r"[a-zäöü]+|\d{4}[-:.]\d{2}[-:.]\d{2}|\d{4}", name)
    tokens.extend([normalize_token(t) for t in raw_tokens])

# Build vocabulary of top N most frequent words
vocab_counter = Counter(tokens)
most_common = vocab_counter.most_common(VOCAB_SIZE)
vocab_words = [word for word, _ in most_common]
vocab = {word: idx for idx, word in enumerate(vocab_words)}
print(f"Vocabulary size: {len(vocab)}")

# Filter tokens to only those in the fixed vocabulary
filtered_tokens = [t for t in tokens if t in vocab]

# Export vocabulary as a JSON file
vocab_file_path = "vocab.json"
with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
    json.dump(vocab, vocab_file, ensure_ascii=False, indent=4)
print(f"Vocabulary exported to {vocab_file_path}")

class SkipGramDataset(Dataset):
    def __init__(self, tokens, vocab, window_size=2):
        self.tokens = tokens
        self.vocab = vocab
        self.window_size = window_size
        self.pairs = []
        for i, target_word in enumerate(tokens):
            if target_word not in vocab:
                continue
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            for j in range(start, end):
                if j != i and tokens[j] in vocab:
                    self.pairs.append((vocab[target_word], vocab[tokens[j]]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)


batch_size = 32768
window_size = 2
dataset = SkipGramDataset(filtered_tokens, vocab, window_size=window_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training
vocab_size = len(vocab)
embedding_dim = 256
model = SkipGramModel(vocab_size, embedding_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

print(f"Training with vocabulary size: {vocab_size}")

for epoch in range(80):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")
    for targets, contexts in progress_bar:
        targets, contexts = targets.to(device), contexts.to(device)
        optimizer.zero_grad()
        output = model(targets)
        loss = criterion(output, contexts)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

print("Default: ")
print(model.output.weight)

print("Numpy: ")
weights = model.output.weight
weights = weights.to("cpu").detach().numpy()
weights.tofile("weights")
weights_as_bytes = weights.tobytes()

# Number of words to plot (top-N most frequent)
N = 25  # Adjust as needed for clarity

# Get embeddings and labels
embedding_weights = model.embeddings.weight.data.cpu().numpy()
words = list(vocab.keys())[:N]
indices = [vocab[word] for word in words]
embeddings_subset = embedding_weights[indices]

# Number of words to plot (top-N most frequent)
N = 25  # Number of words to plot

# Get embeddings and labels
embedding_weights = model.embeddings.weight.data.cpu().numpy()

# Select words 101 to 125 (Python is zero-indexed, so 100:125)
words = list(vocab.keys())[1000:1000+N]
indices = [vocab[word] for word in words]

# t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=20, init='pca')
embeddings_2d = tsne.fit_transform(embeddings_subset)

# Plot
plt.figure(figsize=(16, 12))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=140)

for i, word in enumerate(words):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=28, alpha=0.7)

plt.title("Visualisierung der Worteinbettungen", fontsize=32)
plt.xlabel("Dimension 1", fontsize=28)
plt.ylabel("Dimension 2", fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.tight_layout()
plt.savefig("Visualization-Embeddings.png")
plt.show()
