from typing import TextIO

import torch
import torch.nn as nn
import torch.optim as optim
import json
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# Check GPU availability and set device
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define SkipGram Model
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target):
        embed = self.embeddings(target)
        return self.output(embed)

    # Mark get_embedding for export
    @torch.jit.export
    def get_embedding(self, target):
        return self.embeddings(target)

# Read file names from a .txt file
file_path = "eng-simple_wikipedia_2021_100K/eng-simple_wikipedia_2021_100K-sentences.txt"

with open(file_path, "r", encoding="utf-8") as f:
    file_names = [line.strip() for line in f.readlines()]  # Strip whitespace and newlines

# Optimized tokenization: process sentences line by line and split at "_", "-", or spaces
tokens = []
for name in file_names:
    tokens.extend(name.split())  # Extend the token list directly to avoid intermediate lists

# Create vocabulary mapping each unique word to a unique index
vocab_counter = Counter(tokens)
vocab = {word: idx for idx, word in enumerate(vocab_counter.keys())}

# Export vocabulary as a JSON file
vocab_file_path = "vocab.json"
vocab_file: TextIO
with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
    json.dump(vocab, vocab_file, ensure_ascii=False, indent=4)

print(f"Vocabulary exported to {vocab_file_path}")

# Prepare training data (pairs of target and context word indices) in batches to reduce memory usage
data = [(vocab[tokens[i]], vocab[tokens[i + 1]]) for i in range(len(tokens) - 1)]
class SkipGramDataset(Dataset):
    def __init__(self, tokens, vocab):
        self.tokens = tokens
        self.vocab = vocab

    def __len__(self):
        return len(self.tokens) - 1

    def __getitem__(self, idx):
        target = self.vocab[self.tokens[idx]]
        context = self.vocab[self.tokens[idx + 1]]
        return torch.tensor(target, dtype=torch.long), torch.tensor(context, dtype=torch.long)

batch_size = 1028  # Process data in batches to reduce memory usage
dataset = SkipGramDataset(tokens, vocab)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training
vocab_size = len(vocab)
embedding_dim = 384
model = SkipGramModel(vocab_size, embedding_dim).to(device)  # Move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


for epoch in range(1):  # Training loop
    print(f"Epoch {epoch + 1} started")
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")

    for targets, contexts in progress_bar:
        targets, contexts = targets.to(device), contexts.to(device)

        optimizer.zero_grad()
        output = model(targets)
        loss = criterion(output, contexts)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()
        total_loss += loss.item()

        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}, Total Loss: {total_loss}")

# Save model and state dict (no need to move them explicitly)
scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")

