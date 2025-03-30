import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
import json
import re
import gc
from tqdm import tqdm  # Import tqdm for progress tracking

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
    tokens.extend(re.split(r'[_\-\s]+', name))  # Extend the token list directly to avoid intermediate lists

# Create vocabulary mapping each unique word to a unique index
vocab = {word: idx for idx, word in enumerate(set(tokens))}

# Export vocabulary as a JSON file
vocab_file_path = "vocab.json"
with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
    json.dump(vocab, vocab_file, ensure_ascii=False, indent=4)  # Save vocab as JSON

print(f"Vocabulary exported to {vocab_file_path}")

# Prepare training data (pairs of target and context word indices) in batches to reduce memory usage
data = [(vocab[tokens[i]], vocab[tokens[i + 1]]) for i in range(len(tokens) - 1)]

# Training
vocab_size = len(vocab)
embedding_dim = 256
model = SkipGramModel(vocab_size, embedding_dim).to(device)  # Move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

batch_size = 1028  # Process data in batches to reduce memory usage

for epoch in range(1):  # Training loop
    print(f"Epoch {epoch + 1} started")
    total_loss = 0

    # Use tqdm to show progress for batches within each epoch
    progress_bar = tqdm(range(0, len(data), batch_size), desc=f"Epoch {epoch + 1}", unit="batch")

    for i in progress_bar:
        batch_data = data[i:i + batch_size]  # Get a batch of data

        # Move tensors to GPU
        targets = torch.tensor([pair[0] for pair in batch_data], dtype=torch.long).to(device)
        contexts = torch.tensor([pair[1] for pair in batch_data], dtype=torch.long).to(device)

        optimizer.zero_grad()
        output = model(targets)  # Model is already on GPU
        loss = criterion(output, contexts)  # Loss computation on GPU
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Update tqdm description with current loss value
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch + 1}, Total Loss: {total_loss}")

# Save model and state dict (no need to move them explicitly)
torch.save(model, "skipgram_model.pt")
torch.save(model.state_dict(), "skipgram_model_state.pt")
scripted_model = torch.jit.script(model)
scripted_model.save("skipgram_model_script.pt")