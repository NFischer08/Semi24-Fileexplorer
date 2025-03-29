import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from itertools import chain

# Define SkipGram Model
class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, target):
        embed = self.embeddings(target)
        return self.output(embed)

# Read file names from a .txt file
file_path = "deu_wikipedia_2021_10K/deu_wikipedia_2021_10K-sentences.txt"  # Path to your .txt file

with open(file_path, "r", encoding="utf-8") as f:
    file_names = [line.strip() for line in f.readlines()]  # Strip whitespace and newlines

tokens = list(chain.from_iterable([name.split('_') for name in file_names]))
vocab = {word: idx for idx, word in enumerate(set(tokens))}
data = [(vocab[tokens[i]], vocab[tokens[i + 1]]) for i in range(len(tokens) - 1)]

# Training
vocab_size = len(vocab)
embedding_dim = 64
model = SkipGramModel(vocab_size, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):  # Training loop
    total_loss = 0
    for target, context in data:
        target_tensor = torch.tensor([target], dtype=torch.long)
        context_tensor = torch.tensor([context], dtype=torch.long)

        optimizer.zero_grad()
        output = model(target_tensor)
        loss = criterion(output, context_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch}, Loss: {total_loss}")

torch.save(model, "skipgram_model.pt")
torch.save(model.state_dict(), "skipgram_model_state.pt")
