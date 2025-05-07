import re
import json
import numpy as np
import os
import GPUtil

if not GPUtil.getGPUs():
    os.putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.putenv("AMD_SERIALIZE_KERNEL", "3")

from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# 1. Data Preparation
# -------------------------------
# Parameters
VOCAB_SIZE = 50_000
UNK_TOKEN = "UNK"

file_path = "deu_wikipedia_2021_10K/deu_wikipedia_2021_10K-sentences.txt"

def normalize_token(token):
    if re.fullmatch(r"\d{4}([-:.])\d{2}\1\d{2}", token):
        return "DATE"
    elif re.fullmatch(r"\d{4}", token):
        return "YEAR"
    else:
        return token

with open(file_path, "r", encoding="utf-8") as f:
    file_names = [line.strip() for line in f.readlines()]

tokens = []
for name in file_names:
    name = name.lower()
    raw_tokens = re.findall(r"[a-zäöü]+|\d{4}[-:.]\d{2}[-:.]\d{2}|\d{4}", name)
    tokens.extend([normalize_token(t) for t in raw_tokens])

vocab_counter = Counter(tokens)
most_common = vocab_counter.most_common(VOCAB_SIZE - 1)  # -1 to leave space for UNK at 0
vocab_words = [word for word, _ in most_common]

# Build vocab with <UNK> at index 0
vocab = {UNK_TOKEN: 0}
for idx, word in enumerate(vocab_words, start=1):
    vocab[word] = idx
unk_idx = 0  # UNK is always 0

# Map tokens to indices, using UNK for OOV
tokens_idx = [vocab.get(t, unk_idx) for t in tokens]

print(f"Vocabulary size: {len(vocab)}")

# Save vocab for reference
with open("deu_vocab.json", "w", encoding="utf-8") as vocab_file:
    json.dump(vocab, vocab_file, ensure_ascii=False, indent=4)

# -------------------------------
# 2. Dataset with Negative Sampling and UNK
# -------------------------------
class SkipGramNegDataset(Dataset):
    def __init__(self, tokens_idx, vocab, vocab_counter, window_size=2, unk_idx=None, subsample_t=1e-4, device='cpu'):
        self.vocab = vocab
        self.window_size = window_size
        self.unk_idx = unk_idx

        # --- Subsampling Setup ---
        total_count = len(tokens_idx)
        idx2word = {idx: word for word, idx in vocab.items()}
        freqs = np.zeros(len(vocab))
        for idx in range(len(vocab)):
            word = idx2word[idx]
            freqs[idx] = vocab_counter.get(word, 0) / total_count
        subsample_probs = 1 - np.sqrt(subsample_t / (freqs + 1e-10))
        subsample_probs = np.clip(subsample_probs, 0, 1)

        filtered_tokens_idx = []
        for idx in tokens_idx:
            if np.random.rand() > subsample_probs[idx]:
                filtered_tokens_idx.append(idx)
        self.tokens_idx = filtered_tokens_idx

        # --- Negative Sampling Distribution ---
        word_counts = np.array([vocab_counter.get(word, 1) for word in vocab])
        word_counts[unk_idx] = 1  # Avoid oversampling UNK
        self.word_prob = word_counts ** 0.75
        self.word_prob = self.word_prob / self.word_prob.sum()
        self.word_prob_tensor = torch.tensor(self.word_prob, dtype=torch.float32)

        # --- Build (target, context) pairs ---
        pairs = []
        for i, target_idx in enumerate(self.tokens_idx):
            if target_idx == self.unk_idx:
                continue
            start = max(0, i - window_size)
            end = min(len(self.tokens_idx), i + window_size + 1)
            for j in range(start, end):
                if j == i:
                    continue
                context_idx = self.tokens_idx[j]
                if context_idx == self.unk_idx:
                    continue
                pairs.append((target_idx, context_idx))
        self.pairs = torch.tensor(pairs, dtype=torch.long, device=device)
        self.device = device

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, idx):
        target, context = self.pairs[idx]
        return target, context
# -------------------------------
# 3. SkipGram Model with Negative Sampling
# -------------------------------
class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, target, context, negatives):
        emb_target = self.target_embeddings(target)
        emb_context = self.context_embeddings(context)
        pos_score = torch.sum(emb_target * emb_context, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score) + 1e-10)

        emb_neg = self.context_embeddings(negatives)
        neg_score = torch.bmm(emb_neg, emb_target.unsqueeze(2)).squeeze(2)
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1)

        loss = -torch.mean(pos_loss + neg_loss)
        return loss

# -------------------------------
# 4. Training
# -------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

batch_size = 8192
window_size = 5
n_neg = 10
embedding_dim = 300
epochs = 20

dataset = SkipGramNegDataset(tokens_idx, vocab, vocab_counter, window_size=window_size, unk_idx=unk_idx, device='cpu')
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12, prefetch_factor=4, pin_memory=True, persistent_workers=True)

model = SkipGramNegSampling(len(vocab), embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.003)

# Get the negative sampling distribution tensor ONCE and move to GPU
word_prob_tensor = torch.tensor(dataset.word_prob, dtype=torch.float32, device=device)

for epoch in range(epochs):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")
    for targets, contexts in progress_bar:
        targets, contexts = targets.to(device), contexts.to(device)
        batch_size = targets.shape[0]
        # Vectorized negative sampling on GPU!
        negatives = torch.multinomial(
            word_prob_tensor,
            num_samples=batch_size * n_neg,
            replacement=True
        ).view(batch_size, n_neg)
        optimizer.zero_grad()
        loss = model(targets, contexts, negatives)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

# -------------------------------
# 5. Save Embeddings and Visualize
# -------------------------------
embedding_weights = model.target_embeddings.weight.data.cpu().numpy()
embedding_weights.tofile("deu_weights")
print("Embeddings saved")

