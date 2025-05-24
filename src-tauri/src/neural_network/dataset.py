import torch
import numpy as np
from torch.utils.data import Dataset

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
