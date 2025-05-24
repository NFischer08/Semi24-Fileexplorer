import torch.nn as nn
import torch

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
