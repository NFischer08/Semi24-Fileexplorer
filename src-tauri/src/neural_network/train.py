import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import GPUtil

from config import *
from data_utils import load_and_tokenize, build_vocab, save_vocab
from dataset import SkipGramNegDataset
from model import SkipGramNegSampling

# GPU env setup
if not GPUtil.getGPUs():
    os.putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.putenv("AMD_SERIALIZE_KERNEL", "3")

def main():
    tokens = load_and_tokenize(FILE_PATH)
    vocab, vocab_counter = build_vocab(tokens, VOCAB_SIZE, UNK_TOKEN)
    save_vocab(vocab, VOCAB_NAME)
    unk_idx = 0
    tokens_idx = [vocab.get(t, unk_idx) for t in tokens]
    print(f"Vocabulary size: {len(vocab)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = SkipGramNegDataset(tokens_idx, vocab, vocab_counter, window_size=WINDOW_SIZE, unk_idx=unk_idx, device='cpu')
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATASET_WORKERS, prefetch_factor=4, pin_memory=True, persistent_workers=True)

    model = SkipGramNegSampling(len(vocab), EMBEDDING_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    word_prob_tensor = torch.tensor(dataset.word_prob, dtype=torch.float32, device=device)

    start_epoch = 0
    prev_loss = None
    consecutive_increases = 0

    if os.path.exists(CHECKPOINT_PATH) and LOAD_CHECKPOINT == True:
        print(f"Loading checkpoint from {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        prev_loss = checkpoint.get('loss', None)
        print(f"Resuming from epoch {start_epoch} with previous loss {prev_loss}")

    start_time = time.time()

    for epoch in range(start_epoch, MAX_EPOCHS):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", unit="batch")
        for targets, contexts in progress_bar:
            targets, contexts = targets.to(device), contexts.to(device)
            batch_size = targets.shape[0]
            negatives = torch.multinomial(
                word_prob_tensor,
                num_samples=batch_size * N_NEG,
                replacement=True
            ).view(batch_size, N_NEG)
            optimizer.zero_grad()
            loss = model(targets, contexts, negatives)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss,
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

        if prev_loss is not None and prev_loss < total_loss:
            consecutive_increases += 1
            print("consecutive_increased")
        else:
            consecutive_increases = 0
            prev_loss = total_loss

        if consecutive_increases == 2:
            print(f"Stopping early: loss increased two epochs in a row at epoch {epoch + 1}")
            break

    embedding_weights = model.target_embeddings.weight.data.cpu().numpy()
    embedding_weights.tofile("eng_weights_D300")
    print("Embeddings saved")
    print("Seconds since epoch =", time.time() - start_time)

if __name__ == "__main__":
    main()
