# config.py
import os

VOCAB_SIZE = 50_000
UNK_TOKEN = "UNK"
BATCH_SIZE = 8192
WINDOW_SIZE = 5
N_NEG = 10
EMBEDDING_DIM = 75
MAX_EPOCHS = 200
DATASET_WORKERS = 10
LOAD_CHECKPOINT = False
CHECKPOINT_PATH = "checkpoint_latest.pt"


FILE_PATH = "/home/magnus/RustroverProjects/Semi24-Fileexplorer/src-tauri/src/neural_network/deu_wikipedia_2021_1M/deu_wikipedia_2021_1M-sentences.txt"
FILE_NAME = os.path.basename(FILE_PATH)
VOCAB_NAME = "deu_vocab.json"
WEIGHTS_NAME = "deu_weights_D75"