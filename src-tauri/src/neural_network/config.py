# config.py
import os

VOCAB_SIZE = 50_000
UNK_TOKEN = "UNK"
BATCH_SIZE = 8192
WINDOW_SIZE = 5
N_NEG = 10
EMBEDDING_DIM = 150
MAX_EPOCHS = 100
DATASET_WORKERS = 10
LOAD_CHECKPOINT = True
CHECKPOINT_PATH = "checkpoint_latest.pt"


FILE_PATH = "/home/magnus/RustroverProjects/Semi24-Fileexplorer/src-tauri/src/neural_network/eng_wikipedia_2016_1M/eng_wikipedia_2016_1M-sentences.txt"
FILE_NAME = os.path.basename(FILE_PATH)
VOCAB_NAME = "eng_vocab.json"
WEIGHTS_NAME = "eng_weights_D150"