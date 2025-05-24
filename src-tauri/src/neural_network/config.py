# config.py

VOCAB_SIZE = 50_000
UNK_TOKEN = "UNK"
BATCH_SIZE = 8192
WINDOW_SIZE = 5
N_NEG = 10
EMBEDDING_DIM = 300
MAX_EPOCHS = 50
DATASET_WORKERS = 10
LOAD_CHECKPOINT = False
CHECKPOINT_PATH = "checkpoint_latest.pt"


FILE_PATH = "/home/magnus/RustroverProjects/Semi24-Fileexplorer/src-tauri/src/neural_network/deu_wikipedia_2021_1M/deu_wikipedia_2021_1M-sentences.txt"
VOCAB_NAME = "eng_vocab.json"