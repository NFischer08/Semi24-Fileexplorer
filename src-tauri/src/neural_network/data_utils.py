import re
import json
from collections import Counter

def normalize_token(token):
    if re.fullmatch(r"\d{4}([-:.])\d{2}\1\d{2}", token):
        return "DATE"
    elif re.fullmatch(r"\d{4}", token):
        return "YEAR"
    else:
        return token

def load_and_tokenize(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        file_names = [line.strip() for line in f.readlines()]
    tokens = []
    for name in file_names:
        name = name.lower()
        raw_tokens = re.findall(r"[a-zäöü]+|\d{4}[-:.]\d{2}[-:.]\d{2}|\d{4}", name)
        tokens.extend([normalize_token(t) for t in raw_tokens])
    return tokens

def build_vocab(tokens, vocab_size, unk_token):
    vocab_counter = Counter(tokens)
    most_common = vocab_counter.most_common(vocab_size - 1)
    vocab_words = [word for word, _ in most_common]
    vocab = {unk_token: 0}
    for idx, word in enumerate(vocab_words, start=1):
        vocab[word] = idx
    return vocab, vocab_counter

def save_vocab(vocab, vocab_name):
    with open(vocab_name, "w", encoding="utf-8") as vocab_file:
        json.dump(vocab, vocab_file, ensure_ascii=False, indent=4)
