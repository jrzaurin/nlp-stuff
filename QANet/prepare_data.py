import numpy as np
import os
import pickle
import spacy
import ujson as json

from pathlib import Path
from tqdm import tqdm
from collections import Counter
from utils.preprocessing import (
    process_file,
    get_embeddings_v1,
    get_embeddings_v2,
    build_sequences,
)
from utils.text import Vocab

import pdb


if __name__ == "__main__":

    data_dir = Path("data")
    squad_dir = data_dir / "squad"
    # I will use dev as test and split train into train/dev
    train_file = "train-v1.1.json"
    test_file = "dev-v1.1.json"

    embed_dir = data_dir / "glove"
    # word_vectors_path = None
    # char_vectors_path = None
    word_vectors_path = embed_dir / "glove.6B.300d.txt"
    char_vectors_path = embed_dir / "glove.840B.300d-char.txt"

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    valid_dir = data_dir / "valid"

    logs_dir = Path("logs")
    model_dir = Path("models")

    enforce = True

    if not os.path.exists(squad_dir):
        os.makedirs(squad_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        os.makedirs(valid_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # c = context, q = question, a = answer
    if os.path.exists(data_dir / "word_counter.p") and not enforce:
        word_counter = pickle.load(open(data_dir / "word_counter.p", "rb"))
        char_counter = pickle.load(open(data_dir / "char_counter.p", "rb"))
    else:
        word_counter, char_counter = Counter(), Counter()
        process_file(squad_dir / train_file, word_counter, char_counter, "train")
        process_file(squad_dir / test_file, word_counter, char_counter, "test")
        pickle.dump(word_counter, open(data_dir / "word_counter.p", "wb"))
        pickle.dump(char_counter, open(data_dir / "char_counter.p", "wb"))

    if word_vectors_path is None:
        word_vocab = Vocab.create(word_counter, max_vocab=50000, min_freq=5)
    else:
        word_vocab, word_emb_mat = get_embeddings_v1(
            word_counter, emb_file=word_vectors_path, max_vocab=90000, min_freq=1
        )
        np.savez(data_dir / "word_emb_mat.npz", word_emb_mat=word_emb_mat)
    pickle.dump(word_vocab, open(data_dir / "word_vocab.p", "wb"))

    if char_vectors_path is None:
        char_vocab = Vocab.create(char_counter, max_vocab=200, min_freq=-1)
    else:
        char_vocab, char_emb_mat = get_embeddings_v1(
            char_counter, emb_file=char_vectors_path, max_vocab=1000, min_freq=-1
        )
        np.savez(data_dir / "char_emb_mat.npz", char_emb_mat=char_emb_mat)
    pickle.dump(char_vocab, open(data_dir / "char_vocab.p", "wb"))

    build_sequences(
        train_dir / "full_train_c_q.p",
        train_dir / "full_train_seq.npz",
        word_vocab,
        char_vocab,
    )
    build_sequences(
        train_dir / "train_c_q.p", train_dir / "train_seq.npz", word_vocab, char_vocab,
    )
    build_sequences(
        valid_dir / "valid_c_q.p", valid_dir / "valid_seq.npz", word_vocab, char_vocab,
    )
    build_sequences(
        test_dir / "test_c_q.p", test_dir / "test_seq.npz", word_vocab, char_vocab
    )
