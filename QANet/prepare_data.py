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
from utils.text_utils import Vocab
from utils import config

if __name__ == "__main__":

    data_dir = config.data_dir

    # I will use dev as test and split train into train/dev
    orig_train_fpath = config.orig_train_fpath
    orig_test_fpath = config.orig_test_fpath

    # word_vectors_path = config.glove_wordv_fpath
    word_vectors_path = config.fastt_wordv_fpath
    char_vectors_path = config.glove_charv_fpath

    train_dir = config.train_dir
    test_dir = config.test_dir
    valid_dir = config.valid_dir

    # useful when I was experimenting
    enforce = True

    # c = context, q = question, a = answer
    if os.path.exists(data_dir / "word_counter.p") and not enforce:
        word_counter = pickle.load(open(data_dir / "word_counter.p", "rb"))
        char_counter = pickle.load(open(data_dir / "char_counter.p", "rb"))
    else:
        word_counter, char_counter = Counter(), Counter()
        process_file(orig_train_fpath, word_counter, char_counter, "train")
        process_file(orig_test_fpath, word_counter, char_counter, "test")
        pickle.dump(word_counter, open(data_dir / "word_counter.p", "wb"))
        pickle.dump(char_counter, open(data_dir / "char_counter.p", "wb"))

    if word_vectors_path is None:
        word_vocab = Vocab.create(word_counter, max_vocab=50000, min_freq=5)
    else:
        word_vocab, word_emb_mtx = get_embeddings_v1(
            word_counter,
            emb_file=word_vectors_path,
            max_vocab=len(word_counter),
            min_freq=-1,
        )
        pickle.dump(word_emb_mtx, open(data_dir / "word_emb_mtx.p", "wb"))
    word_vocab.save(data_dir / "word_vocab.p")

    if char_vectors_path is None:
        char_vocab = Vocab.create(char_counter, max_vocab=200, min_freq=-1)
    else:
        char_vocab, char_emb_mtx = get_embeddings_v1(
            char_counter,
            emb_file=char_vectors_path,
            max_vocab=len(char_counter),
            min_freq=-1,
        )
        pickle.dump(char_emb_mtx, open(data_dir / "char_emb_mtx.p", "wb"))
    char_vocab.save(data_dir / "char_vocab.p")
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
