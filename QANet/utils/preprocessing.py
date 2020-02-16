"""
The content of this file is an adaptation from
https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py
"""

import os
import spacy
import pickle
import ujson as json
import numpy as np

from tqdm import tqdm
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

from .text_utils import Vocab

import pdb

nlp = spacy.blank("en")
PAD, UNK = "xxpad", "xxunk"


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filepath, word_counter, char_counter, data_type):

    examples = []
    eval_examples = {}
    total = 0
    with open(filepath, "r") as fh:
        source = json.load(fh)
        print("preparing (context, question, answer) triplets.")
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace("''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    # there are len(para["qas"]) per paragraph
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace("''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer["answer_start"]
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            # if answer is within span, add span idx
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {
                        "context_tokens": context_tokens,
                        "context_chars": context_chars,
                        "ques_tokens": ques_tokens,
                        "ques_chars": ques_chars,
                        "y1s": y1s,
                        "y2s": y2s,
                        "id": total,
                    }
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context,
                        "spans": spans,
                        "answers": answer_texts,
                        "uuid": qa["id"],
                    }
        print("{} questions in total".format(len(examples)))
    # c = context, q = question, a = answer
    if data_type == "train":
        pickle.dump(examples, open("data/train/full_train_c_q.p", "wb"))
        pickle.dump(eval_examples, open("data/train/full_train_c_a.p", "wb"))
        train_c_q, valid_c_q = train_test_split(
            examples, test_size=10000, random_state=1
        )
        train_ids = [te["id"] for te in train_c_q]
        valid_ids = [ve["id"] for ve in valid_c_q]
        train_c_a = {str(i): eval_examples[str(i)] for i in train_ids}
        valid_c_a = {str(i): eval_examples[str(i)] for i in valid_ids}
        pickle.dump(train_c_q, open("data/train/train_c_q.p", "wb"))
        pickle.dump(train_c_a, open("data/train/train_c_a.p", "wb"))
        pickle.dump(valid_c_q, open("data/valid/valid_c_q.p", "wb"))
        pickle.dump(valid_c_a, open("data/valid/valid_c_a.p", "wb"))
    else:
        pickle.dump(examples, open("data/test/test_c_q.p", "wb"))
        pickle.dump(eval_examples, open("data/test/test_c_a.p", "wb"))


def get_embeddings_v1(counter, max_vocab, min_freq, emb_file, vec_size=300):

    filtered_elements = [t for t, c in counter.most_common(max_vocab) if c >= min_freq]
    embeddings_index = {}
    f = open(emb_file)
    print("indexing embeddings.")
    for line in tqdm(f):
        values = line.split()
        token = "".join(values[0:-vec_size])
        coefs = np.asarray(values[-vec_size:], dtype="float32")
        if token in counter and counter[token] > min_freq:
            embeddings_index[token] = coefs
    f.close()
    print(
        "{} / {} tokens have corresponding embedding vector".format(
            len(embeddings_index), len(filtered_elements)
        )
    )
    embedding_dim = len(list(embeddings_index.values())[0])
    vocab = Vocab.create(embeddings_index.keys())
    embeddings_index[PAD] = [0.0 for _ in range(embedding_dim)]
    embeddings_index[UNK] = [0.0 for _ in range(embedding_dim)]
    embedding_matrix = np.array([embeddings_index[tok] for tok in vocab.itos])
    return vocab, embedding_matrix


def get_embeddings_v2(counter, max_vocab, min_freq, emb_file):

    vocab = Vocab.create(counter, max_vocab, min_freq)

    embeddings_index = {}
    f = open(emb_file)
    print("indexing embeddings.")
    for line in tqdm(f):
        values = line.split()
        token = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[token] = coefs
    f.close()

    mean_token_vector = np.mean(list(embeddings_index.values()), axis=0)
    embedding_dim = len(list(embeddings_index.values())[0])
    num_tokens = len(vocab.stoi)
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    found_tokens = 0
    for i, token in enumerate(vocab.stoi):
        embedding_vector = embeddings_index.get(token)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_tokens += 1
        else:
            embedding_matrix[i] = mean_token_vector
    print(
        "{} of {} tokens have corresponding embedding vector".format(
            found_tokens, len(vocab.itos)
        )
    )
    return vocab, embedding_matrix


def build_sequences(
    filepath,
    out_file,
    word_vocab,
    char_vocab,
    only_pretrained=True,
    para_limit=400,
    ques_limit=50,
    ans_limit=30,
    char_limit=16,
):
    def filter_func(example, is_test=False):
        return (
            len(example["context_tokens"]) > para_limit
            or len(example["ques_tokens"]) > ques_limit
            or (example["y2s"][0] - example["y1s"][0]) > ans_limit
        )

    def _get_word(word):
        if only_pretrained:
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word_vocab.stoi:
                    return word_vocab.stoi[each]
        elif word in word_vocab.stoi:
            return word_vocab.stoi[word]
        return 1

    def _get_char(char):
        if char in char_vocab.stoi:
            return char_vocab.stoi[char]
        return 1

    examples = pickle.load(open(filepath, "rb"))
    context_word_seqs = []
    context_char_seqs = []
    ques_word_seqs = []
    ques_char_seqs = []
    y1s = []
    y2s = []
    ids = []

    total = 0
    total_ = 0
    print("building padding 2D (N,word) and 3D (N, word, char) sequences")
    for example in tqdm(examples):
        total_ += 1
        if filter_func(example):
            continue
        total += 1

        context_word_seq = np.zeros([para_limit], dtype=np.int64)
        context_char_seq = np.zeros([para_limit, char_limit], dtype=np.int64)
        ques_word_seq = np.zeros([ques_limit], dtype=np.int64)
        ques_char_seq = np.zeros([ques_limit, char_limit], dtype=np.int64)

        for i, token in enumerate(example["context_tokens"]):

            context_word_seq[i] = _get_word(token)
        context_word_seqs.append(context_word_seq)

        for i, token in enumerate(example["ques_tokens"]):
            ques_word_seq[i] = _get_word(token)
        ques_word_seqs.append(ques_word_seq)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_seq[i, j] = _get_char(char)
        context_char_seqs.append(context_char_seq)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_seq[i, j] = _get_char(char)
        ques_char_seqs.append(ques_char_seq)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    print("Built {} of {} examples in total".format(total, total_))

    np.savez(
        out_file,
        context_word_seqs=np.array(context_word_seqs, dtype=np.int64),
        context_char_seqs=np.array(context_char_seqs, dtype=np.int64),
        ques_word_seqs=np.array(ques_word_seqs, dtype=np.int64),
        ques_char_seqs=np.array(ques_char_seqs, dtype=np.int64),
        y1s=np.array(y1s, dtype=np.int64),
        y2s=np.array(y2s, dtype=np.int64),
        ids=np.array(ids, dtype=np.int64),
    )
    # meta = {"total": total}
    # return meta
