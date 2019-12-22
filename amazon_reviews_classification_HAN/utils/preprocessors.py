import pandas as pd
import numpy as np
import spacy
import os

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from gensim.utils import tokenize
from fastai.text import Tokenizer, Vocab
from sklearn.exceptions import NotFittedError


n_cpus = os.cpu_count()


def test_idx_groups(nested_list, flat_list, idx_seq):
    res_nested_list = [
        flat_list[idx_seq[i] : idx_seq[i + 1]] for i in range(len(idx_seq[:-1]))
    ]
    rand_ids = np.random.choice(len(idx_seq), 100)
    res = [res_nested_list[i] == nested_list[i] for i in rand_ids]
    return all(res)


def simple_preprocess(doc, lower=False, deacc=False, min_len=2, max_len=15):
    tokens = [
        token
        for token in tokenize(doc, lower=False, deacc=deacc, errors="ignore")
        if min_len <= len(token) <= max_len and not token.startswith("_")
    ]
    return tokens


def get_texts(texts, with_preprocess=False):
    if with_preprocess:
        texts = [" ".join(simple_preprocess(s)) for s in texts]
    tokens = Tokenizer().process_all(texts)
    return tokens


def pad_sequences(seq, maxlen, pad_first=True, pad_idx=1):
    if len(seq) >= maxlen:
        res = np.array(seq[-maxlen:]).astype("int32")
        return res
    else:
        res = np.zeros(maxlen, dtype="int32") + pad_idx
        if pad_first:
            res[-len(seq) :] = seq
        else:
            res[: len(seq) :] = seq
        return res


def pad_nested_sequences(
    seq, maxlen_sent, maxlen_doc, pad_sent_first=True, pad_doc_first=False, pad_idx=1
    ):
    seq = [s for s in seq if len(s) >= 1]
    if len(seq) == 0:
        return np.array([[pad_idx] * maxlen_sent] * maxlen_doc).astype("int32")
    seq = [pad_sequences(s, maxlen_sent, pad_sent_first, pad_idx) for s in seq]
    if len(seq) >= maxlen_doc:
        return np.array(seq[:maxlen_doc])
    else:
        res = np.array([[pad_idx] * maxlen_sent] * maxlen_doc).astype("int32")
        if pad_doc_first:
            res[-len(seq) :] = seq
        else:
            res[: len(seq) :] = seq
        return res


class BasePreprocessor(object):
    def __init__(self):
        super(BasePreprocessor, self).__init__()

    def tokenize(self, texts):
        raise NotImplemented

    def transform(self, texts):
        raise NotImplemented


class HANPreprocessor(BasePreprocessor):
    def __init__(
        self,
        batch_size=1000,
        max_vocab=30000,
        min_freq=5,
        q=0.8,
        pad_sent_first=True,
        pad_doc_first=False,
        pad_idx=1,
        tok_func=None,
        n_cpus=None,
        verbose=True,
    ):
        super(HANPreprocessor, self).__init__()
        self.batch_size = batch_size
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.q = q
        self.pad_sent_first = pad_sent_first
        self.pad_doc_first = pad_doc_first
        self.pad_idx = pad_idx
        self.tok_func = spacy.load("en_core_web_sm") if tok_func is None else tok_func
        self.n_cpus = os.cpu_count() if n_cpus is None else n_cpus
        self.verbose = verbose

    def _sentencizer(self, texts):
        texts_sents = []
        for doc in self.tok_func.pipe(texts, n_process=self.n_cpus, batch_size=self.batch_size):
            sents = [str(s) for s in list(doc.sents)]
            texts_sents.append(sents)
        return texts_sents

    def tokenize(self, texts):
        if self.verbose:
            print("Running sentence tokenizer for {} documents...".format(len(texts)))
        texts_sents = self._sentencizer(texts)
        # from nested to flat list. For speed purposes
        all_sents = [s for sents in texts_sents for s in sents]
        #  saving the lengths of the documents: 1) for padding purposes and 2) to
        #  compute consecutive ranges so we can "fold" the list again
        texts_length = [0] + [len(s) for s in texts_sents]
        range_idx = [sum(texts_length[: i + 1]) for i in range(len(texts_length))]
        if self.verbose:
            print("Tokenizing {} sentences...".format(len(all_sents)))
        sents_tokens = get_texts(all_sents)
        #  saving the lengths of sentences for padding purposes
        sents_length = [len(s) for s in sents_tokens]
        try:
            self.vocab
            if self.verbose:
                print("Using existing vocabulary")
        except:
            if self.verbose:
                print("Building Vocab...")
            self.vocab = Vocab.create(
                sents_tokens, max_vocab=self.max_vocab, min_freq=self.min_freq
            )
            # 'numericalize' each sentence
        sents_numz = [self.vocab.numericalize(s) for s in sents_tokens]
        # group the sentences again into documents
        texts_numz = [
            sents_numz[range_idx[i] : range_idx[i + 1]]
            for i in range(len(range_idx[:-1]))
        ]
        # compute max lengths for padding purposes
        sorted_sent_length = sorted(sents_length)
        sorted_review_length = sorted(texts_length[1:])
        self.maxlen_sent = sorted_sent_length[int(self.q * len(sorted_sent_length))]
        self.maxlen_doc = sorted_review_length[int(self.q * len(sorted_review_length))]
        if self.verbose:
            print("Padding sentences and documents...")
        padded_texts = [
            pad_nested_sequences(
                r,
                self.maxlen_sent,
                self.maxlen_doc,
                pad_sent_first=self.pad_sent_first,
                pad_doc_first=self.pad_doc_first,
                pad_idx=self.pad_idx,
            )
            for r in texts_numz
        ]
        return np.stack(padded_texts, axis=0)

    def transform(self, texts):
        try:
            self.vocab
        except:
            raise NotFittedError(
                "This HANTokenizer instance is not trained yet. "
                "Call 'tokenize' with appropriate arguments before using this estimator."
            )
        return self.tokenize(texts)


class TextPreprocessor(BasePreprocessor):
    def __init__(self, max_vocab=30000, min_freq=5, q=0.8, verbose=1):
        super(TextPreprocessor, self).__init__()
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.verbose = verbose
        self.q = q

    def tokenize(self, texts):
        if self.verbose:
            print("Tokenizing {} documents...".format(len(texts)))
        tokens = get_texts(texts)
        texts_length = [len(t) for t in tokens]
        try:
            self.vocab
            if self.verbose:
                print("Using existing vocabulary")
        except:
            if self.verbose:
                print("Building Vocab...")
            self.vocab = Vocab.create(
                tokens, max_vocab=self.max_vocab, min_freq=self.min_freq
            )
        texts_numz = [self.vocab.numericalize(t) for t in texts]
        sorted_texts_length = sorted(texts_length)
        self.maxlen = sorted_texts_length[int(self.q * len(sorted_texts_length))]
        if self.verbose:
            print("Padding documents...")
        padded_texts = [pad_sequences(t, self.maxlen) for t in texts_numz]
        return np.stack(padded_texts, axis=0)

    def transform(self, texts):
        try:
            self.vocab
        except:
            raise NotFittedError(
                "This HANTokenizer instance is not trained yet. "
                "Call 'tokenize' with appropriate arguments before using this estimator."
            )
        return self.tokenize(texts)
