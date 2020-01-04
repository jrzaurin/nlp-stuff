import numpy as np
import spacy
import os

from gensim.utils import tokenize
from fastai.text import Tokenizer, Vocab
from sklearn.exceptions import NotFittedError


n_cpus = os.cpu_count()


class BasePreprocessor(object):
    def __init__(self):
        super(BasePreprocessor, self).__init__()

    def tokenize(self, texts):
        raise NotImplementedError

    def transform(self, texts):
        raise NotImplementedError


class HANPreprocessor(BasePreprocessor):
    r"""
    Preprocessor to prepare the data for Hierarchical Attention Networks.
    It will "tokenize" a document into sentences and sentences into tokens

    Parameters:
    ----------
    batch_size: Int
        Int indicating the batch size for Spacy's pipe parallel processes
    max_vocab: Int. Default=30000
        max vocabulary size
    min_freq: Int. Default=5
        min token frequency for a token to be considered
    q: Float. Default=0.8
        quantile used to select the padding maxlen
    pad_sent_first: Boolean. Default=True
    pad_doc_first: Boolean. Default=True
    pad_idx: Int. Default=1
    tok_func: Any. Default=None
        Custom tokenizer. Must be an spacy.lang type or equivalent
    n_cpus: Int
    verbose: Boolean. Default=True

    Returns:
    -------
    np.ndarray with stacked padded sequences
    """

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
        except AttributeError:
            if self.verbose:
                print("Building Vocab...")
            self.vocab = Vocab.create(
                sents_tokens, max_vocab=self.max_vocab, min_freq=self.min_freq
            )
        # 'numericalize' each sentence
        sents_numz = [self.vocab.numericalize(s) for s in sents_tokens]
        # group the sentences again into documents
        texts_numz = [
            sents_numz[range_idx[i] : range_idx[i + 1]] for i in range(len(range_idx[:-1]))
        ]
        # compute max lengths for padding purposes
        self.maxlen_sent = int(np.quantile(sents_length, q=self.q))
        self.maxlen_doc = int(np.quantile(texts_length[1:], q=self.q))

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
        except AttributeError:
            raise NotFittedError(
                "This HANTokenizer instance is not trained yet. "
                "Call 'tokenize' with appropriate arguments before using this estimator."
            )
        return self.tokenize(texts)


class TextPreprocessor(BasePreprocessor):
    r"""
    Preprocessor to prepare the data for a standard RNN classification process.

    Parameters:
    ----------
    max_vocab: Int. Default=30000
        max vocabulary size
    min_freq: Int. Default=5
        min token frequency for a token to be considered
    q: Float. Default=0.8
        quantile used to select the padding maxlen
    pad_first: Boolean. Default=True
    pad_idx: Int. Default=1
    verbose: Boolean. Default=True

    Returns:
    -------
    np.ndarray with stacked padded sequences
    """

    def __init__(self, max_vocab=30000, min_freq=5, q=0.8, pad_first=True, pad_idx=1, verbose=1):
        super(TextPreprocessor, self).__init__()
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.q = q
        self.pad_first = pad_first
        self.pad_idx = pad_idx
        self.verbose = verbose

    def tokenize(self, texts):
        if self.verbose:
            print("Tokenizing {} documents...".format(len(texts)))
        tokens = get_texts(texts)
        texts_length = [len(t) for t in tokens]
        try:
            self.vocab
            if self.verbose:
                print("Using existing vocabulary")
        except AttributeError:
            if self.verbose:
                print("Building Vocab...")
            self.vocab = Vocab.create(tokens, max_vocab=self.max_vocab, min_freq=self.min_freq)
        texts_numz = [self.vocab.numericalize(t) for t in texts]
        sorted_texts_length = sorted(texts_length)
        self.maxlen = int(np.quantile(sorted_texts_length, q=self.q))
        # self.maxlen = sorted_texts_length[int(self.q * len(sorted_texts_length))]
        if self.verbose:
            print("Padding documents...")
        padded_texts = [
            pad_sequences(t, self.maxlen, pad_first=self.pad_first, pad_idx=self.pad_idx)
            for t in texts_numz
        ]
        return np.stack(padded_texts, axis=0)

    def transform(self, texts):
        try:
            self.vocab
        except AttributeError:
            raise NotFittedError(
                "This HANTokenizer instance is not trained yet. "
                "Call 'tokenize' with appropriate arguments before using this estimator."
            )
        return self.tokenize(texts)


def simple_preprocess(doc, lower=False, deacc=False, min_len=2, max_len=15):
    r"""
    Gensim's simple_preprocess adding a 'lower' param to indicate wether or not to
    lower case all the token in the texts
    For more informations see: https://radimrehurek.com/gensim/utils.html
    """
    tokens = [
        token
        for token in tokenize(doc, lower=False, deacc=deacc, errors="ignore")
        if min_len <= len(token) <= max_len and not token.startswith("_")
    ]
    return tokens


def get_texts(texts, with_preprocess=False, pre_rules=None):
    r"""
    Uses fastai's Tokenizer because it does a series of very convenients things
    during the tokenization process
    See here: https://docs.fast.ai/text.transform.html#Tokenizer
    """
    tok_func = Tokenizer()
    if with_preprocess:
        texts = [" ".join(simple_preprocess(s)) for s in texts]
    if pre_rules:
        tok_func.pre_rules = pre_rules + tok_func.pre_rules
    tokens = tok_func.process_all(texts)
    return tokens


def pad_sequences(seq, maxlen, pad_first=True, pad_idx=1):
    r"""
    Given a List of tokenized and 'numericalised' sequences it will return padded sequences
    according to the input parameters maxlen, pad_first and pad_idx

    Parameters
    ----------
    seq: List
        List of int tokens
    maxlen: Int
        Maximum length of the padded sequences
    pad_first: Boolean. Default=True
        Indicates whether the padding index will be added at the beginning or the
        end of the sequences
    pad_idx: Int. Default=1
        padding index. Default=1, Fastai's Tokenizer leaves 0 for the 'unknown' token.

    Returns:
    res: np.ndarray
        Padded sequences
    """
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
    r"""
    Same as pad_sequences but for nested Lists
    """
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


def build_embeddings_matrix(vocab, word_vectors_path, verbose=1):
    r"""
    Build the embedding matrix using pretrained word vectors

    Parameters
    ----------
    vocab: Fastai's Vocab object
        see: https://docs.fast.ai/text.transform.html#Vocab
    word_vectors_path:str
        path to the pretrained word embeddings
    verbose: Int. Default=1

    Returns
    -------
    embedding_matrix: np.ndarray
        pretrained word embeddings. If a word in our vocabulary is not among the
        pretrained embeddings it will be assigned the mean pretrained
        word-embeddings vector
    """
    if not os.path.isfile(word_vectors_path):
        raise FileNotFoundError("{} not found".format(word_vectors_path))
    if verbose: print('Indexing word vectors...')

    embeddings_index = {}
    f = open(word_vectors_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    if verbose:
        print('Loaded {} word vectors'.format(len(embeddings_index)))
        print('Preparing embeddings matrix...')

    mean_word_vector = np.mean(list(embeddings_index.values()), axis=0)
    embedding_dim = len(list(embeddings_index.values())[0])
    num_words = len(vocab.itos)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    found_words=0
    for i,word in enumerate(vocab.itos):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_words+=1
        else:
            embedding_matrix[i] = mean_word_vector

    return embedding_matrix


def test_idx_groups(nested_list, flat_list, idx_seq):
    r"""
    ***CAN BE IGNORED***
    Helper function I used to check that the folding/unfolding process on
    nested list was working properly.
    """
    res_nested_list = [flat_list[idx_seq[i] : idx_seq[i + 1]] for i in range(len(idx_seq[:-1]))]
    rand_ids = np.random.choice(len(idx_seq), 100)
    res = [res_nested_list[i] == nested_list[i] for i in rand_ids]
    return all(res)
