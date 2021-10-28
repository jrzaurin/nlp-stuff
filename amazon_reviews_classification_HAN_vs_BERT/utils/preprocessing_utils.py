import numpy as np
import os
import re
import html

from gensim.utils import tokenize

from pytorch_widedeep.utils import Tokenizer


def spec_add_spaces(text):
    "Add spaces around / and # in `t`. \n"
    return re.sub(r"([/#\n])", r" \1 ", text)


def rm_useless_spaces(text):
    "Remove multiple spaces in `t`."
    return re.sub(" {2,}", " ", text)


def fix_html(text):
    "List of replacements from html strings in `x`."
    re1 = re.compile(r"  +")
    text = (
        text.replace("#39;", "'")
        .replace("amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace(" @,@ ", ",")
        .replace("\\", " \\ ")
    )
    return re1.sub(" ", html.unescape(text))


def simple_preprocess(doc, lower=False, deacc=False, min_len=2, max_len=15):
    tokens = [
        token
        for token in tokenize(doc, lower=False, deacc=deacc, errors="ignore")
        if min_len <= len(token) <= max_len and not token.startswith("_")
    ]
    return tokens


def get_texts(texts, with_preprocess=False):
    tok_func = Tokenizer()
    if with_preprocess:
        texts = [" ".join(simple_preprocess(s)) for s in texts]
    tokens = tok_func.process_all(texts)
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


def build_embeddings_matrix(vocab, word_vectors_path, verbose=1):
    if not os.path.isfile(word_vectors_path):
        raise FileNotFoundError("{} not found".format(word_vectors_path))
    if verbose:
        print("Indexing word vectors...")

    embeddings_index = {}
    f = open(word_vectors_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()

    if verbose:
        print("Loaded {} word vectors".format(len(embeddings_index)))
        print("Preparing embeddings matrix...")

    mean_word_vector = np.mean(list(embeddings_index.values()), axis=0)
    embedding_dim = len(list(embeddings_index.values())[0])
    num_words = len(vocab.itos)
    embedding_matrix = np.zeros((num_words, embedding_dim))
    found_words = 0
    for i, word in enumerate(vocab.itos):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            found_words += 1
        else:
            embedding_matrix[i] = mean_word_vector

    return embedding_matrix
