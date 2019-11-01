from __future__ import print_function
import os
import sys
import numpy as np
import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from data_utils import pad_sequences, one_hot

import pdb


def clean_text( doc, remove_stopwords=False):
    # remove HTML
    doc_text = BeautifulSoup(doc).get_text()
    # remove non-letters
    doc_text = re.sub("[^a-zA-Z]"," ", doc_text)
    # remove multiple white spaces and trailing white spaces
    doc_text = re.sub(" +"," ",doc_text)
    doc_text = doc_text.strip()
    # convert words to lower case and split them
    words = doc_text.lower().split()
    # optionally remove stop words.
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return " ".join(words)


def prepare_data(GLOVE_DIR,TEXT_DATA_DIR,MAX_SEQUENCE_LENGTH,MAX_NB_WORDS
    ,EMBEDDING_DIM,VALIDATION_SPLIT,categorical=True):
    """mostly the same preprocessing as in the original post with a couple of differences.
    sklearn's CountVectorizer is used instead of keras tokenizer and the train/test split
    is done using sklearn's train_test_split().
    """
    # build index mapping words in the embeddings set to their embedding
    # vector
    print('Indexing word vectors.')
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    # second, prepare text samples and their labels
    print('Processing text dataset')
    texts = []         # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []        # list of label ids
    for name in sorted(os.listdir(TEXT_DATA_DIR)):
        path = os.path.join(TEXT_DATA_DIR, name)
        if os.path.isdir(path):
            label_id = len(labels_index)
            labels_index[name] = label_id
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    t = f.read()
                    i = t.find('\n\n')  # skip header
                    if 0 < i:
                        t = t[i:]
                    texts.append(t)
                    f.close()
                    labels.append(label_id)

    print('Found %s texts.' % len(texts))

    # vectorize the text samples into a 2D integer tensor. using sklearn
    # CountVectorizer instead of the keras tokenizer (as in the original post)
    # we obtain a slightly higher overlap between the selected words and the
    # words that have glove vectors. Ultimately this does not make a major
    # difference
    processed_texts = [clean_text(t) for t in texts]
    vectorizer = CountVectorizer(max_features=MAX_NB_WORDS)
    vectorizer_fit = vectorizer.fit_transform(processed_texts)

    # We index the words so the most common word ("the") has index 1. This is
    # irrelevant in reality I do it simply because I wanted to compare with
    # the dictionary obtained using keras.
    words  = vectorizer.get_feature_names()
    counts = vectorizer_fit.toarray().sum(axis=0)
    counts_words = list(zip(counts,words))
    counts_words.sort(reverse=True)

    vocabulary = [str(w[1]) for w in counts_words]
    word_index = dict(zip(vocabulary, range(MAX_NB_WORDS)))

    sequences = []
    for doc in processed_texts:
        sequence=[]
        for word in doc.split():
            if word not in word_index:
                continue
            sequence.append(word_index[word])
        sequences.append(sequence)

    data = np.vstack([pad_sequences(s,MAX_SEQUENCE_LENGTH) for s in sequences])
    labels = np.asarray(labels)

    # split the data into a training set and a validation set.
    x_train,x_val,y_train,y_val = train_test_split(
        data, labels, stratify=labels, test_size=VALIDATION_SPLIT)

    if categorical:
        y_train = one_hot(np.asarray(y_train))
        y_val = one_hot(np.asarray(y_val))

    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', y_train.shape)

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = MAX_NB_WORDS
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return x_train, y_train, x_val, y_val, embedding_matrix