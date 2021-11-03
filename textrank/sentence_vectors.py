import numpy as np
import pandas as pd
import pickle
import spacy
import os
import re

from multiprocessing import Pool
from gensim.utils import simple_preprocess
from pathlib import Path

n_cpus = os.cpu_count()
tok = spacy.blank('en', disable=["parser","tagger","ner"])


def normalize_sents(sents):
	nsents = []
	for s in sents: nsents.append(' '.join([t.norm_ for t in tok.tokenizer(s)]))
	return nsents


def rm_non_alpha(sents):
	return [re.sub("[^a-zA-Z]", " ", s).strip() for s in sents]


def rm_single_chars(sents):
	return [s for s in sents if len(s)>1]


def rm_useless_spaces(sents):
    return [re.sub(' {2,}', ' ', s) for s in sents]


def parallel_apply(df, func, n_cores=n_cpus):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    return df


def process_sents(df, coln='processed_sents'):
	df[coln] = df['review_sents'].apply(lambda x: normalize_sents(x))
	df[coln] = df[coln].apply(lambda x: rm_non_alpha(x))
	df[coln] = df[coln].apply(lambda x: rm_single_chars(x))
	df[coln] = df[coln].apply(lambda x: rm_useless_spaces(x))
	return df


def word_vectors(path, fname):
	embeddings_index = {}
	f = open(path/fname)
	for line in f:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float32')
	    embeddings_index[word] = coefs
	f.close()
	return embeddings_index


def sentence_vector(sent, embeddings_index, dim=100):
	if len(sent)>0:
		return sum([embeddings_index.get(w, np.zeros((dim,))) for w in sent])/len(sent)
	else:
		return np.zeros((dim,))


if __name__ == '__main__':

	DATA_PATH = Path('data')
	WORDVEC_PATH = DATA_PATH/'glove.6B'
	wordvec_fname= 'glove.6B.100d.txt'
	embeddings_index = word_vectors(WORDVEC_PATH, wordvec_fname)

	df = pickle.load(open(DATA_PATH/'df_reviews_text.p', 'rb'))
	df = parallel_apply(df, process_sents)
	df.to_pickle(DATA_PATH/'df_processed_reviews.p')

	all_sents = [s for sents in df.processed_sents for s in sents]
	idx2sent = {k:v for k,v in enumerate(all_sents)}
	pickle.dump(idx2sent, open(DATA_PATH/'idx2sent.p', 'wb'))

	idx2toksents = {}
	for i,s in idx2sent.items(): idx2toksents[i] = simple_preprocess(s)
	sent2vec = {}
	for i,s in idx2toksents.items(): sent2vec[i] = sentence_vector(s, embeddings_index)
	pickle.dump(sent2vec, open(DATA_PATH/'sent2vec.p', 'wb'))
