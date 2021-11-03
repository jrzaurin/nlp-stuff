import numpy as np
import os
import numpy as np
import pandas as pd
import multiprocessing
import en_core_web_sm
import pickle
import spacy

from pathlib import Path
from multiprocessing import Pool
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phraser, Phrases
from nltk.stem import WordNetLemmatizer
from spacy.lang.en.stop_words import STOP_WORDS


cores = multiprocessing.cpu_count()

def simple_tokenizer(doc):
	return [t for t in simple_preprocess(doc, min_len=2) if t not in STOP_WORDS]


class NLTKLemmaTokenizer(object):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.lemmatizer.lemmatize(t, pos="v") for t in simple_tokenizer(doc)]


class SpacyLemmaTokenizer(object):

	def __init__(self):
		self.tok = spacy.blank('en', disable=["parser","tagger","ner"])

	@staticmethod
	def condition(t, min_len=2):
		return not (t.is_punct | t.is_space | (t.lemma_ != '-PRON-') | len(t)<=min_len |
			t.is_stop |  t.is_digit)

	def __call__(self, doc):
		return [t.lemma_.lower() for t in self.tok(doc) if self.condition(t)]


class Bigram(object):

	def __init__(self):
	    self.phraser = Phraser

	@staticmethod
	def append_bigram(doc, phrases_model):
		doc += [t for t in phrases_model[doc] if '_' in t]
		return doc

	def __call__(self, docs):
		phrases = Phrases(docs,min_count=10)
		bigram = self.phraser(phrases)
		p = Pool(cores)
		docs = p.starmap(self.append_bigram, zip(docs, [bigram]*len(docs)))
		pool.close()
		return docs


def count_nouns(tokens):
	return sum([t.pos_ is 'NOUN' for t in tokens])/len(tokens)


def count_adjectives(tokens):
	return sum([t.pos_ is 'ADJ' for t in tokens])/len(tokens)


def count_adverbs(tokens):
	return sum([t.pos_ is 'ADV' for t in tokens])/len(tokens)


def count_verbs(tokens):
	return sum([t.pos_ is 'VERB' for t in tokens])/len(tokens)


def sentence_metric(tokens):
	slen = [len(s) for s in tokens.sents]
	metrics = np.array([np.mean(slen), np.median(slen), np.min(slen), np.max(slen)])/len(tokens)
	return metrics


def xtra_features(doc):
	tokens = nlp(doc)
	n_nouns = count_nouns(tokens)
	n_adj   = count_adjectives(tokens)
	n_adv   = count_adverbs(tokens)
	n_verb  = count_verbs(tokens)
	sent_m  = sentence_metric(tokens)
	return [n_nouns, n_adj, n_adv, n_verb] + list(sent_m)


if __name__ == '__main__':

	DATA_PATH = Path("../datasets/amazon_reviews")
	OUT_PATH  = Path("data")
	if not os.path.exists(OUT_PATH): os.makedirs(OUT_PATH)

	df = pd.read_csv(DATA_PATH/'reviews_Clothing_Shoes_and_Jewelry.csv')
	df = df[~df.reviewText.isna()].sample(frac=1, random_state=1).reset_index(drop=True)
	reviews = df.reviewText.tolist()

	nltk_tok  = NLTKLemmaTokenizer()
	spacy_tok = SpacyLemmaTokenizer()

	pool = Pool(cores)
	nltk_docs  = pool.map(nltk_tok, reviews)
	spacy_docs = pool.map(spacy_tok, reviews)
	pool.close()

	nltk_tok_reviews  = pd.DataFrame({'tokenized_reviews': nltk_docs,  'score': df.overall})
	spacy_tok_reviews = pd.DataFrame({'tokenized_reviews': spacy_docs, 'score': df.overall})
	reviews_len   = [len(r) for r in nltk_tok_reviews.tokenized_reviews]
	drop_index    = np.where([l==0 for l in reviews_len])[0]

	pickle.dump(nltk_tok_reviews,  open(OUT_PATH/'nltk_tok_reviews.p', 'wb'))
	pickle.dump(spacy_tok_reviews, open(OUT_PATH/'spacy_tok_reviews.p', 'wb'))
	pickle.dump(drop_index,    open(OUT_PATH/'drop_index.p', 'wb'))

	nltk_pdocs  = Bigram()(nltk_docs)
	spacy_pdocs = Bigram()(spacy_docs)

	nltk_tok_reviews_bigram  = pd.DataFrame({'tokenized_reviews': nltk_pdocs,  'score': df.overall})
	spacy_tok_reviews_bigram = pd.DataFrame({'tokenized_reviews': spacy_pdocs, 'score': df.overall})

	pickle.dump(nltk_tok_reviews_bigram, open(OUT_PATH/'nltk_tok_reviews_bigram.p', 'wb'))
	pickle.dump(spacy_tok_reviews_bigram, open(OUT_PATH/'spacy_tok_reviews_bigram.p', 'wb'))

	nlp = spacy.load('en_core_web_sm')
	pool = Pool(cores)
	xf = pool.map(xtra_features, reviews)
	pool.close()

	xf_df = pd.DataFrame(np.array(xf))
	xf_df.columns = ['n_nouns', 'n_adj', 'n_adv', 'n_verb', 'sent_len_mean', 'sent_len_median',
		'sent_len_min', 'sent_len_max']
	xf_df['score'] = df.overall
	pickle.dump(xf_df, open(OUT_PATH/'extra_features.p','wb'))