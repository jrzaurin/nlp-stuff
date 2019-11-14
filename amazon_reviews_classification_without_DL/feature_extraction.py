import pdb
import pandas as pd
import os
import pickle
import warnings

from pathlib import Path
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import Bunch
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.utils.validation import check_is_fitted
from enstop import EnsembleTopics


warnings.filterwarnings("ignore")
cores = os.cpu_count()


class FeatureExtraction(object):
	def __init__(self, algo, n_topics=None, max_vocab_size=50000):
		super(FeatureExtraction, self).__init__()

		if algo is 'tfidf':
			vectorizer = TfidfVectorizer(max_features=max_vocab_size, preprocessor = lambda x: x,
				tokenizer = lambda x: x)
			self.fe = Pipeline([('vectorizer', vectorizer)])
		else:
			assert n_topics is not None
			vectorizer = CountVectorizer(max_features=max_vocab_size, preprocessor = lambda x: x,
				tokenizer = lambda x: x)
			if algo is 'lda':
				model = LDA(n_components=n_topics, n_jobs=-1, random_state=0)
			elif algo is 'ensemb':
				model = EnsembleTopics(n_components=n_topics, n_jobs=cores, random_state=0)
			self.fe = Pipeline([('vectorizer', vectorizer), ('model', model)])

	def fit(self, X):
		self.fe.fit(X)
		return self

	def transform(self, X):
		out = self.fe.transform(X)
		return out

	def fit_transform(self,X):
		return self.fit(X).transform(X)


def extract_features(dataset, algo, n_topics=None, max_vocab_size=50000, root='data',
	train_dir='train', valid_dir='valid', test_dir='test', save_dir='features'):

	if n_topics is None:
		print('Extracting Feature for {} using {}'.format(dataset, algo))
	else:
		print('Extracting Feature for {} using {} with {} components'.format(dataset, algo, n_topics))

	TRAIN_PATH = Path('/'.join([root, train_dir]))
	VALID_PATH = Path('/'.join([root, valid_dir]))
	TEST_PATH  = Path('/'.join([root, test_dir]))

	TRAIN_PATH_OUT = Path('/'.join([save_dir, train_dir]))
	VALID_PATH_OUT = Path('/'.join([save_dir, valid_dir]))
	TEST_PATH_OUT  = Path('/'.join([save_dir, test_dir]))
	if not os.path.exists(TRAIN_PATH_OUT):
		os.makedirs(TRAIN_PATH_OUT)
		os.makedirs(VALID_PATH_OUT)
		os.makedirs(TEST_PATH_OUT)

	dtrain = pickle.load(open(TRAIN_PATH/(dataset+'_tr.p'), 'rb'))
	dvalid = pickle.load(open(VALID_PATH/(dataset+'_val.p'), 'rb'))
	dtest  = pickle.load(open(TEST_PATH/(dataset+'_te.p'), 'rb'))

	feature_extractor = FeatureExtraction(algo, n_topics, max_vocab_size)
	X_tr  = feature_extractor.fit_transform(dtrain.X)
	X_val = feature_extractor.transform(dvalid.X)
	X_te  = feature_extractor.transform(dtest.X)

	train_feat = Bunch(X=X_tr, y=dtrain.y)
	valid_feat = Bunch(X=X_val, y=dvalid.y)
	test_feat  = Bunch(X=X_te, y=dtest.y)

	rootname = dataset.replace('_df', '').split('.p')[0]
	if algo is 'tfidf':
		tr_fname  = dataset + '_'  + algo + '_feat_tr.p'
		val_fname = dataset + '_'  + algo + '_feat_val.p'
		te_fname  = dataset + '_'  + algo + '_feat_te.p'
	else:
		tr_fname  = dataset + '_'  + algo + '_' + str(n_topics) + '_feat_tr.p'
		val_fname = dataset + '_'  + algo + '_' + str(n_topics) + '_feat_val.p'
		te_fname  = dataset + '_'  + algo + '_' + str(n_topics) + '_feat_te.p'

	pickle.dump(train_feat, open(TRAIN_PATH_OUT/tr_fname, 'wb'))
	pickle.dump(valid_feat, open(VALID_PATH_OUT/val_fname, 'wb'))
	pickle.dump(test_feat,  open(TEST_PATH_OUT/te_fname, 'wb'))


if __name__ == '__main__':

	DATA_PATH = 'data'
	datasets = [f.split('.p')[0] for f in os.listdir(DATA_PATH) if "nltk" in f or "spacy" in f]
	methods  = [('tfidf', None), ('lda', 20), ('lda', 50), ('ensemb', 20)]
	set_ups  = [(d, m) for d in datasets for m in methods]

	for d,m in set_ups:
		a,n = m[0],m[1]
		extract_features(d, a, n, max_vocab_size=20000)