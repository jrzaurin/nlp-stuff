import pandas as pd
import numpy as np
import os
import pickle

from pathlib import Path
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from fastai.text import Tokenizer
from eda import EDA

tok = Tokenizer()

def extract_features(df, out_path, max_features=30000, vectorizer=None):

	reviews = df['reviewText'].tolist()
	tok_reviews = tok.process_all(reviews)

	if vectorizer is None:
		vectorizer = TfidfVectorizer(max_features=max_features, preprocessor=lambda x: x,
			tokenizer = lambda x: x, min_df=5)
		X = vectorizer.fit_transform(tok_reviews)
	else:
		X = vectorizer.transform(tok_reviews)

	featset = Bunch(X=X, y=df.overall.values)
	pickle.dump(featset, open(out_path, 'wb'))

	return vectorizer


if __name__ == '__main__':

	DATA_PATH  = Path('data')
	FEAT_PATH  = Path('features')
	FTRAIN_PATH = FEAT_PATH/'train'
	FVALID_PATH = FEAT_PATH/'valid'
	FTEST_PATH  = FEAT_PATH/'test'

	paths = [FEAT_PATH, FTRAIN_PATH, FVALID_PATH, FTEST_PATH]
	for p in paths:
		if not os.path.exists(p): os.makedirs(p)

	# ORIGINAL
	train = pd.read_csv(DATA_PATH/'train/train.csv')
	valid = pd.read_csv(DATA_PATH/'valid/valid.csv')
	test  = pd.read_csv(DATA_PATH/'test/test.csv')

	# we will tune parameters with the 80% train and 10% validation
	print("building train/valid features for original dataset")
	vec = extract_features(train, out_path=FTRAIN_PATH/'ftrain.p')
	_ = extract_features(valid, vectorizer=vec, out_path=FVALID_PATH/'fvalid.p')

	# once we have tuned parameters we will train on 'train+valid' (90%) and test
	# on 'test' (10%)
	print("building train/test features for original dataset")
	full_train = pd.concat([train, valid]).sample(frac=1).reset_index(drop=True)
	fvec = extract_features(full_train, out_path=FTRAIN_PATH/'fftrain.p')
	_ = extract_features(test, vectorizer=fvec, out_path=FTEST_PATH/'ftest.p')
	del (train, vec, fvec)

	# AUGMENTED
	# Only the training set "at the time" must be augmented
	a_train = (pd.read_csv(DATA_PATH/'train/a_train.csv', engine='python', sep="::",
		names=['overall', 'reviewText'])
		.sample(frac=1)
		.reset_index(drop=True))

	# we will tune parameters with the 80% train and 10% validation. At this
	# stage, the validation set should not be augmented, but we need to compute
	# the validation features with the "augmented vectorizer"
	print("building train/valid features for augmented dataset")
	a_vec = extract_features(a_train, out_path=FTRAIN_PATH/'a_ftrain.p')
	_ = extract_features(valid, vectorizer=a_vec, out_path=FVALID_PATH/'a_fvalid.p')

	# once we have tuned parameters we will:
	# 1-augment 'train+valid'
	# 2-train the vectorizer on augmented dataset
	# 3-use the augmented vectorizer on 'test'
	print("building augmented dataset for train+valid")
	eda = EDA(rs=False)
	full_train = list(full_train[['reviewText', 'overall']].itertuples(index=False, name=None))
	eda.augment(full_train, out_file=DATA_PATH/'train/a_full_train.csv')
	del (valid, full_train)

	print("building train/test features for augmented dataset")
	a_full_train = (pd.read_csv(DATA_PATH/'train/a_full_train.csv', engine='python', sep="::",
		names=['overall', 'reviewText'])
		.sample(frac=1)
		.reset_index(drop=True))
	a_fvec = extract_features(a_full_train, out_path=FTRAIN_PATH/'a_fftrain.p')
	del a_full_train
	_ = extract_features(test, vectorizer=a_fvec, out_path=FTEST_PATH/'a_ftest.p')

