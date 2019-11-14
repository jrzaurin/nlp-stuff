import pandas as pd
import os
import pickle

from pathlib import Path
from sklearn.utils import Bunch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


drop_idx = pickle.load(open("data/drop_index.p", "rb"))


def train_test_split_(dataset, root='data', train_dir='train',	valid_dir='valid',
	test_dir='test'):

	print('splitting {}'.format(dataset))

	ROOT = Path(root)
	TRAIN_PATH = Path('/'.join([root,train_dir]))
	VALID_PATH = Path('/'.join([root,valid_dir]))
	TEST_PATH  = Path('/'.join([root,test_dir]))

	if not os.path.exists(TRAIN_PATH): 	os.makedirs(TRAIN_PATH)
	if not os.path.exists(VALID_PATH): 	os.makedirs(VALID_PATH)
	if not os.path.exists(TEST_PATH): 	os.makedirs(TEST_PATH)

	df = pickle.load(open(ROOT/dataset, 'rb'))
	df.drop(df.index[drop_idx], inplace=True)
	y = df.score
	df.drop('score', axis=1, inplace=True)
	if 'extra' in dataset:
		scaler = StandardScaler()
		df = pd.DataFrame(scaler.fit_transform(df))
		pickle.dump(scaler, open(TRAIN_PATH/'extra_feat_scaler.p', 'wb'))

	# 80% train
	X_tr, X_val, y_tr, y_val = train_test_split(df, y, train_size=0.8, random_state=1,
		stratify=y)

	# 10% valid, 10% testing
	X_val, X_te, y_val, y_te = train_test_split(X_val, y_val, train_size=0.5, random_state=1,
		stratify=y_val)

	if 'extra' in dataset:
		X_tr, X_val, X_te = X_tr.values, X_val.values,  X_te.values
	else:
		X_tr, X_val, X_te = X_tr.tokenized_reviews.tolist(), X_val.tokenized_reviews.tolist(),\
		X_te.tokenized_reviews.tolist()

	dtrain = Bunch(X=X_tr,  y=y_tr.values)
	dvalid = Bunch(X=X_val, y=y_val.values)
	dtest  = Bunch(X=X_te,  y=y_te.values)
	fname_tr  = dataset.split('.')[0] + '_tr.p'
	fname_val = dataset.split('.')[0] + '_val.p'
	fname_te  = dataset.split('.')[0] + '_te.p'
	pickle.dump(dtrain, open(TRAIN_PATH/fname_tr, 'wb'))
	pickle.dump(dvalid, open(VALID_PATH/fname_val, 'wb'))
	pickle.dump(dtest,  open(TEST_PATH/fname_te, 'wb'))


if __name__ == '__main__':

	DATA_PATH = 'data'
	datasets = [f for f in os.listdir(DATA_PATH) if f.endswith('.p') and 'drop' not in f]
	for d in datasets: train_test_split_(d)