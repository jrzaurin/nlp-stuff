import pandas as pd
import os

from pathlib import Path
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

	DATA_PATH  = Path('../datasets/amazon_reviews/')

	OUT_PATH   = Path('data/')
	TRAIN_PATH = OUT_PATH/'train'
	VALID_PATH = OUT_PATH/'valid'
	TEST_PATH  = OUT_PATH/'test'

	paths = [OUT_PATH, TRAIN_PATH, VALID_PATH, TEST_PATH]
	for p in paths:
		if not os.path.exists(p): os.makedirs(p)

	# DATA_PATH = Path('/home/ubuntu/projects/nlp-stuff/datasets/amazon_reviews')
	df_org = pd.read_json(DATA_PATH/'reviews_Clothing_Shoes_and_Jewelry_5.json.gz', lines=True)

	# LightGBM expect classes from [0,num_class)
	df = df_org.copy()
	df['overall'] = (df['overall']-1).astype('int64')

	# group reviews with 1 and 2 scores into one class
	df.loc[df.overall==0, 'overall'] = 1

	# and back again to [0,num_class)
	df['overall'] = (df['overall']-1).astype('int64')

	# agressive preprocessing: drop short reviews
	df['reviewLength'] = df.reviewText.apply(lambda x: len(x.split(' ')))
	df = df[df.reviewLength >= 5]
	df = df.drop('reviewLength', axis=1).reset_index()

	df.to_csv(OUT_PATH/'reviews_Clothing_Shoes_and_Jewelry.csv', index=False)

	train, valid = train_test_split(df, train_size=0.8, random_state=1, stratify=df.overall)
	valid, test  = train_test_split(valid, train_size=0.5, random_state=1, stratify=valid.overall)
	train[['reviewText', 'overall']].to_csv(TRAIN_PATH/'train.csv', index=False)
	valid[['reviewText', 'overall']].to_csv(VALID_PATH/'valid.csv', index=False)
	test[['reviewText', 'overall']].to_csv(TEST_PATH/'test.csv', index=False)