import pandas as pd

from pathlib import Path


if __name__ == '__main__':

	DATA_PATH = Path('/home/ubuntu/projects/nlp-stuff/datasets/amazon_reviews')
	df_org = pd.read_json(DATA_PATH/'reviews_Clothing_Shoes_and_Jewelry_5.json.gz', lines=True)

	# LightGBM expect classes from [0,num_class)
	df = df_org.copy()
	df['overall'] = (df['overall']-1).astype('int64')

	# group reviews with 1 and 2 scores into one class
	df.loc[df.overall==0, 'overall'] = 1

	# and back again to [0,num_class)
	df['overall'] = (df['overall']-1).astype('int64')

	df.to_csv(DATA_PATH/'reviews_Clothing_Shoes_and_Jewelry.csv', index=False)
