import pandas as pd
import argparse

from pathlib import Path
from eda import EDA

if __name__ == '__main__':

	DATA_PATH   = Path('data/')
	TRAIN_PATH = DATA_PATH/'train'

	train = pd.read_csv(TRAIN_PATH/'train.csv')
	train = list(train.itertuples(index=False, name=None))
	eda = EDA(alpha_sr=0.2, num_aug=1, rs=False)
	eda.augment(train, out_file=TRAIN_PATH/'a_train.csv')
