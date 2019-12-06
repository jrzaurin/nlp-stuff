import pickle

from pathlib import Path
from summa import summarizer
from summarize import Summarizer


if __name__ == '__main__':

	DATA_PATH = Path('data')
	reviewtext = pd.read_pickle(DATA_PATH/'df_processed_reviews.p')
	idx2sent = pickle.load(open(DATA_PATH/'idx2sent.p', 'rb'))
	sent2vec = pickle.load(open(DATA_PATH/'sent2vec.p', 'rb'))

	idx = 198
	my_summarizer = Summarizer(reviewtext, idx2sent, sent2vec)
	review, my_summary = my_summarizer.summarize(idx, n=100)
	summary = summarizer.summarize(review, ratio=0.2)