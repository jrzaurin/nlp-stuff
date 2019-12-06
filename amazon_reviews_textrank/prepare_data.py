import pandas as pd
import gzip
import os
import spacy

from pathlib import Path

n_cpus = os.cpu_count()


if __name__ == '__main__':

	DATA_PATH  = Path('../datasets/amazon_reviews/')
	OUT_PATH   = Path('data/')

	nlp = spacy.load("en_core_web_sm")
	MIN_TOKENS = 5
	MIN_SENTS  = 1

	if not os.path.exists(OUT_PATH): os.makedirs(OUT_PATH)
	df_org = pd.read_json(DATA_PATH/'reviews_Clothing_Shoes_and_Jewelry_5.json.gz', lines=True)
	df_summary = df_org[['reviewText', 'summary']]
	df_summary = df_summary[~df_summary.reviewText.isna()]

	# Keep reviews with more than MIN_TOKENS 'tokens'
	keep_review_1 = [len(r.split(' '))>MIN_TOKENS for r in df_summary.reviewText]
	df_summary = df_summary.loc[keep_review_1].reset_index(drop=True)

	# Keep reviews with more than MIN_SENTS sentences
	review_sents = []
	for doc in nlp.pipe(df_summary.reviewText.tolist(), n_process=n_cpus, batch_size=1000):
		# to str so it can be pickled
		sents = [str(s) for s in list(doc.sents)]
		review_sents.append(sents)
	df_summary['review_sents'] = review_sents
	# let's save here since sentence tokenization takes some time
	df_summary.to_pickle(OUT_PATH/'df_reviews_text.p')
	keep_review_2 = [len(rs)>MIN_SENTS for rs in review_sents]
	df_summary = df_summary.loc[keep_review_2].reset_index(drop=True)
	# final save
	df_summary.to_pickle(OUT_PATH/'df_reviews_text.p')
