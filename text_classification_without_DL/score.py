import numpy as np
import pickle
import argparse
import pdb

from pathlib import Path
from utils.lightgbm_optimizer import LGBOptimizer

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--packg", type=str, default='nltk',
		help="python packaged used for pre-pressing: nltk or spacy")

	parser.add_argument("--with_bigrams", action='store_true',
		help="bigram phraser used during pre-pressing")
	parser.add_argument("--algo", type=str, default='tfidf',
		help="algorithm/method used to generate the features: tfidf, lda or ensemb.")
	parser.add_argument("--n_topics", type=str, default='',
		help="number of topics used for lda or ensemb methods: 20 (lda and ensemb) or 50 (only lda)")

	parser.add_argument("--with_cv", action='store_true',
		help="whether lightgbm will use CV when running within hyperopt.")
	parser.add_argument("--is_unbalance", action='store_false',
		help="boolean directly passed to the 'is_unbalance' param in lightgbm.")
	parser.add_argument("--with_focal_loss", action='store_true',
		help="Use the focal loss during hyperoptimization.")
	parser.add_argument("--eval_with_metric", action='store_true',
		help="Optimize against metric or loss.")
	parser.add_argument("--maxevals", type=int, default=100,
		help="number of iterations during hyperoptimization.")
	parser.add_argument("--save", action='store_true')

	args = parser.parse_args()

	FEAT_PATH = Path('features')

	wbigram = 'bigram_' if args.with_bigrams else ''
	dataname = args.packg + '_tok_reviews_' + wbigram + args.algo
	if args.algo is not 'tfidf': dataname = dataname + '_' +  args.n_topics

	dtrain = pickle.load(open(FEAT_PATH/('train/'+ dataname+'_feat_tr.p'), 'rb'))
	dvalid = pickle.load(open(FEAT_PATH/('valid/'+ dataname+'_feat_val.p'), 'rb'))
	dtest  = pickle.load(open(FEAT_PATH/('test/' + dataname+'_feat_te.p'),  'rb'))

	opt = LGBOptimizer(
		dataname,
		dtrain,
		dvalid,
		dtest,
		with_cv=args.with_cv,
		is_unbalance=args.is_unbalance,
		with_focal_loss=args.with_focal_loss,
		eval_with_metric=args.eval_with_metric,
		save=args.save)
	opt.optimize(maxevals=args.maxevals)