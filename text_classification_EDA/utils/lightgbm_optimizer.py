import numpy as np
import pandas as pd
import lightgbm as lgb
import pdb
import os
import pickle
import warnings

from pathlib import Path
from time import time
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
	recall_score, confusion_matrix, log_loss)
from sklearn.utils import Bunch
from hyperopt import hp, tpe, fmin, Trials
from fastai.text import Tokenizer

warnings.filterwarnings("ignore")


class LGBOptimizer(object):

	def __init__(self, dataset='original', out_dir=Path('results'), save=False):

		self.out_dir = out_dir
		self.dataset = dataset
		self.save = save
		self.early_stop_dict = {}

	def load_data(self):

		if self.dataset is 'original':
			train = pickle.load(open('features/train/ftrain.p', 'rb'))
			valid = pickle.load(open('features/valid/fvalid.p', 'rb'))
			self.lgtrain = lgb.Dataset(train.X.astype('float'), train.y, free_raw_data=False)
			self.lgvalid = self.lgtrain.create_valid(valid.X.astype('float'), valid.y)
			self.num_class = len(np.unique(train.y))

			full_train = pickle.load(open('features/train/fftrain.p', 'rb'))
			test = pickle.load(open('features/test/ftest.p', 'rb'))
			self.lgftrain = lgb.Dataset(full_train.X.astype('float'), full_train.y, free_raw_data=False)
			self.lgtest = self.lgftrain.create_valid(test.X.astype('float'), test.y)

		elif self.dataset is 'augmented':
			train = pickle.load(open('features/train/a_ftrain.p', 'rb'))
			valid = pickle.load(open('features/valid/a_fvalid.p', 'rb'))
			self.lgtrain = lgb.Dataset(train.X.astype('float'), train.y, free_raw_data=False)
			self.lgvalid = self.lgtrain.create_valid(valid.X.astype('float'), valid.y)
			self.num_class = len(np.unique(train.y))

			full_train = pickle.load(open('features/train/a_fftrain.p', 'rb'))
			test = pickle.load(open('features/test/a_ftest.p', 'rb'))
			self.lgftrain = lgb.Dataset(full_train.X.astype('float'), full_train.y, free_raw_data=False)
			self.lgtest = self.lgftrain.create_valid(test.X.astype('float'), test.y)

	def optimize(self, maxevals=200):

		self.load_data()
		param_space = self.hyperparameter_space()
		objective = self.get_objective(self.lgtrain, self.lgvalid)
		objective.i=0

		start = time()
		trials = Trials()
		best = fmin(fn=objective,
		            space=param_space,
		            algo=tpe.suggest,
		            max_evals=maxevals,
		            trials=trials)
		best['num_boost_round'] = self.early_stop_dict[trials.best_trial['tid']]
		best['num_leaves'] = int(best['num_leaves'])
		best['verbose'] = -1
		best['num_class'] = self.num_class
		best['objective'] = 'multiclass'

		model = lgb.train(best, self.lgftrain)
		preds = model.predict(self.lgtest.data)
		preds = np.argmax(preds, axis=1).astype('int')

		acc  = accuracy_score(self.lgtest.label, preds)
		f1   = f1_score(self.lgtest.label, preds, average='weighted')
		prec = precision_score(self.lgtest.label, preds, average='weighted')
		rec  = recall_score(self.lgtest.label, preds, average='weighted')
		cm   = confusion_matrix(self.lgtest.label, preds)

		print('acc: {:.4f}, f1 score: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(
			acc, f1, prec, rec))
		print('confusion_matrix')
		print(cm)
		running_time = round((time()-start)/maxevals, 2)

		if self.save:
			if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)
			results = Bunch(acc=acc, f1=f1, prec=prec, rec=rec, cm=cm)
			out_fname = self.dataset + '_results'
			out_fname += '.p'
			results.model = model
			results.best_params = best
			results.running_time = running_time
			pickle.dump(results, open(self.out_dir/out_fname, 'wb'))

		self.best = best
		self.model = model

	def get_objective(self, dtrain, deval):

		def objective(params):

			# hyperopt casts as float
			params['num_boost_round'] = int(params['num_boost_round'])
			params['num_leaves'] = int(params['num_leaves'])
			params['verbose'] = -1
			params['seed'] = 1

			# number of classes for mutli-class classification
			params['objective'] = 'multiclass'
			params['num_class'] = self.num_class
			params['is_unbalance'] = True

			model = lgb.train(
				params,
				dtrain,
				num_boost_round=params['num_boost_round'],
				valid_sets=[deval],
				early_stopping_rounds=10,
				verbose_eval=False)

			preds = model.predict(deval.data)
			score = log_loss(deval.label, preds)
			self.early_stop_dict[objective.i] = model.best_iteration
			objective.i+=1

			return score

		return objective

	def hyperparameter_space(self, param_space=None):

		space = {
			'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
			'num_boost_round': hp.quniform('num_boost_round', 50, 500, 20),
			'num_leaves': hp.quniform('num_leaves', 31, 255, 4),
		    'min_child_weight': hp.uniform('min_child_weight', 0.1, 10),
		    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.),
		    'subsample': hp.uniform('subsample', 0.5, 1.),
		    'reg_alpha': hp.uniform('reg_alpha', 0.01, 0.1),
		    'reg_lambda': hp.uniform('reg_lambda', 0.01, 0.1),
		}
		if param_space:
			return param_space
		else:
			return space