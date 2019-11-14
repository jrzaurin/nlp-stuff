import numpy as np
import pandas as pd
import lightgbm as lgb
import pdb
import os
import pickle
import warnings

from pathlib import Path
from time import time
from .metrics import (focal_loss_lgb, focal_loss_lgb_eval_error, lgb_f1_score,
	lgb_focal_f1_score, softmax)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
	recall_score, confusion_matrix, log_loss)
from sklearn.utils import Bunch
from hyperopt import hp, tpe, fmin, Trials
from scipy import sparse

warnings.filterwarnings("ignore")


class LGBOptimizer(object):

	def __init__(self, dataname, train_set, valid_set, test_set, out_dir=Path('results'),
		with_cv=False, is_unbalance=False, with_focal_loss=False, eval_with_metric=False,
		save=False):

		self.PATH = out_dir
		self.dataname = dataname
		self.with_cv = with_cv
		self.is_unbalance = is_unbalance
		self.with_focal_loss = with_focal_loss
		self.eval_with_metric = eval_with_metric
		self.save = save
		self.early_stop_dict = {}
		self.num_class = len(np.unique(train_set.y))

		if 'tfidf' in dataname:	X_tr = sparse.vstack([train_set.X, valid_set.X])
		else: X_tr = np.vstack([train_set.X, valid_set.X])
		y_tr = np.hstack([train_set.y, valid_set.y])
		self.cvlgtrain = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
		if with_cv:
			self.lgtest = self.cvlgtrain.create_valid(test_set.X, test_set.y)
		else:
			self.lgtrain = lgb.Dataset(train_set.X, train_set.y, free_raw_data=False)
			self.lgvalid = self.lgtrain.create_valid(valid_set.X, valid_set.y)
			self.lgtest  = self.lgtrain.create_valid(test_set.X, test_set.y)

	def optimize(self, maxevals=200):

		param_space = self.hyperparameter_space()

		if self.with_cv: objective = self.get_objective_cv(self.cvlgtrain)
		else: objective = self.get_objective(self.lgtrain, self.lgvalid)
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

		if self.with_focal_loss:
			focal_loss = lambda x,y: focal_loss_lgb(x, y, best['alpha'], best['gamma'], self.num_class)
			model = lgb.train(best, self.cvlgtrain, fobj=focal_loss)
			preds = model.predict(self.lgtest.data)
			preds = np.argmax(softmax(preds), axis=1).astype('int')
		else:
			best['objective'] = 'multiclass'
			model = lgb.train(best, self.cvlgtrain)
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
			if not os.path.exists(self.PATH): os.makedirs(self.PATH)
			results = Bunch(acc=acc, f1=f1, prec=prec, rec=rec, cm=cm)
			out_fname = self.dataname + '_results'
			if self.is_unbalance:
				out_fname += '_unb'
			if self.with_focal_loss:
				out_fname += '_fl'
			out_fname += '.p'
			results.model = model
			results.best_params = best
			results.running_time = running_time
			pickle.dump(results, open(self.PATH/out_fname, 'wb'))

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

			if self.with_focal_loss:
				focal_loss = lambda x,y: focal_loss_lgb(x, y, params['alpha'], params['gamma'],
					self.num_class)
				focal_loss_eval = lambda x,y: focal_loss_lgb_eval_error(x, y, params['alpha'],
					params['gamma'], self.num_class)
				lgb_fl_f1 = lambda x,y: lgb_focal_f1_score(x,y, self.num_class)
				model = lgb.train(
					params,
					dtrain,
					num_boost_round=params['num_boost_round'],
					valid_sets=[deval],
					fobj  = focal_loss,
					feval = lgb_fl_f1 if self.eval_with_metric else focal_loss_eval,
					early_stopping_rounds=10,
					verbose_eval=False)
				if self.eval_with_metric:
					preds = np.argmax(softmax(model.predict(deval.data)), axis=1)
					score = f1_score(deval.label, preds, average='weighted')
				else:
					preds = model.predict(deval.data).ravel('F')
					score = focal_loss_eval(preds, deval)
			else:
				if self.is_unbalance: params['is_unbalance'] = True
				lgb_f1 = lambda x,y: lgb_f1_score(x,y, self.num_class)
				model = lgb.train(
					params,
					dtrain,
					num_boost_round=params['num_boost_round'],
					valid_sets=[deval],
					feval = lgb_f1 if self.eval_with_metric else None,
					early_stopping_rounds=10,
					verbose_eval=False)
				if self.eval_with_metric:
					preds = np.argmax(model.predict(deval.data), axis=1)
					score = f1_score(deval.label, preds, average='weighted')
				else:
					preds = model.predict(deval.data)
					score = log_loss(deval.label, preds)
			self.early_stop_dict[objective.i] = model.best_iteration
			objective.i+=1

			if self.eval_with_metric:
				return -score
			else:
				return score

		return objective


	def get_objective_cv(self, dtrain):

		def objective(params):
			# hyperopt casts as float
			params['num_boost_round'] = int(params['num_boost_round'])
			params['num_leaves'] = int(params['num_leaves'])
			params['verbose'] = -1
			params['seed'] = 1

			# number of classes for mutli-class classification
			params['objective'] = 'multiclass'
			params['num_class'] = self.num_class

			if self.with_focal_loss:
				focal_loss = lambda x,y: focal_loss_lgb(x, y, params['alpha'], params['gamma'],
					self.num_class)
				focal_loss_eval = lambda x,y: focal_loss_lgb_eval_error(x, y, params['alpha'],
					params['gamma'], self.num_class)
				lgb_fl_f1 = lambda x,y: lgb_focal_f1_score(x,y, self.num_class)
				cv_result = lgb.cv(
					params,
					dtrain,
					num_boost_round=params['num_boost_round'],
					fobj = focal_loss,
					feval = lgb_fl_f1 if self.eval_with_metric else focal_loss_eval,
					nfold=3,
					stratified=True,
					early_stopping_rounds=20)
			else:
				if self.is_unbalance: params['is_unbalance'] = True
				lgb_f1 = lambda x,y: lgb_f1_score(x,y, self.num_class)
				cv_result = lgb.cv(
					params,
					dtrain,
					num_boost_round=params['num_boost_round'],
					metrics='multi_logloss',
					feval = lgb_f1 if self.eval_with_metric else None,
					nfold=3,
					stratified=True,
					early_stopping_rounds=20)

			if self.eval_with_metric: result_key = 'f1-mean'
			elif self.with_focal_loss: result_key = 'focal_loss-mean'
			else: result_key = 'multi_logloss-mean'
			self.early_stop_dict[objective.i] = len(cv_result[result_key])
			score = round(cv_result[result_key][-1], 4)
			objective.i+=1

			if self.eval_with_metric:
				return -score
			else:
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
		if self.with_focal_loss:
			space['alpha'] = hp.uniform('alpha', 0.1, 0.75)
			space['gamma'] = hp.uniform('gamma', 0.5, 5)
		if param_space:
			return param_space
		else:
			return space