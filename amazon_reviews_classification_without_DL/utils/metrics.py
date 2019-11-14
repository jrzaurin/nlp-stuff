import numpy as np
import lightgbm as lgb

from sklearn.metrics import f1_score
from scipy.misc import derivative


def sigmoid(x): return 1./(1. +  np.exp(-x))


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-6)


def focal_loss_lgb(y_pred, dtrain, alpha, gamma, num_class):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = dtrain.label
    # N observations x num_class arrays
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1,num_class, order='F')
    # alpha and gamma multiplicative factors with BCEWithLogitsLoss
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    # flatten in column-major (Fortran-style) order
    return grad.flatten('F'), hess.flatten('F')


def focal_loss_lgb_eval_error(y_pred, deval, alpha, gamma, num_class):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    deval: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    num_class: int
        number of classes
    """
    a,g = alpha, gamma
    y_true = deval.label
    y_true = np.eye(num_class)[y_true.astype('int')]
    y_pred = y_pred.reshape(-1, num_class, order='F')
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    # a variant can be np.sum(loss)/num_class
    return 'focal_loss', np.mean(loss), False


def lgb_f1_score(preds, lgbDataset, num_class):
	"""
	Implementation of the f1 score to be used as evaluation score for lightgbm
	Parameters:
	-----------
	preds: numpy.ndarray
		array with the predictions
	lgbDataset: lightgbm.Dataset
	"""
	preds = preds.reshape(-1, num_class, order='F')
	cat_preds = np.argmax(preds, axis=1)
	y_true = lgbDataset.get_label()
	return 'f1', f1_score(y_true, cat_preds, average='weighted'), True


def lgb_focal_f1_score(preds, lgbDataset, num_class):
	"""
	Adaptation of the implementation of the f1 score to be used as evaluation
	score for lightgbm. The adaptation is required since when using custom losses
	the row prediction needs to passed through a sigmoid to represent a
	probability
	Parameters:
	-----------
	preds: numpy.ndarray
		array with the predictions
	lgbDataset: lightgbm.Dataset
	"""
	preds = softmax(preds.reshape(-1, num_class, order='F'))
	cat_preds = np.argmax(preds, axis=1)
	y_true = lgbDataset.get_label()
	return 'f1', f1_score(y_true, cat_preds, average='weighted'), True