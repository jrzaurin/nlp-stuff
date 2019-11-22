import pickle

from utils.lightgbm_optimizer import LGBOptimizer
from pathlib import Path

if __name__ == '__main__':

	opt = LGBOptimizer(dataset='original', save=True)
	opt.optimize(maxevals=50)

	opt = LGBOptimizer(dataset='augmented', save=True)
	opt.optimize(maxevals=50)