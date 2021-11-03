"""
Code here is taken from my repo: https://github.com/jrzaurin/pytorch-widedeep/blob/master/pytorch_widedeep/metrics.py
which is in itself inspired by torchsample: https://github.com/ncullen93/torchsample
which is in itself inpired in many aspects by Keras.
"""

import numpy as np


class CategoricalAccuracy(object):
    def __init__(self, top_k=1):
        self.top_k = top_k
        self.correct_count = 0
        self.total_count = 0

        self._name = "acc"

    def reset(self):
        self.correct_count = 0
        self.total_count = 0

    def __call__(self, y_pred, y_true):
        top_k = y_pred.topk(self.top_k, 1)[1]
        true_k = y_true.view(len(y_true), 1).expand_as(top_k)
        self.correct_count += top_k.eq(true_k).float().sum().item()
        self.total_count += len(y_pred)
        accuracy = float(self.correct_count) / float(self.total_count)
        return np.round(accuracy, 4)
