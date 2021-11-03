from __future__ import print_function
import os
import sys
import numpy as np
import bcolz
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes


def pad_sequences(seq, maxlen):
    if len(seq) >= maxlen:
        return np.array(seq[-maxlen:]).astype('int32')
    else:
        return np.pad(seq, (maxlen - len(seq)%maxlen, 0), 'constant').astype('int32')


def one_hot(x):
    return np.array(OneHotEncoder().fit_transform(x.reshape(-1,1)).todense())


def save_array(fname, arr):
    carr=bcolz.carray(arr, rootdir=fname, mode='w'); carr.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


class get_batch(object):
    """This class is a "stripped-out" version of the DataSet class within the
    tensorflow mnist module, used to generate batches"""

    def __init__(self,documents,labels,dtype=dtypes.float32,seed=None):

        seed1, seed2 = random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)

        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
          raise TypeError('Invalid dtype %r, expected uint8 or float32' % dtype)

        assert documents.shape[0] == labels.shape[0], (
            'documents.shape: %s labels.shape: %s' % (documents.shape, labels.shape))

        self._num_examples = documents.shape[0]
        self._documents = documents
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def documents(self):
        return self._documents

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch

        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:

            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._documents = self.documents[perm0]
            self._labels = self.labels[perm0]

        # Go to the next epoch
        if start + batch_size > self._num_examples:

            # Finished epoch
            self._epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            documents_rest_part = self._documents[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]

            # Shuffle the data
            if shuffle:
              perm = np.arange(self._num_examples)
              np.random.shuffle(perm)
              self._documents = self.documents[perm]
              self._labels = self.labels[perm]

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            documents_new_part = self._documents[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((documents_rest_part, documents_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._documents[start:end], self._labels[start:end]



